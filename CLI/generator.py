
import torch
import numpy as np
import os
from datetime import datetime
from dataclasses import asdict
from typing import Union, Dict, Any, Optional, Tuple
import torch.nn.functional as F
from config import ModelConfig, DiffusionConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, MissionEncoder, AeroSurrogate
from mesh_utils import voxels_to_stl
from geometry import TypedAircraftGeometry, AircraftPart
from constraints import ConstraintProjector, ConstraintReport

class SemanticAdapter:
    """Heuristic adapter to convert single-channel model output to TypedAircraftGeometry (Issue #16).

    WARNING: This is a heuristic interpretation layer. In future versions, the model
    should be trained to output multi-channel semantic geometry directly.
    """
    def __init__(self, resolution: int, device: torch.device):
        self.res = resolution
        self.device = device

    def adapt(self, voxel_grid: torch.Tensor) -> TypedAircraftGeometry:
        """Splits a single-channel occupancy grid into semantic parts based on geometric heuristics."""
        typed_geom = TypedAircraftGeometry(self.res, device=self.device)

        mid = self.res // 2

        # 1. Fuselage: Central core
        y_coords = torch.arange(self.res, device=self.device).view(1, self.res, 1)
        z_coords = torch.arange(self.res, device=self.device).view(self.res, 1, 1)
        dist_axis = torch.sqrt((y_coords - mid)**2 + (z_coords - mid)**2)
        is_fuselage = dist_axis < (self.res // 6)

        # 2. Wing: Everything else that is solid
        is_wing = ~is_fuselage

        # Apply masks
        typed_geom.set_part_mask(AircraftPart.FUSELAGE, voxel_grid * is_fuselage.float())
        typed_geom.set_part_mask(AircraftPart.WING, voxel_grid * is_wing.float())

        # 3. Skin: Boundary layer of solid volume
        skin_mask = voxel_grid - (F.max_pool3d(-voxel_grid.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1) * -1).squeeze()
        typed_geom.set_part_mask(AircraftPart.SKIN, skin_mask)

        # 4. Spar: Structural spine (heuristic)
        spar_mask = torch.zeros_like(voxel_grid)
        spar_mask[mid, :, mid] = voxel_grid[mid, :, mid]
        typed_geom.set_part_mask(AircraftPart.SPAR, spar_mask)

        return typed_geom

class OptimizedAircraftGenerator:
    """Optimized inference engine with mission-conditioned 4-step generation and semantic constraints."""

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_payload = dict(checkpoint['model_config'])
        if 'grid_resolution' not in model_payload and 'base_grid_resolution' not in model_payload:
            converter_state = checkpoint.get('converter', {})
            output_bias = converter_state.get('decoder.4.bias')
            if output_bias is not None:
                voxel_count = int(output_bias.numel())
                inferred_side = int(round(voxel_count ** (1.0 / 3.0)))
                if inferred_side ** 3 != voxel_count:
                    raise ValueError(
                        "Legacy dense checkpoint converter output is not a cubic voxel lattice"
                    )
                model_payload['base_grid_resolution'] = inferred_side
                model_payload['grid_resolution'] = inferred_side
        self.model_config = ModelConfig(**model_payload)
        self.diffusion_config = DiffusionConfig(**checkpoint['diffusion_config'])

        self.diffusion_model = LatentDiffusionUNet(self.model_config, self.diffusion_config).to(self.device)
        self.converter = LatentTo3DConverter(self.model_config.latent_dim, self.model_config.grid_resolution).to(self.device)
        self.consistency_model = ConsistencyModel(self.model_config, self.diffusion_config).to(self.device)

        if 'mission_encoder' not in checkpoint:
            raise RuntimeError("Checkpoint missing 'mission_encoder'.")

        self.mission_encoder = MissionEncoder(self.model_config.condition_dim).to(self.device)
        self.mission_encoder.load_state_dict(checkpoint['mission_encoder'], strict=False)

        self.consistency_model.load_state_dict(checkpoint['consistency_model'], strict=False)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'], strict=False)
        self.converter.load_state_dict(checkpoint['converter'], strict=False)

        self.diffusion_model.eval()
        self.converter.eval()
        self.mission_encoder.eval()
        self.consistency_model.student_model.eval()

        self.adapter = SemanticAdapter(self.model_config.grid_resolution, self.device)
        self.surrogate = AeroSurrogate(self.model_config.condition_dim, self.model_config.grid_resolution).to(self.device)
        self.surrogate_mission_encoder = MissionEncoder(self.model_config.condition_dim).to(self.device)

        # Default to main encoder if no standalone surrogate is loaded
        self.surrogate_mission_encoder.load_state_dict(self.mission_encoder.state_dict())

        if 'aero_surrogate' in checkpoint:
            self.surrogate.load_state_dict(checkpoint['aero_surrogate'], strict=False)

    @torch.no_grad()
    def generate(self, mission: Union[MissionProfile, DesignSpec], num_steps: int = 4,
                 initial_noise: torch.Tensor = None, return_typed: bool = False,
                 existing_report: ConstraintReport = None,
                 num_candidates: int = 1, top_k: int = 1,
                 return_results: bool = False,
                 external_val_mode: str = 'none') -> Any:

        if isinstance(mission, DesignSpec):
            mission = mission.to_mission_profile()

        condition = self.mission_encoder(mission).to(self.device)
        if initial_noise is not None:
            initial_noise = initial_noise.to(self.device)

        if num_candidates > 1:
            return self.generate_candidates(mission, condition, num_steps, num_candidates, top_k, return_typed, existing_report, return_results, external_val_mode)

        latent_shape = (1, self.model_config.latent_dim)
        print(f"Generating mission-conditioned design ({mission.aircraft_class})")
        generated_latent = self.consistency_model.fast_inference(
            latent_shape,
            num_steps=num_steps,
            condition=condition,
            initial_noise=initial_noise
        )
        generated_latent = generated_latent.to(self.device)
        voxel_grid = torch.sigmoid(self.converter(generated_latent))
        voxel_grid = voxel_grid.squeeze(0)

        # 1. Adapt to semantic parts
        typed_geom = self.adapter.adapt(voxel_grid)

        # 2. Project and repair constraints
        projector = ConstraintProjector(self.model_config.grid_resolution, device=self.device, existing_report=existing_report)
        typed_geom = projector.project(typed_geom, mission)

        results = None
        if return_results:
            # For num_candidates=1, we still need to simulate if results are requested
            from cfd_simulator import AdvancedCFDSimulator
            from config import CFDConfig
            from data_utils import GroundTruthExporter
            from dataclasses import replace

            # Apply external validation trigger (Review Feedback)
            # Use replace to avoid in-place mutation of the original mission object
            sim_mission = mission
            if external_val_mode in ('final', 'top-k'):
                sim_mission = replace(mission, force_external_validation=True)

            sim_config = CFDConfig(base_grid_resolution=self.model_config.grid_resolution)
            simulator = AdvancedCFDSimulator(sim_config, self.device)
            results = simulator.simulate_aerodynamics(typed_geom, steps=100, mission=sim_mission, existing_report=existing_report)

            # Export single-candidate results to GroundTruthExporter (Review Feedback Fix 3)
            exporter = GroundTruthExporter()
            run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
            sample_id = f"gen_single_{mission.aircraft_class}_{run_id}"
            # Ensure exact sim_mission is exported to preserve forced validation metadata
            exporter.export_sample(sample_id, typed_geom.get_combined_occupancy(), results.get('velocity_fields'), results.get('pressure_field'), results | {'mission': asdict(sim_mission)})

        final_geom = typed_geom if return_typed else typed_geom.get_combined_occupancy()

        if return_results:
            return final_geom, results

        return final_geom

    @torch.no_grad()
    def generate_candidates(self, mission, condition, num_steps, num_candidates, top_k, return_typed, existing_report, return_results=False, external_val_mode='none'):
        """Sample many candidates, rank with surrogate, and validate top-k (Issue #15)."""
        from datetime import datetime, timezone
        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        print(f"Sampling {num_candidates} candidates for ranking (Run ID: {run_timestamp})...")
        latent_shape = (num_candidates, self.model_config.latent_dim)

        # Separate condition for ranking (Isolated surrogate encoder)
        surr_condition = self.surrogate_mission_encoder(mission).to(self.device)

        # 1. Sample many
        latents = self.consistency_model.fast_inference(
            latent_shape,
            num_steps=num_steps,
            condition=condition.repeat(num_candidates, 1)
        )
        latents = latents.to(self.device)
        voxel_grids = torch.sigmoid(self.converter(latents))

        # 2. Batch Project (Heuristic project for all)
        projected_occupancies = []
        typed_candidates = []
        projected_reports = []

        for i in range(num_candidates):
            # Re-init projector per candidate to ensure total isolation (Review Feedback)
            cand_report = ConstraintReport()
            projector = ConstraintProjector(self.model_config.grid_resolution, device=self.device, existing_report=cand_report)

            tg = self.adapter.adapt(voxel_grids[i])
            tg = projector.project(tg, mission)

            typed_candidates.append(tg)
            projected_occupancies.append(tg.get_combined_occupancy())
            projected_reports.append(cand_report)

        projected_batch = torch.stack(projected_occupancies)

        # 3. Rank with surrogate
        min_samples = self.model_config.surrogate_min_samples
        max_loss = self.model_config.surrogate_max_loss

        if not self.surrogate.is_ready(min_samples=min_samples, max_loss=max_loss):
            print(f"⚠️ AeroSurrogate is not ready (Samples: {self.surrogate.sample_count.item()}, Loss EMA: {self.surrogate.train_loss_ema.item():.4f}).")
            print(f"Evaluating ALL {min(top_k, num_candidates)} candidates directly with D3Q27 (Review Feedback Fix 2/5).")
            # When surrogate is not ready, skip pre-ranking and just take first k
            top_indices = np.arange(min(top_k, num_candidates))
        else:
            print(f"Ranking candidates with AeroSurrogate (Loss EMA: {self.surrogate.train_loss_ema.item():.4f})...")
            scores = self.surrogate.rank(projected_batch, surr_condition.repeat(num_candidates, 1))
            # 4. Select top-k
            top_indices = torch.topk(scores, min(top_k, num_candidates)).indices.cpu().numpy()
            print(f"Selected top {len(top_indices)} candidates for D3Q27 validation.")

        from cfd_simulator import AdvancedCFDSimulator
        from config import CFDConfig
        from data_utils import GroundTruthExporter

        sim_config = CFDConfig(base_grid_resolution=self.model_config.grid_resolution)
        simulator = AdvancedCFDSimulator(sim_config, self.device)
        exporter = GroundTruthExporter()

        best_results = None
        best_geom = None
        best_idx = -1

        from dataclasses import replace
        for idx in top_indices:
            tg = typed_candidates[idx]
            cand_report = projected_reports[idx]

            # Handle staged external validation (Review Feedback)
            # Use replace to avoid cross-candidate side effects
            if external_val_mode == 'top-k':
                sim_mission = replace(mission, force_external_validation=True)
            else:
                sim_mission = replace(mission, force_external_validation=False)

            # 5. Run D3Q27 on top candidates
            print(f"Validating candidate {idx} with D3Q27...")
            res = simulator.simulate_aerodynamics(tg, steps=100, mission=sim_mission, existing_report=cand_report)

            # 6. Save results to reusable label dataset (Fix 4: unique IDs)
            sample_id = f"gen_{mission.aircraft_class}_{run_timestamp}_{idx}"
            # Use a more compatible way to merge dicts for safety
            # Export exact sim_mission to preserve forced validation metadata (Review Feedback)
            metadata = {**res, 'mission': asdict(sim_mission)}
            exporter.export_sample(sample_id, tg.get_combined_occupancy(), res.get('velocity_fields'), res.get('pressure_field'), metadata)

            # 7. Feasibility-aware selection (Fix 5)
            is_feasible = res['constraints'].get('valid', True)
            is_converged = res.get('lbm_converged', False)

            def calculate_overall_score(r):
                # Prefer feasible and converged. Tie-break with Cd.
                f_score = 100.0 if r['constraints'].get('valid', True) else 0.0
                c_score = 50.0 if r.get('lbm_converged', False) else 0.0
                drag_score = max(0.0, 1.0 - r['drag_coefficient'])
                return f_score + c_score + drag_score

            current_score = calculate_overall_score(res)

            if best_results is None or current_score > calculate_overall_score(best_results):
                best_results = res
                best_geom = tg
                best_idx = idx

        # Final Selected Candidate External Validation (if mode is 'final')
        if external_val_mode == 'final' and best_results:
            # Check if it was already externally validated (it wouldn't be if mode was 'final')
            if best_results.get('label_tier') != 'external_pde':
                print(f"🚀 Running final external PDE validation for selected candidate...")
                sim_mission = replace(mission, force_external_validation=True)
                best_results = simulator.simulate_aerodynamics(best_geom, steps=100, mission=sim_mission, existing_report=projected_reports[best_idx])

                # Re-export to the SAME ID to leverage GroundTruthExporter promotion logic (Review Feedback)
                # This ensures multi-fidelity history is maintained in one record.
                sample_id = f"gen_{mission.aircraft_class}_{run_timestamp}_{best_idx}"
                exporter.export_sample(sample_id, best_geom.get_combined_occupancy(), best_results.get('velocity_fields'), best_results.get('pressure_field'), best_results | {'mission': asdict(sim_mission)})

        if existing_report is not None and best_results:
            # Use the selected candidate's isolated report for final status (Review Feedback)
            best_report = projected_reports[best_idx]
            existing_report.violations = best_report.violations
            existing_report.repaired = best_report.repaired
            existing_report.metrics = best_report.metrics
            existing_report.export_status = best_report.export_status

        final_geom = best_geom if return_typed else best_geom.get_combined_occupancy()

        if return_results:
            return final_geom, best_results

        return final_geom

    def load_surrogate(self, surrogate_path: str):
        """Load a standalone surrogate checkpoint (Issue #15 Review Feedback)."""
        # Isolated loading: Do not corrupt the main mission_encoder (diffusion path)
        print(f"Loading standalone surrogate from {surrogate_path}...")
        checkpoint = torch.load(surrogate_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.surrogate.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.surrogate.load_state_dict(checkpoint)

        if 'encoder_state_dict' in checkpoint:
            self.surrogate_mission_encoder.load_state_dict(checkpoint['encoder_state_dict'])

    def save_stl(self, voxel_grid: Union[torch.Tensor, TypedAircraftGeometry], output_path: str,
                 use_marching_cubes: bool = True, report: ConstraintReport = None):
        """Saves geometry to STL, performing watertight checks and reporting violations (Issue #16)."""
        if isinstance(voxel_grid, TypedAircraftGeometry):
            voxel_grid = voxel_grid.get_combined_occupancy()

        # We'll use a specialized version of voxels_to_stl that handles reports
        from mesh_utils import voxels_to_stl_checked
        return voxels_to_stl_checked(voxel_grid, output_path, resolution=self.model_config.grid_resolution,
                                    use_marching_cubes=use_marching_cubes, report=report)
