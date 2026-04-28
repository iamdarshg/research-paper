
import torch
import numpy as np
import os
from typing import Union, Dict, Any, Optional, Tuple
import torch.nn.functional as F
from config import ModelConfig, DiffusionConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, MissionEncoder
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
        self.model_config = ModelConfig(**checkpoint['model_config'])
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

    @torch.no_grad()
    def generate(self, mission: Union[MissionProfile, DesignSpec], num_steps: int = 4,
                 initial_noise: torch.Tensor = None, return_typed: bool = False,
                 existing_report: ConstraintReport = None) -> Union[torch.Tensor, TypedAircraftGeometry, Tuple[TypedAircraftGeometry, ConstraintReport]]:

        if isinstance(mission, DesignSpec):
            mission = mission.to_mission_profile()

        condition = self.mission_encoder(mission)
        latent_shape = (1, self.model_config.latent_dim)

        print(f"Generating mission-conditioned design ({mission.aircraft_class})")
        generated_latent = self.consistency_model.fast_inference(
            latent_shape,
            num_steps=num_steps,
            condition=condition,
            initial_noise=initial_noise
        )
        voxel_grid = torch.sigmoid(self.converter(generated_latent))
        voxel_grid = voxel_grid.squeeze(0)

        # 1. Adapt to semantic parts
        typed_geom = self.adapter.adapt(voxel_grid)

        # 2. Project and repair constraints
        projector = ConstraintProjector(self.model_config.grid_resolution, device=self.device, existing_report=existing_report)
        typed_geom = projector.project(typed_geom, mission)

        if return_typed:
            return typed_geom

        return typed_geom.get_combined_occupancy()

    def save_stl(self, voxel_grid: Union[torch.Tensor, TypedAircraftGeometry], output_path: str,
                 use_marching_cubes: bool = True, report: ConstraintReport = None):
        """Saves geometry to STL, performing watertight checks and reporting violations (Issue #16)."""
        if isinstance(voxel_grid, TypedAircraftGeometry):
            voxel_grid = voxel_grid.get_combined_occupancy()

        # We'll use a specialized version of voxels_to_stl that handles reports
        from mesh_utils import voxels_to_stl_checked
        return voxels_to_stl_checked(voxel_grid, output_path, resolution=self.model_config.grid_resolution,
                                    use_marching_cubes=use_marching_cubes, report=report)
