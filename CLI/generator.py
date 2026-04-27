
import torch
import numpy as np
import os
from typing import Union
from config import ModelConfig, DiffusionConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, MissionEncoder
from mesh_utils import voxels_to_stl
from geometry import TypedAircraftGeometry, AircraftPart
from constraints import ConstraintProjector

class OptimizedAircraftGenerator:
    """Optimized inference engine with mission-conditioned 4-step generation"""

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = ModelConfig(**checkpoint['model_config'])
        self.diffusion_config = DiffusionConfig(**checkpoint['diffusion_config'])

        self.diffusion_model = LatentDiffusionUNet(self.model_config, self.diffusion_config).to(self.device)
        self.converter = LatentTo3DConverter(self.model_config.latent_dim, self.model_config.grid_resolution).to(self.device)
        self.consistency_model = ConsistencyModel(self.model_config, self.diffusion_config).to(self.device)

        # Load mission encoder (Required for conditioned generation)
        if 'mission_encoder' not in checkpoint:
            raise RuntimeError("Checkpoint missing 'mission_encoder'. This model does not support mission-conditioned generation.")

        self.mission_encoder = MissionEncoder(self.model_config.condition_dim).to(self.device)
        self.mission_encoder.load_state_dict(checkpoint['mission_encoder'])

        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.converter.load_state_dict(checkpoint['converter'])

        self.diffusion_model.eval()
        self.converter.eval()
        self.mission_encoder.eval()
        self.consistency_model.student_model.eval()

    @torch.no_grad()
    def generate(self, mission: Union[MissionProfile, DesignSpec], num_steps: int = 4, initial_noise: torch.Tensor = None, return_typed: bool = False) -> Union[torch.Tensor, TypedAircraftGeometry]:
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

        # Apply Constraint Projection (Issue #16)
        # In a real model, the converter would output multi-channel.
        # Here we heuristically split the single-channel output for semantic labeling.
        typed_geom = TypedAircraftGeometry(self.model_config.grid_resolution, device=self.device)

        # Heuristic semantic labeling for synthetic model
        res = self.model_config.grid_resolution
        mid = res // 2

        # Central core is fuselage
        fuselage_mask = voxel_grid.clone()
        y_coords = torch.arange(res, device=self.device).view(1, res, 1)
        z_coords = torch.arange(res, device=self.device).view(res, 1, 1)
        dist_axis = torch.sqrt((y_coords - mid)**2 + (z_coords - mid)**2)
        is_fuselage = dist_axis < (res // 6)

        typed_geom.set_part_mask(AircraftPart.FUSELAGE, voxel_grid * is_fuselage.float())
        typed_geom.set_part_mask(AircraftPart.WING, voxel_grid * (~is_fuselage).float())

        # Add a synthetic spar for structural checks
        spar_mask = torch.zeros_like(voxel_grid)
        spar_mask[mid, :, mid] = voxel_grid[mid, :, mid]
        typed_geom.set_part_mask(AircraftPart.SPAR, spar_mask)

        projector = ConstraintProjector(self.model_config.grid_resolution, device=self.device)
        typed_geom = projector.project(typed_geom, mission)

        if return_typed:
            return typed_geom

        # Return combined occupancy for backward compatibility
        return typed_geom.get_combined_occupancy()

    def save_stl(self, voxel_grid, output_path, use_marching_cubes=True):
        voxels_to_stl(voxel_grid, output_path, resolution=self.model_config.grid_resolution, use_marching_cubes=use_marching_cubes)
