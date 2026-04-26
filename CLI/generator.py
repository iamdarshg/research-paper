
import torch
import numpy as np
import os
from typing import Union
from config import ModelConfig, DiffusionConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, MissionEncoder
from mesh_utils import voxels_to_stl

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

        # Load mission encoder if present
        self.mission_encoder = MissionEncoder(self.model_config.condition_dim).to(self.device)
        if 'mission_encoder' in checkpoint:
            self.mission_encoder.load_state_dict(checkpoint['mission_encoder'])
        else:
            print("⚠️ Checkpoint missing mission_encoder. Conditioning may fail or use default weights.")

        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.converter.load_state_dict(checkpoint['converter'])

        self.diffusion_model.eval()
        self.converter.eval()
        self.mission_encoder.eval()
        self.consistency_model.student_model.eval()

    @torch.no_grad()
    def generate(self, mission: Union[MissionProfile, DesignSpec], num_steps: int = 4, initial_noise: torch.Tensor = None) -> torch.Tensor:
        if isinstance(mission, DesignSpec):
            mission = mission.to_mission_profile()

        condition = self.mission_encoder(mission)
        latent_shape = (1, self.model_config.latent_dim)

        print(f"Generating mission-conditioned design ({mission.aircraft_type})")
        generated_latent = self.consistency_model.fast_inference(
            latent_shape,
            num_steps=num_steps,
            condition=condition,
            initial_noise=initial_noise
        )
        voxel_grid = torch.sigmoid(self.converter(generated_latent))
        return voxel_grid.squeeze(0)

    def save_stl(self, voxel_grid, output_path, use_marching_cubes=True):
        voxels_to_stl(voxel_grid, output_path, resolution=self.model_config.grid_resolution, use_marching_cubes=use_marching_cubes)
