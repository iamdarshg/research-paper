
import torch
import numpy as np
import os
from config import ModelConfig, DiffusionConfig, DesignSpec
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel
from mesh_utils import voxels_to_stl

class OptimizedAircraftGenerator:
    """Optimized inference engine with 4-step generation"""

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_config = ModelConfig(**checkpoint['model_config'])
        self.diffusion_config = DiffusionConfig(**checkpoint['diffusion_config'])

        self.diffusion_model = LatentDiffusionUNet(self.model_config, self.diffusion_config).to(self.device)
        self.converter = LatentTo3DConverter(self.model_config.latent_dim, self.model_config.grid_resolution).to(self.device)
        self.consistency_model = ConsistencyModel(self.model_config, self.diffusion_config).to(self.device)
        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.converter.load_state_dict(checkpoint['converter'])

        self.diffusion_model.eval()
        self.converter.eval()
        self.consistency_model.student_model.eval()

    @torch.no_grad()
    def generate(self, design_spec: DesignSpec, num_steps: int = 4) -> torch.Tensor:
        latent_shape = (1, self.model_config.latent_dim)
        print(f"Generating with 4-step consistency model")
        generated_latent = self.consistency_model.fast_inference(latent_shape, num_steps=num_steps)
        voxel_grid = torch.sigmoid(self.converter(generated_latent))
        return voxel_grid.squeeze(0)

    def save_stl(self, voxel_grid, output_path, use_marching_cubes=True):
        voxels_to_stl(voxel_grid, output_path, resolution=self.model_config.grid_resolution, use_marching_cubes=use_marching_cubes)
