#!/usr/bin/env python3
"""
Aircraft Structural Design via Diffusion Models + FluidX3D CFD
Combines TRM/HRM principles with diffusion-based 3D voxel generation,
GPU-accelerated CFD simulation, and marching cubes STL export.

Optimized for 8-13GB VRAM with pipelined training and inference.
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import yaml
from scipy.ndimage import label, binary_dilation
from skimage import measure

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG & DATACLASSES
# ============================================================================

@dataclass
class DiffusionConfig:
    """Diffusion model hyperparameters"""
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    sampling_timesteps: int = 250
    guidance_scale: float = 7.5
    
@dataclass
class ModelConfig:
    """Model architecture parameters"""
    latent_dim: int = 128
    xyz_dim: int = 3
    encoder_channels: List[int] = None
    decoder_channels: List[int] = None
    attention_heads: int = 8
    num_attention_layers: int = 4
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [64, 128, 256]
        if self.decoder_channels is None:
            self.decoder_channels = [256, 128, 64]

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    ema_decay: float = 0.999
    disconnection_penalty: float = 10.0
    use_mixed_precision: bool = True
    save_interval: int = 5
    val_interval: int = 1

@dataclass
class CFDConfig:
    """FluidX3D simulation parameters"""
    resolution: int = 32  # Will be doubled during training
    mach_number: float = 0.3
    reynolds_number: float = 1e6
    simulation_steps: int = 500
    output_interval: int = 50
    device_id: int = 0

@dataclass
class DesignSpec:
    """Aircraft design specification"""
    target_speed: float  # m/s
    space_weight: float = 0.33
    drag_weight: float = 0.33
    lift_weight: float = 0.34
    bounding_box: Tuple[int, int, int] = (64, 64, 64)
    vital_components: np.ndarray = None  # Sparse matrix with component locations


# ============================================================================
# NOISE SCHEDULING & DIFFUSION UTILITIES
# ============================================================================

class NoiseSchedule:
    """Linear noise schedule for diffusion"""
    
    def __init__(self, config: DiffusionConfig):
        self.timesteps = config.timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion process: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise"""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        return self


# ============================================================================
# ARCHITECTURE: LATENT DIFFUSION + 3D CONVERTER
# ============================================================================

class SpatialAttention(nn.Module):
    """Self-attention for spatial feature maps"""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.scale = (channels // num_heads) ** -0.5
        
        self.to_qkv = nn.Conv3d(channels, channels * 3, 1)
        self.to_out = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        
        qkv = self.to_qkv(x)
        qkv = qkv.view(b, self.num_heads, -1, d * h * w)
        q, k, v = qkv.chunk(3, dim=2)
        
        sim = torch.einsum('bhcd,bhce->bhde', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = torch.einsum('bhde,bhce->bhcd', attn, v)
        out = out.view(b, c, d, h, w)
        out = self.to_out(out)
        
        return x + out


class ResidualBlock3D(nn.Module):
    """3D residual block with optional attention"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, use_attention: bool = False):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = SpatialAttention(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb).view(-1, -1, 1, 1, 1)
        h = self.block2(h)
        h = h + self.res_conv(x)
        h = self.attention(h)
        return h


class LatentDiffusionUNet(nn.Module):
    """UNet for diffusion on latent codes"""
    
    def __init__(self, config: ModelConfig, diffusion_config: DiffusionConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.diffusion_config = diffusion_config
        
        time_emb_dim = config.latent_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder: project latent to spatial
        self.encoder = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )
        
        channels = [64, 128, 256]
        self.down_layers = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.down_layers.append(nn.Sequential(
                ResidualBlock3D(channels[i], channels[i+1], time_emb_dim, use_attention=(i > 0)),
                nn.Conv3d(channels[i+1], channels[i+1], 4, stride=2, padding=1)
            ))
        
        self.mid_block = ResidualBlock3D(channels[-1], channels[-1], time_emb_dim, use_attention=True)
        
        self.up_layers = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_layers.append(nn.Sequential(
                nn.ConvTranspose3d(channels[i], channels[i-1], 4, stride=2, padding=1),
                ResidualBlock3D(channels[i-1], channels[i-1], time_emb_dim, use_attention=(i > 1))
            ))
        
        self.out_conv = nn.Conv3d(channels[0], 1, 1)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, latent_dim] - noisy latent codes
        timestep: [B] - diffusion timesteps
        condition: [B, C, D, H, W] - optional spatial conditioning
        """
        b = x.shape[0]
        
        t_emb = self.time_embedding(timestep.float().unsqueeze(1) / self.diffusion_config.timesteps)
        
        # Expand latent to 3D spatial (8x8x8)
        h = self.encoder(x)  # [B, 512]
        h = h.view(b, 1, 8, 8, 8).expand(-1, 1, -1, -1, -1)
        
        if condition is not None:
            # Adaptive average pooling to match spatial dims
            h = h + condition
        
        # U-Net forward pass
        skip_connections = []
        for down_layer in self.down_layers:
            h = down_layer(h)
            skip_connections.append(h)
        
        h = self.mid_block(h, t_emb)
        
        for up_layer in reversed(self.up_layers):
            h = up_layer(h)
        
        out = self.out_conv(h)
        return out


class LatentTo3DConverter(nn.Module):
    """Convert n-dimensional latent codes to 3D spatial representation"""
    
    def __init__(self, latent_dim: int, output_shape: Tuple[int, int, int] = (32, 32, 32)):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        total_voxels = np.prod(output_shape)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, total_voxels)
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert latent code to voxel grid"""
        batch_size = latent.shape[0]
        voxels = self.decoder(latent)  # [B, total_voxels]
        voxels = voxels.view(batch_size, *self.output_shape)
        return voxels


# ============================================================================
# CFD SIMULATION (FluidX3D-like GPU CFD)
# ============================================================================

class SimplifiedCFDSimulator:
    """
    Simplified GPU-accelerated CFD for aircraft aerodynamics.
    Uses grid-based lattice Boltzmann-inspired approach.
    """
    
    def __init__(self, config: CFDConfig, device: torch.device):
        self.config = config
        self.device = device
        self.resolution = config.resolution
        
        # Initialize flow field
        self.init_flow_field()
    
    def init_flow_field(self):
        """Initialize flow field for incompressible flow"""
        self.velocity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=self.device)
        self.pressure = torch.zeros(1, self.resolution, self.resolution, self.resolution, device=self.device)
        
        # Freestream conditions
        self.velocity[0, :, :, :] = self.config.mach_number * 343.0  # Speed of sound at sea level
    
    def simulate_aerodynamics(self, geometry: torch.Tensor, steps: int = 100) -> Dict[str, float]:
        """
        Simulate flow around geometry and compute aerodynamic coefficients.
        geometry: [D, H, W] binary voxel grid (1 = solid, 0 = fluid)
        """
        h = 1.0  # Grid spacing
        dt = 0.01  # Time step
        
        device = geometry.device
        geom_expanded = geometry.unsqueeze(0).float()
        
        total_drag = 0.0
        total_lift = 0.0
        
        for step in range(min(steps, self.config.simulation_steps)):
            # Enforce boundary conditions at solid cells
            boundary_mask = geom_expanded
            self.velocity = self.velocity * (1 - boundary_mask)
            
            # Simple pressure-based correction (approximation)
            grad_p = torch.zeros_like(self.velocity)
            grad_p[:, 1:, :, :] -= self.pressure[:, 1:, :, :] - self.pressure[:, :-1, :, :]
            grad_p[:, :, 1:, :] -= self.pressure[:, :, 1:, :] - self.pressure[:, :, :-1, :]
            grad_p[:, :, :, 1:] -= self.pressure[:, :, :, 1:] - self.pressure[:, :, :, :-1]
            
            # Update velocity (simplified Navier-Stokes)
            self.velocity = self.velocity + dt * grad_p / 1.225  # Air density
            self.velocity[0, :, :, :] = torch.clamp(self.velocity[0, :, :, :], min=self.config.mach_number * 343.0 * 0.99)
            
            # Compute forces on boundary
            if step % self.config.output_interval == 0:
                # Force estimation from pressure difference
                drag_pressure = (self.pressure * boundary_mask).sum().item()
                lift_pressure = (self.pressure[:, :, self.resolution//2:, :] * boundary_mask[:, :, self.resolution//2:, :]).sum().item()
                
                total_drag += abs(drag_pressure)
                total_lift += abs(lift_pressure)
        
        # Normalize by reference area (simplified)
        ref_area = self.resolution ** 2
        ref_dynamic_pressure = 0.5 * 1.225 * (self.config.mach_number * 343.0) ** 2
        
        cd = total_drag / (ref_dynamic_pressure * ref_area + 1e-6) if total_drag > 0 else 0.1
        cl = total_lift / (ref_dynamic_pressure * ref_area + 1e-6) if total_lift > 0 else 0.0
        
        return {
            'drag_coefficient': min(cd, 1.0),
            'lift_coefficient': cl,
            'pressure_sum': self.pressure.sum().item()
        }


# ============================================================================
# DATASET & DATA LOADING
# ============================================================================

class AircraftDesignDataset(Dataset):
    """Synthetic dataset for aircraft structure training"""
    
    def __init__(self, num_samples: int = 100, grid_size: int = 32, seed: int = 42):
        self.num_samples = num_samples
        self.grid_size = grid_size
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.latent_codes = torch.randn(num_samples, 128)  # Random latent codes
        self.geometries = self._generate_geometries()
    
    def _generate_geometries(self) -> List[torch.Tensor]:
        """Generate synthetic aircraft geometries"""
        geometries = []
        for i in range(self.num_samples):
            # Create fuselage-like structure
            geom = torch.zeros(self.grid_size, self.grid_size, self.grid_size)
            
            # Central fuselage
            cx, cy, cz = self.grid_size // 2, self.grid_size // 2, self.grid_size // 2
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    for z in range(self.grid_size):
                        dist_center = ((x - cx) ** 2 + (z - cz) ** 2) ** 0.5
                        if dist_center < 6 and 10 < y < 22:
                            geom[x, y, z] = 1.0
            
            # Wings
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    for z in range(self.grid_size):
                        if 8 < y < 24 and (z < 4 or z > self.grid_size - 4):
                            geom[x, y, z] = 1.0
            
            # Add some noise for variation
            noise = torch.rand_like(geom)
            geom = (geom + 0.1 * noise > 0.5).float()
            
            geometries.append(geom)
        
        return geometries
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'latent': self.latent_codes[idx],
            'geometry': self.geometries[idx],
            'target_speed': torch.tensor(self.grid_size / 32 * 50.0)  # Normalized speed
        }


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ConnectivityLoss(nn.Module):
    """Penalize disconnected voxel groups"""
    
    def __init__(self, penalty: float = 10.0):
        super().__init__()
        self.penalty = penalty
    
    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """Compute connectivity penalty for batch of voxel grids"""
        batch_size = voxel_grid.shape[0]
        total_penalty = 0.0
        
        for b in range(batch_size):
            geom = (voxel_grid[b] > 0.5).int().cpu().numpy()
            
            # Label connected components
            labeled, num_components = label(geom)
            
            if num_components > 1:
                # Count voxels in each component
                component_sizes = np.bincount(labeled.flatten())
                
                # Largest component should dominate
                largest_size = component_sizes[1:].max() if num_components > 1 else 0
                total_size = geom.sum()
                
                if largest_size > 0:
                    disconnected_fraction = (total_size - largest_size) / (total_size + 1e-6)
                    total_penalty += disconnected_fraction
        
        return self.penalty * total_penalty / batch_size if batch_size > 0 else torch.tensor(0.0, device=voxel_grid.device)


class AerodynamicLoss(nn.Module):
    """Loss based on aerodynamic properties"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, voxel_grid: torch.Tensor, design_spec: DesignSpec, cfd_results: Dict) -> torch.Tensor:
        """
        Compute aerodynamic loss balancing drag, lift, and volume.
        """
        if not cfd_results:
            return torch.tensor(0.0, device=voxel_grid.device)
        
        batch_size = voxel_grid.shape[0]
        loss = torch.tensor(0.0, device=voxel_grid.device)
        
        for b in range(batch_size):
            # Volume penalty (space weight)
            volume = voxel_grid[b].sum() / np.prod(voxel_grid.shape[1:])
            volume_loss = design_spec.space_weight * volume
            
            # Drag coefficient penalty (drag weight)
            cd = cfd_results.get('drag_coefficient', 0.1)
            drag_loss = design_spec.drag_weight * cd
            
            # Lift coefficient encouragement (lift weight - we want nonzero but not excessive)
            cl = abs(cfd_results.get('lift_coefficient', 0.0))
            lift_loss = design_spec.lift_weight * (1.0 - torch.clamp(torch.tensor(cl), 0, 1))
            
            loss += volume_loss + drag_loss + lift_loss
        
        return loss / batch_size


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class DiffusionTrainer:
    """Main training orchestrator with pipelined execution"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        training_config: TrainingConfig,
        cfd_config: CFDConfig,
        device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        self.cfd_config = cfd_config
        
        self.noise_schedule = NoiseSchedule(diffusion_config).to(self.device)
        
        # Models
        self.diffusion_model = LatentDiffusionUNet(model_config, diffusion_config).to(self.device)
        self.converter = LatentTo3DConverter(model_config.latent_dim, (32, 32, 32)).to(self.device)
        
        # Initialize EMA model
        self.ema_model = self._copy_model(self.diffusion_model)
        
        # Optimizer
        params = list(self.diffusion_model.parameters()) + list(self.converter.parameters())
        self.optimizer = AdamW(params, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=training_config.num_epochs)
        
        # Losses
        self.mse_loss = nn.MSELoss()
        self.connectivity_loss = ConnectivityLoss(penalty=training_config.disconnection_penalty)
        self.aero_loss = AerodynamicLoss()
        
        # CFD simulator
        self.cfd_simulator = SimplifiedCFDSimulator(cfd_config, self.device)
        
        # Logging
        self.writer = SummaryWriter(log_dir='./runs')
        self.global_step = 0
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create an independent copy of the model"""
        import copy
        return copy.deepcopy(model)
    
    def _update_ema(self):
        """Update exponential moving average model"""
        decay = self.training_config.ema_decay
        for ema_param, param in zip(self.ema_model.parameters(), self.diffusion_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def train_epoch(self, train_loader: DataLoader, grid_size: int = 32) -> Dict[str, float]:
        """Train for one epoch with specified grid size (progressive training)"""
        self.diffusion_model.train()
        self.converter.train()
        
        total_loss = 0.0
        total_mse = 0.0
        total_connectivity = 0.0
        total_aero = 0.0
        
        pbar = tqdm(train_loader, desc=f"Training (grid={grid_size}x{grid_size}x{grid_size})")
        
        for batch_idx, batch in enumerate(pbar):
            latent = batch['latent'].to(self.device)
            geometry_target = batch['geometry'].to(self.device)
            
            # Resize geometry to current grid size
            if grid_size != geometry_target.shape[1]:
                geometry_target = F.interpolate(
                    geometry_target.unsqueeze(1),
                    size=(grid_size, grid_size, grid_size),
                    mode='nearest'
                ).squeeze(1)
            
            # Convert latent to voxel grid
            voxel_grid = self.converter(latent)
            voxel_grid = torch.sigmoid(voxel_grid)  # Normalize to [0, 1]
            
            # Random timestep for diffusion training
            t = torch.randint(0, self.diffusion_config.timesteps, (latent.shape[0],), device=self.device)
            
            # Forward diffusion
            noise = torch.randn_like(latent)
            noisy_latent = self.noise_schedule.q_sample(latent, t, noise)
            
            # Model prediction
            pred_noise = self.diffusion_model(noisy_latent, t)
            
            # MSE loss
            mse_loss_val = self.mse_loss(pred_noise, noise)
            
            # Connectivity loss
            connectivity_loss_val = self.connectivity_loss(voxel_grid)
            
            # CFD-based aerodynamic loss (computed on low-res grid for speed)
            aero_loss_val = torch.tensor(0.0, device=self.device)
            if batch_idx % 5 == 0:  # Compute aerodynamics every 5 batches
                design_spec = DesignSpec(target_speed=50.0)
                geom_aero = (voxel_grid[0] > 0.5)
                if geom_aero.sum() > 10:  # Only if enough voxels
                    cfd_results = self.cfd_simulator.simulate_aerodynamics(geom_aero, steps=50)
                    aero_loss_val = self.aero_loss(voxel_grid[:1], design_spec, cfd_results)
            
            # Combined loss
            total_loss_val = mse_loss_val + connectivity_loss_val + aero_loss_val
            
            # Backward
            self.optimizer.zero_grad()
            total_loss_val.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.training_config.gradient_clip)
            torch.nn.utils.clip_grad_norm_(self.converter.parameters(), self.training_config.gradient_clip)
            self.optimizer.step()
            
            # EMA update
            self._update_ema()
            
            # Logging
            total_loss += total_loss_val.item()
            total_mse += mse_loss_val.item()
            total_connectivity += connectivity_loss_val.item()
            total_aero += aero_loss_val.item()
            
            pbar.set_postfix({
                'loss': total_loss_val.item(),
                'mse': mse_loss_val.item(),
                'conn': connectivity_loss_val.item(),
                'aero': aero_loss_val.item()
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/total', avg_loss, self.global_step)
        self.writer.add_scalar('Loss/mse', total_mse / len(train_loader), self.global_step)
        self.writer.add_scalar('Loss/connectivity', total_connectivity / len(train_loader), self.global_step)
        self.writer.add_scalar('Loss/aerodynamic', total_aero / len(train_loader), self.global_step)
        
        return {
            'loss': avg_loss,
            'mse': total_mse / len(train_loader),
            'connectivity': total_connectivity / len(train_loader),
            'aerodynamic': total_aero / len(train_loader)
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Progressive training: start small, scale up"""
        grid_sizes = [16, 24, 32]  # Progressive grid refinement
        
        for grid_size in grid_sizes:
            print(f"\n{'='*60}")
            print(f"Training with grid size: {grid_size}x{grid_size}x{grid_size}")
            print(f"{'='*60}\n")
            
            epochs = self.training_config.num_epochs if grid_size == 32 else self.training_config.num_epochs // 2
            
            for epoch in range(epochs):
                print(f"\nGrid {grid_size} - Epoch {epoch + 1}/{epochs}")
                metrics = self.train_epoch(train_loader, grid_size=grid_size)
                
                print(f"Epoch {epoch + 1} Metrics: {metrics}")
                
                if (epoch + 1) % self.training_config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_grid{grid_size}_ep{epoch+1}.pt')
            
            self.scheduler.step()
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'diffusion_model': self.diffusion_model.state_dict(),
            'converter': self.converter.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'model_config': asdict(self.model_config),
            'diffusion_config': asdict(self.diffusion_config),
            'training_config': asdict(self.training_config),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.converter.load_state_dict(checkpoint['converter'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {path}")


# ============================================================================
# INFERENCE & MARCHING CUBES
# ============================================================================

class AircraftGenerator:
    """Inference engine for aircraft design generation"""
    
    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model_config = ModelConfig(**checkpoint['model_config'])
        self.diffusion_config = DiffusionConfig(**checkpoint['diffusion_config'])
        
        self.diffusion_model = LatentDiffusionUNet(self.model_config, self.diffusion_config).to(self.device)
        self.converter = LatentTo3DConverter(self.model_config.latent_dim, (32, 32, 32)).to(self.device)
        
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.converter.load_state_dict(checkpoint['converter'])
        
        self.noise_schedule = NoiseSchedule(self.diffusion_config).to(self.device)
        
        self.diffusion_model.eval()
        self.converter.eval()
    
    @torch.no_grad()
    def generate(self, design_spec: DesignSpec, num_steps: int = 250, guidance_scale: float = 7.5) -> torch.Tensor:
        """
        Generate aircraft design via reverse diffusion process.
        """
        latent_shape = (1, self.model_config.latent_dim)
        x_t = torch.randn(latent_shape, device=self.device)
        
        timesteps = np.linspace(self.diffusion_config.timesteps - 1, 0, num_steps).astype(int)
        
        print("Reverse diffusion process:")
        for t in tqdm(timesteps):
            t_tensor = torch.tensor([t], device=self.device)
            
            # Predict noise
            pred_noise = self.diffusion_model(x_t, t_tensor)
            
            # DDIM sampling
            alpha = self.noise_schedule.alphas_cumprod[t]
            alpha_prev = self.noise_schedule.alphas_cumprod_prev[t]
            
            sigma = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            c = (1 / torch.sqrt(alpha_prev)) * (1 - alpha / torch.sqrt(alpha * (1 - alpha)))
            
            x_t = (1 / torch.sqrt(alpha)) * (x_t - (1 - alpha) / torch.sqrt(1 - alpha) * pred_noise) + c * torch.randn_like(x_t)
        
        # Convert latent to geometry
        voxel_grid = self.converter(x_t)
        voxel_grid = torch.sigmoid(voxel_grid)
        
        return voxel_grid.squeeze(0)
    
    def voxels_to_stl(self, voxel_grid: torch.Tensor, output_path: str, use_marching_cubes: bool = True):
        """Convert voxel grid to STL file using marching cubes"""
        
        # Convert to numpy
        voxel_np = voxel_grid.cpu().numpy()
        
        # Threshold to get binary grid
        binary_grid = (voxel_np > 0.5).astype(np.float32)
        
        if use_marching_cubes:
            print("Applying marching cubes algorithm...")
            try:
                vertices, faces, normals, values = measure.marching_cubes(
                    binary_grid,
                    level=0.5,
                    spacing=(1.0, 1.0, 1.0)
                )
                
                print(f"Generated mesh: {len(vertices)} vertices, {len(faces)} faces")
                
                self._write_stl(output_path, vertices, faces)
                print(f"STL file written to {output_path}")
            except Exception as e:
                print(f"Marching cubes failed: {e}. Writing voxel representation instead.")
                self._write_voxel_stl(output_path, binary_grid)
        else:
            self._write_voxel_stl(output_path, binary_grid)
    
    def _write_stl(self, path: str, vertices: np.ndarray, faces: np.ndarray):
        """Write mesh to binary STL file"""
        with open(path, 'wb') as f:
            # Header
            f.write(b'\0' * 80)
            # Number of triangles
            f.write(np.uint32(len(faces)).tobytes())
            
            # Write each triangle
            for face in faces:
                v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                
                f.write(normal.astype(np.float32).tobytes())
                f.write(v0.astype(np.float32).tobytes())
                f.write(v1.astype(np.float32).tobytes())
                f.write(v2.astype(np.float32).tobytes())
                f.write(b'\0\0')  # Attribute byte count
    
    def _write_voxel_stl(self, path: str, binary_grid: np.ndarray):
        """Write voxel grid as STL cubes"""
        triangles = []
        
        for x in range(binary_grid.shape[0]):
            for y in range(binary_grid.shape[1]):
                for z in range(binary_grid.shape[2]):
                    if binary_grid[x, y, z] > 0.5:
                        # Create cube at this voxel
                        vertices = np.array([
                            [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
                            [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
                        ], dtype=np.float32)
                        
                        # Cube face indices
                        faces = [
                            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
                            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
                        ]
                        
                        for face in faces:
                            triangles.append(vertices[face])
        
        if triangles:
            triangles = np.array(triangles)
            vertices = triangles.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            self._write_stl(path, vertices, faces)


# ============================================================================
# CLI INTERFACE
# ============================================================================

import click

@click.group()
def cli():
    """Aircraft Structural Design via Diffusion Models + CFD"""
    pass

@cli.command()
@click.option('--num-epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=128, help='Batch size')
@click.option('--learning-rate', default=2e-4, help='Learning rate')
@click.option('--latent-dim', default=128, help='Latent dimension')
@click.option('--disconnection-penalty', default=50.0, help='Penalty for disconnected voxels')
@click.option('--num-samples', default=500, help='Number of training samples')
@click.option('--resume-from', default=None, help='Resume from checkpoint')
@click.option('--save-dir', default='./checkpoints', help='Directory to save checkpoints')
def train(num_epochs, batch_size, learning_rate, latent_dim, disconnection_penalty, num_samples, resume_from, save_dir):
    """Train the diffusion model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Configs
    model_config = ModelConfig(latent_dim=latent_dim)
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        disconnection_penalty=disconnection_penalty
    )
    cfd_config = CFDConfig(resolution=16)
    
    # Dataset
    dataset = AircraftDesignDataset(num_samples=num_samples, grid_size=32)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Trainer
    trainer = DiffusionTrainer(model_config, diffusion_config, training_config, cfd_config, device=device)
    
    if resume_from:
        trainer.load_checkpoint(resume_from)
        print(f"Resumed from {resume_from}")
    
    # Train
    trainer.train(train_loader)
    
    # Save final model
    final_checkpoint = os.path.join(save_dir, 'final_model.pt')
    trainer.save_checkpoint(final_checkpoint)
    print(f"Training complete. Final model saved to {final_checkpoint}")

@cli.command()
@click.option('--checkpoint', required=True, help='Path to model checkpoint')
@click.option('--output', default='aircraft.stl', help='Output STL file path')
@click.option('--target-speed', default=50.0, help='Target aircraft speed (m/s)')
@click.option('--num-steps', default=250, help='Number of diffusion steps for generation')
@click.option('--use-marching-cubes', is_flag=True, default=True, help='Use marching cubes for STL conversion')
def generate(checkpoint, output, target_speed, num_steps, use_marching_cubes):
    """Generate aircraft design and export to STL"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found at {checkpoint}")
        sys.exit(1)
    
    # Load generator
    print(f"Loading checkpoint from {checkpoint}...")
    generator = AircraftGenerator(checkpoint, device=device)
    
    # Design specification
    design_spec = DesignSpec(
        target_speed=target_speed,
        space_weight=0.33,
        drag_weight=0.33,
        lift_weight=0.34
    )
    
    # Generate
    print(f"Generating aircraft design with {num_steps} diffusion steps...")
    voxel_grid = generator.generate(design_spec, num_steps=num_steps)
    
    print(f"Generated voxel grid shape: {voxel_grid.shape}")
    print(f"Occupied voxels: {(voxel_grid > 0.5).sum().item()} / {np.prod(voxel_grid.shape)}")
    
    # Export to STL
    print(f"Converting to STL mesh...")
    generator.voxels_to_stl(voxel_grid, output, use_marching_cubes=use_marching_cubes)

@cli.command()
@click.option('--checkpoint', required=True, help='Path to model checkpoint')
@click.option('--output-dir', default='./generations', help='Output directory for generated designs')
@click.option('--num-designs', default=5, help='Number of designs to generate')
def batch_generate(checkpoint, output_dir, num_designs):
    """Generate multiple aircraft designs"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Loading checkpoint from {checkpoint}...")
    generator = AircraftGenerator(checkpoint, device=device)
    
    design_spec = DesignSpec(target_speed=50.0)
    
    for i in range(num_designs):
        print(f"\nGenerating design {i+1}/{num_designs}...")
        voxel_grid = generator.generate(design_spec, num_steps=250)
        
        output_path = os.path.join(output_dir, f'aircraft_{i+1:03d}.stl')
        generator.voxels_to_stl(voxel_grid, output_path, use_marching_cubes=True)

@cli.command()
def info():
    """Print system and CUDA information"""
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

if __name__ == '__main__':
    cli()
