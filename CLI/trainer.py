
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any
from dataclasses import asdict

from config import ModelConfig, DiffusionConfig, TrainingConfig, CFDConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, NoiseSchedule, MissionEncoder
from data_utils import ConnectivityLoss, AerodynamicLoss, GroundTruthExporter
from cfd_simulator import AdvancedCFDSimulator

class OptimizedDiffusionTrainer:
    """Main training orchestrator with all TRM/HRM optimizations"""

    def __init__(self, model_config, diffusion_config, training_config, cfd_config, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.training_config = training_config
        self.cfd_config = cfd_config
        self.dtype = getattr(torch, training_config.precision, torch.float32)

        self.noise_schedule = NoiseSchedule(diffusion_config).to(self.device, self.dtype)
        self.diffusion_model = LatentDiffusionUNet(model_config, diffusion_config).to(self.device).to(self.dtype)
        self.converter = LatentTo3DConverter(model_config.latent_dim, model_config.grid_resolution).to(self.device).to(self.dtype)
        self.mission_encoder = MissionEncoder(model_config.condition_dim).to(self.device).to(self.dtype)

        self.current_grid_size = model_config.grid_resolution
        self.consistency_model = ConsistencyModel(model_config, diffusion_config, self.dtype).to(self.device)
        self.ema_model = self._copy_model(self.diffusion_model)

        params = (list(self.diffusion_model.parameters()) +
                  list(self.converter.parameters()) +
                  list(self.consistency_model.student_model.parameters()) +
                  list(self.mission_encoder.parameters()))
        self.optimizer = AdamW(params, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=training_config.num_epochs)
        self.scaler = GradScaler()

        self.mse_loss = nn.MSELoss()
        self.connectivity_loss = ConnectivityLoss(penalty=training_config.disconnection_penalty)
        self.aero_loss = AerodynamicLoss()
        self.cfd_simulator = AdvancedCFDSimulator(cfd_config, self.device)

        import copy
        val_cfd_config = copy.deepcopy(cfd_config)
        val_cfd_config.solver_type = "D3Q27"
        val_cfd_config.use_amr = True
        self.val_cfd_simulator = AdvancedCFDSimulator(val_cfd_config, self.device)
        self.gt_exporter = GroundTruthExporter()
        self.writer = SummaryWriter(log_dir='./runs')
        self.global_step = 0

    def _copy_model(self, model):
        import copy
        return copy.deepcopy(model)

    def _update_ema(self):
        decay = self.training_config.ema_decay
        for ema_param, param in zip(self.ema_model.parameters(), self.diffusion_model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def train_epoch(self, train_loader, grid_size=32):
        if self.current_grid_size != grid_size:
            self.converter.set_resolution(grid_size)
            self.cfd_simulator.set_resolution(grid_size)
            self.current_grid_size = grid_size
        self.diffusion_model.train()
        self.converter.train()
        self.mission_encoder.train()

        pbar = tqdm(train_loader, desc=f"Training Grid {grid_size}")
        for batch_idx, batch in enumerate(pbar):
            latent = batch['latent'].to(self.device, dtype=self.dtype)

            # Generate MissionProfiles for condition encoding
            # In real usage, these would come from the dataset.
            # Load mission profiles from batch if available, else fallback
            batch_size = latent.shape[0]
            if 'mission_profile' in batch:
                 mission_profiles = batch['mission_profile']
            else:
                # Synthesis fallback for testing/initial training
                mission_profiles = []
                speeds = batch.get('target_speed', torch.full((batch_size,), 50.0))
                for i in range(batch_size):
                    mission_profiles.append(MissionProfile(cruise_speed=float(speeds[i].item())))

            condition = self.mission_encoder(mission_profiles)

            voxel_grid = torch.sigmoid(self.converter(latent)).nan_to_num(0.0)
            t = torch.randint(0, self.diffusion_config.timesteps, (batch_size,), device=self.device)
            noise = torch.randn_like(latent)
            noisy_latent = self.noise_schedule.q_sample(latent, t, noise)

            pred_noise = self.diffusion_model(noisy_latent, t, condition=condition)
            mse_loss_val = self.mse_loss(pred_noise, noise)

            # Consistency loss with conditioning
            t_student = torch.randint(0, self.consistency_model.student_steps, (batch_size,), device=self.device)
            t_teacher = t_student * (self.consistency_model.teacher_steps // self.consistency_model.student_steps)
            consist_loss_val = self.consistency_model.consistency_loss(latent, t_student, t_teacher, condition=condition)

            connectivity_loss_val = self.connectivity_loss(voxel_grid)
            aero_loss_val = torch.tensor(0.0, device=self.device)
            if batch_idx % 10 == 0:
                design_spec = DesignSpec(target_speed=grid_size / 32.0 * 50.0)
                aero_loss_val = self.aero_loss(voxel_grid[:1], design_spec, self.cfd_simulator, gt_exporter=self.gt_exporter, sample_prefix=f"grid{grid_size}_step{self.global_step}")

            total_loss_val = mse_loss_val + consist_loss_val + connectivity_loss_val + aero_loss_val
            self.optimizer.zero_grad()
            total_loss_val.backward()
            self.optimizer.step()
            self._update_ema()
            self.global_step += 1
        return {'loss': total_loss_val.item()}

    def train(self, train_loader, val_loader=None):
        from utils import get_vram_limit_resolution
        max_vram_res = get_vram_limit_resolution(max_usage=0.9)
        grid_sizes = [32]
        curr = 64
        while curr <= max_vram_res and curr <= 512:
            grid_sizes.append(curr)
            curr *= 2
        for grid_size in grid_sizes:
            if hasattr(train_loader.dataset, 'set_resolution'):
                train_loader.dataset.set_resolution(grid_size)
            epochs = self.training_config.num_epochs if grid_size == 32 else max(1, self.training_config.num_epochs // 2)
            for epoch in range(epochs):
                self.train_epoch(train_loader, grid_size=grid_size)
            self.scheduler.step()

    def save_checkpoint(self, path):
        checkpoint = {
            'diffusion_model': self.diffusion_model.state_dict(),
            'consistency_model': self.consistency_model.state_dict(),
            'converter': self.converter.state_dict(),
            'mission_encoder': self.mission_encoder.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'model_config': asdict(self.model_config),
            'diffusion_config': asdict(self.diffusion_config),
            'training_config': asdict(self.training_config),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.converter.load_state_dict(checkpoint['converter'])
        if 'mission_encoder' in checkpoint:
            self.mission_encoder.load_state_dict(checkpoint['mission_encoder'])
