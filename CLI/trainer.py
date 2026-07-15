
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import asdict

from config import ModelConfig, DiffusionConfig, TrainingConfig, CFDConfig, DesignSpec, MissionProfile
from models import LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel, NoiseSchedule, MissionEncoder, AeroSurrogate
from data_utils import ConnectivityLoss, AerodynamicLoss, GroundTruthExporter
from cfd_simulator import AdvancedCFDSimulator
from aircraft_diffusion_cfd import LatentTo3DConverter as ScalableLatentTo3DConverter


def _make_grad_scaler(device_type: str):
    """Use the modern AMP GradScaler API when available without breaking older torch versions."""
    enabled = device_type == "cuda"
    amp_namespace = getattr(torch, "amp", None)
    grad_scaler_cls = getattr(amp_namespace, "GradScaler", None) if amp_namespace else None
    if grad_scaler_cls is not None:
        for args, kwargs in (
            ((), {"device": device_type, "enabled": enabled}),
            (((device_type,), {"enabled": enabled})),
            ((), {"enabled": enabled}),
        ):
            try:
                return grad_scaler_cls(*args, **kwargs)
            except TypeError:
                continue

    from torch.cuda.amp import GradScaler as CudaGradScaler
    return CudaGradScaler(enabled=enabled)

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
        self.converter = ScalableLatentTo3DConverter(
            model_config.latent_dim,
            model_config.grid_resolution,
        ).to(self.device).to(self.dtype)
        self.mission_encoder = MissionEncoder(model_config.condition_dim).to(self.device).to(self.dtype)
        self.surrogate = AeroSurrogate(model_config.condition_dim, model_config.grid_resolution).to(self.device).to(self.dtype)

        self.current_grid_size = model_config.grid_resolution
        self.consistency_model = ConsistencyModel(model_config, diffusion_config, self.dtype).to(self.device)
        self.ema_model = self._copy_model(self.diffusion_model)

        params = (list(self.diffusion_model.parameters()) +
                  list(self.converter.parameters()) +
                  list(self.consistency_model.student_model.parameters()) +
                  list(self.mission_encoder.parameters()) +
                  list(self.surrogate.parameters()))
        self.optimizer = AdamW(params, lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=training_config.num_epochs)
        self.scaler = _make_grad_scaler(self.device.type)

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

    def _normalize_mission_batch(self, batch: Dict[str, Any], batch_size: int) -> List[MissionProfile]:
        """Normalize mission data from dataset batch into MissionProfile objects"""
        if 'mission_profile' in batch:
            profiles = batch['mission_profile']

            if isinstance(profiles, MissionProfile):
                return [profiles] * batch_size

            if isinstance(profiles, list):
                if all(isinstance(p, MissionProfile) for p in profiles):
                    if len(profiles) != batch_size:
                         raise ValueError(f"Mission profile batch size mismatch: {len(profiles)} vs {batch_size}")
                    return profiles
                if all(isinstance(p, dict) for p in profiles):
                    return [MissionProfile(**p) for p in profiles]

            if isinstance(profiles, dict):
                # dict of tensors/lists/scalars -> list[MissionProfile]
                out = []
                for i in range(batch_size):
                    kwargs = {}
                    for key, value in profiles.items():
                        if torch.is_tensor(value):
                            kwargs[key] = value[i].item() if value.ndim > 0 else value.item()
                        elif isinstance(value, list):
                            kwargs[key] = value[i]
                        else:
                            kwargs[key] = value
                    out.append(MissionProfile(**kwargs))
                return out

        # Fallback to synthesizing from available fields
        profiles = []
        speeds = batch.get('target_speed', torch.full((batch_size,), 50.0))
        for i in range(batch_size):
            # Safe default stall speed
            cruise = float(speeds[i].item())
            stall = max(0.1, min(0.5 * cruise, cruise - 1.0))
            profiles.append(MissionProfile(cruise_speed_mps=cruise, stall_speed_mps=stall))
        return profiles

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
            batch_size = latent.shape[0]

            mission_profiles = self._normalize_mission_batch(batch, batch_size)
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

            # Online surrogate training (Issue #15)
            surrogate_loss_val = torch.tensor(0.0, device=self.device)
            if len(self.gt_exporter._cache) >= self.training_config.batch_size:
                # Sample a mini-batch from the labels cache
                import random
                samples = random.sample(self.gt_exporter._cache, self.training_config.batch_size)

                # We need to load geometries and targets
                batch_geoms = []
                batch_targets = {'Cd': [], 'Cl': [], 'Cm': [], 'convergence_score': [], 'separation_risk': []}
                batch_missions = []

                for s in samples:
                    geom_path = self.gt_exporter.output_dir / s['geometry_ref']
                    if geom_path.exists():
                        batch_geoms.append(torch.from_numpy(np.load(geom_path)).float())
                        batch_targets['Cd'].append(s.get('cd', 0.0))
                        batch_targets['Cl'].append(s.get('cl', 0.0))
                        batch_targets['Cm'].append(s.get('cm', 0.0) or 0.0)
                        batch_targets['convergence_score'].append(float(s.get('converged', False)))
                        batch_targets['separation_risk'].append(s.get('separation_risk', 0.0))
                        batch_missions.append(MissionProfile(**s.get('mission_profile', {})))

                if batch_geoms:
                    geoms_t = torch.stack(batch_geoms).to(self.device)
                    targets_t = {k: torch.tensor(v, device=self.device, dtype=self.dtype) for k, v in batch_targets.items()}
                    cond_surr = self.mission_encoder(batch_missions)

                    # Track tiers for online training too
                    tiers = [s.get('tier', 'lbm_raw') for s in samples]

                    preds = self.surrogate(geoms_t, cond_surr)
                    surr_loss = F.mse_loss(preds['Cd'], targets_t['Cd']) + \
                                F.mse_loss(preds['Cl'], targets_t['Cl']) + \
                                F.binary_cross_entropy(preds['convergence_score'], targets_t['convergence_score'])

                    # Register step but we backward the total_loss_val later
                    # We manually call train_step-like logic here without a separate optimizer call
                    self.surrogate.is_trained.fill_(True)
                    self.surrogate.sample_count += batch_geoms[0].shape[0] # roughly

                    surrogate_loss_val = surr_loss
                    self.surrogate.last_val_mse.copy_(0.95 * self.surrogate.last_val_mse + 0.05 * surr_loss.detach())

            optimization_loss_val = (mse_loss_val + consist_loss_val + surrogate_loss_val).nan_to_num(0.0)
            diagnostic_total_loss_val = (
                optimization_loss_val.detach()
                + connectivity_loss_val.detach()
                + aero_loss_val.detach()
            ).nan_to_num(0.0)
            self.optimizer.zero_grad()
            optimization_loss_val.backward()
            self.optimizer.step()
            self._update_ema()
            self.global_step += 1
        return {
            'loss': optimization_loss_val.item(),
            'optimization_loss': optimization_loss_val.item(),
            'diagnostic_total': diagnostic_total_loss_val.item(),
            'connectivity': connectivity_loss_val.item(),
            'aerodynamic': aero_loss_val.item(),
        }

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
            'aero_surrogate': self.surrogate.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'model_config': asdict(self.model_config),
            'diffusion_config': asdict(self.diffusion_config),
            'training_config': asdict(self.training_config),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, allow_legacy: bool = False):
        checkpoint = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
        self.consistency_model.load_state_dict(checkpoint['consistency_model'])
        self.converter.load_state_dict(checkpoint['converter'])

        if 'mission_encoder' in checkpoint:
            self.mission_encoder.load_state_dict(checkpoint['mission_encoder'])
        elif not allow_legacy:
            raise RuntimeError("Checkpoint missing 'mission_encoder'. Use allow_legacy=True to skip.")

        if 'aero_surrogate' in checkpoint:
            self.surrogate.load_state_dict(checkpoint['aero_surrogate'])
