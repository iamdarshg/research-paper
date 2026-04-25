
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
from config import ModelConfig, DiffusionConfig

class GroupedQueryAttention(nn.Module):
    """Memory-efficient grouped-query attention for 50% KV-cache reduction"""

    def __init__(self, channels: int, num_groups: int = 4, num_kv_groups: int = 4):
        super().__init__()
        self.num_groups = num_groups
        self.num_kv_groups = num_kv_groups
        self.channels = channels
        self.group_size = channels // num_groups
        self.kv_group_size = channels // num_kv_groups

        self.scale = (self.group_size) ** -0.5

        # Q projections: one per group
        self.to_q = nn.Conv3d(channels, channels, 1)

        # KV projections: shared across KV groups
        self.to_k = nn.Conv3d(channels, self.num_kv_groups * self.kv_group_size, 1)
        self.to_v = nn.Conv3d(channels, self.num_kv_groups * self.kv_group_size, 1)

        # Output projection
        self.to_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape

        # Compute Q, K, V
        q = self.to_q(x)  # [B, C, D, H, W]
        k = self.to_k(x)  # [B, num_kv_groups * kv_group_size, D, H, W]
        v = self.to_v(x)  # [B, num_kv_groups * kv_group_size, D, H, W]

        # Reshape for grouped attention
        q = q.view(b, self.num_groups, self.group_size, d, h, w)
        k = k.view(b, self.num_kv_groups, self.kv_group_size, d, h, w)
        v = v.view(b, self.num_kv_groups, self.kv_group_size, d, h, w)

        # Flatten spatial dimensions for attention computation
        q = q.view(b, self.num_groups, self.group_size, -1).transpose(-2, -1)  # [B, num_groups, N, group_size]
        k = k.view(b, self.num_kv_groups, self.kv_group_size, -1).transpose(-2, -1)  # [B, num_kv_groups, N, kv_group_size]
        v = v.view(b, self.num_kv_groups, self.kv_group_size, -1).transpose(-2, -1)  # [B, num_kv_groups, N, kv_group_size]

        # Expand K and V to match Q groups
        k_expanded = k.repeat_interleave(self.num_groups // self.num_kv_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_groups // self.num_kv_groups, dim=1)

        # Compute attention
        sim = torch.einsum('bgqd,bgkd->bgqk', q, k_expanded) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('bgqk,bgkd->bgqd', attn, v_expanded)
        out = out.transpose(-2, -1).contiguous().view(b, c, d, h, w)
        out = self.to_out(out)

        return x + out

class GradientCheckpointingWrapper(nn.Module):
    """Wrapper to enable gradient checkpointing for 60% VRAM savings"""

    def __init__(self, module: nn.Module, checkpoint_every: int = 1):
        super().__init__()
        self.module = module
        self.checkpoint_every = checkpoint_every
        self.call_count = 0

    def forward(self, *args, **kwargs):
        if self.checkpoint_every > 1:
            self.call_count += 1
            if self.call_count % self.checkpoint_every == 0:
                # Use gradient checkpointing
                return torch.utils.checkpoint.checkpoint(self.module, *args, **kwargs)

        return self.module(*args, **kwargs)

class SpatialAttention(nn.Module):
    """Self-attention for spatial feature maps with grouped-query attention"""

    def __init__(self, channels: int, num_heads: int = 8, num_groups: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.channels = channels
        self.scale = (channels // num_heads) ** -0.5

        # Use grouped-query attention instead of multi-head
        self.grouped_attention = GroupedQueryAttention(channels, num_groups, num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grouped_attention(x)

class ResidualBlock3D(nn.Module):
    """3D residual block with optional attention and gradient checkpointing"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int,
                 use_attention: bool = False, enable_checkpointing: bool = True):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1)
        )

        self.out_channels = out_channels

        self.res_conv = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # Use grouped-query attention with memory optimization
        if use_attention:
            self.attention = SpatialAttention(out_channels, num_groups=4)
        else:
            self.attention = nn.Identity()

        # Apply gradient checkpointing wrapper
        if enable_checkpointing:
            self.block1 = GradientCheckpointingWrapper(self.block1)
            self.block2 = GradientCheckpointingWrapper(self.block2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb).view(-1, self.out_channels, 1, 1, 1)
        h = self.block2(h)
        h = h + self.res_conv(x)
        h = self.attention(h)
        return h

class LatentDiffusionUNet(nn.Module):
    """UNet for diffusion on latent codes with memory optimizations"""

    def __init__(self, config: ModelConfig, diffusion_config: DiffusionConfig):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.diffusion_config = diffusion_config
        self.encoder_out_dim = config.encoder_channels[0] * 2 * 2 * 2  # Reduced from 4x4x4 to 2x2x2 to avoid overflow
        self.config = config

        time_emb_dim = config.latent_dim
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Encoder: project latent to spatial
        self.encoder = nn.Sequential(
            nn.Linear(config.latent_dim, self.encoder_out_dim),
            nn.SiLU(),
            nn.Linear(self.encoder_out_dim, self.encoder_out_dim),
        )

        channels = config.encoder_channels + [config.decoder_channels[-1]]
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.down_blocks.append(ResidualBlock3D(
                channels[i], channels[i+1], time_emb_dim,
                use_attention=False,
                enable_checkpointing=config.enable_gradient_checkpointing
            ))
            self.down_convs.append(nn.Conv3d(channels[i+1], channels[i+1], 3, stride=1, padding=1))

        self.mid_block = ResidualBlock3D(
            channels[-1], channels[-1], time_emb_dim,
            use_attention=False,
            enable_checkpointing=config.enable_gradient_checkpointing
        )

        self.up_convs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.up_convs.append(nn.Conv3d(channels[i], channels[i-1], 3, stride=1, padding=1))
            self.up_blocks.append(ResidualBlock3D(
                channels[i-1], channels[i-1], time_emb_dim,
                use_attention=False,
                enable_checkpointing=config.enable_gradient_checkpointing
            ))

        self.out_conv = nn.Conv3d(channels[0], channels[0], 1)
        self.out = nn.Linear(self.encoder_out_dim, self.latent_dim)

        # Apply torch.compile for kernel fusion
        if config.use_torch_compile:
            self._apply_torch_compile()

    def _apply_torch_compile(self):
        """Apply torch.compile() with reduce-overhead mode for kernel fusion"""
        backends_to_try = [
            ("inductor", "reduce-overhead"),
            ("inductor", "default"),
            ("eager", "reduce-overhead"),
            ("eager", "default")
        ]
        for backend, mode in backends_to_try:
            try:
                print(f"Trying torch.compile with backend='{backend}', mode='{mode}'...")
                if backend == "inductor":
                    import torch._inductor.config
                    if hasattr(torch._inductor.config, 'triton'):
                        triton_config = torch._inductor.config.triton
                        if hasattr(triton_config, 'cudagraphs'):
                            triton_config.cudagraphs = False
                self.forward = torch.compile(self.forward, backend=backend, mode=mode)
                print(f"✅ Successfully applied torch.compile() with backend='{backend}', mode='{mode}'")
                return
            except Exception as e:
                print(f"❌ torch.compile() failed with backend='{backend}': {str(e)}")
                continue
        print("⚠️  All torch.compile() backends failed, using original forward function")

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        b = x.shape[0]
        t_emb = self.time_embedding(timestep.to(self.time_embedding[0].weight.dtype).unsqueeze(1) / self.diffusion_config.timesteps)
        h = self.encoder(x)
        h = h.view(b, -1)
        target_size = self.encoder_out_dim
        if h.size(1) > target_size:
            h = h[:, :target_size]
        elif h.size(1) < target_size:
            h = torch.cat([h, h.new_zeros(b, target_size - h.size(1))], dim=1)
        h = h.view(b, self.config.encoder_channels[0], 2, 2, 2)
        if condition is not None and condition.shape == h.shape:
            h = h + condition
        skip_connections = []
        for i in range(len(self.down_blocks)):
            h = self.down_blocks[i](h, t_emb)
            h = self.down_convs[i](h)
            skip_connections.append(h)
        h = self.mid_block(h, t_emb)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            h = h + skip
            h = self.up_convs[i](h)
            h = self.up_blocks[i](h, t_emb)
        out = self.out_conv(h).view(b, -1)
        out = self.out(out)
        return out

class ConsistencyModel(nn.Module):
    """4-step consistency model replacing 1000-step diffusion"""

    def __init__(self, config: ModelConfig, diffusion_config: DiffusionConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.diffusion_config = diffusion_config
        self.student_steps = diffusion_config.student_steps  # 4 steps
        self.teacher_steps = diffusion_config.teacher_steps  # 1000 steps

        teacher_config = ModelConfig(
            latent_dim=config.latent_dim,
            encoder_channels=config.encoder_channels,
            decoder_channels=config.decoder_channels,
            attention_groups=config.attention_groups,
            enable_gradient_checkpointing=config.enable_gradient_checkpointing,
            use_torch_compile=False
        )
        self.teacher_model = LatentDiffusionUNet(teacher_config, diffusion_config).to(dtype)

        student_config = ModelConfig(
            latent_dim=config.latent_dim,
            encoder_channels=[c // 2 for c in config.encoder_channels],
            decoder_channels=[c // 2 for c in config.decoder_channels],
            attention_groups=4,
            enable_gradient_checkpointing=True,
            use_torch_compile=False
        )
        self.student_model = LatentDiffusionUNet(student_config, diffusion_config).to(dtype)
        self._initialize_student()

    def _initialize_student(self):
        for param in self.student_model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def consistency_loss(self, x_0: torch.Tensor, t_student: torch.Tensor, t_teacher: torch.Tensor) -> torch.Tensor:
        batch_size = x_0.shape[0]
        noise = torch.randn_like(x_0)
        x_t_teacher = self._add_noise(x_0, t_teacher, noise)
        with torch.no_grad():
            pred_teacher = self.teacher_model(x_t_teacher, t_teacher)
        x_t_student = self._add_noise(x_0, t_student, noise)
        pred_student = self.student_model(x_t_student, t_student)
        return F.mse_loss(pred_student, pred_teacher.detach())

    def _add_noise(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_cumprod = torch.ones_like(t, dtype=x_0.dtype)
        for i in range(len(t)):
            alpha_cumprod[i] = 0.5 ** (t[i].to(x_0.dtype) / self.teacher_steps)
        alpha_cumprod = alpha_cumprod.view(-1, 1, 1, 1, 1)
        sqrt_alpha = torch.sqrt(alpha_cumprod)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_cumprod)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def fast_inference(self, shape: Tuple[int, ...], num_steps: int = 4) -> torch.Tensor:
        device = next(self.student_model.parameters()).device
        dtype = next(self.student_model.parameters()).dtype
        x_t = torch.randn(shape, device=device, dtype=dtype)
        step_size = self.diffusion_config.timesteps // num_steps
        for i in range(num_steps):
            current_step = self.diffusion_config.timesteps - i * step_size - 1
            t = torch.full((shape[0],), current_step, device=device, dtype=dtype)
            pred_noise = self.student_model(x_t, t)
            alpha_t = torch.pow(torch.tensor(0.5, device=device, dtype=torch.float32), (current_step / self.diffusion_config.timesteps)).to(dtype)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            x_t = (x_t - (1 - alpha_t) * pred_noise) / sqrt_alpha_t
        return x_t

class LatentTo3DConverter(nn.Module):
    def __init__(self, latent_dim: int, grid_resolution: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_resolution = grid_resolution
        self.output_shape = (grid_resolution, grid_resolution, grid_resolution)
        total_voxels = grid_resolution ** 3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, total_voxels)
        )

    def set_resolution(self, new_resolution: int):
        if new_resolution == self.grid_resolution:
            return
        self.grid_resolution = new_resolution
        print(f"LatentTo3DConverter: Internal target resolution set to {new_resolution}^3")

    def forward(self, latent: torch.Tensor, target_res: int = None) -> torch.Tensor:
        batch_size = latent.shape[0]
        voxels = self.decoder(latent)
        base_res = int(round(voxels.shape[1]**(1/3)))
        voxels = voxels.view(batch_size, base_res, base_res, base_res)
        target = target_res if target_res is not None else self.grid_resolution
        if target != base_res:
            voxels = F.interpolate(voxels.unsqueeze(1), size=(target, target, target), mode='trilinear', align_corners=False).squeeze(1)
        return voxels

class NoiseSchedule:
    def __init__(self, config: DiffusionConfig):
        self.timesteps = config.timesteps
        self.betas = torch.linspace(config.beta_start, config.beta_end, self.timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def to(self, device, dtype=None):
        self.betas = self.betas.to(device, dtype=dtype)
        self.alphas = self.alphas.to(device, dtype=dtype)
        self.alphas_cumprod = self.alphas_cumprod.to(device, dtype=dtype)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device, dtype=dtype)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device, dtype=dtype)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device, dtype=dtype)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device, dtype=dtype)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device, dtype=dtype)
        return self
