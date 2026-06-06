
import torch
import numpy as np
from typing import Dict
from typing import TYPE_CHECKING

from lbm_utils import D3Q27Lattice, _compute_force_coefficients
from lbm_diagnostics import compute_strain_rate_tensor, compute_vorticity, compute_velocity_gradients
from sdf_utils import compute_all_link_distances
from lbm_logger import LBMLogger
from utils import compute_tensor_content_hash

try:
    from d3q27_kernels import stream_bounce_d3q27
except Exception:  # pragma: no cover - optional acceleration path
    stream_bounce_d3q27 = None


def _scale_momentum_exchange_force(force, grid_spacing: float, mach_number: float, density: float = 1.225):
    """Convert raw lattice momentum exchange into a physical force scale (Issue #16).

    Standard LBM force scaling: F_phys = rho_phys * (dx^4 / dt^2) * F_lattice.
    Using consistent dt derived from sound speed: dt = dx / (343.0 * sqrt(3)).
    This results in force_scale = rho_phys * dx^2 * (343.0 * sqrt(3))^2.

    The D3Q27 momentum exchange sum corresponds to Delta p / Delta t in lattice units.
    We apply the c_s^2 = 1/3 factor to align with the physical dynamic pressure definition.
    """
    dx = float(grid_spacing)
    # velocity_ratio = sound_speed_phys / sound_speed_lattice = 343.0 * sqrt(3)
    velocity_ratio = 343.0 * np.sqrt(3.0)
    force_scale = float(density) * (dx**2) * (velocity_ratio**2)
    return (1.0 / 3.0) * force * force_scale


class D3Q27Solver:
    """Complete D3Q27 LBM solver with vectorized Cascaded MRT collision operator."""
    def __init__(
        self,
        resolution,
        device,
        inlet_velocity_lu: float = 0.0,
        use_triton_streaming: bool = False,
    ):
        self.res = resolution
        self.device = device
        self.inlet_velocity_lu = float(inlet_velocity_lu)

        self.ex, self.ey, self.ez = D3Q27Lattice.get_vectors()
        self.ex = self.ex.to(device, dtype=torch.long)
        self.ey = self.ey.to(device, dtype=torch.long)
        self.ez = self.ez.to(device, dtype=torch.long)
        self.ex_f = self.ex.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.ey_f = self.ey.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.ez_f = self.ez.to(dtype=torch.float32).view(-1, 1, 1, 1)
        self.w = D3Q27Lattice.get_weights().to(device)
        self.opposite = D3Q27Lattice.get_opposite().to(device)
        self._opposite_list = D3Q27Lattice.get_opposite().tolist()
        self._stream_shifts = [
            (int(dx), int(dy), int(dz))
            for dx, dy, dz in zip(self.ex.tolist(), self.ey.tolist(), self.ez.tolist())
        ]
        self._force_dirs = [i for i in range(27) if i != 0]
        self._force_dir_index = torch.tensor(self._force_dirs, dtype=torch.long, device=device)
        self._force_ex = self.ex[self._force_dir_index].to(dtype=torch.float32).view(-1, 1, 1, 1)
        self._force_ez = self.ez[self._force_dir_index].to(dtype=torch.float32).view(-1, 1, 1, 1)
        self._force_speed = torch.sqrt(
            self.ex[self._force_dir_index].to(dtype=torch.float32)**2
            + self.ey[self._force_dir_index].to(dtype=torch.float32)**2
            + self.ez[self._force_dir_index].to(dtype=torch.float32)**2
        ).view(-1, 1, 1, 1)

        # Precompute moment basis for vectorized MRT Collision (Issue #16)
        self.moment_keys = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        basis_rows = []
        ex_f_vec = self.ex.to(dtype=torch.float32)
        ey_f_vec = self.ey.to(dtype=torch.float32)
        ez_f_vec = self.ez.to(dtype=torch.float32)
        for a, b, c in self.moment_keys:
            basis_rows.append((ex_f_vec ** a) * (ey_f_vec ** b) * (ez_f_vec ** c))
        self.moment_basis = torch.stack(basis_rows, dim=0).to(device)
        self.moment_basis_inv = torch.inverse(self.moment_basis)

        # Precompute relaxation indices for MRT
        # 0: Conserved, 1: Energy, 2: Shear, 3: High-order
        self.s_indices = torch.zeros(27, dtype=torch.long, device=device)
        self.conserved_indices = []
        for i, (a, b, c) in enumerate(self.moment_keys):
            # Conserved moments: rho(000), jx(100), jy(010), jz(001)
            if (a == 0 and b == 0 and c == 0) or (a == 1 and b == 0 and c == 0) or \
               (a == 0 and b == 1 and c == 0) or (a == 0 and b == 0 and c == 1):
                self.s_indices[i] = 0
                self.conserved_indices.append(i)
            elif (a, b, c) in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]:
                self.s_indices[i] = 1  # Energy
            elif (a, b, c) in [(1, 1, 0), (1, 0, 1), (0, 1, 1)]:
                self.s_indices[i] = 2  # Shear (determines viscosity)
            else:
                self.s_indices[i] = 3  # High-order

        self.conserved_indices = torch.tensor(self.conserved_indices, dtype=torch.long, device=device)
        self.s_e = 1.2
        self.s_h = 1.6

        # Cache masks for vectorized boundary population updates
        self._inlet_mask = self.ex > 0
        self._outlet_mask = self.ex < 0

        self.drag_link_metric_exponent = None
        self._boundary_cache_key = None
        self._boundary_link_cache = None
        self.use_triton_streaming = bool(use_triton_streaming and stream_bounce_d3q27 is not None and device.type == "cuda")

        # 27 populations
        self.f = torch.zeros(27, resolution, resolution, resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)

        # Macroscopic fields stored for diagnostics/synchronization
        self.velocity_x = torch.zeros(resolution, resolution, resolution, device=device)
        self.velocity_y = torch.zeros_like(self.velocity_x)
        self.velocity_z = torch.zeros_like(self.velocity_x)
        self.pressure = torch.zeros_like(self.velocity_x)
        self.rho = torch.ones_like(self.velocity_x)

        self.reset_force_accounting()
        self.logger = LBMLogger()

        # Cache for BFL distances keyed by geometry hash (Fix A)
        self._q_cache = {}

        # Precompute Guo directions
        self.ei_guo = torch.stack([self.ex_f, self.ey_f, self.ez_f], dim=1).view(27, 3, 1, 1, 1)

    def _get_q(self, geometry_mask, geom_hash=None):
        """Get or compute sub-voxel distances for the given geometry (Fix A/Issue #15)."""
        # Fix A: Use a true content hash for the geometry key (Review Feedback)
        # Issue #22: Support pre-computed hash to reduce CPU-GPU sync overhead
        geom_key = geom_hash if geom_hash is not None else compute_tensor_content_hash(geometry_mask)

        if geom_key not in self._q_cache:
            # CPU/SciPy cost is explicit here.
            self._q_cache[geom_key] = compute_all_link_distances(geometry_mask, self.ex, self.ey, self.ez)

        # Optional: Limit cache size to prevent OOM
        if len(self._q_cache) > 100:
            self._q_cache.pop(next(iter(self._q_cache)))

        return self._q_cache[geom_key]

    def compute_equilibrium(self, rho, ux, uy, uz):
        cu = self.ex_f * ux + self.ey_f * uy + self.ez_f * uz
        u_sq = ux**2 + uy**2 + uz**2
        return self.w.view(-1, 1, 1, 1) * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

    def compute_moment_equilibrium(self, rho, ux, uy, uz):
        """Equilibrium tensor-product moments for D3Q27 (Issue #16)."""
        cs2 = 1.0 / 3.0
        m1d_x = [torch.ones_like(rho), ux, ux * ux + cs2]
        m1d_y = [torch.ones_like(rho), uy, uy * uy + cs2]
        m1d_z = [torch.ones_like(rho), uz, uz * uz + cs2]

        keq_list = []
        for a, b, c in self.moment_keys:
            keq_list.append(rho * m1d_x[a] * m1d_y[b] * m1d_z[c])
        return torch.stack(keq_list, dim=0)

    def reset_force_accounting(self, sample_start: int = 0):
        """Reset momentum-exchange bookkeeping for a new simulation run."""
        self.force_x_accum = torch.tensor(0.0, device=self.device)
        self.force_z_accum = torch.tensor(0.0, device=self.device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=self.device)
        self.force_z_last = torch.tensor(0.0, device=self.device)
        self.projected_drag_accum = torch.tensor(0.0, device=self.device)
        self.projected_drag_last = torch.tensor(0.0, device=self.device)
        self._force_sample_start = max(0, int(sample_start))
        self._force_step = 0

    def _accumulate_momentum_exchange_force(self, geometry_mask, geom_hash=None):
        """Compute wall force from fluid-solid links using bounce-back exchange and BFL correction."""
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)

        # BFL wall distance q (shape: [26, D, H, W])
        q = self._get_q(geometry_mask, geom_hash=geom_hash)

        step_force_x = torch.tensor(0.0, device=self.device)
        step_force_z = torch.tensor(0.0, device=self.device)

        for i in range(1, 27):
            link_idx = i - 1
            active = boundary_links[link_idx]
            if not torch.any(active):
                continue

            qi = q[i][active]

            # Momentum exchange for BFL:
            # The simple BB exchange is 2 * fi_in.
            # For BFL, we use the interpolated population that actually hits the wall.
            # Weighting factor 4/(1+2q) ensures consistency with bounce-back (2*fi) at q=0.5.
            # Simplified version: Use q to scale the momentum transfer.
            bfl_factor = 4.0 / (1.0 + 2.0 * qi)

            f_in = self.f_pre_stream[i][active]
            step_force_x += torch.sum(bfl_factor * self.ex[i] * f_in)
            step_force_z += torch.sum(bfl_factor * self.ez[i] * f_in)

        return step_force_x, step_force_z

    def _effective_drag_link_metric_exponent(self, geometry_mask):
        if self.drag_link_metric_exponent is not None:
            return float(self.drag_link_metric_exponent)
        projected_cells = torch.sum(torch.any(geometry_mask > 0.5, dim=0).float()).item()
        projected_side = float(np.sqrt(max(projected_cells, 1.0)))
        return float(np.clip(1.68 - 0.295 * (projected_side - 13.0), 0.5, 1.68))

    def _accumulate_projected_pressure_drag_proxy(self, geometry_mask, geom_hash=None):
        """Coarse-grid pressure-drag proxy from upwind-facing D3Q27 wall links."""
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)
        flow_sign = 1.0 if self.inlet_velocity_lu >= 0.0 else -1.0
        upwind = (self._force_ex * flow_sign) > 0.0
        metric_exponent = self._effective_drag_link_metric_exponent(geometry_mask)
        metric = torch.pow(self._force_speed.clamp_min(1.0), -metric_exponent)
        projected = 2.0 * torch.abs(self._force_ex) * metric * self.f_pre_stream[self._force_dir_index]
        return torch.sum(torch.where(upwind, projected * boundary_links, torch.zeros_like(projected)))

    def _boundary_links(self, geometry_mask, geom_hash=None):
        """Cache static fluid-solid links without boundary wraparound (Issue #15)."""
        # Use true content hash for cache key (Review Feedback)
        # Issue #22: Support pre-computed hash to reduce CPU-GPU sync overhead
        cache_key = geom_hash if geom_hash is not None else compute_tensor_content_hash(geometry_mask)
        if cache_key == self._boundary_cache_key and self._boundary_link_cache is not None:
            return self._boundary_link_cache

        mask = geometry_mask > 0.5

        links = []
        fluid = ~mask
        D, H, W = mask.shape
        # Pad mask to detect neighbors without wraparound
        mask_padded = torch.nn.functional.pad(mask, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        for i in self._force_dirs:
            dx, dy, dz = self._stream_shifts[i]
            # Neighbor at (x+dx, y+dy, z+dz).
            # In mask_padded, this corresponds to (1+x+dx, 1+y+dy, 1+z+dz)
            d_s, d_e = 1+dx, 1+dx+D
            h_s, h_e = 1+dy, 1+dy+H
            w_s, w_e = 1+dz, 1+dz+W
            neighbor_is_solid = mask_padded[d_s:d_e, h_s:h_e, w_s:w_e]
            links.append(fluid & neighbor_is_solid)

        self._boundary_link_cache = torch.stack(links, dim=0)
        self._boundary_cache_key = cache_key
        return self._boundary_link_cache

    def _apply_domain_boundaries(self):
        """Apply vectorized non-periodic domain boundaries (Equilibrium Inlet, Neumann Outlet)."""
        # Equilibrium Inlet at X=0
        if self.inlet_velocity_lu != 0.0:
            inlet_shape = self.f[:, 0, :, :].shape[1:]
            rho_in = torch.ones(inlet_shape, device=self.device, dtype=self.f.dtype)
            ux_in = torch.full_like(rho_in, self.inlet_velocity_lu)
            uy_in = torch.zeros_like(rho_in)
            uz_in = torch.zeros_like(rho_in)

            cu = (
                self.ex.to(dtype=self.f.dtype).view(-1, 1, 1) * ux_in
                + self.ey.to(dtype=self.f.dtype).view(-1, 1, 1) * uy_in
                + self.ez.to(dtype=self.f.dtype).view(-1, 1, 1) * uz_in
            )
            u_sq = ux_in**2 + uy_in**2 + uz_in**2
            feq_in = self.w.to(dtype=self.f.dtype).view(-1, 1, 1) * rho_in * (
                1 + 3 * cu + 4.5 * cu**2 - 1.5 * u_sq
            )
            # Vectorized inlet update: overwrite populations streaming INTO the domain
            self.f_temp[self._inlet_mask, 0, :, :] = feq_in[self._inlet_mask]

        # Vectorized Neumann (Zero-Gradient) Outlet at X=-1 (Issue #16 fix)
        # Use interior plane after streaming but before BC for true zero-gradient
        self.f_temp[self._outlet_mask, -1, :, :] = self.f_temp[self._outlet_mask, -2, :, :]

        # Slip Walls (Mirror) or Neumann for other boundaries - Vectorized slices
        self.f_temp[:, :, 0, :] = self.f_temp[:, :, 1, :]
        self.f_temp[:, :, -1, :] = self.f_temp[:, :, -2, :]
        self.f_temp[:, :, :, 0] = self.f_temp[:, :, :, 1]
        self.f_temp[:, :, :, -1] = self.f_temp[:, :, :, -2]

    def _apply_bfl_boundary(self, geometry_mask, geom_hash=None):
        """Bouzidi-Firdaouss-Lallemand (BFL) boundary condition."""
        q = self._get_q(geometry_mask, geom_hash=geom_hash)

        # Standard bounce-back links
        boundary_links = self._boundary_links(geometry_mask, geom_hash=geom_hash)

        # Issue #22: Optimize padding by padding the entire tensor once if possible,
        # but since directions vary, we still need to shift.
        # We can at least pad the spatial dimensions of f_pre_stream once.
        # self.f_pre_stream: [27, D, H, W]
        D, H, W = geometry_mask.shape
        f_pre_padded = torch.nn.functional.pad(self.f_pre_stream, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

        for i in range(1, 27):
            opp_i = self._opposite_list[i]
            link_idx = i - 1

            # Identify boundary links
            active = boundary_links[link_idx]
            if not torch.any(active):
                continue

            qi = q[i][active]

            # Identify neighbors without wraparound (Issue #15)
            dx, dy, dz = self._stream_shifts[i]

            # Neighbor at x-e_i is at (1-dx, 1-dy, 1-dz) in padded coords
            # f_pre_padded has shape [27, D+2, H+2, W+2]
            f_neighbor = f_pre_padded[i, 1-dx:1-dx+D, 1-dy:1-dy+H, 1-dz:1-dz+W][active]

            # BFL interpolation based on q
            q_low = qi < 0.5
            q_high = ~q_low

            # Simplified BFL Linear Interpolation
            # f_opp(x, t+1) = (1-2q)f_i(x, t) + 2q f_i(x, t)_bb (if q < 0.5)
            # f_opp(x, t+1) = (1 - 1/(2q))f_opp(x, t) + (1/(2q))f_i(x, t) (if q >= 0.5)

            res = torch.zeros_like(qi)

            # q < 0.5
            if torch.any(q_low):
                res[q_low] = (1 - 2*qi[q_low]) * f_neighbor[q_low] + 2*qi[q_low] * self.f_pre_stream[i][active][q_low]

            # q >= 0.5
            if torch.any(q_high):
                res[q_high] = (1 / (2*qi[q_high])) * self.f_pre_stream[i][active][q_high] + (1 - 1/(2*qi[q_high])) * self.f_temp[opp_i][active][q_high]

            self.f_temp[opp_i][active] = res

    def _compute_guo_forcing(self, rho, u, F, omega):
        """Guo's forcing source term."""
        # F is the external force field [3, D, H, W]
        # u is macroscopic velocity
        # omega is 1/tau
        cs2 = 1.0 / 3.0

        # Precompute u.F
        uF = torch.sum(u * F, dim=0)

        # Factor (1 - 1/(2*tau)) = (1 - omega/2)
        factor = (1.0 - 0.5 * omega)

        # Vectorized Guo source term
        # S_i = w_i * factor * [ (e_i - u)/cs2 + (e_i . u)*e_i / cs2^2 ] . F

        # self.ei_guo: [27, 3, 1, 1, 1]
        # u: [3, D, H, W]
        # F: [3, D, H, W]

        # e_i . u: [27, D, H, W]
        ei_u = torch.sum(self.ei_guo * u.unsqueeze(0), dim=1)
        # e_i . F: [27, D, H, W]
        ei_F = torch.sum(self.ei_guo * F.unsqueeze(0), dim=1)

        term1 = (ei_F - uF) / cs2
        term2 = (ei_u * ei_F) / (cs2**2)

        S = factor * self.w.view(27, 1, 1, 1) * (term1 + term2)
        return S

    def collide_and_stream(self, omega, geometry_mask, ext_force=None, geom_hash=None):
        geometry_mask = geometry_mask.to(self.device, non_blocking=True)
        # Guard against runaway non-finite populations from previous steps.
        self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        # Macroscopic variables
        rho = torch.sum(self.f, dim=0).clamp_min(1e-8)

        # Macroscopic velocity with forcing offset (Guo's definition)
        # u = (sum fi ci + 0.5 F) / rho
        if ext_force is None:
            ext_force = torch.zeros((3, *rho.shape), device=self.device)

        ux_raw = torch.sum(self.f * self.ex_f, dim=0)
        uy_raw = torch.sum(self.f * self.ey_f, dim=0)
        uz_raw = torch.sum(self.f * self.ez_f, dim=0)

        ux = (ux_raw + 0.5 * ext_force[0]) / (rho + 1e-12)
        uy = (uy_raw + 0.5 * ext_force[1]) / (rho + 1e-12)
        uz = (uz_raw + 0.5 * ext_force[2]) / (rho + 1e-12)

        ux = ux.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uy = uy.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        uz = uz.nan_to_num(0.0, posinf=0.0, neginf=0.0)

        # Cascaded MRT Collision (Issue #16)
        K = torch.tensordot(self.moment_basis, self.f, dims=([1], [0]))
        Keq = self.compute_moment_equilibrium(rho, ux, uy, uz)

        # Build S-vector (relaxation rates)
        s_nu = float(omega)
        s_vec = torch.tensor([0.0, self.s_e, s_nu, self.s_h], device=self.device)
        S = s_vec[self.s_indices].view(27, 1, 1, 1)

        # MRT relaxation towards equilibrium
        K_post = K + S * (Keq - K)

        # Actually apply Guo's forcing if ext_force is non-zero
        if torch.any(ext_force != 0):
            S_guo = self._compute_guo_forcing(rho, torch.stack([ux, uy, uz]), ext_force, omega)
            # Transform Guo source term to moment space if using MRT,
            # or just add to f if using SRT.
            # For MRT, it's easier to add it to populations AFTER MRT relaxation.
            # fi = fi + Si
            # But we are in moment space K_post. So we transform Si to moment space.
            K_S = torch.tensordot(self.moment_basis, S_guo, dims=([1], [0]))
            K_post += K_S

        # Enforce exact conservation of mass and momentum (Issue #16 fix)
        K_post[self.conserved_indices] = Keq[self.conserved_indices]

        # Transform back to populations using in-place copy
        self.f.copy_(torch.tensordot(self.moment_basis_inv, K_post, dims=([1], [0])))
        self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        self.f_pre_stream.copy_(self.f)
        
        used_triton = False
        if self.use_triton_streaming:
            used_triton = stream_bounce_d3q27(
                self.f,
                self.f_pre_stream,
                self.f_temp,
                geometry_mask,
                self.ex,
                self.ey,
                self.ez,
                self.opposite,
            )

        if not used_triton:
            # Streaming
            for i in range(27):
                self.f_temp[i] = torch.roll(self.f[i], shifts=self._stream_shifts[i], dims=(0,1,2))

            # Sub-voxel boundary conditions
            self._apply_bfl_boundary(geometry_mask, geom_hash=geom_hash)

        self._apply_domain_boundaries()

        step_force_x, step_force_z = self._accumulate_momentum_exchange_force(geometry_mask, geom_hash=geom_hash)
        step_projected_drag = self._accumulate_projected_pressure_drag_proxy(geometry_mask, geom_hash=geom_hash)

        self.f.copy_(self.f_temp)
        self.f.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)

        # Recompute macroscopic fields from updated populations (Issue #15 Fix 10)
        rho_new = torch.sum(self.f, dim=0).clamp_min(1e-8)
        # Use Guo's velocity definition: u = (sum fi ci + 0.5 F) / rho
        ux_new = (torch.sum(self.f * self.ex_f, dim=0) + 0.5 * ext_force[0]) / (rho_new + 1e-12)
        uy_new = (torch.sum(self.f * self.ey_f, dim=0) + 0.5 * ext_force[1]) / (rho_new + 1e-12)
        uz_new = (torch.sum(self.f * self.ez_f, dim=0) + 0.5 * ext_force[2]) / (rho_new + 1e-12)

        # Update stored macroscopic fields
        cs2 = 1.0 / 3.0
        self.velocity_x = ux_new.nan_to_num(0.0)
        self.velocity_y = uy_new.nan_to_num(0.0)
        self.velocity_z = uz_new.nan_to_num(0.0)
        self.pressure = rho_new * cs2
        self.rho = rho_new

        self.force_x_last = step_force_x
        self.force_z_last = step_force_z
        self.projected_drag_last = step_projected_drag
        if self._force_step >= self._force_sample_start:
            self.force_x_accum += step_force_x
            self.force_z_accum += step_force_z
            self.projected_drag_accum += step_projected_drag
            self.force_samples += 1
        self._force_step += 1
        return ux, uy, uz, rho


class D3Q27CascadedSolver:
    """Adapter to provide a D3Q27 cascaded solver API compatible with the CFD
    simulator. Wraps the simpler `D3Q27Solver` defined above and exposes the
    same high-level methods used by `AdvancedCFDSimulator`.
    """

    def __init__(self, config, device: torch.device, phys_config):
        self.config = config
        self.device = device
        if callable(phys_config):
            self.phys_config = phys_config()
        else:
            self.phys_config = phys_config

        # resolution: config may provide `resolution` or `base_grid_resolution`
        self.resolution = getattr(self.config, 'resolution', getattr(self.config, 'base_grid_resolution', 32))

        self.inlet_velocity_lu = self._estimate_lattice_freestream_velocity()

        # instantiate the core D3Q27 solver using its expected constructor
        self._solver = D3Q27Solver(
            self.resolution,
            device,
            inlet_velocity_lu=self.inlet_velocity_lu,
            use_triton_streaming=bool(getattr(self.phys_config, "use_triton_streaming", False)),
        )
        # Issue #23: Correctly pass relaxation parameters
        self._solver.s_e = float(getattr(self.phys_config, 's_e_d3q27', 1.2))
        self._solver.s_h = float(getattr(self.phys_config, 's_h_d3q27', 1.6))

        self._solver.drag_link_metric_exponent = getattr(
            self.phys_config, "drag_link_metric_exponent", self._solver.drag_link_metric_exponent
        )

        # Expose population arrays and buffers expected by tests and external code
        # so callers can access `solver.f` directly and observe shapes/values.
        self.f = self._solver.f
        self.f_temp = self._solver.f_temp
        self.f_pre_stream = self._solver.f_pre_stream

        # store last macroscopic fields for diagnostics (create before initialization
        # so _initialize_equilibrium can safely copy into them)
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_y = torch.zeros_like(self.velocity_x)
        self.velocity_z = torch.zeros_like(self.velocity_x)
        self.pressure = torch.zeros_like(self.velocity_x)
        self.rho = torch.ones_like(self.velocity_x)
        self.force_x_accum = torch.tensor(0.0, device=device)
        self.force_z_accum = torch.tensor(0.0, device=device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=device)
        self.force_z_last = torch.tensor(0.0, device=device)
        self.projected_drag_accum = torch.tensor(0.0, device=device)
        self.projected_drag_last = torch.tensor(0.0, device=device)
        self.nu_turb = torch.zeros_like(self.velocity_x)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros_like(self.velocity_x)
        self.nu = self._estimate_kinematic_viscosity()

        # Initialize populations to equilibrium immediately so tests see valid data
        # (non-NaN, correct shape) on solver construction.
        self._initialize_equilibrium()

    def _estimate_kinematic_viscosity(self):
        """Estimate lattice viscosity from lattice freestream and Reynolds."""
        Re = max(float(getattr(self.config, 'reynolds_number', 1e6)), 1e-12)
        u_lu = max(abs(float(self.inlet_velocity_lu)), 1e-6)
        # Use the lattice domain size as the reference length in lattice units.
        L_lu = float(max(self.resolution, 1))
        return max(u_lu * L_lu / Re, 1e-9)

    def _estimate_lattice_freestream_velocity(self):
        """Convert configured physical freestream to lattice units (Issue #16).

        For D3Q27, the sound speed c_s = 1/sqrt(3).
        To maintain consistent Mach number, u_lattice = Ma * c_s = Ma / sqrt(3).
        """
        mach = getattr(self.config, 'mach_number', 0.0)
        u_lattice = float(mach) / np.sqrt(3.0)

        max_mach = float(getattr(self.phys_config, "max_mach", 0.3))
        target_lattice_velocity = float(getattr(self.phys_config, "target_lattice_velocity", 0.12))
        max_lattice_velocity = max(1e-4, min(0.85 * (max_mach / np.sqrt(3.0)), target_lattice_velocity))

        return float(np.clip(u_lattice, -max_lattice_velocity, max_lattice_velocity))

    def _initialize_equilibrium(self):
        """Initialize solver populations to equilibrium with a small freestream."""
        rho = torch.ones(self.resolution, self.resolution, self.resolution, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)

        if self.inlet_velocity_lu:
            ux = torch.full_like(rho, self.inlet_velocity_lu)

        # compute and set equilibrium populations
        feq = self._solver.compute_equilibrium(rho, ux, uy, uz)
        # assign into underlying solver f (expecting shape [27, D, H, W])
        with torch.no_grad():
            self._solver.f.copy_(feq)

        # initialize stored fields
        self.velocity_x.copy_(ux)
        self.velocity_y.copy_(uy)
        self.velocity_z.copy_(uz)
        self.pressure.copy_(rho * (1.0 / 3.0))
        self.rho.copy_(rho)

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100, ext_force=None):
        """Run collide/stream for a number of steps. This adapts the simpler
        D3Q27 solver's `collide_and_stream(omega, geometry_mask)` API.
        """
        geometry_mask = geometry_mask.to(self.device, non_blocking=True)
        # compute a nominal relaxation rate from config
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        nu = self._estimate_kinematic_viscosity()
        self.nu = nu
        tau_min = float(getattr(self.phys_config, "tau_min_d3q27", 0.52))
        tau = max(3.0 * nu + 0.5, tau_min)
        omega = 1.0 / max(tau, 1e-12)

        sample_window = max(10, steps // 4)
        sample_start = max(0, steps - sample_window)
        self._solver.reset_force_accounting(sample_start=sample_start)

        # Issue #22: Compute hash once
        geom_hash = compute_tensor_content_hash(geometry_mask)

        # Issue #23: Convergence tracking
        tol = float(getattr(self.phys_config, 'convergence_tolerance', 1e-5))
        check_every = int(getattr(self.phys_config, 'check_convergence_every', 10))
        # Handle 0 case to avoid crash (Review Feedback)
        check_every = max(1, check_every)

        # run steps
        for step in range(steps):
            v_prev = torch.stack([self.velocity_x, self.velocity_y, self.velocity_z])

            ux, uy, uz, rho = self._solver.collide_and_stream(
                omega, geometry_mask, ext_force=ext_force, geom_hash=geom_hash
            )

            # store fields for diagnostics BEFORE potential break (Review Feedback)
            self.velocity_x = ux
            self.velocity_y = uy
            self.velocity_z = uz
            self.pressure = rho * (1.0 / 3.0)
            self.rho = rho
            self.force_x_last = self._solver.force_x_last
            self.force_z_last = self._solver.force_z_last
            self.force_x_accum = self._solver.force_x_accum
            self.force_z_accum = self._solver.force_z_accum
            self.projected_drag_accum = self._solver.projected_drag_accum
            self.projected_drag_last = self._solver.projected_drag_last
            self.force_samples = self._solver.force_samples

            # Issue #23: Relative L2 norm convergence check
            if step > 0 and step % check_every == 0:
                v_curr = torch.stack([ux, uy, uz])
                du = v_curr - v_prev
                u_mag = torch.sqrt(torch.sum(v_curr**2, dim=0) + 1e-12)
                rel_change = torch.norm(du) / (torch.norm(u_mag) + 1e-12)
                if rel_change < tol:
                    self._solver.logger.log_info(f"LBM Converged at step {step} (rel_change={rel_change:.2e} < {tol})")
                    break

    def _refresh_flow_diagnostics(self):
        """Update vorticity, Q-criterion, and turbulence proxy from the fields."""
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        gradients = compute_velocity_gradients(self.velocity_x, self.velocity_y, self.velocity_z, spacing=h)
        S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor(
            self.velocity_x, self.velocity_y, self.velocity_z, spacing=h, gradients=gradients
        )
        omega_x, omega_y, omega_z = compute_vorticity(
            self.velocity_x, self.velocity_y, self.velocity_z, spacing=h, gradients=gradients
        )

        self.vorticity[0] = omega_x
        self.vorticity[1] = omega_y
        self.vorticity[2] = omega_z

        omega_sq = omega_x**2 + omega_y**2 + omega_z**2
        strain_sq = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2))
        vorticity_mag = torch.sqrt(omega_sq + 1e-12)
        strain_mag = torch.sqrt(strain_sq + 1e-12)

        cs = float(getattr(self.phys_config, 'smagorinsky_constant', 0.17))
        self.nu_turb = ((cs * h) ** 2 * strain_mag).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        q_criterion = 0.5 * (omega_sq - strain_sq)
        self.q_criterion = q_criterion.nan_to_num(0.0, posinf=1e18, neginf=-1e18)

        return vorticity_mag

    def _shape_drag_correction(self, geometry_mask: torch.Tensor, projected_area_lattice: float):
        """Geometry-aware drag correction for non-cube voxelized bodies."""
        if not bool(getattr(self.phys_config, 'use_shape_drag_correction', True)):
            return 1.0, {}

        solid = geometry_mask > 0.5
        solid_volume = float(torch.sum(solid.float()).item())
        if solid_volume <= 0.0:
            return 1.0, {
                'shape_drag_fullness': 0.0,
                'shape_drag_blockage': 0.0,
                'shape_drag_surface_to_volume': 0.0,
            }

        x_presence = torch.any(solid, dim=(1, 2))
        x_idx = torch.where(x_presence)[0]
        x_extent = int((x_idx[-1] - x_idx[0] + 1).item()) if x_idx.numel() > 0 else 1
        fullness = float(solid_volume / max(projected_area_lattice * max(x_extent, 1), 1.0))
        blockage = float(projected_area_lattice / max(float(self.resolution * self.resolution), 1.0))

        surface_proxy = 0.0
        for axis in (0, 1, 2):
            surface_proxy += float(torch.sum(solid != torch.roll(solid, shifts=1, dims=axis)).item())
        surface_to_volume = float(surface_proxy / max(solid_volume, 1.0))
        projected_side = float(np.sqrt(max(projected_area_lattice, 1.0)))
        log_projected_side = float(np.log(max(projected_side, 1.0)))

        # Preserve cube-like compact bodies around scale 1.0 so we do not
        # degrade already-validated baseline behavior.
        if fullness >= 0.95 and surface_to_volume <= 0.8:
            scale = 1.0
        else:
            coeffs = tuple(float(v) for v in getattr(
                self.phys_config,
                'shape_drag_correction_coefficients',
                (
                    -12.633030612111941, 27.87582461044955, -10.247055184812014,
                    22.962648171191816, -17.337224317584685, -3.946645931513679,
                    0.08323209768046214, 4.548014973469924, -5.179313884992105,
                    -7.623947231425998,
                ),
            ))
            if len(coeffs) >= 10:
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = coeffs[:10]
                b_over_f = blockage / max(fullness, 1e-6)
                log_scale = (
                    c0
                    + c1 * fullness
                    + c2 * blockage
                    + c3 * surface_to_volume
                    + c4 * (fullness * surface_to_volume)
                    + c5 * (surface_to_volume * surface_to_volume)
                    + c6 * b_over_f
                    + c7 * log_projected_side
                    + c8 * (surface_to_volume * log_projected_side)
                    + c9 * (fullness * log_projected_side)
                )
            elif len(coeffs) >= 7:
                c0, c1, c2, c3, c4, c5, c6 = coeffs[:7]
                b_over_f = blockage / max(fullness, 1e-6)
                log_scale = (
                    c0
                    + c1 * fullness
                    + c2 * blockage
                    + c3 * surface_to_volume
                    + c4 * (fullness * surface_to_volume)
                    + c5 * (surface_to_volume * surface_to_volume)
                    + c6 * b_over_f
                )
            else:
                c0, c1, c2, c3 = coeffs[:4]
                log_scale = c0 + c1 * fullness + c2 * blockage + c3 * surface_to_volume
            scale = float(np.exp(log_scale))

        min_scale = float(getattr(self.phys_config, 'shape_drag_correction_min', 0.1))
        max_scale = float(getattr(self.phys_config, 'shape_drag_correction_max', 3.0))
        scale = float(np.clip(scale, min_scale, max_scale))

        return scale, {
            'shape_drag_fullness': fullness,
            'shape_drag_blockage': blockage,
            'shape_drag_surface_to_volume': surface_to_volume,
            'shape_drag_projected_side': projected_side,
        }

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        """Compute approximate aerodynamic coefficients from the last simulated
        macroscopic fields. This mirrors the interface used by the training
        CFD simulator.
        """
        # conservative reference area and freestream speed
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        solid = geometry_mask > 0.5
        ref_area = torch.sum(torch.any(solid, dim=0).float()).item() * h**2
        ref_area = max(ref_area, h**2)

        # Issue #23: Handle zero-sample case for early convergence (Review Feedback)
        if self._solver.force_samples > 0:
            projected_drag = self._solver.projected_drag_accum / self._solver.force_samples
            net_drag_force = self._solver.force_x_accum / self._solver.force_samples
            lift_force = self._solver.force_z_accum / self._solver.force_samples
            force_definition = 'raw bounce-back momentum exchange averaged over the last-quarter window'
        else:
            projected_drag = self._solver.projected_drag_last
            net_drag_force = self._solver.force_x_last
            lift_force = self._solver.force_z_last
            force_definition = 'raw bounce-back momentum exchange from last streaming step'

        projected_area_lattice = max(torch.sum(torch.any(solid, dim=0).float()).item(), 1.0)
        raw_projected_drag_coefficient = float(projected_drag.item() / projected_area_lattice)
        freestream_speed = float(getattr(self.config, 'mach_number', 0.0) * 343.0)
        drag_reference_speed = float(getattr(self.phys_config, 'drag_reference_speed', 80.0))
        speed_exponent = float(getattr(self.phys_config, 'drag_speed_normalization_exponent', 1.0))
        if freestream_speed > 1e-12 and drag_reference_speed > 0.0 and speed_exponent != 0.0:
            speed_normalization = (drag_reference_speed / freestream_speed) ** speed_exponent
        else:
            speed_normalization = 1.0
        # 1. Pure momentum exchange (Raw PDE Ground Truth)
        physical_net_drag_force = _scale_momentum_exchange_force(net_drag_force, h, getattr(self.config, 'mach_number', 0.0))
        physical_lift_force = _scale_momentum_exchange_force(lift_force, h, getattr(self.config, 'mach_number', 0.0))

        # 2. Upwind pressure proxy (Fallback/Diagnostic)
        physical_pressure_fallback_force = raw_projected_drag_coefficient * (0.5 * 1.225 * freestream_speed**2 * ref_area)

        # 3. Tuned Surrogate version (Heuristic correction)
        shape_drag_scale, shape_drag_metrics = self._shape_drag_correction(geometry_mask, projected_area_lattice)
        drag_coefficient_surrogate = raw_projected_drag_coefficient * speed_normalization * shape_drag_scale
        physical_surrogate_force = drag_coefficient_surrogate * (0.5 * 1.225 * freestream_speed**2 * ref_area)

        # Label Tiering Logic (Issue #12)
        # 1. lbm_raw: Pure PDE momentum exchange from internal solver
        # 2. lbm_calibrated: Heuristically corrected result for stable training

        lbm_raw_force = physical_net_drag_force
        lbm_calibrated_force = torch.tensor(physical_surrogate_force, device=self.device, dtype=self.f.dtype)

        # Issue #16: Use pure PDE momentum exchange by default
        physical_drag_force = lbm_raw_force

        coeffs = _compute_force_coefficients(
            physical_drag_force,
            physical_lift_force,
            getattr(self.config, 'mach_number', 0.0),
            ref_area=max(ref_area, 1e-12),
            rho_ref=1.225
        )

        # PINN-ready check: requires low divergence, stable forces, and sampling convergence
        force_stability = 1.0
        if self._solver.force_samples > 20:
            avg_fx = float(self._solver.force_x_accum.item()) / self._solver.force_samples
            last_fx = float(self._solver.force_x_last.item())
            force_stability = abs(last_fx - avg_fx) / (abs(avg_fx) + 1e-6)

        lbm_converged = bool(
            not torch.isnan(self.velocity_x).any() and
            abs(float(self.force_x_last.item())) < 1e5 and
            self._solver.force_samples > 50 and
            force_stability < 0.1 # Relaxed for small test resolutions
        )

        vorticity_mag = self._refresh_flow_diagnostics()
        vortex_cells = torch.sum((self.q_criterion > getattr(self.phys_config, 'q_threshold', 0.0)).float()).item()
        v_inf = coeffs.get('freestream_speed', 0.0)
        nu_turb_mean = float(self.nu_turb.mean().item())
        reynolds_turbulent = float(v_inf * h * self.resolution / max(self.nu + nu_turb_mean, 1e-12))

        drag_coefficient = float(coeffs['drag_coefficient'])
        lift_coefficient = float(coeffs['lift_coefficient'])
        calibrated_drag_coefficient = float(drag_coefficient_surrogate)
        training_drag_coefficient = calibrated_drag_coefficient
        training_drag_source = 'lbm_calibrated'
        if lbm_converged and np.isfinite(drag_coefficient) and drag_coefficient > 0.0:
            training_drag_coefficient = drag_coefficient
            training_drag_source = 'lbm_raw'
        lift_to_drag = float(lift_coefficient / max(abs(drag_coefficient), 1e-12))
        solver_quality_checks = {
            'finite_coefficients': bool(np.isfinite(drag_coefficient) and np.isfinite(lift_coefficient)),
            'positive_reference_area': bool(ref_area > 0.0),
            'nonempty_geometry': bool(torch.sum(solid.float()).item() > 0.0),
            'finite_force_outputs': bool(
                np.isfinite(float(physical_net_drag_force.item()))
                and np.isfinite(float(physical_lift_force.item()))
            ),
        }

        return {
            'force_x': float(physical_drag_force.item() if isinstance(physical_drag_force, torch.Tensor) else physical_drag_force),
            'force_z': float(physical_lift_force.item() if isinstance(physical_lift_force, torch.Tensor) else physical_lift_force),

            # Tiered Labeling (Issue #12)
            'label_source': 'lbm_d3q27',
            'label_tier': 'lbm_raw', # Updated to raw for Issue #16
            'lbm_converged': lbm_converged,
            'force_stability': force_stability,

            'physical_force_source': float(physical_net_drag_force.item()),
            'pressure_only_fallback': float(physical_pressure_fallback_force),
            'surrogate_proxy_force': float(physical_surrogate_force),

            'raw_force_x': float(projected_drag.item() if isinstance(projected_drag, torch.Tensor) else projected_drag),
            'raw_force_z': float(lift_force.item() if isinstance(lift_force, torch.Tensor) else lift_force),
            'drag_coefficient': drag_coefficient,
            'calibrated_drag_coefficient': calibrated_drag_coefficient,
            'training_drag_coefficient': training_drag_coefficient,
            'training_drag_source': training_drag_source,
            'lift_coefficient': lift_coefficient,
            'lift_to_drag': lift_to_drag,
            'net_momentum_exchange_force_x': float(physical_net_drag_force.item() if isinstance(physical_net_drag_force, torch.Tensor) else physical_net_drag_force),
            'raw_net_momentum_exchange_force_x': float(net_drag_force.item() if isinstance(net_drag_force, torch.Tensor) else net_drag_force),
            'projected_area_lattice': projected_area_lattice,
            'raw_projected_drag_coefficient': raw_projected_drag_coefficient,
            'drag_speed_normalization': speed_normalization,
            'drag_reference_speed': drag_reference_speed,
            'drag_speed_normalization_exponent': speed_exponent,
            'shape_drag_scale': shape_drag_scale,
            'drag_link_metric_exponent': float(self._solver._effective_drag_link_metric_exponent(geometry_mask)),
            'force_definition': force_definition,
            'pressure_sum': float(self.pressure.sum().item()),
            'max_turbulent_viscosity': float(self.nu_turb.max().item()),
            'mean_smagorinsky_constant': float(getattr(self.phys_config, 'smagorinsky_constant', 0.17)),
            'max_vorticity': float(vorticity_mag.max().item()),
            'vortex_core_volume': float(vortex_cells * h**3),
            'reference_area': ref_area,
            'reference_area_source': 'projected_frontal_voxel_area_yz',
            'reference_area_lattice': projected_area_lattice,
            'reference_length': h * self.resolution,
            'reference_length_source': 'grid_spacing_times_resolution',
            'freestream_speed': v_inf,
            'density': coeffs['density'],
            'reynolds_number_turbulent': reynolds_turbulent,
            'empty_geometry': bool(torch.sum(solid.float()).item() <= 0.0),
            'claim_bearing_cfd': False,
            'solver_quality_checks': solver_quality_checks,
            'solver_provenance': {
                'primary_solver': 'D3Q27',
                'label_tier': 'lbm_raw',
                'lbm_converged': lbm_converged,
                'grid_resolution': int(self.resolution),
                'force_samples': int(self._solver.force_samples),
                'reference_area_source': 'projected_frontal_voxel_area_yz',
            },
        } | shape_drag_metrics


if __name__ == '__main__':
    import argparse
    import os
    import time
    try:
        import trimesh
    except Exception:
        trimesh = None
    from scipy.ndimage import zoom

    parser = argparse.ArgumentParser(description='Run the D3Q27 LBM solver on an input STL')
    parser.add_argument('stl', help='Path to input STL file')
    parser.add_argument('--grid', type=int, default=64, help='Solver grid resolution (default: 64)')
    parser.add_argument('--steps', type=int, default=500, help='Number of simulation steps (default: 500)')
    parser.add_argument('--mach', type=float, default=0.025, help='Mach number (default: 0.025)')
    parser.add_argument('--re', type=float, default=1e5, help='Reynolds number (default: 1e5)')
    parser.add_argument('--body-size', type=float, default=1.0, help='Physical body size in meters (default: 1.0)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'], help='Device to run on')
    args = parser.parse_args()

    if not os.path.exists(args.stl):
        raise FileNotFoundError(f"STL file not found: {args.stl}")

    if trimesh is None:
        raise ImportError('trimesh is required to voxelize STL files; please install trimesh')

    # Load mesh
    mesh = trimesh.load_mesh(args.stl)

    # Domain and grid spacing
    grid_resolution = int(args.grid)
    body_size = float(args.body_size)
    # Keep a fluid buffer around the body so the mesh does not fill the full box.
    domain_scale = 2.0
    domain_size = [body_size * domain_scale, body_size * domain_scale, body_size * domain_scale]
    grid_spacing = domain_size[0] / float(grid_resolution)

    # Preserve mesh physical size and center in domain
    bounds = mesh.bounds
    mesh_extent = bounds[1] - bounds[0]
    max_mesh_extent = float(np.max(mesh_extent)) if mesh_extent is not None else 1.0
    if max_mesh_extent > 1e-12:
        scale_factor = body_size / max_mesh_extent
        mesh.vertices = (mesh.vertices - bounds[0]) * scale_factor

    mesh_center = np.mean(mesh.vertices, axis=0)
    domain_center = np.array(domain_size) / 2.0
    mesh.vertices = mesh.vertices - mesh_center + domain_center

    # Voxelize using trimesh voxelization
    voxel_pitch = grid_spacing
    try:
        voxel_grid = mesh.voxelized(voxel_pitch).fill()
        voxel_np = voxel_grid.matrix.view(np.ndarray)
    except Exception:
        # fall back to coarser pitch
        voxel_pitch = voxel_pitch * 2.0
        voxel_grid = mesh.voxelized(voxel_pitch).fill()
        voxel_np = voxel_grid.matrix.view(np.ndarray)

    # Resize to requested grid by placing occupied voxel centers into the domain grid.
    # This preserves the body size and leaves surrounding fluid cells for the wake.
    target_shape = (grid_resolution, grid_resolution, grid_resolution)
    resized = np.zeros(target_shape, dtype=np.float32)
    try:
        voxel_points = voxel_grid.points
        voxel_indices = np.rint(voxel_points / voxel_pitch).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_resolution - 1)
        resized[
            voxel_indices[:, 0],
            voxel_indices[:, 1],
            voxel_indices[:, 2],
        ] = 1.0
    except Exception:
        # Fallback: preserve the old behavior if point-based voxel placement fails.
        voxel_np = voxel_grid.matrix.view(np.ndarray)
        zoom_factors = np.array(target_shape) / np.array(voxel_np.shape)
        resized = zoom(voxel_np.astype(np.float32), zoom_factors, order=1)
        resized = (resized > 0.5).astype(np.float32)

    # Convert to torch tensor and device
    import torch
    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    geometry_mask = torch.from_numpy(resized).float().to(device)

    # Minimal config objects expected by D3Q27CascadedSolver
    class _Cfg:
        def __init__(self, resolution, mach, reynolds):
            self.resolution = resolution
            self.mach_number = mach
            self.reynolds_number = reynolds
            self.lbm_config = type('LC', (), {})()

    # Minimal phys config
    class _Phys:
        def __init__(self):
            self.smagorinsky_constant = 0.17
            self.test_filter_ratio = 2.0
            self.dynamic_cs_clip_min = 0.0
            self.dynamic_cs_clip_max = 0.2
            self.wale_constant = 0.5
            self.use_les_turbulence = True
            self.turbulence_model = 'dynamic_smagorinsky'
            self.use_vorticity_confinement = True
            self.vc_adaptive = True
            self.vorticity_confinement_epsilon = 0.1
            self.compute_q_criterion = True
            self.check_convergence_every = 250
            self.convergence_tolerance = 1e-5
            self.q_threshold = 0.0
            self.s_nu = None
            self.s_bulk = 1.0
            self.s_energy = 1.2
            self.s_higher = 1.4
            self.tau_min_d3q27 = 0.52
            self.use_triton_streaming = False
            self.drag_link_metric_exponent = None
            self.drag_reference_speed = 80.0
            self.drag_speed_normalization_exponent = 1.0
            self.use_shape_drag_correction = True
            self.shape_drag_correction_coefficients = (
                -12.633030612111941, 27.87582461044955, -10.247055184812014,
                22.962648171191816, -17.337224317584685, -3.946645931513679,
                0.08323209768046214, 4.548014973469924, -5.179313884992105,
                -7.623947231425998,
            )
            self.shape_drag_correction_min = 0.1
            self.shape_drag_correction_max = 3.0
            self.target_lattice_velocity = 0.12
            self.max_mach = 0.3

    # Populate lbm_config fields commonly used
    cfg = _Cfg(grid_resolution, args.mach, args.re)
    cfg.lbm_config.grid_spacing = grid_spacing
    cfg.lbm_config.time_step = 1e-3
    cfg.lbm_config.physical_length_scale = body_size
    cfg.lbm_config.compute_q_criterion = True
    cfg.lbm_config.use_vorticity_confinement = True

    phys = _Phys()

    # Create solver and run
    print(f"Running solver on {args.stl} with grid={grid_resolution}, steps={args.steps}, device={device}")
    solver = D3Q27CascadedSolver(cfg, device, phys)

    t0 = time.time()
    solver.collide_stream(geometry_mask, steps=int(args.steps))
    t1 = time.time()

    # Compute and print aerodynamic coefficients
    try:
        coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
        print('Simulation complete:')
        print(f"  Force X: {coeffs['force_x']:.6e}")
        print(f"  Force Z: {coeffs['force_z']:.6e}")
        print(f"  Coefficient of Drag (Cd): {coeffs['drag_coefficient']:.6e}")
        print(f"  Coefficient of Lift (Cl): {coeffs['lift_coefficient']:.6e}")
        print(f"  Force Definition: {coeffs['force_definition']}")
        print(f"  Pressure Sum: {coeffs['pressure_sum']:.6e}")
        print(f"  Max Turbulent Viscosity: {coeffs['max_turbulent_viscosity']:.6e}")
        print(f"  Mean Smagorinsky Constant: {coeffs['mean_smagorinsky_constant']:.6e}")
        print(f"  Max Vorticity: {coeffs['max_vorticity']:.6e}")
        print(f"  Vortex Core Volume: {coeffs['vortex_core_volume']:.6e}")
        print(f"  Reference Area: {coeffs['reference_area']:.6e}")
        print(f"  Reference Length: {coeffs['reference_length']:.6e}")
        print(f"  Freestream Speed: {coeffs['freestream_speed']:.6e}")
        print(f"  Density: {coeffs['density']:.6e}")
        print(f"  Reynolds Number (turbulent): {coeffs['reynolds_number_turbulent']:.6e}")
    except Exception as e:
        print('Could not compute aerodynamic coefficients:', e)

    print(f"Elapsed time: {t1 - t0:.2f}s")
