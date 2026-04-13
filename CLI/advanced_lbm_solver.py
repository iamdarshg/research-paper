
import torch
import numpy as np
import math
from typing import Dict
from typing import TYPE_CHECKING

from lbm_utils import D3Q27Lattice, _compute_force_coefficients
from lbm_diagnostics import compute_strain_rate_tensor, compute_vorticity, compute_velocity_gradients
from lbm_logger import lbm_debug_logger


def _scale_momentum_exchange_force(force, grid_spacing: float, mach_number: float, density: float = 1.225):
    """Convert raw lattice momentum exchange into a physical force scale."""
    freestream_speed = float(mach_number) * 343.0
    # Use analytic momentum-exchange → physical scaling:
    # physical_force = raw_lattice_sum * (0.5 * rho * U_inf^2 * dx^2)
    # The 0.5 factor aligns the momentum-exchange definition with the
    # aerodynamic dynamic pressure used in the coefficient denominator.
    force_scale = 0.5 * float(density) * freestream_speed * freestream_speed * float(grid_spacing) * float(grid_spacing)
    return force * force_scale


class D3Q27Solver:
    """Complete D3Q27 LBM solver with Cascaded Central Moment MRT"""
    def __init__(self, resolution, device):
        self.res = resolution
        self.device = device

        self.ex, self.ey, self.ez = D3Q27Lattice.get_vectors()
        self.ex_f = self.ex.to(device, dtype=torch.float32)
        self.ey_f = self.ey.to(device, dtype=torch.float32)
        self.ez_f = self.ez.to(device, dtype=torch.float32)
        self.ex = self.ex.to(device, dtype=torch.long)
        self.ey = self.ey.to(device, dtype=torch.long)
        self.ez = self.ez.to(device, dtype=torch.long)
        self.w = D3Q27Lattice.get_weights().to(device)
        self.opposite = D3Q27Lattice.get_opposite().to(device)

        # 27 populations
        self.f = torch.zeros(27, resolution, resolution, resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)
        self.reset_force_accounting()

        # Precompute moment basis for D3Q27
        self._setup_moment_basis()

    def _setup_moment_basis(self):
        """Setup 27-moment basis and its inverse for D3Q27"""
        # We use tensor product basis: e_x^a * e_y^b * e_z^c for a,b,c in {0,1,2}
        moments = []
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    moments.append((a, b, c))
        self.moment_indices = moments

        M = torch.zeros((27, 27), device=self.device)
        for k, (a, b, c) in enumerate(moments):
            M[k] = (self.ex_f**a) * (self.ey_f**b) * (self.ez_f**c)

        self.M_matrix = M
        self.M_inv = torch.inverse(M)

        # Relaxation rates (default values, can be optimized)
        # s_nu is set by viscosity, others are for higher-order/ghost moments
        self.s_e = 1.19
        self.s_eps = 1.4
        self.s_q = 1.2
        self.s_ghost = 1.5

    def compute_equilibrium(self, rho, ux, uy, uz):
        # Use spatial shape of input for feq
        spatial_shape = rho.shape
        feq = torch.zeros((27, *spatial_shape), device=self.device, dtype=self.f.dtype)

        u_sq = ux**2 + uy**2 + uz**2

        for i in range(27):
            cu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
            feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        return feq

    def reset_force_accounting(self, sample_start: int = 0):
        """Reset momentum-exchange bookkeeping for a new simulation run."""
        self.force_x_accum = torch.tensor(0.0, device=self.device)
        self.force_z_accum = torch.tensor(0.0, device=self.device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=self.device)
        self.force_z_last = torch.tensor(0.0, device=self.device)
        self._force_sample_start = max(0, int(sample_start))
        self._force_step = 0

    def _accumulate_momentum_exchange_force(self, geometry_mask):
        """Compute wall force from fluid-solid links using bounce-back exchange."""
        mask = geometry_mask > 0.5
        step_force_x = torch.tensor(0.0, device=self.device)
        step_force_z = torch.tensor(0.0, device=self.device)

        for i in range(27):
            opp_i = int(self.opposite[i].item())
            if i == 0 or i > opp_i:
                continue

            dx = int(self.ex[i].item())
            dy = int(self.ey[i].item())
            dz = int(self.ez[i].item())
            neighbor_is_solid = torch.roll(mask, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))
            boundary_link = (~mask) & neighbor_is_solid
            if not torch.any(boundary_link):
                continue

            step_force_x += torch.sum(2.0 * float(self.ex[i].item()) * self.f_pre_stream[i, boundary_link])
            step_force_z += torch.sum(2.0 * float(self.ez[i].item()) * self.f_pre_stream[i, boundary_link])

        return step_force_x, step_force_z

    def apply_boundary_conditions(self, geometry_mask, u_inlet=None):
        """Apply modular boundary conditions.
        Currently supports:
        - Bounce-back on geometry_mask (solid body)
        - Periodic in Y and Z
        - Inlet (Zou-He or Equilibrium) on X-min
        - Outlet (Neumann or Equilibrium) on X-max
        """
        mask = geometry_mask > 0.5

        # 1. Body Bounce-back (already handled in collide_and_stream by default)
        # However, we can move it here for modularity.

        # 2. X-Inlet (X=0)
        if u_inlet is not None:
            # Simple equilibrium inlet for now
            rho_in = torch.sum(self.f[:, 0, :, :], dim=0)
            ux_in = torch.full_like(rho_in, u_inlet)
            uy_in = torch.zeros_like(rho_in)
            uz_in = torch.zeros_like(rho_in)
            feq_in = self.compute_equilibrium(rho_in, ux_in, uy_in, uz_in)
            # Apply to populations that point into the domain from X=0
            # Directions with ex[i] > 0: 1, 7, 9, 11, 13, 19, 21, 23, 25
            for i in [1, 7, 9, 11, 13, 19, 21, 23, 25]:
                self.f[i, 0, :, :] = feq_in[i]

        # 3. X-Outlet (X=res-1)
        # Simple Neumann (zero-gradient) for populations pointing out
        # Directions with ex[i] < 0: 2, 8, 10, 12, 14, 20, 22, 24, 26
        for i in [2, 8, 10, 12, 14, 20, 22, 24, 26]:
            self.f[i, -1, :, :] = self.f[i, -2, :, :]

    def collide_and_stream(self, omega, geometry_mask):
        # Macroscopic variables
        rho = torch.sum(self.f, dim=0)
        ux = torch.sum(self.f * self.ex_f.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)
        uy = torch.sum(self.f * self.ey_f.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)
        uz = torch.sum(self.f * self.ez_f.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)

        # 1. Transform to Central Moments K_prime
        dx = self.ex_f.view(27, 1, 1, 1) - ux.unsqueeze(0)
        dy = self.ey_f.view(27, 1, 1, 1) - uy.unsqueeze(0)
        dz = self.ez_f.view(27, 1, 1, 1) - uz.unsqueeze(0)

        pow_x = [torch.ones_like(dx[0]), dx, dx**2]
        pow_y = [torch.ones_like(dy[0]), dy, dy**2]
        pow_z = [torch.ones_like(dz[0]), dz, dz**2]

        K_prime = []
        for (i, j, m) in self.moment_indices:
            K_prime.append(torch.sum(self.f * pow_x[i] * pow_y[j] * pow_z[m], dim=0))
        K_prime = torch.stack(K_prime, dim=0)

        # 2. Relax Central Moments
        cs2 = 1.0/3.0
        m_eq = [torch.ones_like(rho), torch.zeros_like(rho), torch.full_like(rho, cs2)]

        for k, (i, j, m) in enumerate(self.moment_indices):
            if i + j + m <= 1: continue # Conserved moments

            keq = rho * m_eq[i] * m_eq[j] * m_eq[m]

            s = self.s_ghost
            if i+j+m == 2:
                if (i==1 and j==1) or (i==1 and m==1) or (j==1 and m==1):
                    s = omega
                else:
                    s = self.s_e

            K_prime[k] += s * (keq - K_prime[k])

        # 3. Transform back to Populations via Raw Moments
        ux_pow = [torch.ones_like(ux), ux, ux**2]
        uy_pow = [torch.ones_like(uy), uy, uy**2]
        uz_pow = [torch.ones_like(uz), uz, uz**2]

        idx_map = {(i,j,m): k for k, (i,j,m) in enumerate(self.moment_indices)}
        K_raw = torch.zeros_like(K_prime)
        for (i, j, m), k in idx_map.items():
            res_k = torch.zeros_like(rho)
            for p in range(i + 1):
                for q in range(j + 1):
                    for r in range(m + 1):
                        coeff = math.comb(i, p) * math.comb(j, q) * math.comb(m, r)
                        res_k += coeff * (ux_pow[i-p] * uy_pow[j-q] * uz_pow[m-r]) * K_prime[idx_map[(p, q, r)]]
            K_raw[k] = res_k

        self.f.copy_(torch.matmul(self.M_inv, K_raw.reshape(27, -1)).reshape(self.f.shape))

        self.f_pre_stream.copy_(self.f)
        
        # Streaming
        for i in range(27):
            shifts = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
            self.f_temp[i] = torch.roll(self.f[i], shifts=shifts, dims=(0,1,2))
        
        # Bounce-back using PRE-STREAM populations
        mask = geometry_mask > 0.5
        for i in range(27):
            opp_i = int(self.opposite[i].item())
            self.f_temp[i] = torch.where(mask, self.f_pre_stream[opp_i], self.f_temp[i])

        step_force_x, step_force_z = self._accumulate_momentum_exchange_force(geometry_mask)

        self.f.copy_(self.f_temp)
        self.force_x_last = step_force_x
        self.force_z_last = step_force_z
        if self._force_step >= self._force_sample_start:
            self.force_x_accum += step_force_x
            self.force_z_accum += step_force_z
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

        # instantiate the core D3Q27 solver using its expected constructor
        self._solver = D3Q27Solver(self.resolution, device)

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
        self.nu_turb = torch.zeros_like(self.velocity_x)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros_like(self.velocity_x)
        self.nu = self._estimate_kinematic_viscosity()
        lbm_debug_logger.debug(f"Initialized D3Q27CascadedSolver: resolution={self.resolution}, nu={self.nu}")

        # Initialize populations to equilibrium immediately so tests see valid data
        # (non-NaN, correct shape) on solver construction.
        self._initialize_equilibrium()

    def _estimate_kinematic_viscosity(self):
        """Estimate the lattice kinematic viscosity from the current config."""
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        Re = getattr(self.config, 'reynolds_number', 1e6)
        # Use physical speed from Mach
        U_phys = getattr(self.config, 'mach_number', 0.0) * 343.0
        # Reference length should be the body size, but often defaults to domain size
        # in the current config.
        L_phys = getattr(self.config.lbm_config, 'physical_length_scale', h * self.resolution)
        nu_phys = (U_phys * L_phys) / max(Re, 1e-12)
        nu_lattice = nu_phys * dt / (h * h)
        lbm_debug_logger.debug(f"Viscosity estimate: Mach={U_phys/343:.4f}, Re={Re}, L_phys={L_phys:.4f}, nu_phys={nu_phys:.6e}, nu_lattice={nu_lattice:.6e}")
        return nu_lattice

    def _initialize_equilibrium(self):
        """Initialize solver populations to equilibrium with a small freestream."""
        # Use dtype of solver's population for consistency
        dtype = self._solver.f.dtype
        rho = torch.ones(self.resolution, self.resolution, self.resolution, device=self.device, dtype=dtype)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)

        # Seed the equilibrium in lattice units for stability.
        # u_lattice = Mach_phys * cs, where cs = 1/sqrt(3)
        mach_phys = getattr(self.config, 'mach_number', 0.0)
        cs_lattice = 1.0 / (3.0**0.5)
        u_lattice = mach_phys * cs_lattice

        if mach_phys > 0:
            ux = torch.full_like(rho, u_lattice)
            lbm_debug_logger.debug(f"Initializing equilibrium: mach_phys={mach_phys:.4f}, u_lattice={u_lattice:.6e}")

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

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100, use_inlet_outlet: bool = False):
        """Run collide/stream for a number of steps. This adapts the simpler
        D3Q27 solver's `collide_and_stream(omega, geometry_mask)` API.
        """
        # compute a nominal relaxation rate from config
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        nu = self._estimate_kinematic_viscosity()
        self.nu = nu
        tau = 3.0 * nu + 0.5
        omega = 1.0 / max(tau, 1e-12)

        sample_window = max(10, steps // 4)
        sample_start = max(0, steps - sample_window)
        self._solver.reset_force_accounting(sample_start=sample_start)

        # Inlet velocity in lattice units
        mach_phys = getattr(self.config, 'mach_number', 0.0)
        u_inlet = mach_phys / (3.0**0.5) if use_inlet_outlet else None

        # run steps
        for _ in range(steps):
            if use_inlet_outlet:
                self._solver.apply_boundary_conditions(geometry_mask, u_inlet=u_inlet)
            ux, uy, uz, rho = self._solver.collide_and_stream(omega, geometry_mask)
            # store fields for diagnostics
            self.velocity_x = ux
            self.velocity_y = uy
            self.velocity_z = uz
            self.pressure = rho * (1.0 / 3.0)
            self.rho = rho
            self.force_x_last = self._solver.force_x_last
            self.force_z_last = self._solver.force_z_last
            self.force_x_accum = self._solver.force_x_accum
            self.force_z_accum = self._solver.force_z_accum
            self.force_samples = self._solver.force_samples

    def _refresh_flow_diagnostics(self):
        """Update vorticity, Q-criterion, and turbulence proxy from the fields."""
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)

        # Ensure input fields have consistent dtype
        vx, vy, vz = self.velocity_x, self.velocity_y, self.velocity_z

        gradients = compute_velocity_gradients(vx, vy, vz, spacing=h)
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

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor, ref_area_override: float = None) -> Dict[str, float]:
        """Compute approximate aerodynamic coefficients from the last simulated
        macroscopic fields. Derives Cd directly in lattice units for accuracy.
        """
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)

        # 1. Determine reference area in lattice units (voxels)
        solid = geometry_mask > 0.5
        if ref_area_override is not None:
            # ref_area_override is physical m^2, convert to lattice dx^2
            ref_area_lat = ref_area_override / (h**2)
        else:
            ref_area_lat = torch.sum(torch.any(solid, dim=0).float()).item()
        ref_area_lat = max(ref_area_lat, 1.0)

        # 2. Get raw lattice forces
        if self._solver.force_samples > 0:
            drag_force_lat = self._solver.force_x_accum / self._solver.force_samples
            lift_force_lat = self._solver.force_z_accum / self._solver.force_samples
            force_definition = 'bounce-back momentum exchange averaged over the last-quarter window'
        else:
            drag_force_lat = self._solver.force_x_last
            lift_force_lat = self._solver.force_z_last
            force_definition = 'bounce-back momentum exchange from last streaming step'

        # 3. Measure far-field lattice velocity for dynamic pressure
        # Use a shell of 2 layers at the domain boundaries (excluding inlet/outlet faces if any)
        # For now, just average the whole domain's velocity_x as a proxy, or better,
        # a specific region.
        u_inf_lat = torch.mean(self.velocity_x).item() # Rough proxy
        # More robust: average of the first 2 and last 2 layers in Y and Z
        u_inf_lat = (torch.mean(self.velocity_x[:, :2, :]) +
                     torch.mean(self.velocity_x[:, -2:, :]) +
                     torch.mean(self.velocity_x[:, :, :2]) +
                     torch.mean(self.velocity_x[:, :, -2:])) / 4.0
        u_inf_lat = abs(u_inf_lat)
        if u_inf_lat < 1e-9:
            mach_phys = getattr(self.config, 'mach_number', 0.0)
            u_inf_lat = mach_phys / (3.0**0.5)

        # 4. Compute Cd, Cl directly in lattice units
        # Cd = F / (0.5 * rho_lat * u_inf_lat^2 * A_ref_lat)
        # In standard LBM, rho_lat approx 1.0
        rho_lat = 1.0
        q_inf_lat = 0.5 * rho_lat * (u_inf_lat**2)

        cd = drag_force_lat.item() / (q_inf_lat * ref_area_lat + 1e-12)
        cl = lift_force_lat.item() / (q_inf_lat * ref_area_lat + 1e-12)

        # 5. Convert to physical units for compatibility
        # F_phys = F_lat * (rho_phys * dx^4 / dt^2)
        rho_phys = 1.225
        force_scale = rho_phys * (h**4) / (dt**2)
        physical_drag_force = drag_force_lat * force_scale
        physical_lift_force = lift_force_lat * force_scale

        v_inf_phys = u_inf_lat * (h / dt)

        lbm_debug_logger.debug(f"Cd computation: drag_lat={drag_force_lat.item():.6e}, u_inf_lat={u_inf_lat:.6f}, q_inf_lat={q_inf_lat:.6e}, A_ref_lat={ref_area_lat:.2f}, Cd={cd:.6f}")

        vorticity_mag = self._refresh_flow_diagnostics()
        vortex_cells = torch.sum((self.q_criterion > getattr(self.phys_config, 'q_threshold', 0.0)).float()).item()
        nu_turb_mean = float(self.nu_turb.mean().item())
        reynolds_turbulent = float(v_inf_phys * h * self.resolution / max((self.nu + nu_turb_mean) * (h**2/dt), 1e-12))

        return {
            'force_x': float(physical_drag_force.item()),
            'force_z': float(physical_lift_force.item()),
            'raw_force_x': float(drag_force_lat.item()),
            'raw_force_z': float(lift_force_lat.item()),
            'drag_coefficient': float(cd),
            'lift_coefficient': float(cl),
            'force_definition': force_definition,
            'pressure_sum': float(self.pressure.sum().item()),
            'max_turbulent_viscosity': float(self.nu_turb.max().item()),
            'mean_smagorinsky_constant': float(getattr(self.phys_config, 'smagorinsky_constant', 0.17)),
            'max_vorticity': float(vorticity_mag.max().item()),
            'vortex_core_volume': float(vortex_cells * h**3),
            'reference_area': ref_area_lat * (h**2),
            'reference_length': h * self.resolution,
            'freestream_speed': v_inf_phys,
            'density': rho_phys,
            'reynolds_number_turbulent': reynolds_turbulent,
            'u_inf_lat': u_inf_lat
        }


if __name__ == '__main__':
    import argparse
    import os
    import time
    try:
        import trimesh
    except Exception:
        trimesh = None
    from scipy.ndimage import zoom

    parser = argparse.ArgumentParser(description='Run GPULBMSolver on an input STL')
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

    # Minimal config objects expected by GPULBMSolver
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

