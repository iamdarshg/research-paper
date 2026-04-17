import math

import torch
import numpy as np
from typing import Dict
from typing import TYPE_CHECKING

from lbm_utils import D3Q27Lattice, _compute_force_coefficients
from lbm_diagnostics import compute_strain_rate_tensor, compute_vorticity, compute_velocity_gradients


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
    """Complete D3Q27 LBM solver"""
    def __init__(self, resolution, device):
        self.res = resolution
        self.device = device

        self.ex, self.ey, self.ez = D3Q27Lattice.get_vectors()
        self.ex = self.ex.to(device, dtype=torch.long)
        self.ey = self.ey.to(device, dtype=torch.long)
        self.ez = self.ez.to(device, dtype=torch.long)
        self.w = D3Q27Lattice.get_weights().to(device)
        self.opposite = D3Q27Lattice.get_opposite().to(device)

        # 27 populations instead of 19
        self.f = torch.zeros(27, resolution, resolution, resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)
        self.reset_force_accounting()

    def compute_equilibrium(self, rho, ux, uy, uz):
        feq = torch.zeros_like(self.f)
        for i in range(27):
            cu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
            u_sq = ux**2 + uy**2 + uz**2
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

            step_force_x += torch.sum(2.0 * float(self.ex[i].item()) * self.f_pre_stream[i][boundary_link])
            step_force_z += torch.sum(2.0 * float(self.ez[i].item()) * self.f_pre_stream[i][boundary_link])

        return step_force_x, step_force_z

    def collide_and_stream(self, omega, geometry_mask):
        # Macroscopic variables
        rho = torch.sum(self.f, dim=0)
        ux = torch.sum(self.f * self.ex.view(-1,1,1,1), dim=0) / (rho + 1e-12)
        uy = torch.sum(self.f * self.ey.view(-1,1,1,1), dim=0) / (rho + 1e-12)
        uz = torch.sum(self.f * self.ez.view(-1,1,1,1), dim=0) / (rho + 1e-12)

        # Collision
        feq = self.compute_equilibrium(rho, ux, uy, uz)
        self.f += omega * (feq - self.f)

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

class GPULBMSolver:
    """GPU-resident LBM solver with Dynamic Smagorinsky, Vorticity Confinement, and improved vorticity resolution"""

    def __init__(self, config, device: torch.device, phys_config):
        self.config = config
        self.device = device
        self.resolution = config.resolution
        # If phys_config is callable (class), instantiate it; otherwise use directly (instance)
        if callable(phys_config):
            self.phys_config = phys_config()
        else:
            self.phys_config = phys_config

        self._setup_physics_constants()

        # Structure of Arrays (SoA) layout
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.velocity_y = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.velocity_z = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.pressure = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12

        # LBM populations (D3Q27)
        self.f = torch.zeros(27, self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.f_temp = torch.zeros_like(self.f)+1e-12
        self.f_pre_stream = torch.empty_like(self.f)

        # Turbulence and vorticity fields
        self.nu_turb = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.q_criterion = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.cs_dynamic = torch.full((self.resolution, self.resolution, self.resolution),
                                     self.phys_config.smagorinsky_constant, device=device)

        # Convergence tracking
        self.velocity_prev = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)

        self._setup_d3q27_lattice()
        self._setup_mrt_matrices()
        self._initialize_equilibrium()

        self.force_x_accum = torch.tensor(0.0, device=device)
        self.force_z_accum = torch.tensor(0.0, device=device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=device)
        self.force_z_last = torch.tensor(0.0, device=device)

    def _setup_physics_constants(self):
        """Compute physics constants from config with improved stability mapping."""
        h = self.config.lbm_config.grid_spacing
        self.cs2 = 1.0 / 3.0
        U_phys = self.config.mach_number * 343.0
        Re = getattr(self.config, 'reynolds_number', 1000)
        self.u_lat = self.config.mach_number / math.sqrt(3.0)
        if self.u_lat > 0.1: self.u_lat = 0.1
        self.nu = self.u_lat * self.resolution / max(float(Re), 1e-12)
        self.tau = max(3.0 * self.nu + 0.5, 0.501)
        self.phys_config.s_nu = 1.0 / self.tau
    def _setup_d3q27_lattice(self):
        """Setup D3Q27 lattice"""
        ex, ey, ez = D3Q27Lattice.get_vectors()
        self.ex, self.ey, self.ez = ex.to(self.device, dtype=torch.int32), ey.to(self.device, dtype=torch.int32), ez.to(self.device, dtype=torch.int32)
        self.w = D3Q27Lattice.get_weights().to(self.device, dtype=torch.float32)
        self.opposite = D3Q27Lattice.get_opposite().to(self.device, dtype=torch.int64)

    def _setup_mrt_matrices(self):
        self.s_relax = torch.full((27,), 1.0 / self.tau, device=self.device)

    def _initialize_equilibrium(self):
        ux, uy, uz = self.u_lat, 0.0, 0.0
        rho = 1.0
        for i in range(27):
            eu = self.ex[i].item() * ux + self.ey[i].item() * uy + self.ez[i].item() * uz
            u_sq = ux*ux + uy*uy + uz*uz
            feq_val = self.w[i].item() * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
            self.f[i].fill_(feq_val)
    def _compute_strain_rate_tensor(self, ux, uy, uz, gradients=None):
        return compute_strain_rate_tensor(ux, uy, uz, spacing=self.config.lbm_config.grid_spacing, gradients=gradients)

    def _compute_vorticity(self, ux, uy, uz, gradients=None):
        return compute_vorticity(ux, uy, uz, spacing=self.config.lbm_config.grid_spacing, gradients=gradients)

    def _compute_q_criterion(self, ux, uy, uz):
        """Compute Q-criterion for vortex identification [web:44][web:47]
        Q = 0.5*(||Omega||^2 - ||S||^2) where Omega is rotation rate tensor
        Positive Q indicates vortex regions (rotation > strain)
        """
        gradients = compute_velocity_gradients(ux, uy, uz, spacing=self.config.lbm_config.grid_spacing)
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz, gradients=gradients)
        S_mag_sq = S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)

        omega_x, omega_y, omega_z = self._compute_vorticity(ux, uy, uz, gradients=gradients)
        omega_mag_sq = omega_x**2 + omega_y**2 + omega_z**2

        # Q-criterion: Q > 0 indicates vortex regions
        Q = 0.5 * (omega_mag_sq - S_mag_sq).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        return Q

    def _compute_dynamic_smagorinsky(self, ux, uy, uz):
        """Compute dynamic Smagorinsky constant using Germano identity [web:43][web:46]"""
        Delta = self.config.lbm_config.grid_spacing
        Delta_test = self.phys_config.test_filter_ratio * Delta

        # Grid-scale strain rate
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
        S11 = S11.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S22 = S22.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S33 = S33.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S12 = S12.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S13 = S13.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S23 = S23.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Test filter (simple box filter approximation via pooling)
        kernel_size = int(self.phys_config.test_filter_ratio)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        # Apply test filter to velocities (approximation)
        ux_test = torch.nn.functional.avg_pool3d(
            ux.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze().nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        uy_test = torch.nn.functional.avg_pool3d(
            uy.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze().nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        uz_test = torch.nn.functional.avg_pool3d(
            uz.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze().nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Test-scale strain rate
        S11_test, S22_test, S33_test, S12_test, S13_test, S23_test = self._compute_strain_rate_tensor(
            ux_test, uy_test, uz_test
        )
        S11_test = S11_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S22_test = S22_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S33_test = S33_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S12_test = S12_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S13_test = S13_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S23_test = S23_test.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S_mag_test = torch.sqrt(2.0 * (S11_test**2 + S22_test**2 + S33_test**2 +
                                       2.0*(S12_test**2 + S13_test**2 + S23_test**2)) + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Leonard stress (Germano identity)
        # L_ij = test_filter(u_i * u_j) - test_filter(u_i) * test_filter(u_j)
        L11 = torch.nn.functional.avg_pool3d((ux*ux).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - ux_test*ux_test
        L22 = torch.nn.functional.avg_pool3d((uy*uy).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - uy_test*uy_test
        L33 = torch.nn.functional.avg_pool3d((uz*uz).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - uz_test*uz_test
        L12 = torch.nn.functional.avg_pool3d((ux*uy).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - ux_test*uy_test
        L13 = torch.nn.functional.avg_pool3d((ux*uz).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - ux_test*uz_test
        L23 = torch.nn.functional.avg_pool3d((uy*uz).unsqueeze(0).unsqueeze(0),
                                            kernel_size, 1, padding).squeeze() - uy_test*uz_test
        L11 = L11.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        L22 = L22.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        L33 = L33.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        L12 = L12.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        L13 = L13.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        L23 = L23.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        # Model coefficient (least squares fit)
        # C_s^2 = <L_ij * M_ij> / <M_ij * M_ij>
        # where M_ij = -2 * Delta^2 * |S| * S_ij (grid) + 2 * Delta_test^2 * |S_test| * S_ij_test
        M11 = -2.0*Delta**2*S_mag*S11 + 2.0*Delta_test**2*S_mag_test*S11_test
        M22 = -2.0*Delta**2*S_mag*S22 + 2.0*Delta_test**2*S_mag_test*S22_test
        M33 = -2.0*Delta**2*S_mag*S33 + 2.0*Delta_test**2*S_mag_test*S33_test
        M12 = -2.0*Delta**2*S_mag*S12 + 2.0*Delta_test**2*S_mag_test*S12_test
        M13 = -2.0*Delta**2*S_mag*S13 + 2.0*Delta_test**2*S_mag_test*S13_test
        M23 = -2.0*Delta**2*S_mag*S23 + 2.0*Delta_test**2*S_mag_test*S23_test
        M11 = M11.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        M22 = M22.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        M33 = M33.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        M12 = M12.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        M13 = M13.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        M23 = M23.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        # Compute Cs^2 locally
        LM = L11*M11 + L22*M22 + L33*M33 + 2.0*(L12*M12 + L13*M13 + L23*M23)
        MM = M11**2 + M22**2 + M33**2 + 2.0*(M12**2 + M13**2 + M23**2)
        LM = LM.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        MM = MM.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        Cs_squared = LM / (MM + 1e-12)
        Cs_squared = torch.clamp(Cs_squared, min=1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)  # Ensure non-negative
        Cs = torch.sqrt(Cs_squared+1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Clip to reasonable bounds
        Cs = torch.clamp(Cs,
                        min=self.phys_config.dynamic_cs_clip_min,
                        max=self.phys_config.dynamic_cs_clip_max)

        return Cs, S_mag

    def _compute_wale_model(self, ux, uy, uz):
        """WALE (Wall-Adapting Local Eddy-viscosity) model [web:36]
        Better near-wall behavior than Smagorinsky
        """
        Delta = self.config.lbm_config.grid_spacing
        Cw = self.phys_config.wale_constant

        # Velocity gradient tensor
        gradients = compute_velocity_gradients(ux, uy, uz, spacing=self.config.lbm_config.grid_spacing)
        dux_dx, dux_dy, dux_dz = gradients[0]
        duy_dx, duy_dy, duy_dz = gradients[1]
        duz_dx, duz_dy, duz_dz = gradients[2]

        # Traceless symmetric part of velocity gradient squared
        # S_d = 0.5*(grad_u + grad_u^T) - (1/3)*tr(grad_u)*I
        gij_gji_11 = (dux_dx**2 + dux_dy*duy_dx + dux_dz*duz_dx).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        gij_gji_22 = (duy_dx*dux_dy + duy_dy**2 + duy_dz*duz_dy).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        gij_gji_33 = (duz_dx*dux_dz + duz_dy*duy_dz + duz_dz**2).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        trace_third = (dux_dx + duy_dy + duz_dz) / 3.0

        Sd_11 = 0.5*(gij_gji_11 + gij_gji_11) - 2.0*trace_third*dux_dx
        Sd_22 = 0.5*(gij_gji_22 + gij_gji_22) - 2.0*trace_third*duy_dy
        Sd_33 = 0.5*(gij_gji_33 + gij_gji_33) - 2.0*trace_third*duz_dz

        Sd_mag = torch.sqrt(Sd_11**2 + Sd_22**2 + Sd_33**2 + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Strain rate magnitude
        S_mag = torch.sqrt(2.0*(dux_dx**2 + duy_dy**2 + duz_dz**2) + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # WALE turbulent viscosity
        nu_turb = (Cw * Delta)**2 * (Sd_mag**1.5) / (S_mag**2.5 + Sd_mag**1.25 + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        return nu_turb

    def _compute_turbulent_viscosity(self, ux, uy, uz):
        """Compute turbulent viscosity using selected model"""
        if not self.phys_config.use_les_turbulence:
            return torch.zeros_like(ux)

        if self.phys_config.turbulence_model == "smagorinsky":
            # Standard Smagorinsky
            S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
            S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)
            Cs = self.phys_config.smagorinsky_constant
            Delta = self.config.lbm_config.grid_spacing
            nu_turb = (Cs * Delta)**2 * S_mag

        elif self.phys_config.turbulence_model == "dynamic_smagorinsky":
            # Dynamic Smagorinsky (Germano)
            Cs, S_mag = self._compute_dynamic_smagorinsky(ux, uy, uz)
            Cs, S_mag = Cs.nan_to_num(1e-12, posinf=1e18, neginf=-1e18), S_mag.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
            self.cs_dynamic = Cs  # Store for diagnostics
            Delta = self.config.lbm_config.grid_spacing
            nu_turb = ((Cs * Delta)**2 * S_mag).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        elif self.phys_config.turbulence_model == "wale":
            # WALE model
            nu_turb = self._compute_wale_model(ux, uy, uz)

        else:
            raise ValueError(f"Unknown turbulence model: {self.phys_config.turbulence_model}")

        return nu_turb

    def _apply_vorticity_confinement(self, ux, uy, uz):
        """Apply vorticity confinement to preserve vortices [web:38][web:41]
        Adds anti-dissipation force F = epsilon * (eta x omega)
        """
        if not self.phys_config.use_vorticity_confinement:
            return torch.zeros_like(ux), torch.zeros_like(uy), torch.zeros_like(uz)

        gradients = compute_velocity_gradients(ux, uy, uz, spacing=self.config.lbm_config.grid_spacing)
        omega_x, omega_y, omega_z = self._compute_vorticity(ux, uy, uz, gradients=gradients)
        self.vorticity[0] = omega_x
        self.vorticity[1] = omega_y
        self.vorticity[2] = omega_z

        # Vorticity magnitude
        omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        # Confinement direction: eta = grad(|omega|) / |grad(|omega|)|
        grad_spacing = (self.config.lbm_config.grid_spacing,) * 3
        grad_omega_x, grad_omega_y, grad_omega_z = torch.gradient(omega_mag, dim=(0, 1, 2), spacing=grad_spacing)
        grad_omega_mag = torch.sqrt(grad_omega_x**2 + grad_omega_y**2 + grad_omega_z**2 + 1e-12).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        eta_x = torch.clamp(grad_omega_x / grad_omega_mag, -1e12, 1e12).nan_to_num(0.0, posinf=1e18, neginf=-1e18)
        eta_y = torch.clamp(grad_omega_y / grad_omega_mag, -1e12, 1e12).nan_to_num(0.0, posinf=1e18, neginf=-1e18)
        eta_z = torch.clamp(grad_omega_z / grad_omega_mag, -1e12, 1e12).nan_to_num(0.0, posinf=1e18, neginf=-1e18)

        # Adaptive epsilon based on local vorticity (preserve strong vortices more)
        if self.phys_config.vc_adaptive:
            # Scale epsilon by vorticity magnitude
            omega_mean = torch.mean(omega_mag)
            epsilon_local = self.phys_config.vorticity_confinement_epsilon * (omega_mag / (omega_mean + 1e-12))
        else:
            epsilon_local = self.phys_config.vorticity_confinement_epsilon

        # Confinement force: F = epsilon * (eta x omega)
        Fx = epsilon_local * (eta_y * omega_z - eta_z * omega_y).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        Fy = epsilon_local * (eta_z * omega_x - eta_x * omega_z).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        Fz = epsilon_local * (eta_x * omega_y - eta_y * omega_x).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

        return Fx, Fy, Fz

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        h = self.config.lbm_config.grid_spacing
        Fx, Fy, Fz = torch.zeros_like(self.velocity_x), torch.zeros_like(self.velocity_x), torch.zeros_like(self.velocity_x)
        self.force_x_accum, self.force_z_accum, self.force_samples = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0
        sample_window = max(10, steps // 4)
        sample_start = steps - sample_window
        for step in range(steps):
            rho = torch.sum(self.f, dim=0)
            mx = torch.sum(self.f * self.ex.view(-1,1,1,1), dim=0)
            my = torch.sum(self.f * self.ey.view(-1,1,1,1), dim=0)
            mz = torch.sum(self.f * self.ez.view(-1,1,1,1), dim=0)
            ux, uy, uz = (mx + 0.5*Fx)/(rho+1e-12), (my + 0.5*Fy)/(rho+1e-12), (mz + 0.5*Fz)/(rho+1e-12)
            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz).nan_to_num(0.0)
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)
            omega = 1.0 / torch.clamp(3.0 * (self.nu + self.nu_turb) + 0.5, min=0.501)
            u_sq = ux**2 + uy**2 + uz**2
            for i in range(27):
                eu, eF, uF = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz, self.ex[i]*Fx + self.ey[i]*Fy + self.ez[i]*Fz, ux*Fx + uy*Fy + uz*Fz
                feq = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                force_term = self.w[i] * (1.0 - 0.5*omega) * (3.0*eF + 9.0*eu*eF - 3.0*uF)
                self.f[i] += omega * (feq - self.f[i]) + force_term
            self.f_pre_stream.copy_(self.f)
            for i in range(27):
                dx, dy, dz = int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item())
                self.f_temp[i] = torch.roll(self.f[i], shifts=(dx,dy,dz), dims=(0,1,2))
                if dx > 0: self.f_temp[i][0,:,:] = self.w[i]*(1.0+3.0*self.ex[i]*self.u_lat+4.5*(self.ex[i]*self.u_lat)**2-1.5*self.u_lat**2)
                if dx < 0: self.f_temp[i][-1,:,:] = self.f_pre_stream[i][-1,:,:]
                if dy != 0: self.f_temp[i][:,0,:], self.f_temp[i][:,-1,:] = self.f_pre_stream[i][:,0,:], self.f_pre_stream[i][:,-1,:]
                if dz != 0: self.f_temp[i][:,:,0], self.f_temp[i][:,:,-1] = self.f_pre_stream[i][:,:,0], self.f_pre_stream[i][:,:,-1]
            mask = geometry_mask > 0.5
            sfx, sfz = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
            for i in range(27):
                opp_i = int(self.opposite[i].item())
                dx, dy, dz = int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item())
                neighbor_is_solid = torch.roll(mask, shifts=(-dx,-dy,-dz), dims=(0,1,2))
                boundary_link = (~mask) & neighbor_is_solid
                if i != 0:
                    self.f_temp[opp_i] = torch.where(boundary_link, self.f_pre_stream[i], self.f_temp[opp_i])
                    if torch.any(boundary_link):
                        sfx += torch.sum(1.0 * float(self.ex[i].item()) * self.f_pre_stream[i][boundary_link])
                        sfz += torch.sum(1.0 * float(self.ez[i].item()) * self.f_pre_stream[i][boundary_link])
            self.force_x_last, self.force_z_last = sfx, sfz
            if step >= sample_start: self.force_x_accum += sfx; self.force_z_accum += sfz; self.force_samples += 1
            self.f.copy_(self.f_temp); self.velocity_x, self.velocity_y, self.velocity_z, self.pressure = ux, uy, uz, rho*self.cs2
            if step % self.phys_config.check_convergence_every == 0 and step > 0:
                u_curr = torch.stack([ux, uy, uz], dim=0)
                if torch.norm(u_curr - self.velocity_prev)/(torch.norm(u_curr)+1e-12) < self.phys_config.convergence_tolerance: break
                self.velocity_prev = u_curr.clone()
        return
    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        h = self.config.lbm_config.grid_spacing
        solid = geometry_mask > 0.5
        ref_area_lat = max(torch.sum(torch.any(solid, dim=0).float()).item(), 1.0)
        df_lat, lf_lat = (self.force_x_accum / self.force_samples, self.force_z_accum / self.force_samples) if self.force_samples > 0 else (self.force_x_last, self.force_z_last)
        q_lat = 0.5 * 1.0 * (self.u_lat**2) * ref_area_lat
        cd, cl = df_lat.item()/q_lat, lf_lat.item()/q_lat
        up = self.config.mach_number * 343.0
        qp = 0.5 * 1.225 * (up**2) * (ref_area_lat * h**2)
        return {"drag_coefficient": cd, "lift_coefficient": cl, "force_x": cd*qp, "force_z": cl*qp, "freestream_speed": up, "density": 1.225, "reference_area": ref_area_lat*h**2, "force_definition": "lattice-native"}
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

        # Initialize populations to equilibrium immediately so tests see valid data
        # (non-NaN, correct shape) on solver construction.
        self._initialize_equilibrium()

    def _estimate_kinematic_viscosity(self):
        """Estimate the lattice kinematic viscosity from the current config."""
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        dt = getattr(self.config.lbm_config, 'time_step', 0.001)
        Re = getattr(self.config, 'reynolds_number', 1e6)
        U_ref = getattr(self.config, 'mach_number', 0.0) * 343.0
        L_ref = h * float(self.resolution)
        nu_phys = (U_ref * L_ref) / max(Re, 1e-12)
        return nu_phys * dt / (h * h)

    def _initialize_equilibrium(self):
        """Initialize solver populations to equilibrium with a small freestream."""
        rho = torch.ones(self.resolution, self.resolution, self.resolution, device=self.device)
        ux = torch.zeros_like(rho)
        uy = torch.zeros_like(rho)
        uz = torch.zeros_like(rho)

        # Seed the equilibrium in lattice units for stability.
        mach = getattr(self.config, 'mach_number', getattr(self.config, 'lbm_config', None) and getattr(self.config.lbm_config, 'mach_number', 0.0))
        if mach:
            ux = torch.full_like(rho, mach / 3.0)

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

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
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

        # run steps
        for _ in range(steps):
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

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        """Compute approximate aerodynamic coefficients from the last simulated
        macroscopic fields. This mirrors the interface used by `GPULBMSolver`.
        """
        # conservative reference area and freestream speed
        h = getattr(self.config.lbm_config, 'grid_spacing', 0.01)
        solid = geometry_mask > 0.5
        ref_area = torch.sum(torch.any(solid, dim=0).float()).item() * h**2
        ref_area = max(ref_area, h**2)

        if self._solver.force_samples > 0:
            drag_force = self._solver.force_x_accum / self._solver.force_samples
            lift_force = self._solver.force_z_accum / self._solver.force_samples
            force_definition = 'bounce-back momentum exchange averaged over the last-quarter window'
        else:
            drag_force = self._solver.force_x_last
            lift_force = self._solver.force_z_last
            force_definition = 'bounce-back momentum exchange from last streaming step'

        physical_drag_force = _scale_momentum_exchange_force(drag_force, h, getattr(self.config, 'mach_number', 0.0))
        physical_lift_force = _scale_momentum_exchange_force(lift_force, h, getattr(self.config, 'mach_number', 0.0))
        coeffs = _compute_force_coefficients(
            physical_drag_force,
            physical_lift_force,
            getattr(self.config, 'mach_number', 0.0),
            ref_area=max(ref_area, 1e-12),
            rho_ref=1.225
        )

        vorticity_mag = self._refresh_flow_diagnostics()
        vortex_cells = torch.sum((self.q_criterion > getattr(self.phys_config, 'q_threshold', 0.0)).float()).item()
        v_inf = coeffs.get('freestream_speed', 0.0)
        nu_turb_mean = float(self.nu_turb.mean().item())
        reynolds_turbulent = float(v_inf * h * self.resolution / max(self.nu + nu_turb_mean, 1e-12))

        return {
            'force_x': float(physical_drag_force.item() if isinstance(physical_drag_force, torch.Tensor) else physical_drag_force),
            'force_z': float(physical_lift_force.item() if isinstance(physical_lift_force, torch.Tensor) else physical_lift_force),
            'raw_force_x': float(drag_force.item() if isinstance(drag_force, torch.Tensor) else drag_force),
            'raw_force_z': float(lift_force.item() if isinstance(lift_force, torch.Tensor) else lift_force),
            'drag_coefficient': coeffs['drag_coefficient'],
            'lift_coefficient': coeffs['lift_coefficient'],
            'force_definition': force_definition,
            'pressure_sum': float(self.pressure.sum().item()),
            'max_turbulent_viscosity': float(self.nu_turb.max().item()),
            'mean_smagorinsky_constant': float(getattr(self.phys_config, 'smagorinsky_constant', 0.17)),
            'max_vorticity': float(vorticity_mag.max().item()),
            'vortex_core_volume': float(vortex_cells * h**3),
            'reference_area': ref_area,
            'reference_length': h * self.resolution,
            'freestream_speed': v_inf,
            'density': coeffs['density'],
            'reynolds_number_turbulent': reynolds_turbulent
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

