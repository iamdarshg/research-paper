
import torch
import numpy as np
from typing import Dict
from scipy.ndimage import binary_dilation
from typing import TYPE_CHECKING

from lbm_utils import D3Q27Lattice, _compute_force_coefficients
from lbm_diagnostics import compute_strain_rate_tensor, compute_vorticity, compute_velocity_gradients

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

    def compute_equilibrium(self, rho, ux, uy, uz):
        feq = torch.zeros_like(self.f)
        for i in range(27):
            cu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
            u_sq = ux**2 + uy**2 + uz**2
            feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        return feq

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
        
        self.f.copy_(self.f_temp)
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

        # LBM populations (D3Q19)
        self.f = torch.zeros(19, self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.f_temp = torch.zeros_like(self.f)+1e-12
        self.f_pre_stream = torch.empty_like(self.f)

        # Turbulence and vorticity fields
        self.nu_turb = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.q_criterion = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)+1e-12
        self.cs_dynamic = torch.full((self.resolution, self.resolution, self.resolution), 
                                     self.phys_config.smagorinsky_constant, device=device)

        # Convergence tracking
        self.velocity_prev = torch.zeros_like(self.velocity_x)

        self._setup_d3q19_lattice()
        self._setup_mrt_matrices()
        self._initialize_equilibrium()

        self.force_x_accum = torch.tensor(0.0, device=device)
        self.force_z_accum = torch.tensor(0.0, device=device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=device)
        self.force_z_last = torch.tensor(0.0, device=device)

    def _setup_physics_constants(self):
        """Compute physics constants from config"""
        h = self.config.lbm_config.grid_spacing
        dt = self.config.lbm_config.time_step

        self.cs2 = 1.0 / 3.0

        U_ref = self.config.mach_number * 343.0
        L_ref = h * self.resolution
        Re = getattr(self.config, 'reynolds_number', 1000)
        nu_phys = U_ref * L_ref / Re

        self.nu = nu_phys * dt / (h * h)
        tau = 3.0 * self.nu + 0.5
        self.phys_config.s_nu = 1.0 / tau  # Ensure this gets set properly

        max_velocity_lattice = self.config.mach_number * 343.0 * dt / h
        if max_velocity_lattice > self.phys_config.max_mach:
            print(f"WARNING: Lattice velocity {max_velocity_lattice:.3f} exceeds stability limit")

    def _setup_d3q19_lattice(self):
        """Setup D3Q19 lattice"""
        ex = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0]
        ey = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1]
        ez = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1]

        self.ex = torch.tensor(ex, dtype=torch.int32, device=self.device)
        self.ey = torch.tensor(ey, dtype=torch.int32, device=self.device)
        self.ez = torch.tensor(ez, dtype=torch.int32, device=self.device)

        w = [1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 
             1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36]
        self.w = torch.tensor(w, dtype=torch.float32, device=self.device)

        self.opposite = torch.tensor([0, 2, 1, 4, 3, 6, 5, 9, 10, 7, 8, 13, 14, 11, 12, 17, 18, 15, 16], 
                                     dtype=torch.int64, device=self.device)

    def _setup_mrt_matrices(self):
        """Setup MRT transformation matrices"""
        s_nu = self.phys_config.s_nu
        s_bulk = self.phys_config.s_bulk
        s_energy = self.phys_config.s_energy
        s_higher = self.phys_config.s_higher

        self.s_relax = torch.tensor([
            1.0, 1.0, 1.0, 1.0, s_energy, s_energy, s_energy,
            s_nu, s_nu, s_nu, s_nu, s_nu, s_nu,
            s_higher, s_higher, s_higher, s_higher, s_higher, s_higher
        ], device=self.device)

    def _initialize_equilibrium(self):
        """Initialize with corrected D3Q19 equilibrium"""
        rho = 1.0
        ux = self.config.mach_number * 343.0
        uy, uz = 0.0, 0.0

        for i in range(19):
            eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
            u_sq = ux*ux + uy*uy + uz*uz

            # Standard D3Q19 equilibrium (ALL directions use same formula)
            feq = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)

            self.f[i] = feq.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

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
        """MRT collision with LES, vorticity confinement, and improved turbulence"""
        h = self.config.lbm_config.grid_spacing
        dt = self.config.lbm_config.time_step

        Fx = torch.zeros_like(self.velocity_x)
        Fy = torch.zeros_like(self.velocity_y)
        Fz = torch.zeros_like(self.velocity_z)
        for step in range(steps):
            # === 1. Compute macroscopic variables ===
            rho = torch.sum(self.f, dim=0)
            momentum_x = torch.sum(self.f * self.ex.view(-1, 1, 1, 1), dim=0)
            momentum_y = torch.sum(self.f * self.ey.view(-1, 1, 1, 1), dim=0)
            momentum_z = torch.sum(self.f * self.ez.view(-1, 1, 1, 1), dim=0)
            ux = (momentum_x + 0.5 * Fx) / (rho + 1e-12)
            uy = (momentum_y + 0.5 * Fy) / (rho + 1e-12)
            uz = (momentum_z + 0.5 * Fz) / (rho + 1e-12)

            # === 2. Turbulence modeling (Dynamic Smagorinsky / WALE) ===
            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
            nu_eff = self.nu+ self.nu_turb

            # === 3. Vorticity confinement force ===
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)

            # === 4. Update relaxation time ===
            tau_eff = 3.0 * nu_eff + 0.5
            omega_eff = 1.0 / tau_eff

            # === 5. MRT Collision with corrected equilibrium ===
            for i in range(19):
                eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
                u_sq = ux**2 + uy**2 + uz**2

                # Correct Guo forcing scheme (exact)
                eF = self.ex[i]*Fx + self.ey[i]*Fy + self.ez[i]*Fz
                uF = ux*Fx + uy*Fy + uz*Fz

                # Standard D3Q19 equilibrium (ALL directions use same formula)
                feq = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                force_term = self.w[i] * (1.0 - 0.5*omega_eff) * (3.0*eF + 9.0*eu*eF - 3.0*uF)

                # Collision with force
                self.f[i] += omega_eff * (feq - self.f[i]) + force_term

            # === 6. Store pre-stream populations for bounce-back ===
            self.f_pre_stream.copy_(self.f)

            # === 7. Streaming ===
            for i in range(19):
                shifts = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
                self.f_temp[i] = torch.roll(self.f[i], shifts=shifts, dims=(0, 1, 2))

                # Prevent periodic wraparound at the outer domain boundary.
                if shifts[0] > 0:
                    self.f_temp[i][0, :, :] = self.f_pre_stream[i][0, :, :]
                elif shifts[0] < 0:
                    self.f_temp[i][-1, :, :] = self.f_pre_stream[i][-1, :, :]
                if shifts[1] > 0:
                    self.f_temp[i][:, 0, :] = self.f_pre_stream[i][:, 0, :]
                elif shifts[1] < 0:
                    self.f_temp[i][:, -1, :] = self.f_pre_stream[i][:, -1, :]
                if shifts[2] > 0:
                    self.f_temp[i][:, :, 0] = self.f_pre_stream[i][:, :, 0]
                elif shifts[2] < 0:
                    self.f_temp[i][:, :, -1] = self.f_pre_stream[i][:, :, -1]

            # === 8. Boundary conditions - bounce-back using pre-stream values ===
            mask = geometry_mask > 0.5
            for i in range(19):
                opp_i = int(self.opposite[i].item())
                self.f_temp[i] = torch.where(mask, self.f_pre_stream[opp_i], self.f_temp[i])

            self.f.copy_(self.f_temp)

            # === 8. Update fields ===
            self.velocity_x = ux
            self.velocity_y = uy
            self.velocity_z = uz
            self.pressure = rho * self.cs2

            # === 9. Compute Q-criterion for vortex detection ===
            if self.phys_config.compute_q_criterion:
                self.q_criterion = self._compute_q_criterion(ux, uy, uz)

            # === 10. Convergence check ===
            if step % self.phys_config.check_convergence_every == 0 and step > 0:
                vel_change = torch.max(torch.abs(ux - self.velocity_prev))
                if vel_change < self.phys_config.convergence_tolerance:
                    print(f"Converged at step {step}, max velocity change: {vel_change:.2e}")
                    break
                self.velocity_prev = ux.clone()

            # === 11. Quick diagnostic (suggested by user) ===
            if step % 100 == 0:
                print(f"Step {step}:")
                print(f"  max vorticity: {torch.max(torch.sqrt(torch.sum(self.vorticity**2, dim=0))):.4f}")
                print(f"  s_nu: {self.phys_config.s_nu}")
                print(f"  nu_turb mean: {self.nu_turb.mean():.6f}")
                print(f"  any NaN: {torch.any(torch.isnan(self.f))}")

            if step % 500 == 0:
                vortex_volume = torch.sum((self.q_criterion > self.phys_config.q_threshold).float()).item()
                print(f"Step {step}/{steps}, max vel: {torch.max(torch.sqrt(ux**2 + uy**2 + uz**2)):.4f}, "
                      f"vortex cells: {vortex_volume:.0f}, mean Cs: {self.cs_dynamic.mean():.4f}")

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        """Compute total hydrodynamic force from fluid-solid links.

        This uses the standard momentum-exchange interpretation of bounce-back: for each
        fluid cell adjacent to the solid, the reflected population transfers 2*f_i*e_i
        to the wall. The result is a total force (pressure + viscous) on the body.
        """
        h = self.config.lbm_config.grid_spacing

        solid = geometry_mask > 0.5
        ref_area = torch.sum(torch.any(solid, dim=0).float()).item() * h**2
        ref_area = max(ref_area, h**2)

        if self.force_samples > 0:
            drag_force = self.force_x_accum / self.force_samples
            lift_force = self.force_z_accum / self.force_samples
            force_definition = 'bounce-back momentum exchange averaged over the last-quarter window'
        else:
            drag_force = self.force_x_last
            lift_force = self.force_z_last
            force_definition = 'bounce-back momentum exchange from last streaming step'

        coeffs = _compute_force_coefficients(
            drag_force,
            lift_force,
            self.config.mach_number,
            ref_area=max(ref_area, 1e-12),
            rho_ref=1.225,
        )

        vorticity_mag = torch.sqrt(torch.sum(self.vorticity**2, dim=0))
        vortex_cells = torch.sum((self.q_criterion > self.phys_config.q_threshold).float()).item()
        v_inf = coeffs['freestream_speed']

        return {
            'force_x': drag_force.item(),
            'force_z': lift_force.item(),
            'drag_coefficient': coeffs['drag_coefficient'],
            'lift_coefficient': coeffs['lift_coefficient'],
            'force_definition': force_definition,
            'pressure_sum': self.pressure.sum().item(),
            'max_turbulent_viscosity': self.nu_turb.max().item(),
            'mean_smagorinsky_constant': self.cs_dynamic.mean().item(),
            'max_vorticity': vorticity_mag.max().item(),
            'vortex_core_volume': vortex_cells * h**3,
            'reference_area': ref_area,
            'reference_length': h * self.resolution,
            'freestream_speed': v_inf,
            'density': coeffs['density'],
            'reynolds_number_turbulent': v_inf * h * self.resolution / (self.nu + self.nu_turb.mean().item())
        }
