
import torch
import numpy as np
from typing import Dict
from scipy.ndimage import binary_dilation
from typing import TYPE_CHECKING

class D3Q27Lattice:
    """D3Q27 velocity vectors and weights"""

    @staticmethod
    def get_vectors():
        # 27 velocity vectors: 1 rest + 6 face + 12 edge + 8 corner
        ex = [0,  # Rest
              1,-1,0,0,0,0,  # Faces (±x, ±y, ±z)
              1,-1,1,-1,1,-1,1,-1,0,0,0,0,  # Edges
              1,-1,1,1,-1,-1,1,-1]  # Corners (±1,±1,±1)

        ey = [0,  # Rest
              0,0,1,-1,0,0,  # Faces
              1,1,-1,-1,0,0,0,0,1,-1,1,-1,  # Edges
              1,1,-1,1,-1,1,-1,-1]  # Corners

        ez = [0,  # Rest
              0,0,0,0,1,-1,  # Faces
              0,0,0,0,1,1,-1,-1,1,1,-1,-1,  # Edges
              1,1,1,-1,1,-1,-1,-1]  # Corners

        return torch.tensor(ex), torch.tensor(ey), torch.tensor(ez)

    @staticmethod
    def get_weights():
        # Weights: 8/27 (rest), 2/27 (face), 1/54 (edge), 1/216 (corner)
        w = [8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8
        return torch.tensor(w, dtype=torch.float32)

    @staticmethod
    def get_opposite():
        # Opposite directions for bounce-back
        opp = [0, 2,1,4,3,6,5, 9,10,7,8,13,14,11,12,17,18,15,16, 26,25,24,23,22,21,20,19]
        return torch.tensor(opp, dtype=torch.int64)

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

        # Streaming
        for i in range(27):
            shifts = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
            self.f_temp[i] = torch.roll(self.f[i], shifts=shifts, dims=(0,1,2))

        # Bounce-back
        for i in range(27):
            mask = geometry_mask > 0.5
            self.f_temp[i] = torch.where(mask, self.f_temp[self.opposite[i]], self.f_temp[i])

        self.f = self.f_temp.clone()
        return ux, uy, uz, rho

class CascadedLBM:
    """Central moments cascaded collision"""

    @staticmethod
    def compute_central_moments(f, ux, uy, uz, ex, ey, ez):
        """Transform populations to central moments"""
        # Shift to moving frame
        cx = ex.view(-1,1,1,1) - ux.unsqueeze(0)
        cy = ey.view(-1,1,1,1) - uy.unsqueeze(0)
        cz = ez.view(-1,1,1,1) - uz.unsqueeze(0)

        K = {}
        # Order 0: density
        K['000'] = torch.sum(f, dim=0)

        # Order 1: momentum (should be ~0 in moving frame)
        K['100'] = torch.sum(f * cx, dim=0)
        K['010'] = torch.sum(f * cy, dim=0)
        K['001'] = torch.sum(f * cz, dim=0)

        # Order 2: energy and stress
        K['200'] = torch.sum(f * cx**2, dim=0)
        K['020'] = torch.sum(f * cy**2, dim=0)
        K['002'] = torch.sum(f * cz**2, dim=0)
        K['110'] = torch.sum(f * cx*cy, dim=0)
        K['101'] = torch.sum(f * cx*cz, dim=0)
        K['011'] = torch.sum(f * cy*cz, dim=0)

        # Order 3+: higher moments (9 more for D3Q19)
        K['111'] = torch.sum(f * cx*cy*cz, dim=0)
        # ... (add remaining moments as needed)
        # Simplified: add basic third order moments
        K['300'] = torch.sum(f * cx**3, dim=0)
        K['030'] = torch.sum(f * cy**3, dim=0)
        K['003'] = torch.sum(f * cz**3, dim=0)
        K['210'] = torch.sum(f * cx**2*cy, dim=0)
        K['201'] = torch.sum(f * cx**2*cz, dim=0)
        K['120'] = torch.sum(f * cx*cy**2, dim=0)
        K['021'] = torch.sum(f * cy**2*cz, dim=0)
        K['102'] = torch.sum(f * cx*cz**2, dim=0)
        K['012'] = torch.sum(f * cy*cz**2, dim=0)

        return K

    @staticmethod
    def equilibrium_central_moments(rho, cs2=1/3):
        """Equilibrium central moments"""
        K_eq = {}
        K_eq['000'] = rho
        K_eq['100'] = K_eq['010'] = K_eq['001'] = torch.zeros_like(rho)
        K_eq['200'] = K_eq['020'] = K_eq['002'] = rho * cs2
        K_eq['110'] = K_eq['101'] = K_eq['011'] = torch.zeros_like(rho)
        K_eq['111'] = torch.zeros_like(rho)
        K_eq['300'] = K_eq['030'] = K_eq['003'] = torch.zeros_like(rho)
        K_eq['210'] = K_eq['201'] = K_eq['120'] = torch.zeros_like(rho)
        K_eq['021'] = K_eq['102'] = K_eq['012'] = torch.zeros_like(rho)
        return K_eq

    @staticmethod
    def cascaded_relax(K, K_eq, s_nu, s_e, s_h):
        """Sequential cascaded relaxation"""
        K_post = {}

        # Step 1: Conserve mass and momentum
        K_post['000'] = K['000']
        K_post['100'] = K['100']
        K_post['010'] = K['010']
        K_post['001'] = K['001']

        # Step 2: Relax energy (uses step 1 results)
        K_post['200'] = K['200'] + s_e * (K_eq['200'] - K['200'])
        K_post['020'] = K['020'] + s_e * (K_eq['020'] - K['020'])
        K_post['002'] = K['002'] + s_e * (K_eq['002'] - K['002'])

        # Step 3: Relax stress (uses steps 1-2 results) → VISCOSITY
        K_post['110'] = K['110'] + s_nu * (K_eq['110'] - K['110'])
        K_post['101'] = K['101'] + s_nu * (K_eq['101'] - K['101'])
        K_post['011'] = K['011'] + s_nu * (K_eq['011'] - K['011'])

        # Step 4: Relax higher (uses steps 1-3)
        K_post['111'] = K['111'] + s_h * (K_eq['111'] - K['111'])
        # Relax higher moments
        for moment in ['300', '030', '003', '210', '201', '120', '021', '102', '012']:
            K_post[moment] = K[moment] + s_h * (K_eq.get(moment, torch.zeros_like(K[moment])) - K[moment])

        return K_post

    @staticmethod
    def moments_to_populations(K, ux, uy, uz, ex, ey, ez, w):
        """Inverse transform: central moments → populations"""
        # This requires precomputed transformation matrix
        # Simplified: use equilibrium + corrections
        rho = K['000']
        feq = torch.zeros(19 if len(ex) == 19 else 27, *rho.shape, device=rho.device)

        n_dirs = len(ex)
        for i in range(n_dirs):
            cu = ex[i]*ux + ey[i]*uy + ez[i]*uz
            u_sq = ux**2 + uy**2 + uz**2
            feq[i] = w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)

        # Add non-equilibrium corrections from K
        # (full implementation needs transformation matrix)
        return feq


class GPULBMSolver:
    """GPU-resident LBM solver with Dynamic Smagorinsky, Vorticity Confinement, and improved vorticity resolution"""

    def __init__(self, config, device: torch.device, phys_config):
        self.config = config
        self.device = device
        self.resolution = config.resolution
        self.phys_config = phys_config()

        self._setup_physics_constants()

        # Structure of Arrays (SoA) layout
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_y = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_z = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.pressure = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)

        # LBM populations (D3Q19)
        self.f = torch.zeros(19, self.resolution, self.resolution, self.resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)

        # Turbulence and vorticity fields
        self.nu_turb = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.cs_dynamic = torch.full((self.resolution, self.resolution, self.resolution), 
                                     self.phys_config.smagorinsky_constant, device=device)

        # Convergence tracking
        self.velocity_prev = torch.zeros_like(self.velocity_x)

        self._setup_d3q19_lattice()
        self._setup_mrt_matrices()
        self._initialize_equilibrium()

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
        self.phys_config.s_nu = 1.0 / tau

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
        """Initialize with improved D3Q19 equilibrium"""
        rho = 1.0
        ux = self.config.mach_number * 343.0
        uy, uz = 0.0, 0.0

        for i in range(19):
            eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
            u_sq = ux*ux + uy*uy + uz*uz

            if i == 0:
                feq = self.w[i] * rho * (1.0 - 1.5 * u_sq)
            elif i <= 6:
                feq = self.w[i] * rho * (1.0 + 3.0*eu - 3.0*u_sq + 6.0*eu*eu)
            else:
                feq = self.w[i] * rho * (1.0 + 3.0*eu - 1.5*u_sq + 4.5*eu*eu)

            self.f[i] = feq

    def _compute_strain_rate_tensor(self, ux, uy, uz):
        """Compute strain rate tensor S_ij = 0.5*(du_i/dx_j + du_j/dx_i)"""
        # Velocity gradients
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2))

        # Strain rate tensor (symmetric)
        S11 = dux_dx
        S22 = duy_dy
        S33 = duz_dz
        S12 = 0.5 * (dux_dy + duy_dx)
        S13 = 0.5 * (dux_dz + duz_dx)
        S23 = 0.5 * (duy_dz + duz_dy)

        return S11, S22, S33, S12, S13, S23

    def _compute_vorticity(self, ux, uy, uz):
        """Compute vorticity omega = curl(u) [web:44]"""
        # Correct gradient unpacking for 3D tensors
        dux_dy, dux_dz = torch.gradient(ux, dim=1)[0], torch.gradient(ux, dim=2)[0]
        duy_dx, duy_dz = torch.gradient(uy, dim=0)[0], torch.gradient(uy, dim=2)[0]
        duz_dx, duz_dy = torch.gradient(uz, dim=0)[0], torch.gradient(uz, dim=1)[0]

        omega_x = duz_dy - duy_dz  # ∂w/∂y - ∂v/∂z
        omega_y = dux_dz - duz_dx  # ∂u/∂z - ∂w/∂x
        omega_z = duy_dx - dux_dy  # ∂v/∂x - ∂u/∂y

        return omega_x, omega_y, omega_z

    def _compute_q_criterion(self, ux, uy, uz):
        """Compute Q-criterion for vortex identification [web:44][web:47]
        Q = 0.5*(||Omega||^2 - ||S||^2) where Omega is rotation rate tensor
        Positive Q indicates vortex regions (rotation > strain)
        """
        # Strain rate tensor magnitude
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag_sq = S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)

        # Vorticity (rotation rate) magnitude
        omega_x, omega_y, omega_z = self._compute_vorticity(ux, uy, uz)
        omega_mag_sq = omega_x**2 + omega_y**2 + omega_z**2

        # Q-criterion: Q > 0 indicates vortex regions
        Q = 0.5 * (omega_mag_sq - S_mag_sq)

        return Q

    def _compute_dynamic_smagorinsky(self, ux, uy, uz):
        """Compute dynamic Smagorinsky constant using Germano identity [web:43][web:46]"""
        Delta = self.config.lbm_config.grid_spacing
        Delta_test = self.phys_config.test_filter_ratio * Delta

        # Grid-scale strain rate
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)

        # Test filter (simple box filter approximation via pooling)
        kernel_size = int(self.phys_config.test_filter_ratio)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        # Apply test filter to velocities (approximation)
        ux_test = torch.nn.functional.avg_pool3d(
            ux.unsqueeze(0).unsqueeze(0), 
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze()
        uy_test = torch.nn.functional.avg_pool3d(
            uy.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze()
        uz_test = torch.nn.functional.avg_pool3d(
            uz.unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size, stride=1, padding=padding
        ).squeeze()

        # Test-scale strain rate
        S11_test, S22_test, S33_test, S12_test, S13_test, S23_test = self._compute_strain_rate_tensor(
            ux_test, uy_test, uz_test
        )
        S_mag_test = torch.sqrt(2.0 * (S11_test**2 + S22_test**2 + S33_test**2 + 
                                       2.0*(S12_test**2 + S13_test**2 + S23_test**2)) + 1e-12)

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

        # Model coefficient (least squares fit)
        # C_s^2 = <L_ij * M_ij> / <M_ij * M_ij>
        # where M_ij = -2 * Delta^2 * |S| * S_ij (grid) + 2 * Delta_test^2 * |S_test| * S_ij_test
        M11 = -2.0*Delta**2*S_mag*S11 + 2.0*Delta_test**2*S_mag_test*S11_test
        M22 = -2.0*Delta**2*S_mag*S22 + 2.0*Delta_test**2*S_mag_test*S22_test
        M33 = -2.0*Delta**2*S_mag*S33 + 2.0*Delta_test**2*S_mag_test*S33_test
        M12 = -2.0*Delta**2*S_mag*S12 + 2.0*Delta_test**2*S_mag_test*S12_test
        M13 = -2.0*Delta**2*S_mag*S13 + 2.0*Delta_test**2*S_mag_test*S13_test
        M23 = -2.0*Delta**2*S_mag*S23 + 2.0*Delta_test**2*S_mag_test*S23_test

        # Compute Cs^2 locally
        LM = L11*M11 + L22*M22 + L33*M33 + 2.0*(L12*M12 + L13*M13 + L23*M23)
        MM = M11**2 + M22**2 + M33**2 + 2.0*(M12**2 + M13**2 + M23**2)

        Cs_squared = LM / (MM + 1e-12)
        Cs_squared = torch.clamp(Cs_squared, min=0.0)  # Ensure non-negative
        Cs = torch.sqrt(Cs_squared)

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
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2))

        # Traceless symmetric part of velocity gradient squared
        # S_d = 0.5*(grad_u + grad_u^T) - (1/3)*tr(grad_u)*I
        gij_gji_11 = dux_dx**2 + dux_dy*duy_dx + dux_dz*duz_dx
        gij_gji_22 = duy_dx*dux_dy + duy_dy**2 + duy_dz*duz_dy
        gij_gji_33 = duz_dx*dux_dz + duz_dy*duy_dz + duz_dz**2

        trace_third = (dux_dx + duy_dy + duz_dz) / 3.0

        Sd_11 = 0.5*(gij_gji_11 + gij_gji_11) - 2.0*trace_third*dux_dx
        Sd_22 = 0.5*(gij_gji_22 + gij_gji_22) - 2.0*trace_third*duy_dy
        Sd_33 = 0.5*(gij_gji_33 + gij_gji_33) - 2.0*trace_third*duz_dz

        Sd_mag = torch.sqrt(Sd_11**2 + Sd_22**2 + Sd_33**2 + 1e-12)

        # Strain rate magnitude
        _, _, _, _, _, _ = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag = torch.sqrt(2.0*(dux_dx**2 + duy_dy**2 + duz_dz**2) + 1e-12)

        # WALE turbulent viscosity
        nu_turb = (Cw * Delta)**2 * (Sd_mag**1.5) / (S_mag**2.5 + Sd_mag**1.25 + 1e-12)

        return nu_turb

    def _compute_turbulent_viscosity(self, ux, uy, uz):
        """Compute turbulent viscosity using selected model"""
        if not self.phys_config.use_les_turbulence:
            return torch.zeros_like(ux)

        if self.phys_config.turbulence_model == "smagorinsky":
            # Standard Smagorinsky
            S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
            S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)
            Cs = self.phys_config.smagorinsky_constant
            Delta = self.config.lbm_config.grid_spacing
            nu_turb = (Cs * Delta)**2 * S_mag

        elif self.phys_config.turbulence_model == "dynamic_smagorinsky":
            # Dynamic Smagorinsky (Germano)
            Cs, S_mag = self._compute_dynamic_smagorinsky(ux, uy, uz)
            self.cs_dynamic = Cs  # Store for diagnostics
            Delta = self.config.lbm_config.grid_spacing
            nu_turb = (Cs * Delta)**2 * S_mag
            
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

        # Compute vorticity
        omega_x, omega_y, omega_z = self._compute_vorticity(ux, uy, uz)
        self.vorticity[0] = omega_x
        self.vorticity[1] = omega_y
        self.vorticity[2] = omega_z

        # Vorticity magnitude
        omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-12)

        # Confinement direction: eta = grad(|omega|) / |grad(|omega|)|
        grad_omega_x, grad_omega_y, grad_omega_z = torch.gradient(omega_mag, dim=(0, 1, 2))
        grad_omega_mag = torch.sqrt(grad_omega_x**2 + grad_omega_y**2 + grad_omega_z**2 + 1e-12)

        eta_x = grad_omega_x / grad_omega_mag
        eta_y = grad_omega_y / grad_omega_mag
        eta_z = grad_omega_z / grad_omega_mag

        # Adaptive epsilon based on local vorticity (preserve strong vortices more)
        if self.phys_config.vc_adaptive:
            # Scale epsilon by vorticity magnitude
            omega_mean = torch.mean(omega_mag)
            epsilon_local = self.phys_config.vorticity_confinement_epsilon * (omega_mag / (omega_mean + 1e-12))
        else:
            epsilon_local = self.phys_config.vorticity_confinement_epsilon

        # Confinement force: F = epsilon * (eta x omega)
        Fx = epsilon_local * (eta_y * omega_z - eta_z * omega_y)
        Fy = epsilon_local * (eta_z * omega_x - eta_x * omega_z)
        Fz = epsilon_local * (eta_x * omega_y - eta_y * omega_x)

        return Fx, Fy, Fz

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        """MRT collision with LES, vorticity confinement, and improved turbulence"""
        h = self.config.lbm_config.grid_spacing
        dt = self.config.lbm_config.time_step

        for step in range(steps):
            # === 1. Compute macroscopic variables ===
            rho = torch.sum(self.f, dim=0)
            ux = torch.sum(self.f * self.ex.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)
            uy = torch.sum(self.f * self.ey.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)
            uz = torch.sum(self.f * self.ez.view(-1, 1, 1, 1), dim=0) / (rho + 1e-12)

            # === 2. Turbulence modeling (Dynamic Smagorinsky / WALE) ===
            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz)
            nu_eff = self.nu + self.nu_turb

            # === 3. Vorticity confinement force ===
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)

            # === 4. Update relaxation time ===
            tau_eff = 3.0 * nu_eff + 0.5
            omega_eff = 1.0 / tau_eff

            # === 5. MRT Collision with force term ===
            for i in range(19):
                eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
                u_sq = ux**2 + uy**2 + uz**2

                # Improved equilibrium
                if i == 0:
                    feq = self.w[i] * rho * (1.0 - 1.5 * u_sq) if self.phys_config.use_incompressible_correction else self.w[i] * rho * (1.0 - 1.5 * u_sq)
                elif i <= 6:
                    feq = self.w[i] * (1.0 + 3.0*eu - 3.0*u_sq + 6.0*eu*eu) if self.phys_config.use_incompressible_correction else self.w[i] * rho * (1.0 + 3.0*eu - 3.0*u_sq + 6.0*eu*eu)
                else:
                    feq = self.w[i] * (1.0 + 3.0*eu - 1.5*u_sq + 4.5*eu*eu) if self.phys_config.use_incompressible_correction else self.w[i] * rho * (1.0 + 3.0*eu - 1.5*u_sq + 4.5*eu*eu)

                # Add vorticity confinement force (Guo forcing scheme)
                Fi = self.ex[i] * Fx + self.ey[i] * Fy + self.ez[i] * Fz
                force_term = self.w[i] * (1.0 - 0.5*omega_eff) * (3.0*Fi)

                # Collision with force
                self.f[i] += omega_eff * (feq - self.f[i]) + force_term

            # === 6. Streaming ===
            for i in range(19):
                shifts = (self.ex[i].item(), self.ey[i].item(), self.ez[i].item())
                self.f_temp[i] = torch.roll(self.f[i], shifts=shifts, dims=(0, 1, 2))

            # === 7. Boundary conditions ===
            for i in range(19):
                opp_i = self.opposite[i]
                mask = geometry_mask > 0.5
                self.f_temp[i] = torch.where(mask, self.f_temp[opp_i], self.f_temp[i])

            self.f = self.f_temp.clone()

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

            if step % 500 == 0:
                vortex_volume = torch.sum((self.q_criterion > self.phys_config.q_threshold).float()).item()
                print(f"Step {step}/{steps}, max vel: {torch.max(torch.sqrt(ux**2 + uy**2 + uz**2)):.4f}, "
                      f"vortex cells: {vortex_volume:.0f}, mean Cs: {self.cs_dynamic.mean():.4f}")

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        """Compute forces using momentum-exchange method with enhanced diagnostics"""
        rho_ref = 1.0
        v_inf = self.config.mach_number * 343.0
        q_inf = 0.5 * rho_ref * v_inf**2
        h = self.config.lbm_config.grid_spacing

        ref_area = torch.sum(torch.any(geometry_mask > 0.5, dim=0).float()) * h**2

        if not self.phys_config.momentum_exchange_correction:
            drag_force = torch.sum(self.velocity_x[geometry_mask > 0.5])
            lift_force = torch.sum(self.velocity_z[geometry_mask > 0.5])
        else:
            # Momentum-Exchange Method
            drag_force = torch.tensor(0.0, device=self.device)
            lift_force = torch.tensor(0.0, device=self.device)

            geom_np = geometry_mask.cpu().numpy().astype(bool)
            dilated = binary_dilation(geom_np, iterations=1)
            boundary_fluid = torch.tensor(dilated & ~geom_np, device=self.device, dtype=torch.bool)

            for i in range(19):
                shifts = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
                geom_shifted = torch.roll(geometry_mask, shifts=shifts, dims=(0,1,2))

                # Fix bitwise AND to logical operations for numpy arrays
                boundary_link_np = np.logical_and(dilated, np.logical_not(geom_np))
                boundary_link = torch.tensor(boundary_link_np, device=self.device, dtype=torch.bool)

                opp_i = self.opposite[i]
                momentum_x = self.ex[i] * (self.f[i] + self.f[opp_i])
                momentum_z = self.ez[i] * (self.f[i] + self.f[opp_i])

                drag_force += torch.sum(momentum_x[boundary_link])
                lift_force += torch.sum(momentum_z[boundary_link])

        cd = abs(drag_force.item()) / (q_inf * ref_area + 1e-10)
        cl = abs(lift_force.item()) / (q_inf * ref_area + 1e-10)

        # Vorticity-based diagnostics
        vorticity_mag = torch.sqrt(torch.sum(self.vorticity**2, dim=0))
        vortex_cells = torch.sum((self.q_criterion > self.phys_config.q_threshold).float()).item()

        return {
            'drag_coefficient': cd,
            'lift_coefficient': cl,
            'pressure_sum': self.pressure.sum().item(),
            'max_turbulent_viscosity': self.nu_turb.max().item(),
            'mean_smagorinsky_constant': self.cs_dynamic.mean().item(),
            'max_vorticity': vorticity_mag.max().item(),
            'vortex_core_volume': vortex_cells * h**3,
            'reynolds_number_turbulent': v_inf * h * self.resolution / (self.nu + self.nu_turb.mean().item())
        }
