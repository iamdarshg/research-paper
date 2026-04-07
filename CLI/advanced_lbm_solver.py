
import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.ndimage import binary_dilation

class D3Q27Lattice:
    """D3Q27 velocity vectors and weights"""

    @staticmethod
    def get_vectors():
        """
        D3Q27 velocity set:
        - 0: rest (0,0,0)
        - 1-6: face neighbors (±1,0,0), (0,±1,0), (0,0,±1)
        - 7-18: edge neighbors (±1,±1,0), (±1,0,±1), (0,±1,±1)
        - 19-26: corner neighbors (±1,±1,±1)
        """
        ex = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1]
        ey = [0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
        ez = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1]
        return torch.tensor(ex), torch.tensor(ey), torch.tensor(ez)

    @staticmethod
    def get_weights():
        # Weights for D3Q27: 8/27, 2/27, 1/54, 1/216
        w = [8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8
        return torch.tensor(w, dtype=torch.float32)

    @staticmethod
    def get_opposite():
        # Corrected opposite indices for D3Q27
        return torch.tensor([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15, 26, 25, 24, 23, 22, 21, 20, 19], dtype=torch.int64)

def _compute_force_coefficients(force_x, force_z, mach_number, ref_area=1.0, rho_ref=1.225):
    """Compute lift and drag coefficients from forces."""
    v_inf = mach_number * 343.0
    q_inf = 0.5 * rho_ref * v_inf**2
    cd = abs(force_x) / (q_inf * ref_area + 1e-12)
    cl = abs(force_z) / (q_inf * ref_area + 1e-12)
    return {
        'drag_coefficient': cd if isinstance(cd, float) else cd.item(),
        'lift_coefficient': cl if isinstance(cl, float) else cl.item(),
        'freestream_speed': v_inf,
        'density': rho_ref
    }

class CascadedLBM:
    """Tensor-product raw-moment collision for D3Q27."""

    MOMENT_KEYS = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    MOMENT_NAMES = [f"{a}{b}{c}" for a, b, c in MOMENT_KEYS]

    @staticmethod
    def build_moment_basis(ex, ey, ez):
        ex = ex.to(dtype=torch.float32)
        ey = ey.to(dtype=torch.float32)
        ez = ez.to(dtype=torch.float32)
        basis_rows = []
        for a, b, c in CascadedLBM.MOMENT_KEYS:
            basis_rows.append((ex ** a) * (ey ** b) * (ez ** c))
        return torch.stack(basis_rows, dim=0)

    @staticmethod
    def compute_raw_moments(f, ex, ey, ez, basis=None):
        if basis is None:
            basis = CascadedLBM.build_moment_basis(ex, ey, ez)
        basis = basis.to(device=f.device, dtype=f.dtype)
        raw = torch.tensordot(basis, f, dims=([1], [0]))
        return {name: raw[i] for i, name in enumerate(CascadedLBM.MOMENT_NAMES)}

    @staticmethod
    def equilibrium_raw_moments(rho, ux, uy, uz, cs2=1/3):
        K_eq = {}
        def m1d(order, u):
            if order == 0: return torch.ones_like(rho)
            if order == 1: return u
            if order == 2: return u * u + cs2
            raise ValueError(f"Unsupported moment order: {order}")
        for a, b, c in CascadedLBM.MOMENT_KEYS:
            key = f"{a}{b}{c}"
            K_eq[key] = rho * m1d(a, ux) * m1d(b, uy) * m1d(c, uz)
        return K_eq

    @staticmethod
    def cascaded_relax(K, K_eq, s_nu, s_e, s_h):
        K_post = {}
        for key in ("000", "100", "010", "001"):
            K_post[key] = K[key]
        K_post["200"] = K["200"] + s_e * (K_eq["200"] - K["200"])
        K_post["020"] = K["020"] + s_e * (K_eq["020"] - K["020"])
        K_post["002"] = K["002"] + s_e * (K_eq["002"] - K["002"])
        K_post["110"] = K["110"] + s_nu * (K_eq["110"] - K["110"])
        K_post["101"] = K["101"] + s_nu * (K_eq["101"] - K["101"])
        K_post["011"] = K["011"] + s_nu * (K_eq["011"] - K["011"])
        for key in CascadedLBM.MOMENT_NAMES:
            if key in K_post: continue
            K_eq_val = K_eq.get(key, torch.zeros_like(K[key]))
            K_post[key] = K[key] + s_h * (K_eq_val - K[key])
        return K_post

    @staticmethod
    def moments_to_populations(K, moment_matrix_inv):
        K_flat = torch.stack([K[name] for name in CascadedLBM.MOMENT_NAMES], dim=0)
        shape = K_flat.shape[1:]
        f_flat = moment_matrix_inv @ K_flat.reshape(27, -1)
        return f_flat.reshape(27, *shape)

class D3Q27CascadedSolver:
    """D3Q27 LBM solver with cascaded raw-moment collision."""
    def __init__(self, config, device: torch.device, phys_config_class):
        self.config = config
        self.device = device
        self.resolution = config.base_grid_resolution
        self.phys_config = phys_config_class()
        self._setup_physics_constants()
        self.ex, self.ey, self.ez = D3Q27Lattice.get_vectors()
        self.ex = self.ex.to(device)
        self.ey = self.ey.to(device)
        self.ez = self.ez.to(device)
        self.w = D3Q27Lattice.get_weights().to(device)
        self.opposite = D3Q27Lattice.get_opposite().to(device)
        self.moment_basis = CascadedLBM.build_moment_basis(self.ex, self.ey, self.ez).to(device=device, dtype=torch.float32)
        self.moment_matrix_inv = torch.inverse(self.moment_basis)
        self.f = torch.zeros(27, self.resolution, self.resolution, self.resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_y = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_z = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.pressure = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.s_nu = 1.0 / 0.6; self.s_e = 1.2; self.s_h = 1.6
        self._initialize_equilibrium()

    def _setup_physics_constants(self):
        self.cs2 = 1.0 / 3.0
        u_lattice = self.config.mach_number / 3.0
        L_lattice = self.resolution
        Re_target = self.config.reynolds_number
        self.nu = u_lattice * L_lattice / Re_target
        tau = 3.0 * self.nu + 0.5
        self.phys_config.s_nu = 1.0 / tau

    def _initialize_equilibrium(self):
        rho = 1.0; ux = self.config.mach_number / 3.0; uy = uz = 0.0
        for i in range(27):
            eu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
            u_sq = ux**2 + uy**2 + uz**2
            self.f[i] = self.w[i]*rho*(1+3*eu+4.5*eu**2-1.5*u_sq)

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        h = self.config.lbm_config.grid_spacing
        for step in range(steps):
            rho = torch.sum(self.f, dim=0)
            ux = torch.sum(self.f * self.ex.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            uy = torch.sum(self.f * self.ey.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            uz = torch.sum(self.f * self.ez.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            self.f_pre_stream.copy_(self.f)
            K = CascadedLBM.compute_raw_moments(self.f, self.ex, self.ey, self.ez, basis=self.moment_basis)
            K_eq = CascadedLBM.equilibrium_raw_moments(rho, ux, uy, uz)
            self.s_nu = 1.0 / (3.0 * self.nu + 0.5)
            K_post = CascadedLBM.cascaded_relax(K, K_eq, self.s_nu, self.s_e, self.s_h)
            self.f = CascadedLBM.moments_to_populations(K_post, self.moment_matrix_inv)
            for i in range(27):
                dx, dy, dz = int(self.ex[i]), int(self.ey[i]), int(self.ez[i])
                self.f_temp[i] = torch.roll(self.f[i], shifts=(dx, dy, dz), dims=(0, 1, 2))
            mask = geometry_mask > 0.5
            for i in range(27):
                opp_i = int(self.opposite[i])
                self.f_temp[i] = torch.where(mask, self.f_pre_stream[opp_i], self.f_temp[i])
            self.f.copy_(self.f_temp)
            self.velocity_x, self.velocity_y, self.velocity_z = ux, uy, uz
            self.pressure = rho * self.cs2

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        h = self.config.lbm_config.grid_spacing
        ref_area = max(torch.sum(torch.any(geometry_mask > 0.5, dim=0).float()).item() * h**2, h**2)
        drag_force = lift_force = 0.0
        mask = geometry_mask > 0.5
        for i in range(27):
            dx, dy, dz = int(self.ex[i]), int(self.ey[i]), int(self.ez[i])
            neighbor_is_solid = torch.roll(mask, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))
            boundary_link = (~mask) & neighbor_is_solid
            if torch.any(boundary_link):
                drag_force += torch.sum(2.0 * self.ex[i] * self.f_pre_stream[i][boundary_link]).item()
                lift_force += torch.sum(2.0 * self.ez[i] * self.f_pre_stream[i][boundary_link]).item()
        return _compute_force_coefficients(drag_force, lift_force, self.config.mach_number, ref_area=ref_area)

class GPULBMSolver:
    """D3Q19 solver implementation"""
    def __init__(self, config, device, phys_config_class):
        self.config = config
        self.device = device
        self.resolution = config.base_grid_resolution
        self.phys_config = phys_config_class()
        self.ex = torch.tensor([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], device=device)
        self.ey = torch.tensor([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], device=device)
        self.ez = torch.tensor([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], device=device)
        self.w = torch.tensor([1/3] + [1/18]*6 + [1/36]*12, device=device)
        self.opposite = torch.tensor([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15], device=device)
        self.f = torch.zeros(19, self.resolution, self.resolution, self.resolution, device=device)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)
        self.cs2 = 1.0/3.0
        self._initialize_equilibrium()

    def _initialize_equilibrium(self):
        rho = 1.0; ux = self.config.mach_number / 3.0; uy = uz = 0.0
        for i in range(19):
            eu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
            u_sq = ux**2 + uy**2 + uz**2
            self.f[i] = self.w[i]*rho*(1+3*eu+4.5*eu**2-1.5*u_sq)

    def collide_stream(self, geometry_mask, steps=100):
        for step in range(steps):
            rho = torch.sum(self.f, dim=0)
            ux = torch.sum(self.f * self.ex.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            uy = torch.sum(self.f * self.ey.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            uz = torch.sum(self.f * self.ez.view(-1,1,1,1), dim=0) / (rho + 1e-12)
            self.f_pre_stream.copy_(self.f)
            omega = 1.0 / (3.0 * (ux.mean()*self.resolution/self.config.reynolds_number) + 0.5)
            for i in range(19):
                eu = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz
                u_sq = ux**2 + uy**2 + uz**2
                feq = self.w[i]*rho*(1+3*eu+4.5*eu**2-1.5*u_sq)
                self.f[i] += omega * (feq - self.f[i])
            for i in range(19):
                self.f_temp[i] = torch.roll(self.f[i], shifts=(int(self.ex[i]), int(self.ey[i]), int(self.ez[i])), dims=(0,1,2))
            mask = geometry_mask > 0.5
            for i in range(19):
                opp_i = int(self.opposite[i])
                self.f_temp[i] = torch.where(mask, self.f_pre_stream[opp_i], self.f_temp[i])
            self.f.copy_(self.f_temp)

    def compute_aerodynamic_coefficients(self, geometry_mask):
        h = self.config.lbm_config.grid_spacing
        ref_area = max(torch.sum(torch.any(geometry_mask > 0.5, dim=0).float()).item() * h**2, h**2)
        drag_force = lift_force = 0.0
        mask = geometry_mask > 0.5
        for i in range(19):
            dx, dy, dz = int(self.ex[i]), int(self.ey[i]), int(self.ez[i])
            neighbor_is_solid = torch.roll(mask, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))
            boundary_link = (~mask) & neighbor_is_solid
            if torch.any(boundary_link):
                drag_force += torch.sum(2.0 * self.ex[i] * self.f_pre_stream[i][boundary_link]).item()
                lift_force += torch.sum(2.0 * self.ez[i] * self.f_pre_stream[i][boundary_link]).item()
        return _compute_force_coefficients(drag_force, lift_force, self.config.mach_number, ref_area=ref_area)
