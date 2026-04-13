import torch
from typing import Dict
from lbm_utils import D3Q27Lattice, _compute_force_coefficients

class GPULBMSolver:
    """GPU-resident LBM solver with D3Q27 MRT, Dynamic Smagorinsky, and Vorticity Confinement"""

    def __init__(self, config, device: torch.device, phys_config):
        self.config = config
        self.device = device
        self.resolution = config.resolution
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
        self.velocity_prev = torch.zeros_like(self.velocity_x)

        self._setup_d3q27_lattice()
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
        self.phys_config.s_nu = 1.0 / tau

        max_velocity_lattice = self.config.mach_number * 343.0 * dt / h
        if max_velocity_lattice > self.phys_config.max_mach:
            print(f"WARNING: Lattice velocity {max_velocity_lattice:.3f} exceeds stability limit")

    def _setup_d3q27_lattice(self):
        """Setup D3Q27 lattice"""
        ex, ey, ez = D3Q27Lattice.get_vectors()
        self.ex = ex.to(self.device, dtype=torch.int32)
        self.ey = ey.to(self.device, dtype=torch.int32)
        self.ez = ez.to(self.device, dtype=torch.int32)
        self.w = D3Q27Lattice.get_weights().to(self.device)
        self.opposite = D3Q27Lattice.get_opposite().to(self.device)

    def _setup_mrt_matrices(self):
        """Setup MRT transformation matrices for D3Q27"""
        moments = []
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    moments.append((a, b, c))
        self.moment_indices = moments

        M = torch.zeros((27, 27), device=self.device)
        ex_f = self.ex.float()
        ey_f = self.ey.float()
        ez_f = self.ez.float()
        for k, (a, b, c) in enumerate(moments):
            M[k] = (ex_f**a) * (ey_f**b) * (ez_f**c)

        self.M_matrix = M
        self.M_inv = torch.inverse(M)

        s_nu = self.phys_config.s_nu
        s_e = getattr(self.phys_config, 's_energy', 1.19)
        s_h = getattr(self.phys_config, 's_higher', 1.4)

        self.s_relax = torch.ones(27, device=self.device) * s_h
        for k, (a, b, c) in enumerate(moments):
            if a + b + c <= 1:
                self.s_relax[k] = 0.0 # Conserved
            elif a + b + c == 2:
                if (a==1 and b==1) or (a==1 and c==1) or (b==1 and c==1):
                    self.s_relax[k] = s_nu
                else:
                    self.s_relax[k] = s_e

    def _initialize_equilibrium(self):
        """Initialize with D3Q27 equilibrium"""
        rho = 1.0
        ux = self.config.mach_number / 3.0
        uy, uz = 0.0, 0.0

        for i in range(27):
            eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
            u_sq = ux*ux + uy*uy + uz*uz
            feq = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
            self.f[i] = feq.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

    def _compute_strain_rate_tensor(self, ux, uy, uz):
        grad_spacing = (self.config.lbm_config.grid_spacing,) * 3
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=grad_spacing)
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=grad_spacing)
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=grad_spacing)
        S11 = dux_dx.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S22 = duy_dy.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S33 = duz_dz.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)
        S12 = 0.5 * (dux_dy + duy_dx)
        S13 = 0.5 * (dux_dz + duz_dx)
        S23 = 0.5 * (duy_dz + duz_dy)
        return S11, S22, S33, S12, S13, S23

    def _compute_vorticity(self, ux, uy, uz):
        grad_spacing = (self.config.lbm_config.grid_spacing,) * 3
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=grad_spacing)
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=grad_spacing)
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=grad_spacing)
        omega_x = duz_dy - duy_dz
        omega_y = dux_dz - duz_dx
        omega_z = duy_dx - dux_dy
        return omega_x.nan_to_num(1e-12, posinf=1e18, neginf=-1e18), omega_y.nan_to_num(1e-12, posinf=1e18, neginf=-1e18), omega_z.nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

    def _compute_q_criterion(self, ux, uy, uz):
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag_sq = S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)
        omega_x, omega_y, omega_z = self._compute_vorticity(ux, uy, uz)
        omega_mag_sq = omega_x**2 + omega_y**2 + omega_z**2
        return 0.5 * (omega_mag_sq - S_mag_sq).nan_to_num(1e-12, posinf=1e18, neginf=-1e18)

    def _compute_dynamic_smagorinsky(self, ux, uy, uz):
        Delta = self.config.lbm_config.grid_spacing
        Delta_test = self.phys_config.test_filter_ratio * Delta
        S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)
        kernel_size = int(self.phys_config.test_filter_ratio)
        if kernel_size % 2 == 0: kernel_size += 1
        padding = kernel_size // 2
        ux_test = torch.nn.functional.avg_pool3d(ux.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze()
        uy_test = torch.nn.functional.avg_pool3d(uy.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze()
        uz_test = torch.nn.functional.avg_pool3d(uz.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=padding).squeeze()
        S11_t, S22_t, S33_t, S12_t, S13_t, S23_t = self._compute_strain_rate_tensor(ux_test, uy_test, uz_test)
        S_mag_t = torch.sqrt(2.0 * (S11_t**2 + S22_t**2 + S33_t**2 + 2.0*(S12_t**2 + S13_t**2 + S23_t**2)) + 1e-12)
        L11 = torch.nn.functional.avg_pool3d((ux*ux).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - ux_test*ux_test
        L22 = torch.nn.functional.avg_pool3d((uy*uy).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - uy_test*uy_test
        L33 = torch.nn.functional.avg_pool3d((uz*uz).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - uz_test*uz_test
        L12 = torch.nn.functional.avg_pool3d((ux*uy).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - ux_test*uy_test
        L13 = torch.nn.functional.avg_pool3d((ux*uz).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - ux_test*uz_test
        L23 = torch.nn.functional.avg_pool3d((uy*uz).unsqueeze(0).unsqueeze(0), kernel_size, 1, padding).squeeze() - uy_test*uz_test
        M11 = -2.0*Delta**2*S_mag*S11 + 2.0*Delta_test**2*S_mag_t*S11_t
        M22 = -2.0*Delta**2*S_mag*S22 + 2.0*Delta_test**2*S_mag_t*S22_t
        M33 = -2.0*Delta**2*S_mag*S33 + 2.0*Delta_test**2*S_mag_t*S33_t
        M12 = -2.0*Delta**2*S_mag*S12 + 2.0*Delta_test**2*S_mag_t*S12_t
        M13 = -2.0*Delta**2*S_mag*S13 + 2.0*Delta_test**2*S_mag_t*S13_t
        M23 = -2.0*Delta**2*S_mag*S23 + 2.0*Delta_test**2*S_mag_t*S23_t
        LM = L11*M11 + L22*M22 + L33*M33 + 2.0*(L12*M12 + L13*M13 + L23*M23)
        MM = M11**2 + M22**2 + M33**2 + 2.0*(M12**2 + M13**2 + M23**2)
        Cs_sq = torch.clamp(LM / (MM + 1e-12), min=1e-12)
        Cs = torch.sqrt(Cs_sq).clamp(min=self.phys_config.dynamic_cs_clip_min, max=self.phys_config.dynamic_cs_clip_max)
        return Cs, S_mag

    def _compute_wale_model(self, ux, uy, uz):
        Delta = self.config.lbm_config.grid_spacing
        Cw = self.phys_config.wale_constant
        grad_spacing = (Delta,) * 3
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0,1,2), spacing=grad_spacing)
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0,1,2), spacing=grad_spacing)
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0,1,2), spacing=grad_spacing)
        gij_gji_11 = dux_dx**2 + dux_dy*duy_dx + dux_dz*duz_dx
        gij_gji_22 = duy_dx*dux_dy + duy_dy**2 + duy_dz*duz_dy
        gij_gji_33 = duz_dx*dux_dz + duz_dy*duy_dz + duz_dz**2
        trace_third = (dux_dx + duy_dy + duz_dz) / 3.0
        Sd_11 = 0.5*(gij_gji_11 + gij_gji_11) - 2.0*trace_third*dux_dx
        Sd_22 = 0.5*(gij_gji_22 + gij_gji_22) - 2.0*trace_third*duy_dy
        Sd_33 = 0.5*(gij_gji_33 + gij_gji_33) - 2.0*trace_third*duz_dz
        Sd_mag = torch.sqrt(Sd_11**2 + Sd_22**2 + Sd_33**2 + 1e-12)
        S_mag = torch.sqrt(2.0*(dux_dx**2 + duy_dy**2 + duz_dz**2) + 1e-12)
        return (Cw * Delta)**2 * (Sd_mag**1.5) / (S_mag**2.5 + Sd_mag**1.25 + 1e-12)

    def _compute_turbulent_viscosity(self, ux, uy, uz):
        if not self.phys_config.use_les_turbulence: return torch.zeros_like(ux)
        if self.phys_config.turbulence_model == "smagorinsky":
            S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)
            S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)
            return (self.phys_config.smagorinsky_constant * self.config.lbm_config.grid_spacing)**2 * S_mag
        elif self.phys_config.turbulence_model == "dynamic_smagorinsky":
            Cs, S_mag = self._compute_dynamic_smagorinsky(ux, uy, uz)
            self.cs_dynamic = Cs
            return (Cs * self.config.lbm_config.grid_spacing)**2 * S_mag
        elif self.phys_config.turbulence_model == "wale":
            return self._compute_wale_model(ux, uy, uz)
        return torch.zeros_like(ux)

    def _apply_vorticity_confinement(self, ux, uy, uz):
        if not self.phys_config.use_vorticity_confinement: return torch.zeros_like(ux), torch.zeros_like(uy), torch.zeros_like(uz)
        ox, oy, oz = self._compute_vorticity(ux, uy, uz)
        self.vorticity[0], self.vorticity[1], self.vorticity[2] = ox, oy, oz
        omega_mag = torch.sqrt(ox**2 + oy**2 + oz**2 + 1e-12)
        grad_spacing = (self.config.lbm_config.grid_spacing,) * 3
        gox, goy, goz = torch.gradient(omega_mag, dim=(0, 1, 2), spacing=grad_spacing)
        gom = torch.sqrt(gox**2 + goy**2 + goz**2 + 1e-12)
        ex, ey, ez = gox/gom, goy/gom, goz/gom
        eps = self.phys_config.vorticity_confinement_epsilon * (omega_mag / (torch.mean(omega_mag) + 1e-12)) if self.phys_config.vc_adaptive else self.phys_config.vorticity_confinement_epsilon
        return eps*(ey*oz - ez*oy), eps*(ez*ox - ex*oz), eps*(ex*oy - ey*ox)

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        Fx, Fy, Fz = torch.zeros_like(self.velocity_x), torch.zeros_like(self.velocity_y), torch.zeros_like(self.velocity_z)
        self.force_x_accum, self.force_z_accum, self.force_samples = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0
        sample_start = max(0, steps - max(10, steps // 4))
        for step in range(steps):
            rho = torch.sum(self.f, dim=0)
            mx = torch.sum(self.f * self.ex.view(-1, 1, 1, 1), dim=0)
            my = torch.sum(self.f * self.ey.view(-1, 1, 1, 1), dim=0)
            mz = torch.sum(self.f * self.ez.view(-1, 1, 1, 1), dim=0)
            ux, uy, uz = (mx + 0.5*Fx)/(rho+1e-12), (my + 0.5*Fy)/(rho+1e-12), (mz + 0.5*Fz)/(rho+1e-12)
            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz)
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)
            s_nu_eff = 1.0 / (3.0 * (self.nu + self.nu_turb) + 0.5)
            f_flat = self.f.reshape(27, -1)
            K = torch.matmul(self.M_matrix, f_flat)
            cs2 = 1.0/3.0
            for k, (a, b, c) in enumerate(self.moment_indices):
                if a + b + c <= 1: continue
                def m1d(order, u):
                    if order == 0: return 1.0
                    if order == 1: return u
                    if order == 2: return u*u + cs2
                    return 0.0
                keq = rho.view(-1) * m1d(a, ux.view(-1)) * m1d(b, uy.view(-1)) * m1d(c, uz.view(-1))
                s = self.s_relax[k]
                if (a+b+c == 2) and ((a==1 and b==1) or (a==1 and c==1) or (b==1 and c==1)): s = s_nu_eff.view(-1)
                eF, uF = self.ex[k]*Fx + self.ey[k]*Fy + self.ez[k]*Fz, ux*Fx + uy*Fy + uz*Fz
                force_moment = self.w[k] * (1.0 - 0.5*s) * (3.0*eF + 9.0*(self.ex[k]*ux+self.ey[k]*uy+self.ez[k]*uz)*eF - 3.0*uF)
                K[k] = K[k] + s * (keq - K[k]) + force_moment.view(-1)
            self.f.copy_(torch.matmul(self.M_inv, K).reshape(self.f.shape))
            self.f_pre_stream.copy_(self.f)
            for i in range(27):
                dx, dy, dz = int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item())
                self.f_temp[i] = torch.roll(self.f[i], shifts=(dx, dy, dz), dims=(0, 1, 2))
                if dx > 0: self.f_temp[i][0, :, :] = self.f_pre_stream[i][0, :, :]
                elif dx < 0: self.f_temp[i][-1, :, :] = self.f_pre_stream[i][-1, :, :]
            mask = geometry_mask > 0.5
            sfx, sfz = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
            for i in range(27):
                opp_i = int(self.opposite[i].item())
                self.f_temp[i] = torch.where(mask, self.f_pre_stream[opp_i], self.f_temp[i])
                if i == 0 or i > opp_i: continue
                dx, dy, dz = int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item())
                boundary_link = (~mask) & torch.roll(mask, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))
                if not torch.any(boundary_link): continue
                sfx += torch.sum(2.0 * float(self.ex[i].item()) * self.f_pre_stream[i][boundary_link])
                sfz += torch.sum(2.0 * float(self.ez[i].item()) * self.f_pre_stream[i][boundary_link])
            self.force_x_last, self.force_z_last = sfx, sfz
            if step >= sample_start: self.force_x_accum += sfx; self.force_z_accum += sfz; self.force_samples += 1
            self.f.copy_(self.f_temp)
            self.velocity_x, self.velocity_y, self.velocity_z, self.pressure = ux, uy, uz, rho * self.cs2
            if self.phys_config.compute_q_criterion: self.q_criterion = self._compute_q_criterion(ux, uy, uz)

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        h = self.config.lbm_config.grid_spacing
        solid = geometry_mask > 0.5
        ref_area = max(torch.sum(torch.any(solid, dim=0).float()).item() * h**2, h**2)
        df = self.force_x_accum / self.force_samples if self.force_samples > 0 else self.force_x_last
        lf = self.force_z_accum / self.force_samples if self.force_samples > 0 else self.force_z_last
        fs = self.config.mach_number * 343.0
        scale = 0.5 * 1.225 * fs * fs * h * h
        p_df, p_lf = df * scale, lf * scale
        coeffs = _compute_force_coefficients(p_df, p_lf, self.config.mach_number, ref_area=ref_area)
        return {
            'force_x': p_df.item(), 'force_z': p_lf.item(), 'drag_coefficient': coeffs['drag_coefficient'],
            'lift_coefficient': coeffs['lift_coefficient'], 'pressure_sum': self.pressure.sum().item(),
            'max_vorticity': torch.max(torch.sqrt(torch.sum(self.vorticity**2, dim=0))).item(),
            'reference_area': ref_area, 'freestream_speed': fs
        }
