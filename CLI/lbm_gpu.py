import torch
import math
from typing import Dict
from lbm_utils import D3Q27Lattice, _compute_force_coefficients

class GPULBMSolver:
    """GPU-resident LBM solver with D3Q27 Central Moment MRT, Mixed Precision (DDF-Shift), and LES."""

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
        self.velocity_x = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_y = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.velocity_z = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.pressure = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)

        # --- Mixed Precision & DDF Shift Setup ---
        # Store populations in float16 to save bandwidth
        self.f = torch.zeros(27, self.resolution, self.resolution, self.resolution, device=device, dtype=torch.float16)
        self.f_temp = torch.zeros_like(self.f)
        self.f_pre_stream = torch.empty_like(self.f)

        # Precompute f_eq_ref for DDF shift (centered at freestream)
        self.f_eq_ref = self._compute_f_eq_ref()

        # Turbulence and vorticity fields
        self.nu_turb = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.vorticity = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)
        self.q_criterion = torch.zeros(self.resolution, self.resolution, self.resolution, device=device)
        self.cs_dynamic = torch.full((self.resolution, self.resolution, self.resolution), 
                                     self.phys_config.smagorinsky_constant, device=device)

        self.velocity_prev = torch.zeros_like(self.velocity_x)

        self._setup_d3q27_lattice()
        self._setup_mrt_matrices()
        self._initialize_equilibrium()

        self.force_x_accum = torch.tensor(0.0, device=device)
        self.force_z_accum = torch.tensor(0.0, device=device)
        self.force_samples = 0
        self.force_x_last = torch.tensor(0.0, device=device)
        self.force_z_last = torch.tensor(0.0, device=device)

    def _compute_f_eq_ref(self):
        """Compute the reference equilibrium for the DDF shift."""
        rho_ref = 1.0
        ux_ref = self.config.mach_number / 3.0
        uy_ref, uz_ref = 0.0, 0.0
        
        ex, ey, ez = D3Q27Lattice.get_vectors()
        w = D3Q27Lattice.get_weights()
        
        f_eq = torch.zeros(27, device=self.device, dtype=torch.float32)
        u_sq = ux_ref**2 + uy_ref**2 + uz_ref**2
        for i in range(27):
            cu = ex[i]*ux_ref + ey[i]*uy_ref + ez[i]*uz_ref
            f_eq[i] = w[i] * rho_ref * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq)
        
        return f_eq.view(27, 1, 1, 1).expand(27, self.resolution, self.resolution, self.resolution).half()

    def _setup_physics_constants(self):
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

    def _setup_d3q27_lattice(self):
        ex, ey, ez = D3Q27Lattice.get_vectors()
        self.ex = ex.to(self.device, dtype=torch.int32)
        self.ey = ey.to(self.device, dtype=torch.int32)
        self.ez = ez.to(self.device, dtype=torch.int32)
        self.w = D3Q27Lattice.get_weights().to(self.device)
        self.opposite = D3Q27Lattice.get_opposite().to(self.device)

    def _setup_mrt_matrices(self):
        moments = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
        self.moment_indices = moments
        self.idx_map = {m: k for k, m in enumerate(moments)}

        M = torch.zeros((27, 27), device=self.device)
        ex_f, ey_f, ez_f = self.ex.float(), self.ey.float(), self.ez.float()
        for k, (a, b, c) in enumerate(moments):
            M[k] = (ex_f**a) * (ey_f**b) * (ez_f**c)

        self.M_matrix = M
        self.M_inv = torch.inverse(M)

        s_nu = self.phys_config.s_nu
        s_e = getattr(self.phys_config, 's_energy', 1.19)
        s_h = getattr(self.phys_config, 's_higher', 1.4)

        self.s_relax = torch.ones(27, device=self.device) * s_h
        for k, (a, b, c) in enumerate(moments):
            if a + b + c <= 1: self.s_relax[k] = 0.0
            elif a + b + c == 2:
                self.s_relax[k] = s_nu if (a+b==2 or a+c==2 or b+c==2) else s_e

    def _initialize_equilibrium(self):
        rho = 1.0
        ux = self.config.mach_number / 3.0
        uy, uz = 0.0, 0.0
        f_eq = torch.zeros(27, self.resolution, self.resolution, self.resolution, device=self.device)
        for i in range(27):
            eu = self.ex[i] * ux + self.ey[i] * uy + self.ez[i] * uz
            u_sq = ux*ux + uy*uy + uz*uz
            f_eq[i] = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
        self.f.copy_((f_eq - self.f_eq_ref.float()).half())

    def _compute_strain_rate_tensor(self, ux, uy, uz):
        h = self.config.lbm_config.grid_spacing
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=(h, h, h))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=(h, h, h))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=(h, h, h))
        return dux_dx, dux_dy, dux_dz, duy_dx, duy_dy, duy_dz, duz_dx, duz_dy, duz_dz

    def _compute_vorticity(self, ux, uy, uz):
        h = self.config.lbm_config.grid_spacing
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=(h, h, h))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=(h, h, h))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=(h, h, h))
        return duz_dy - duy_dz, dux_dz - duz_dx, duy_dx - dux_dy

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        Fx, Fy, Fz = torch.zeros_like(self.velocity_x), torch.zeros_like(self.velocity_y), torch.zeros_like(self.velocity_z)
        self.force_x_accum, self.force_z_accum, self.force_samples = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0
        sample_start = max(0, steps - max(10, steps // 4))

        for step in range(steps):
            f = self.f.float() + self.f_eq_ref.float()
            rho = torch.sum(f, dim=0)
            mx = torch.sum(f * self.ex.view(-1, 1, 1, 1).float(), dim=0)
            my = torch.sum(f * self.ey.view(-1, 1, 1, 1).float(), dim=0)
            mz = torch.sum(f * self.ez.view(-1, 1, 1, 1).float(), dim=0)
            ux, uy, uz = (mx + 0.5*Fx)/(rho+1e-12), (my + 0.5*Fy)/(rho+1e-12), (mz + 0.5*Fz)/(rho+1e-12)

            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz)
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)
            s_nu_eff = 1.0 / (3.0 * (self.nu + self.nu_turb) + 0.5)

            f_flat = f.reshape(27, -1)
            K = torch.matmul(self.M_matrix, f_flat)
            
            cs2 = 1.0/3.0
            K_eq = torch.zeros_like(K)
            K_eq[0] = rho.flatten()
            for k, (a,b,c) in enumerate(self.moment_indices):
                if a+b+c == 2: K_eq[k] = rho.flatten() * cs2

            s_relax = self.s_relax.view(27, 1)
            for k, (a,b,c) in enumerate(self.moment_indices):
                if (a+b+c == 2) and ((a==1 and b==1) or (a==1 and c==1) or (b==1 and c==1)):
                    s_relax[k] = s_nu_eff

            K += s_relax * (K_eq - K)
            f_flat = torch.matmul(self.M_inv, K)
            f = f_flat.reshape(27, self.resolution, self.resolution, self.resolution)

            f_pre = f.clone()
            for i in range(27):
                sh = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
                f[i] = torch.roll(f[i], shifts=sh, dims=(0, 1, 2))
                if sh[0] > 0: f[i][0, :, :] = f_pre[i][0, :, :]
                elif sh[0] < 0: f[i][-1, :, :] = f_pre[i][-1, :, :]

            mask = geometry_mask > 0.5
            sfx, sfz = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
            for i in range(27):
                opp_i = int(self.opposite[i].item())
                f[i] = torch.where(mask, f_pre[opp_i], f[i])
                if i == 0 or i > opp_i: continue
                sh = (int(self.ex[i].item()), int(self.ey[i].item()), int(self.ez[i].item()))
                boundary_link = (~mask) & torch.roll(mask, shifts=(-sh[0], -sh[1], -sh[2]), dims=(0, 1, 2))
                if torch.any(boundary_link):
                    sfx += torch.sum(2.0 * float(self.ex[i].item()) * f_pre[i][boundary_link])
                    sfz += torch.sum(2.0 * float(self.ez[i].item()) * f_pre[i][boundary_link])

            self.force_x_last, self.force_z_last = sfx, sfz
            if step >= sample_start:
                self.force_x_accum += sfx
                self.force_z_accum += sfz
                self.force_samples += 1

            self.f.copy_((f - self.f_eq_ref.float()).half())
            self.velocity_x, self.velocity_y, self.velocity_z, self.pressure = ux, uy, uz, rho * self.cs2
            if self.phys_config.compute_q_criterion: self.q_criterion = self._compute_q_criterion(ux, uy, uz)

    def _compute_turbulent_viscosity(self, ux, uy, uz):
        if not self.phys_config.use_les_turbulence: return torch.zeros_like(ux)
        if self.phys_config.turbulence_model == "smagorinsky":
            S11, S22, S33, S12, S13, S23 = self._compute_strain_rate_tensor(ux, uy, uz)[:6]
            S_mag = torch.sqrt(2.0 * (S11**2 + S22**2 + S33**2 + 2.0*(S12**2 + S13**2 + S23**2)) + 1e-12)
            return (self.phys_config.smagorinsky_constant * self.config.lbm_config.grid_spacing)**2 * S_mag
        elif self.phys_config.turbulence_model == "dynamic_smagorinsky":
            Cs, S_mag = self._compute_dynamic_smagorinsky(ux, uy, uz)
            self.cs_dynamic = Cs
            return (Cs * self.config.lbm_config.grid_spacing)**2 * S_mag
        elif self.phys_config.turbulence_model == "wale":
            return self._compute_wale_model(ux, uy, uz)
        return torch.zeros_like(ux)

    def _compute_strain_rate_tensor(self, ux, uy, uz):
        h = self.config.lbm_config.grid_spacing
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=(h, h, h))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=(h, h, h))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=(h, h, h))
        return dux_dx, dux_dy, dux_dz, duy_dx, duy_dy, duy_dz, duz_dx, duy_dy, duz_dz

    def _compute_vorticity(self, ux, uy, uz):
        h = self.config.lbm_config.grid_spacing
        dux_dx, dux_dy, dux_dz = torch.gradient(ux, dim=(0, 1, 2), spacing=(h, h, h))
        duy_dx, duy_dy, duy_dz = torch.gradient(uy, dim=(0, 1, 2), spacing=(h, h, h))
        duz_dx, duz_dy, duz_dz = torch.gradient(uz, dim=(0, 1, 2), spacing=(h, h, h))
        return duz_dy - duy_dz, dux_dz - duz_dx, duy_dx - dux_dy

    def _compute_q_criterion(self, ux, uy, uz):
        S = self._compute_strain_rate_tensor(ux, uy, uz)
        S_mag_sq = S[0]**2 + S[4]**2 + S[8]**2 + 2.0*(S[1]**2 + S[2]**2 + S[5]**2)
        omega = self._compute_vorticity(ux, uy, uz)
        omega_mag_sq = omega[0]**2 + omega[1]**2 + omega[2]**2
        return 0.5 * (omega_mag_sq - S_mag_sq).nan_to_num(1e-12)

    def _apply_vorticity_confinement(self, ux, uy, uz):
        if not self.phys_config.use_vorticity_confinement: return torch.zeros_like(ux), torch.zeros_like(uy), torch.zeros_like(uz)
        ox, oy, oz = self._compute_vorticity(ux, uy, uz)
        self.vorticity[0], self.vorticity[1], self.vorticity[2] = ox, oy, oz
        omega_mag = torch.sqrt(ox**2 + oy**2 + oz**2 + 1e-12)
        h = self.config.lbm_config.grid_spacing
        gox, goy, goz = torch.gradient(omega_mag, dim=(0, 1, 2), spacing=(h, h, h))
        gom = torch.sqrt(gox**2 + goy**2 + goz**2 + 1e-12)
        ex, ey, ez = gox/gom, goy/gom, goz/gom
        eps = self.phys_config.vorticity_confinement_epsilon * (omega_mag / (torch.mean(omega_mag) + 1e-12)) if self.phys_config.vc_adaptive else self.phys_config.vorticity_confinement_epsilon
        return eps*(ey*oz - ez*oy), eps*(ez*ox - ex*oz), eps*(ex*oy - ey*ox)

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
