import sys

def replace_block(content, start_marker, end_marker, replacement):
    start_idx = content.find(start_marker)
    if start_idx == -1: return content, False
    end_idx = content.find(end_marker, start_idx + len(start_marker))
    if end_idx == -1: return content, False
    new_content = content[:start_idx] + replacement + content[end_idx:]
    return new_content, True

path = 'CLI/advanced_lbm_solver.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Physics Setup
setup_replacement = """    def _setup_physics_constants(self):
        \"\"\"Compute physics constants from config with improved stability mapping.\"\"\"
        h = self.config.lbm_config.grid_spacing
        self.cs2 = 1.0 / 3.0
        U_phys = self.config.mach_number * 343.0
        Re = getattr(self.config, 'reynolds_number', 1000)
        self.u_lat = self.config.mach_number / math.sqrt(3.0)
        if self.u_lat > 0.15: self.u_lat = 0.15
        self.nu = self.u_lat * self.resolution / max(float(Re), 1e-12)
        self.tau = max(3.0 * self.nu + 0.5, 0.501)
        self.phys_config.s_nu = 1.0 / self.tau\n"""
content, _ = replace_block(content, '    def _setup_physics_constants(self):', '    def _setup_d3q19_lattice(self):', setup_replacement)

# 2. Lattice and MRT
lattice_replacement = """    def _setup_d3q27_lattice(self):
        \"\"\"Setup D3Q27 lattice\"\"\"
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
            self.f[i].fill_(feq_val)\n"""
content, _ = replace_block(content, '    def _setup_d3q19_lattice(self):', '    def _compute_strain_rate_tensor(self, ux, uy, uz, gradients=None):', lattice_replacement)

# 3. Solver Init
content = content.replace('LBM populations (D3Q19)', 'LBM populations (D3Q27)')
content = content.replace('self.f = torch.zeros(19,', 'self.f = torch.zeros(27,')
content = content.replace('self._setup_d3q19_lattice()', 'self._setup_d3q27_lattice()')
content = content.replace('self.velocity_prev = torch.zeros_like(self.velocity_x)', 'self.velocity_prev = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)')

# 4. Collide Stream
collide_replacement = """    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):
        h = self.config.lbm_config.grid_spacing
        Fx = torch.zeros_like(self.velocity_x)
        Fy, Fz = torch.zeros_like(Fx), torch.zeros_like(Fx)
        self.force_x_accum, self.force_z_accum, self.force_samples = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), 0
        sample_window, sample_start = max(10, steps // 4), max(0, steps - max(10, steps // 4))
        for step in range(steps):
            rho = torch.sum(self.f, dim=0)
            ux, uy, uz = (torch.sum(self.f * self.ex.view(-1,1,1,1), dim=0)+0.5*Fx)/(rho+1e-12), (torch.sum(self.f * self.ey.view(-1,1,1,1), dim=0)+0.5*Fy)/(rho+1e-12), (torch.sum(self.f * self.ez.view(-1,1,1,1), dim=0)+0.5*Fz)/(rho+1e-12)
            self.nu_turb = self._compute_turbulent_viscosity(ux, uy, uz).nan_to_num(0.0)
            Fx, Fy, Fz = self._apply_vorticity_confinement(ux, uy, uz)
            omega_eff = 1.0 / torch.clamp(3.0 * (self.nu + self.nu_turb) + 0.5, min=0.501)
            u_sq = ux**2 + uy**2 + uz**2
            for i in range(27):
                eu, eF, uF = self.ex[i]*ux + self.ey[i]*uy + self.ez[i]*uz, self.ex[i]*Fx + self.ey[i]*Fy + self.ez[i]*Fz, ux*Fx + uy*Fy + uz*Fz
                feq = self.w[i] * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)
                force_term = self.w[i] * (1.0 - 0.5*omega_eff) * (3.0*eF + 9.0*eu*eF - 3.0*uF)
                self.f[i] += omega_eff * (feq - self.f[i]) + force_term
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
                boundary_link = (~mask) & torch.roll(mask, shifts=(-dx,-dy,-dz), dims=(0,1,2))
                if i != 0:
                    self.f_temp[opp_i] = torch.where(boundary_link, self.f_pre_stream[i], self.f_temp[opp_i])
                    if torch.any(boundary_link):
                        sfx += torch.sum(2.0 * float(self.ex[i].item()) * self.f_pre_stream[i][boundary_link])
                        sfz += torch.sum(2.0 * float(self.ez[i].item()) * self.f_pre_stream[i][boundary_link])
            self.force_x_last, self.force_z_last = sfx, sfz
            if step >= sample_start: self.force_x_accum += sfx; self.force_z_accum += sfz; self.force_samples += 1
            self.f.copy_(self.f_temp); self.velocity_x, self.velocity_y, self.velocity_z, self.pressure = ux, uy, uz, rho*self.cs2
            if step % self.phys_config.check_convergence_every == 0 and step > 0:
                u_curr = torch.stack([ux, uy, uz], dim=0)
                if torch.norm(u_curr - self.velocity_prev)/(torch.norm(u_curr)+1e-12) < self.phys_config.convergence_tolerance: break
                self.velocity_prev = u_curr.clone()
        return\n"""
content, _ = replace_block(content, '    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100):', '    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:', collide_replacement)

# 5. Aero Coeffs
coeffs_replacement = """    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:
        h = self.config.lbm_config.grid_spacing
        solid = geometry_mask > 0.5
        ref_area_lat = max(torch.sum(torch.any(solid, dim=0).float()).item(), 1.0)
        df_lat, lf_lat = (self.force_x_accum / self.force_samples, self.force_z_accum / self.force_samples) if self.force_samples > 0 else (self.force_x_last, self.force_z_last)
        q_lat = 0.5 * 1.0 * (self.u_lat**2) * ref_area_lat
        cd, cl = df_lat.item()/q_lat, lf_lat.item()/q_lat
        up = self.config.mach_number * 343.0
        qp = 0.5 * 1.225 * (up**2) * (ref_area_lat * h**2)
        return {"drag_coefficient": cd, "lift_coefficient": cl, "force_x": cd*qp, "force_z": cl*qp, "freestream_speed": up, "density": 1.225, "reference_area": ref_area_lat*h**2, "force_definition": "lattice-native"}\n"""
content, _ = replace_block(content, '    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:', 'class D3Q27CascadedSolver:', coeffs_replacement)

with open(path, 'w') as f:
    f.write(content)
