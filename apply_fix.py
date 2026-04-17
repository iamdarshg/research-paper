import math

path = 'CLI/advanced_lbm_solver.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = ['import math\n']
for line in lines:
    # 1. Force Scaling Header
    if 'def _scale_momentum_exchange_force(force, grid_spacing: float, mach_number: float, density: float = 1.225):' in line:
        new_lines.append('def _scale_momentum_exchange_force(force, grid_spacing: float, time_step: float, density: float = 1.225):\n')
        new_lines.append('    """Convert raw lattice momentum exchange into a physical force scale.\n')
        new_lines.append('    The correct LBM physical conversion factor for force is rho_phys * (dx^4 / dt^2).\n')
        new_lines.append('    """\n')
        new_lines.append('    force_scale = float(density) * (float(grid_spacing)**4) / (float(time_step)**2)\n')
        new_lines.append('    return force * force_scale\n')
        continue
    if '"""Convert raw lattice momentum exchange' in line or 'freestream_speed =' in line or ' aerodynamic dynamic pressure' in line or 'force_scale =' in line or 'return force * force_scale' in line:
        if '_scale_momentum_exchange_force' not in line:
            continue

    # 2. GPULBMSolver Init
    if 'LBM populations (D3Q19)' in line:
        new_lines.append(line.replace('D3Q19', 'D3Q27'))
        continue
    if 'self.f = torch.zeros(19,' in line:
        new_lines.append(line.replace('19', '27'))
        continue
    if 'self._setup_d3q19_lattice()' in line:
        new_lines.append(line.replace('d3q19', 'd3q27'))
        continue
    if 'self.velocity_prev = torch.zeros_like(self.velocity_x)' in line:
        new_lines.append('        self.velocity_prev = torch.zeros(3, self.resolution, self.resolution, self.resolution, device=device)\n')
        continue

    # 3. Physics Constants
    if 'def _setup_physics_constants(self):' in line:
        new_lines.append(line)
        new_lines.append('        """Compute physics constants with stable mapping."""\n')
        new_lines.append('        h = self.config.lbm_config.grid_spacing\n')
        new_lines.append('        dt = getattr(self.config.lbm_config, \'time_step\', 0.001)\n')
        new_lines.append('        self.cs2 = 1.0 / 3.0\n')
        new_lines.append('        U_phys = self.config.mach_number * 343.0\n')
        new_lines.append('        Re = getattr(self.config, \'reynolds_number\', 1000)\n')
        new_lines.append('        self.u_lat = self.config.mach_number / math.sqrt(3.0)\n')
        new_lines.append('        if self.u_lat > 0.15: self.u_lat = 0.15\n')
        new_lines.append('        self.nu = self.u_lat * self.resolution / max(float(Re), 1e-12)\n')
        new_lines.append('        self.tau = max(3.0 * self.nu + 0.5, 0.501)\n')
        new_lines.append('        self.phys_config.s_nu = 1.0 / self.tau\n')
        skip_until = 'def _setup_d3q19_lattice'
        continue

    # Skip old physics setup
    if 'skip_until' in locals() and skip_until in line:
        del skip_until
    if 'skip_until' in locals():
        continue

    # 4. Lattice Setup
    if 'def _setup_d3q19_lattice(self):' in line:
        new_lines.append('    def _setup_d3q27_lattice(self):\n')
        new_lines.append('        """Setup D3Q27 lattice"""\n')
        new_lines.append('        ex, ey, ez = D3Q27Lattice.get_vectors()\n')
        new_lines.append('        self.ex = ex.to(self.device, dtype=torch.int32)\n')
        new_lines.append('        self.ey = ey.to(self.device, dtype=torch.int32)\n')
        new_lines.append('        self.ez = ez.to(self.device, dtype=torch.int32)\n')
        new_lines.append('        self.w = D3Q27Lattice.get_weights().to(self.device, dtype=torch.float32)\n')
        new_lines.append('        self.opposite = D3Q27Lattice.get_opposite().to(self.device, dtype=torch.int64)\n')
        skip_until_mrt = 'def _setup_mrt_matrices'
        continue
    if 'skip_until_mrt' in locals() and skip_until_mrt in line:
        del skip_until_mrt
    if 'skip_until_mrt' in locals():
        continue

    # 5. MRT matrices
    if 'def _setup_mrt_matrices(self):' in line:
        new_lines.append(line)
        new_lines.append('        self.s_relax = torch.full((27,), 1.0 / self.tau, device=self.device)\n')
        skip_until_init = 'def _initialize_equilibrium'
        continue
    if 'skip_until_init' in locals() and skip_until_init in line:
        del skip_until_init
    if 'skip_until_init' in locals():
        continue

    # 6. Initialize Equilibrium
    if 'def _initialize_equilibrium(self):' in line:
        new_lines.append(line)
        new_lines.append('        ux, uy, uz = self.u_lat, 0.0, 0.0\n')
        new_lines.append('        rho = 1.0\n')
        new_lines.append('        for i in range(27):\n')
        new_lines.append('            eu = self.ex[i].item() * ux + self.ey[i].item() * uy + self.ez[i].item() * uz\n')
        new_lines.append('            u_sq = ux*ux + uy*uy + uz*uz\n')
        new_lines.append('            feq_val = self.w[i].item() * rho * (1.0 + 3.0*eu + 4.5*eu**2 - 1.5*u_sq)\n')
        new_lines.append('            self.f[i].fill_(feq_val)\n')
        skip_until_strain = 'def _compute_strain_rate_tensor'
        continue
    if 'skip_until_strain' in locals() and skip_until_strain in line:
        del skip_until_strain
    if 'skip_until_strain' in locals():
        continue

    new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
