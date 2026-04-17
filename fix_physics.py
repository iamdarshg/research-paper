import sys
import math

path = 'CLI/advanced_lbm_solver.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if 'def _setup_physics_constants(self):' in line:
        new_lines.append(line)
        new_lines.append('        """Compute physics constants from config with improved stability mapping."""\n')
        new_lines.append('        h = self.config.lbm_config.grid_spacing\n')
        new_lines.append('        import math\n')
        new_lines.append('        self.cs2 = 1.0 / 3.0\n')
        new_lines.append('\n')
        new_lines.append('        # Physical reference values\n')
        new_lines.append('        U_phys = self.config.mach_number * 343.0\n')
        new_lines.append('        Re = getattr(self.config, \'reynolds_number\', 1000)\n')
        new_lines.append('\n')
        new_lines.append('        # Map to lattice units using fixed u_lattice = Ma / sqrt(3) for accuracy\n')
        new_lines.append('        self.u_lat = self.config.mach_number / math.sqrt(3.0)\n')
        new_lines.append('        # Ensure u_lat is not too high for stability\n')
        new_lines.append('        if self.u_lat > 0.15:\n')
        new_lines.append('            print(f"WARNING: Lattice velocity {self.u_lat:.3f} reduced for stability.")\n')
        new_lines.append('            self.u_lat = 0.15\n')
        new_lines.append('        \n')
        new_lines.append('        # Lattice viscosity must match Reynolds number\n')
        new_lines.append('        # Re = u_lat * N / nu_lat  => nu_lat = u_lat * N / Re\n')
        new_lines.append('        self.nu = self.u_lat * self.resolution / max(float(Re), 1e-12)\n')
        new_lines.append('        \n')
        new_lines.append('        self.tau = 3.0 * self.nu + 0.5\n')
        new_lines.append('        # LBGK stability limit is tau > 0.5. We use 0.505 for safety.\n')
        new_lines.append('        self.tau = max(self.tau, 0.505)\n')
        new_lines.append('        self.phys_config.s_nu = 1.0 / self.tau\n')
        skip = True
    elif skip and 'def ' in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

lines = new_lines
new_lines = []
skip = False
for line in lines:
    if 'def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:' in line and not hasattr(sys, 'coeffs_done'):
        new_lines.append(line)
        new_lines.append('        """Compute total hydrodynamic force using lattice-native scaling for improved accuracy."""\n')
        new_lines.append('        h = self.config.lbm_config.grid_spacing\n')
        new_lines.append('        \n')
        new_lines.append('        # Reference area in lattice units (pixels)\n')
        new_lines.append('        solid = geometry_mask > 0.5\n')
        new_lines.append('        ref_area_lat = torch.sum(torch.any(solid, dim=0).float()).item()\n')
        new_lines.append('        ref_area_lat = max(ref_area_lat, 1.0)\n')
        new_lines.append('        ref_area_phys = ref_area_lat * (h**2)\n')
        new_lines.append('\n')
        new_lines.append('        if self.force_samples > 0:\n')
        new_lines.append('            drag_force_lat = self.force_x_accum / self.force_samples\n')
        new_lines.append('            lift_force_lat = self.force_z_accum / self.force_samples\n')
        new_lines.append('            force_definition = \'lattice-native momentum exchange (last-quarter window)\'\n')
        new_lines.append('        else:\n')
        new_lines.append('            drag_force_lat = self.force_x_last\n')
        new_lines.append('            lift_force_lat = self.force_z_last\n')
        new_lines.append('            force_definition = \'lattice-native momentum exchange (last step)\'\n')
        new_lines.append('\n')
        new_lines.append('        # Direct coefficient calculation in lattice units\n')
        new_lines.append('        q_lat = 0.5 * 1.0 * (self.u_lat**2) * ref_area_lat\n')
        new_lines.append('        drag_coeff = (drag_force_lat.item() / q_lat) if q_lat > 0 else 0.0\n')
        new_lines.append('        lift_coeff = (lift_force_lat.item() / q_lat) if q_lat > 0 else 0.0\n')
        new_lines.append('\n')
        new_lines.append('        # Back-calculate physical forces for reporting\n')
        new_lines.append('        U_phys = self.config.mach_number * 343.0\n')
        new_lines.append('        q_phys = 0.5 * 1.225 * (U_phys**2) * ref_area_phys\n')
        new_lines.append('        physical_drag_force = drag_coeff * q_phys\n')
        new_lines.append('        physical_lift_force = lift_coeff * q_phys\n')
        new_lines.append('\n')
        new_lines.append('        coeffs = {\n')
        new_lines.append('            "drag_coefficient": drag_coeff,\n')
        new_lines.append('            "lift_coefficient": lift_coeff,\n')
        new_lines.append('            "freestream_speed": U_phys,\n')
        new_lines.append('            "density": 1.225,\n')
        new_lines.append('        }\n')
        skip = True
        sys.coeffs_done = True
    elif skip and 'vorticity_mag =' in line:
        skip = False
        new_lines.append('\n')
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
