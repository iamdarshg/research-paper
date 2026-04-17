import sys

path = 'CLI/advanced_lbm_solver.py'
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
skip = False
for line in lines:
    if 'def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> Dict[str, float]:' in line and not hasattr(sys, 'done'):
        new_lines.append(line)
        new_lines.append('        """Lattice-native coefficient calculation for high precision."""\n')
        new_lines.append('        h = self.config.lbm_config.grid_spacing\n')
        new_lines.append('        solid = geometry_mask > 0.5\n')
        new_lines.append('        ref_area_lat = max(torch.sum(torch.any(solid, dim=0).float()).item(), 1.0)\n')
        new_lines.append('        if self.force_samples > 0:\n')
        new_lines.append('            df_lat, lf_lat = self.force_x_accum / self.force_samples, self.force_z_accum / self.force_samples\n')
        new_lines.append('        else:\n')
        new_lines.append('            df_lat, lf_lat = self.force_x_last, self.force_z_last\n')
        new_lines.append('        q_lat = 0.5 * 1.0 * (self.u_lat**2) * ref_area_lat\n')
        new_lines.append('        cd, cl = df_lat.item()/q_lat, lf_lat.item()/q_lat\n')
        new_lines.append('        up = self.config.mach_number * 343.0\n')
        new_lines.append('        qp = 0.5 * 1.225 * (up**2) * (ref_area_lat * h**2)\n')
        new_lines.append('        return {"drag_coefficient": cd, "lift_coefficient": cl, "force_x": cd*qp, "force_z": cl*qp, "freestream_speed": up, "density": 1.225, "reference_area": ref_area_lat*h**2, "force_definition": "lattice-native"}\n')
        skip = True
        sys.done = True
    elif skip and 'def ' in line:
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)

with open(path, 'w') as f:
    f.writelines(new_lines)
