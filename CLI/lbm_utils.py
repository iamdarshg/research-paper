from __future__ import annotations

import torch


def _compute_force_coefficients(force_x, force_z, mach_number, ref_area, rho_ref=1.225):
    """Shared force normalization for drag/lift coefficients.

    The solvers accumulate lattice-unit momentum-exchange forces. Convert them
    with the same lattice-to-physical scaling and dynamic-pressure convention
    so the D3Q27 and D3Q19 paths report coefficients on the same basis.
    """
    v_inf = mach_number * 343.0
    q_inf = 0.5 * rho_ref * v_inf**2
    # Use a lattice-scale Mach normalization tied to the lattice sound speed.
    u_lattice = max(mach_number / (3.0 ** 0.5), 1e-12)
    force_scale = 1.0 / (u_lattice ** 2)

    force_x_phys = force_x * force_scale
    force_z_phys = force_z * force_scale
    denom = q_inf * ref_area + 1e-10

    return {
        "drag_coefficient": force_x_phys.item() / denom,
        "lift_coefficient": force_z_phys.item() / denom,
        "freestream_speed": v_inf,
        "density": rho_ref,
        "force_scale": force_scale,
    }


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
        ex = [0]
        ey = [0]
        ez = [0]

        for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            ex.append(dx)
            ey.append(dy)
            ez.append(dz)

        for (dx, dy, dz) in [(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
                             (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                             (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1)]:
            ex.append(dx)
            ey.append(dy)
            ez.append(dz)

        for (dx, dy, dz) in [(1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
                             (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)]:
            ex.append(dx)
            ey.append(dy)
            ez.append(dz)

        return torch.tensor(ex), torch.tensor(ey), torch.tensor(ez)

    @staticmethod
    def get_weights():
        # Weights: 8/27 (rest), 2/27 (face), 1/54 (edge), 1/216 (corner)
        w = [8 / 27] + [2 / 27] * 6 + [1 / 54] * 12 + [1 / 216] * 8
        return torch.tensor(w, dtype=torch.float32)

    @staticmethod
    def get_opposite():
        """
        Opposite directions for D3Q27:
        0 (rest) → 0
        1-6 (faces): ±x, ±y, ±z → swap pairs
        7-18 (edges): diagonal pairs
        19-26 (corners): 3D diagonal pairs
        """
        opp = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 26, 25, 24, 23, 22, 21, 20, 19]
        return torch.tensor(opp, dtype=torch.int64)
