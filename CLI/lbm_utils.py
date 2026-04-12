from __future__ import annotations

import torch


def _compute_force_coefficients(force_x, force_z, mach_number, ref_area, rho_ref=1.225):
    """Shared force normalization for drag/lift coefficients.

    The solvers are expected to supply forces in a consistent physical force
    scale. This helper only applies the aerodynamic coefficient denominator:
    C = F / (0.5 * rho_inf * U_inf^2 * A_ref).

    The caller is responsible for supplying a consistent reference area,
    ideally the voxelized projected area of the solid body.
    """
    v_inf = mach_number * 343.0
    q_inf = 0.5 * rho_ref * v_inf**2
    ref_area = max(float(ref_area), 1e-12)
    denom = q_inf * ref_area + 1e-12

    return {
        "drag_coefficient": force_x.item() / denom,
        "lift_coefficient": force_z.item() / denom,
        "freestream_speed": v_inf,
        "density": rho_ref,
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
