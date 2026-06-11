from __future__ import annotations

import math
from typing import Any

import torch


D3Q27_LATTICE_SOUND_SPEED = 1.0 / math.sqrt(3.0)
LOW_MACH_VALIDATED_LIMIT = 0.3
LOW_MACH_VALIDITY_REGIME = "validated_low_mach_envelope"
HIGH_MACH_EXPERIMENTAL_REGIME = "experimental_high_mach_unvalidated"


def mach_to_lattice_velocity(mach_number: float) -> float:
    """Map physical Mach number to D3Q27 lattice velocity for the current isothermal model."""
    return float(mach_number) * D3Q27_LATTICE_SOUND_SPEED


def classify_lbm_regime(mach_number: float, external_validation: str | None = None) -> dict[str, Any]:
    """Classify the current internal LBM result without upgrading the underlying physics."""
    mach = float(mach_number)
    mach_magnitude = abs(mach)
    base = {
        "sound_speed_model": "D3Q27 isothermal cs=1/sqrt(3); physical scaling uses a=343 m/s",
        "compressibility_model": "weakly_compressible_isothermal_lbm",
        "thermal_model": "none_isothermal",
    }
    if external_validation:
        return {
            **base,
            "validity_regime": "external_compressible_reference_available",
            "claim_grade": "external_reference_only",
            "high_mach_warning": (
                "Internal D3Q27 remains weakly compressible/isothermal; high-Mach claim support comes only "
                f"from external validation: {external_validation}."
            ),
        }
    if mach_magnitude <= LOW_MACH_VALIDATED_LIMIT:
        return {
            **base,
            "validity_regime": LOW_MACH_VALIDITY_REGIME,
            "claim_grade": "low_mach_sanity_only",
            "high_mach_warning": None,
        }
    return {
        **base,
        "validity_regime": HIGH_MACH_EXPERIMENTAL_REGIME,
        "claim_grade": "no_claim_experimental",
        "high_mach_warning": (
            f"Mach {mach_magnitude:.3g} exceeds the current internal D3Q27 low-Mach validation envelope; "
            "this run is not validated compressible CFD."
        ),
    }


def build_lbm_compressibility_metadata(
    *,
    mach_number: float,
    u_lattice: float,
    lbm_converged: bool,
    force_stability: float | None,
    external_validation: str | None = None,
) -> dict[str, Any]:
    """Build the standard compressibility metadata block for solver outputs and evidence JSON."""
    regime = classify_lbm_regime(mach_number, external_validation=external_validation)
    lattice_mach = float(u_lattice) / D3Q27_LATTICE_SOUND_SPEED
    if regime["validity_regime"] == HIGH_MACH_EXPERIMENTAL_REGIME:
        training_drag_source = "none_high_mach_internal_lbm_unvalidated"
    elif regime["validity_regime"] == "external_compressible_reference_available":
        training_drag_source = "external_validated_reference"
    else:
        training_drag_source = "internal_lbm_raw_low_mach"
    return {
        "mach_number": float(mach_number),
        "lattice_mach": lattice_mach,
        "u_lattice": float(u_lattice),
        **regime,
        "lbm_converged": bool(lbm_converged),
        "force_stability": None if force_stability is None else float(force_stability),
        "training_drag_source": training_drag_source,
    }


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
