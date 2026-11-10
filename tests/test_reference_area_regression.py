"""Regression tests: reference-area normalization of LBM-vs-OpenFOAM errors.

Defect (publication blocker, fixed 2026-11-01): the internal D3Q27 LBM solver
normalizes C_d/C_l by its own voxel-projected frontal area, which is
resolution-dependent (0.2850196 @ g32 vs 0.2385490 @ g64 on the frozen biplane
corpus), while the OpenFOAM sweep normalizes by a fixed reference area
(0.27734375).  The comparison scripts divided the raw coefficients directly, so
``cd_error_percent`` was contaminated by up to +/-16% and the single "0.25%"
passing row (M0.3/of-g64/lbm-g64) was a coincidence of normalizations.

The corrected convention: renormalize the LBM coefficient onto the OpenFOAM
reference area (C = F/(q*Aref) => C_corrected = C_raw * Aref_lbm / Aref_of)
before computing the error, and record both reference areas per row.  The error
becomes a pure force comparison on one documented convention.
"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLI_DIR = REPO / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

from solver_grid_speed_study import compare_cases, renormalize_coefficient  # noqa: E402
from run_simplefoam_low_mach_suite import make_row  # noqa: E402

# Frozen-corpus values (validation clone, low_mach_openfoam_lbm_sweep_20260613):
SWEEP_REFERENCE_AREA = 0.27734375
LBM_REF_AREA_G32 = 0.2850196089378276
LBM_REF_AREA_G64 = 0.23854902052405133
# simpleFoam suite grids:
LBM_REF_AREA_G16 = 0.3965490211308905
LBM_REF_AREA_G24 = 0.3084270164351371


class TestRenormalizeCoefficient(unittest.TestCase):
    def test_renormalization_is_force_consistent(self):
        # Same physical force: C*Aref must be invariant.
        force_proxy = 1.2345
        for source, target in ((0.2850196089378276, 0.27734375),
                               (0.23854902052405133, 0.27734375)):
            c_source = force_proxy / source
            c_target = renormalize_coefficient(c_source, source, target)
            self.assertAlmostEqual(c_target * target, force_proxy, places=12)
            self.assertAlmostEqual(c_target, force_proxy / target, places=12)

    def test_renormalize_returns_none_for_nonfinite_or_nonpositive(self):
        self.assertIsNone(renormalize_coefficient(float("nan"), 1.0, 1.0))
        self.assertIsNone(renormalize_coefficient(1.0, 0.0, 1.0))
        self.assertIsNone(renormalize_coefficient(1.0, 1.0, -1.0))
        self.assertIsNone(renormalize_coefficient(None, 1.0, 1.0))

    def test_g32_factor_matches_frozen_corpus(self):
        # The g32 LBM area is LARGER than the sweep area, so raw LBM Cd must be
        # scaled UP to the sweep convention.
        self.assertAlmostEqual(
            LBM_REF_AREA_G32 / SWEEP_REFERENCE_AREA, 1.027676, places=5
        )

    def test_g64_factor_matches_frozen_corpus(self):
        # The g64 LBM area is SMALLER than the sweep area, so raw LBM Cd must
        # be scaled DOWN to the sweep convention.
        self.assertAlmostEqual(
            LBM_REF_AREA_G64 / SWEEP_REFERENCE_AREA, 0.860120, places=5
        )


def _lbm_case(grid, mach, cd, cl, ref_area):
    return {
        "solver": "internal_lbm_d3q27",
        "grid": grid,
        "mach": mach,
        "speed_mps": mach * 343.0,
        "cd": cd,
        "cl": cl,
        "reference_area": ref_area,
        "timings": {"total_seconds": 10.0},
        "lbm_converged": True,
        "validity": "validated_low_mach_envelope",
        "claim_grade": "low_mach_sanity_only",
        "compressibility_model": "weakly_compressible_isothermal_lbm",
        "thermal_model": "none_isothermal",
    }


def _of_case(grid, mach, cd, cl, ref_area):
    return {
        "solver": "sonicFoam",
        "grid": grid,
        "mach": mach,
        "speed_mps": mach * 343.0,
        "cd": cd,
        "cl": cl,
        "reference_area": ref_area,
        "timings": {"total_seconds": 50.0},
    }


class TestCompareCasesRenormalizesToOpenfoamArea(unittest.TestCase):
    def test_contaminated_coincidence_row_fails_after_normalization(self):
        # Frozen row M0.3 / of-g64 / lbm-g64: raw LBM Cd 0.260365 (Aref 0.238549)
        # vs OF Cd 0.261027 (Aref 0.27734375) looked like 0.25% agreement.
        lbm = _lbm_case(64, 0.3, 0.2603649993990964, 0.02274169574024362, LBM_REF_AREA_G64)
        of = _of_case(64, 0.3, 0.26102668498728554, -0.021593210761182032, SWEEP_REFERENCE_AREA)
        rows = compare_cases([of], [lbm])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        # Raw comparison would be ~0.25%; corrected force comparison must be
        # material (the audit measured 14.2%).
        self.assertGreater(row["cd_error_percent"], 5.0)
        self.assertAlmostEqual(row["lbm_reference_area"], LBM_REF_AREA_G64, places=9)
        self.assertAlmostEqual(row["openfoam_reference_area"], SWEEP_REFERENCE_AREA, places=9)
        self.assertEqual(
            row["reference_area_convention"], "lbm_renormalized_to_openfoam_reference_area"
        )
        # Corrected Cd equals raw Cd scaled by area ratio (force-consistent).
        self.assertAlmostEqual(
            row["corrected_lbm_cd"],
            row["lbm_cd"] * LBM_REF_AREA_G64 / SWEEP_REFERENCE_AREA,
            places=9,
        )

    def test_identical_force_reports_zero_error(self):
        # Same physical drag force on both solvers must give ~0% error once the
        # areas are handled: raw LBM Cd = F/Aref_lbm, raw OF Cd = F/Aref_of.
        force_proxy = 0.35
        lbm = _lbm_case(32, 0.1, force_proxy / LBM_REF_AREA_G32, 0.0, LBM_REF_AREA_G32)
        of = _of_case(32, 0.1, force_proxy / SWEEP_REFERENCE_AREA, 0.0, SWEEP_REFERENCE_AREA)
        rows = compare_cases([of], [lbm])
        self.assertAlmostEqual(rows[0]["cd_error_percent"], 0.0, places=6)

    def test_legacy_without_areas_declines_to_guess_convention(self):
        lbm = _lbm_case(32, 0.1, 1.0, 0.0, None)
        of = _of_case(32, 0.1, 1.0, 0.0, None)
        rows = compare_cases([of], [lbm])
        self.assertNotEqual(
            rows[0]["reference_area_convention"], "lbm_renormalized_to_openfoam_reference_area"
        )


class TestMakeRowRenormalizesSimpleFoam(unittest.TestCase):
    def _of_result(self, grid, mach, cd, cl):
        return {
            "solver": "simpleFoam",
            "mach": mach,
            "speed_mps": mach * 343.0,
            "grid": grid,
            "case_dir": "",
            "commands": {"solve": {"returncode": 0, "seconds": 5.0}},
            "mesh_ok": True,
            "failed": False,
            "latest_time": "5000",
            "force": {
                "cd_total": cd,
                "cl_total": cl,
                "reference_area": SWEEP_REFERENCE_AREA,
            },
        }

    def _lbm_result(self, grid, mach, cd, cl):
        return {
            "solver": "internal_lbm_d3q27",
            "grid": grid,
            "mach": mach,
            "speed_mps": mach * 343.0,
            "cd": cd,
            "cl": cl,
            "reference_area": LBM_REF_AREA_G16 if grid == 16 else LBM_REF_AREA_G24,
            "timings": {"total_seconds": 20.0},
            "lbm_converged": True,
        }

    def test_grid16_row_is_renormalized(self):
        # Frozen row M0.5/g16: raw LBM Cd 0.4111 (Aref g16=0.39655) vs OF Cd
        # 0.4384 (Aref 0.27734).  The contaminated error was 6.23%; the
        # corrected force-comparison error must reflect the g16 area ratio.
        of = self._of_result(16, 0.5, 0.4384240886746072, 0.029534645356712262)
        lbm = self._lbm_result(16, 0.5, 0.41112027561005826, 0.04107863514506811)
        row = make_row(of, lbm, 16, 0.5)
        self.assertAlmostEqual(
            row["lbm_cd_corrected"],
            0.41112027561005826 * LBM_REF_AREA_G16 / SWEEP_REFERENCE_AREA,
            places=9,
        )
        self.assertGreater(row["cd_error_percent"], 5.0)  # 6.23% contaminated row fails the gate
        self.assertEqual(row["openfoam_reference_area"], SWEEP_REFERENCE_AREA)

    def test_identical_force_zero_error_simplefoam(self):
        force_proxy = 0.2
        of = self._of_result(24, 0.2, force_proxy / SWEEP_REFERENCE_AREA, 0.0)
        lbm = self._lbm_result(24, 0.2, force_proxy / LBM_REF_AREA_G24, 0.0)
        row = make_row(of, lbm, 24, 0.2)
        self.assertAlmostEqual(row["cd_error_percent"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()