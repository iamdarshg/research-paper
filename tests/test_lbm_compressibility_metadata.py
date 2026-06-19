import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from advanced_lbm_solver import D3Q27CascadedSolver
from lbm_utils import build_lbm_compressibility_metadata


class TestPhysicsConfig:
    max_mach = 1.0
    target_lattice_velocity = 1.0
    tau_min_d3q27 = 0.52
    s_e_d3q27 = 1.2
    s_h_d3q27 = 1.6
    drag_link_metric_exponent = None
    use_triton_streaming = False
    convergence_tolerance = 1e-5
    check_convergence_every = 10
    smagorinsky_constant = 0.17
    q_threshold = 0.0
    use_shape_drag_correction = False


def make_config(mach_number: float):
    lbm_config = SimpleNamespace(grid_spacing=0.125, time_step=0.001, physical_length_scale=1.0)
    return SimpleNamespace(
        base_grid_resolution=8,
        resolution=8,
        mach_number=mach_number,
        reynolds_number=100.0,
        simulation_steps=1,
        lbm_config=lbm_config,
    )


class TestLBMCompressibilityMetadata(unittest.TestCase):
    def test_metadata_helper_emits_required_fields_for_high_mach(self):
        metadata = build_lbm_compressibility_metadata(
            mach_number=0.8,
            u_lattice=0.12,
            lbm_converged=True,
            force_stability=0.02,
        )
        for key in (
            "mach_number",
            "lattice_mach",
            "u_lattice",
            "sound_speed_model",
            "compressibility_model",
            "thermal_model",
            "validity_regime",
            "claim_grade",
            "high_mach_warning",
            "lbm_converged",
            "force_stability",
            "training_drag_source",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(metadata["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")

    def test_solver_coefficients_include_regime_metadata(self):
        cfg = make_config(0.5)
        solver = D3Q27CascadedSolver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0
        results = solver.compute_aerodynamic_coefficients(geometry)
        self.assertEqual(results["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["label_tier"], "lbm_raw")
        self.assertEqual(results["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")

    def test_high_mach_internal_lbm_output_is_not_pinn_ready(self):
        cfg = make_config(0.8)
        solver = D3Q27CascadedSolver(cfg, torch.device("cpu"), TestPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0
        solver.collide_stream(geometry, steps=1)
        results = solver.compute_aerodynamic_coefficients(geometry)
        self.assertFalse(results.get("pinn_ready", False))
        self.assertEqual(results["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")


if __name__ == "__main__":
    unittest.main()
