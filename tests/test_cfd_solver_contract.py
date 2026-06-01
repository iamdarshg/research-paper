import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import AdvancedCFDSimulator


class _FakeLBMSolver:
    def __init__(self):
        self.steps = None
        self.mask_shape = None

    def collide_stream(self, geometry_mask, steps=100):
        self.steps = steps
        self.mask_shape = tuple(geometry_mask.shape)

    def compute_aerodynamic_coefficients(self, geometry_mask):
        return {
            "drag_coefficient": 0.31,
            "lift_coefficient": 0.12,
            "reference_area": 2.5,
            "label_tier": "lbm_raw",
            "lbm_converged": False,
        }


class TestCFDSolverContract(unittest.TestCase):
    def _simulator(self):
        simulator = object.__new__(AdvancedCFDSimulator)
        simulator.config = SimpleNamespace(
            solver_type="D3Q27",
            base_grid_resolution=8,
            use_amr=False,
            lbm_config=SimpleNamespace(physical_length_scale=2.0, grid_spacing=0.25),
        )
        simulator.device = torch.device("cpu")
        simulator.resolution = 8
        simulator.lbm_solver = _FakeLBMSolver()
        simulator.amr_solver = None
        return simulator

    def test_heuristic_fluidx3d_proxy_is_reported_but_not_blended_into_primary_coefficients(self):
        simulator = self._simulator()
        geometry = torch.zeros((8, 8, 8))
        geometry[2:6, 3:5, 3:5] = 1.0

        with mock.patch.object(
            simulator,
            "_run_fluidx3d_validation",
            return_value={
                "drag_coefficient": 9.0,
                "lift_coefficient": 9.0,
                "label_tier": "heuristic_proxy",
                "claim_bearing": False,
            },
        ):
            results = simulator.simulate_aerodynamics(geometry, steps=7)

        self.assertEqual(results["drag_coefficient"], 0.31)
        self.assertEqual(results["lift_coefficient"], 0.12)
        self.assertEqual(results["external_validation"]["status"], "heuristic_proxy_not_blended")
        self.assertFalse(results["claim_bearing_cfd"])
        self.assertEqual(results["solver_provenance"]["primary_solver"], "D3Q27")
        self.assertEqual(results["solver_provenance"]["steps"], 7)
        self.assertGreater(results["reference_area"], 0.0)

    def test_fluidx3d_fast_proxy_is_labeled_non_claim_bearing(self):
        simulator = self._simulator()

        proxy = simulator._run_fluidx3d_fast("fake.stl")

        self.assertEqual(proxy["label_tier"], "heuristic_proxy")
        self.assertFalse(proxy["claim_bearing"])
        self.assertIn("not claim-bearing", proxy["claim_boundary"])


if __name__ == "__main__":
    unittest.main()
