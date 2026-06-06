import os
import sys
import unittest

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import AerodynamicLoss, DesignSpec


class _FakeSimulator:
    def __init__(self, results):
        self.results = results

    def simulate_aerodynamics(self, geometry, steps=100):
        return dict(self.results)


class TestAerodynamicLoss(unittest.TestCase):
    def test_prefers_training_drag_coefficient_over_raw_drag(self):
        loss_fn = AerodynamicLoss()
        spec = DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0)
        voxels = torch.ones((1, 2, 2, 2), dtype=torch.float32)
        simulator = _FakeSimulator(
            {
                "drag_coefficient": 6.5,
                "calibrated_drag_coefficient": 0.8,
                "training_drag_coefficient": 0.4,
                "lift_coefficient": 0.0,
            }
        )

        loss = loss_fn(voxels, spec, simulator)

        self.assertAlmostEqual(float(loss.item()), 0.4, places=6)

    def test_falls_back_to_calibrated_drag_then_raw_drag(self):
        loss_fn = AerodynamicLoss()
        spec = DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0)
        voxels = torch.ones((1, 2, 2, 2), dtype=torch.float32)

        calibrated_only = _FakeSimulator(
            {
                "drag_coefficient": 6.5,
                "calibrated_drag_coefficient": 0.8,
                "lift_coefficient": 0.0,
            }
        )
        raw_only = _FakeSimulator(
            {
                "drag_coefficient": 0.6,
                "lift_coefficient": 0.0,
            }
        )

        calibrated_loss = loss_fn(voxels, spec, calibrated_only)
        raw_loss = loss_fn(voxels, spec, raw_only)

        self.assertAlmostEqual(float(calibrated_loss.item()), 0.8, places=6)
        self.assertAlmostEqual(float(raw_loss.item()), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
