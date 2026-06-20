import os
import sys
import unittest

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (
    AerodynamicLoss,
    ConnectivityLoss,
    DesignSpec,
    TrainingConfig,
    combine_training_loss_terms,
)


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

    def test_solver_and_connectivity_scores_are_detached_diagnostics(self):
        voxels = torch.rand((1, 4, 4, 4), dtype=torch.float32, requires_grad=True)
        spec = DesignSpec(space_weight=1.0, drag_weight=1.0, lift_weight=1.0)
        simulator = _FakeSimulator(
            {
                "training_drag_coefficient": 0.4,
                "lift_coefficient": 0.2,
            }
        )

        self.assertFalse(ConnectivityLoss()(voxels).requires_grad)
        self.assertFalse(AerodynamicLoss()(voxels, spec, simulator).requires_grad)

    def test_training_loss_backpropagates_only_differentiable_terms(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(
            geometry_reconstruction_weight=3.0,
            generation_reconstruction_weight=5.0,
        )

        optimization_loss, diagnostic_total = combine_training_loss_terms(
            mse_loss_val=parameter,
            geometry_loss_val=parameter * 2.0,
            generation_geometry_loss_val=parameter * 3.0,
            consistency_loss=parameter * 4.0,
            connectivity_loss_val=torch.tensor(100.0),
            aero_loss_val=torch.tensor(1000.0),
            training_config=config,
        )

        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 26.0, places=6)
        self.assertAlmostEqual(float(diagnostic_total), float(optimization_loss.detach()) + 1100.0, places=6)


if __name__ == "__main__":
    unittest.main()
