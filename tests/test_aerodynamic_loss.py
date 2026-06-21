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
    DirectSolverSPSALoss,
    TrainingConfig,
    combine_training_loss_terms,
)


class _FakeSimulator:
    def __init__(self, results):
        self.results = results

    def simulate_aerodynamics(self, geometry, steps=100):
        return dict(self.results)


class _GeometrySensitiveSimulator:
    def __init__(self):
        self.calls = []

    def simulate_aerodynamics(self, geometry, steps=100):
        geom = geometry.float()
        weights = torch.linspace(0.1, 1.0, geom.numel(), device=geom.device, dtype=geom.dtype).reshape_as(geom)
        weighted_occupancy = float((geom * weights).sum().detach().cpu().item() / weights.sum().detach().cpu().item())
        self.calls.append({"steps": steps, "weighted_occupancy": weighted_occupancy})
        return {
            "training_drag_coefficient": 0.1 + weighted_occupancy,
            "lift_coefficient": 0.0,
        }


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

    def test_direct_solver_spsa_loss_runs_solver_and_backpropagates(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=3,
            perturbation=0.35,
            gradient_clip=10.0,
            connectivity_weight=0.0,
            seed=123,
        )

        loss = loss_fn(voxels, DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0), simulator, seed=123)
        loss.backward()

        self.assertEqual(len(simulator.calls), 3)
        self.assertTrue(all(call["steps"] == 3 for call in simulator.calls))
        self.assertTrue(loss.requires_grad)
        self.assertIsNotNone(voxels.grad)
        self.assertGreater(float(voxels.grad.abs().sum()), 0.0)

    def test_training_loss_includes_direct_solver_term_when_weighted(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(
            direct_solver_loss_weight=7.0,
        )

        optimization_loss, _ = combine_training_loss_terms(
            mse_loss_val=parameter,
            geometry_loss_val=parameter,
            generation_geometry_loss_val=parameter,
            consistency_loss=parameter,
            connectivity_loss_val=torch.tensor(100.0),
            aero_loss_val=torch.tensor(1000.0),
            training_config=config,
            direct_solver_loss_val=parameter * 2.0,
        )
        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 18.0, places=6)


if __name__ == "__main__":
    unittest.main()
