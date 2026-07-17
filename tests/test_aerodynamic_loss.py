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


class _OccupancyRecordingSimulator:
    def __init__(self):
        self.occupancies = []

    def simulate_aerodynamics(self, geometry, steps=100):
        self.occupancies.append(float(geometry.float().mean().item()))
        return {"training_drag_coefficient": 0.0, "lift_coefficient": 1.0}


class _GeometryCacheSimulator(_GeometrySensitiveSimulator):
    def __init__(self):
        super().__init__()
        self.lbm_solver = type("FakeLBMSolver", (), {})()
        self.lbm_solver._q_cache = {}
        self.lbm_solver._boundary_cache_key = "initial"
        self.lbm_solver._boundary_link_cache = torch.ones(1)
        self.lbm_solver._solver = type("FakeNestedSolver", (), {})()
        self.lbm_solver._solver._q_cache = {}
        self.lbm_solver._solver._boundary_cache_key = "nested-initial"
        self.lbm_solver._solver._boundary_link_cache = torch.ones(1)

    def simulate_aerodynamics(self, geometry, steps=100):
        result = super().simulate_aerodynamics(geometry, steps=steps)
        self.lbm_solver._q_cache[f"geometry-{len(self.calls)}"] = torch.ones(1)
        self.lbm_solver._boundary_cache_key = f"geometry-{len(self.calls)}"
        self.lbm_solver._boundary_link_cache = torch.ones(1)
        self.lbm_solver._solver._q_cache[f"nested-geometry-{len(self.calls)}"] = torch.ones(1)
        self.lbm_solver._solver._boundary_cache_key = f"nested-geometry-{len(self.calls)}"
        self.lbm_solver._solver._boundary_link_cache = torch.ones(1)
        return result


class TestAerodynamicLoss(unittest.TestCase):
    def test_default_loss_weights_are_fractional(self):
        spec = DesignSpec()
        self.assertAlmostEqual(spec.space_weight, 0.33)
        self.assertAlmostEqual(spec.drag_weight, 0.33)
        self.assertAlmostEqual(spec.lift_weight, 0.34)

    def test_percentage_style_loss_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fractional weight"):
            DesignSpec(drag_weight=33.0)

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

    def test_training_loss_contains_only_optimizer_terms(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(
            geometry_reconstruction_weight=3.0,
            generation_reconstruction_weight=5.0,
        )

        optimization_loss = combine_training_loss_terms(
            mse_loss_val=parameter,
            geometry_loss_val=parameter * 2.0,
            generation_geometry_loss_val=parameter * 3.0,
            consistency_loss=parameter * 4.0,
            training_config=config,
        )

        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 26.0, places=6)

    def test_training_loss_includes_clean_latent_geometry_reconstruction(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(clean_geometry_reconstruction_weight=3.0)

        optimization_loss = combine_training_loss_terms(
            mse_loss_val=parameter * 0.0,
            geometry_loss_val=parameter * 0.0,
            generation_geometry_loss_val=parameter * 0.0,
            consistency_loss=parameter * 0.0,
            training_config=config,
            clean_geometry_loss_val=parameter,
        )
        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 3.0, places=6)

    def test_training_loss_snr_weights_only_noisy_geometry_branches(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(clean_geometry_reconstruction_weight=2.0)

        optimization_loss = combine_training_loss_terms(
            mse_loss_val=parameter * 0.0,
            geometry_loss_val=parameter,
            generation_geometry_loss_val=parameter,
            consistency_loss=parameter * 0.0,
            training_config=config,
            clean_geometry_loss_val=parameter,
            direct_solver_loss_val=parameter,
            denoising_geometry_confidence=parameter.new_tensor(0.25),
        )
        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 3.5, places=6)

    def test_training_loss_includes_direct_denoised_latent_reconstruction(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(latent_reconstruction_weight=3.0)

        optimization_loss = combine_training_loss_terms(
            mse_loss_val=parameter * 0.0,
            geometry_loss_val=parameter * 0.0,
            generation_geometry_loss_val=parameter * 0.0,
            consistency_loss=parameter * 0.0,
            training_config=config,
            latent_reconstruction_loss_val=parameter,
        )
        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 3.0, places=6)

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

    def test_direct_solver_spsa_clears_geometry_caches_after_solver_calls(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometryCacheSimulator()
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
        self.assertEqual(simulator.lbm_solver._q_cache, {})
        self.assertIsNone(simulator.lbm_solver._boundary_cache_key)
        self.assertIsNone(simulator.lbm_solver._boundary_link_cache)
        self.assertEqual(simulator.lbm_solver._solver._q_cache, {})
        self.assertIsNone(simulator.lbm_solver._solver._boundary_cache_key)
        self.assertIsNone(simulator.lbm_solver._solver._boundary_link_cache)

    def test_direct_solver_averages_multiple_antithetic_directions_sequentially(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.2,
            gradient_clip=10.0,
            connectivity_weight=0.0,
            directions=2,
            seed=9,
        )

        loss_fn(
            voxels,
            DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0),
            simulator,
            seed=9,
        ).backward()

        self.assertEqual(len(simulator.calls), 5)
        self.assertIsNotNone(voxels.grad)

    def test_direct_solver_can_estimate_rank_invariant_gradient_in_logit_space(self):
        logits = torch.linspace(-3.0, 3.0, 64).reshape(1, 4, 4, 4).requires_grad_(True)
        simulator = _OccupancyRecordingSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.2,
            connectivity_weight=0.0,
            aircraft_validity_weight=0.0,
            directions=1,
            seed=9,
            input_is_logits=True,
        )

        loss_fn(
            logits,
            DesignSpec(),
            simulator,
            reference_occupancy=torch.tensor([0.25]),
        ).backward()

        self.assertEqual(len(simulator.occupancies), 3)
        self.assertTrue(all(abs(value - 0.25) < 1.0e-7 for value in simulator.occupancies))
        self.assertIsNotNone(logits.grad)

    def test_direct_solver_reports_components_and_applies_configured_gradient_limit(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.2,
            gradient_clip=1.0,
            connectivity_weight=1.0,
            aircraft_validity_weight=1.0,
            directions=2,
            seed=9,
        )

        loss_fn(voxels, DesignSpec(), simulator, seed=9).backward()

        expected_limit = 1.0
        self.assertAlmostEqual(
            loss_fn.last_components["spsa_gradient_norm_limit"],
            expected_limit,
            places=7,
        )
        self.assertLessEqual(
            loss_fn.last_components["spsa_gradient_norm"],
            expected_limit + 1e-7,
        )
        self.assertIn("aero_loss", loss_fn.last_components)
        self.assertIn("connectivity_loss", loss_fn.last_components)
        self.assertIn("aircraft_validity_loss", loss_fn.last_components)
        for prefix in ("aero", "connectivity", "aircraft_validity"):
            self.assertIn(
                f"{prefix}_spsa_gradient_norm_unclipped",
                loss_fn.last_components,
            )
            self.assertIn(
                f"{prefix}_spsa_gradient_norm",
                loss_fn.last_components,
            )
            self.assertIn(
                f"{prefix}_spsa_gradient_scale",
                loss_fn.last_components,
            )

    def test_direct_solver_applies_component_trust_region_without_extra_solves(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.2,
            gradient_clip=10.0,
            aero_gradient_max_norm=0.01,
            connectivity_weight=0.0,
            aircraft_validity_weight=0.0,
            directions=2,
            seed=9,
        )

        loss_fn(
            voxels,
            DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0),
            simulator,
            seed=9,
        ).backward()

        self.assertEqual(len(simulator.calls), 5)
        self.assertGreater(
            loss_fn.last_components["aero_spsa_gradient_norm_unclipped"],
            loss_fn.last_components["aero_spsa_gradient_norm"],
        )
        self.assertLessEqual(
            loss_fn.last_components["aero_spsa_gradient_norm"],
            0.01 + 1.0e-7,
        )

    def test_direct_solver_fails_closed_on_nonfinite_measured_value(self):
        voxels = torch.full((1, 4, 4, 4), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _FakeSimulator(
            {
                "training_drag_coefficient": float("nan"),
                "lift_coefficient": 0.0,
            }
        )
        loss_fn = DirectSolverSPSALoss(
            connectivity_weight=0.0,
            aircraft_validity_weight=0.0,
        )

        with self.assertRaisesRegex(FloatingPointError, "nonfinite"):
            loss_fn(
                voxels,
                DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0),
                simulator,
            )

    def test_direct_solver_spsa_adds_aircraft_validity_regression_to_loss(self):
        voxels = torch.full((1, 8, 8, 8), 0.5, dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.35,
            gradient_clip=10.0,
            connectivity_weight=0.0,
            aircraft_validity_weight=4.0,
            target_occupancy=0.1,
            seed=123,
        )

        loss = loss_fn(
            voxels,
            DesignSpec(space_weight=0.0, drag_weight=0.0, lift_weight=0.0),
            simulator,
            seed=123,
        )
        loss.backward()

        self.assertEqual(len(simulator.calls), 3)
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(voxels.grad)
        self.assertGreater(float(voxels.grad.abs().sum()), 0.0)

    def test_direct_solver_uses_each_source_geometry_occupancy_not_empty_threshold_output(self):
        voxels = torch.zeros((1, 4, 4, 4), dtype=torch.float32, requires_grad=True)
        simulator = _GeometrySensitiveSimulator()
        loss_fn = DirectSolverSPSALoss(
            cfd_steps=1,
            perturbation=0.1,
            gradient_clip=10.0,
            connectivity_weight=0.0,
            target_occupancy=None,
            seed=7,
        )

        loss = loss_fn(
            voxels,
            DesignSpec(space_weight=0.0, drag_weight=1.0, lift_weight=0.0),
            simulator,
            seed=7,
            reference_occupancy=torch.tensor([0.25]),
        )
        loss.backward()

        self.assertEqual(len(simulator.calls), 3)
        self.assertTrue(all(abs(call["weighted_occupancy"]) > 0.0 for call in simulator.calls))
        self.assertGreater(float(voxels.grad.abs().sum()), 0.0)

    def test_training_loss_includes_direct_solver_term_when_weighted(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        config = TrainingConfig(
            direct_solver_loss_weight=7.0,
        )

        optimization_loss = combine_training_loss_terms(
            mse_loss_val=parameter,
            geometry_loss_val=parameter,
            generation_geometry_loss_val=parameter,
            consistency_loss=parameter,
            training_config=config,
            direct_solver_loss_val=parameter * 2.0,
        )
        optimization_loss.backward()

        self.assertAlmostEqual(float(parameter.grad), 18.0, places=6)


if __name__ == "__main__":
    unittest.main()
