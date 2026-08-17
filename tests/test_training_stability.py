import os
import sys
import unittest

import torch
from torch.utils.data import TensorDataset


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from run_monitored_training import (
    RunLocalCosineScheduler,
    _build_epoch_dataset,
    _run_state_checkpoint_due,
    _geometry_non_regression,
    _geometry_promotion_metrics,
    _restore_best_promotion_rank,
    _sync_best_checkpoint_state,
)
from training_stability import compute_core_loss, summarize_stability
from training_stability import evaluate_directional_promotion_gate


class TestTrainingStability(unittest.TestCase):
    def test_directional_gate_rejects_completed_epoch_topology_regression(self):
        baseline = {
            "generated_mean_occupied_fraction": 0.05,
            "target_mean_occupied_fraction": 0.02,
            "generated_mean_largest_component_fraction": 0.9872801371,
            "reconstruction_recall": 0.2431656392,
            "generated_aircraft_valid_fraction": 0.0,
            "generated_unique_fraction": 0.9479166667,
            "materialization_mode": "fixed_global_threshold",
            "geometry_threshold_calibrated": True,
        }
        candidate = {
            "generated_mean_occupied_fraction": 0.01,
            "target_mean_occupied_fraction": 0.02,
            "generated_mean_largest_component_fraction": 0.5276170658,
            "reconstruction_recall": 0.0,
            "generated_aircraft_valid_fraction": 0.1041666667,
            "generated_unique_fraction": 0.5416666667,
            "materialization_mode": "fixed_global_threshold",
            "geometry_threshold_calibrated": True,
        }

        report = evaluate_directional_promotion_gate(candidate, baseline)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(
            report["failed_conditions"],
            [
                "largest_component_floor",
                "largest_component_non_regression",
                "reconstruction_recall_non_regression",
                "uniqueness_non_regression",
            ],
        )
        self.assertEqual(report["conditions"]["generated_occupancy_error"]["passed"], True)

    def test_directional_gate_requires_strict_validity_improvement_below_half(self):
        baseline = {
            "generated_mean_occupied_fraction": 0.2,
            "target_mean_occupied_fraction": 0.1,
            "generated_mean_largest_component_fraction": 0.8,
            "reconstruction_recall": 0.5,
            "generated_aircraft_valid_fraction": 0.4,
            "generated_unique_fraction": 0.8,
            "materialization_mode": "fixed_global_threshold",
            "geometry_threshold_calibrated": True,
        }
        candidate = {**baseline, "generated_mean_occupied_fraction": 0.1}

        report = evaluate_directional_promotion_gate(candidate, baseline)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["failed_conditions"], ["generated_validity_improvement"])

    def test_run_local_scheduler_decays_each_group_to_nonzero_floor(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(
            [
                {"params": [first], "lr": 2.0e-4},
                {"params": [second], "lr": 5.0e-5},
            ]
        )
        scheduler = RunLocalCosineScheduler(
            optimizer,
            total_updates=4,
            min_lr_ratio=0.1,
        )

        observed = []
        for _ in range(4):
            scheduler.step()
            observed.append([group["lr"] for group in optimizer.param_groups])

        self.assertGreater(observed[0][0], observed[-1][0])
        self.assertAlmostEqual(observed[-1][0], 2.0e-5)
        self.assertAlmostEqual(observed[-1][1], 5.0e-6)
        self.assertTrue(all(value > 0.0 for row in observed for value in row))

    def test_run_state_checkpoint_cadence_is_relative_to_segment_start(self):
        self.assertFalse(_run_state_checkpoint_due(7, 0, 8))
        self.assertTrue(_run_state_checkpoint_due(8, 0, 8))
        self.assertFalse(_run_state_checkpoint_due(15, 8, 8))
        self.assertTrue(_run_state_checkpoint_due(16, 8, 8))

    def test_geometry_promotion_rank_includes_validity_diversity_and_shape(self):
        metrics, rank = _geometry_promotion_metrics(
            {
                "status": "fail",
                "reconstruction_topk_recall": 0.9,
                "generated_topk_recall": 0.7,
                "generated_worst_topk_recall": 0.6,
                "generated_aircraft_valid_fraction": 1.0 / 3.0,
                "generated_unique_fraction": 0.8,
                "generated_mean_largest_component_fraction": 0.9,
                "generated_mean_normalization_boundary_fraction": 0.02,
            }
        )

        self.assertEqual(
            rank,
            (1.0 / 3.0, -0.0, 0.8, 0.9, -0.02, 0.6, 0.7, 0.9),
        )
        self.assertAlmostEqual(metrics["geometry_selection_metric"], 0.3)
        self.assertEqual(metrics["promotion_gate_passed"], 0.0)

    def test_restore_best_promotion_rank_round_trips_and_falls_back(self):
        """R5 (PR 41 review, item 5): a persisted list rank restores to the
        tuple gate used by the lexicographic comparison, and run-states that
        predate the field (or hold garbage) fall back without raising."""
        persisted = [0.75, -0.01, 0.8, 0.9, -0.02, 0.6, 0.7, 0.9]
        self.assertEqual(
            _restore_best_promotion_rank({"best_promotion_rank": persisted}),
            (0.75, -0.01, 0.8, 0.9, -0.02, 0.6, 0.7, 0.9),
        )
        self.assertEqual(_restore_best_promotion_rank({}), (-1.0,) * 8)
        self.assertEqual(
            _restore_best_promotion_rank({"best_promotion_rank": "bad"}),
            (-1.0,) * 8,
        )
        self.assertEqual(
            _restore_best_promotion_rank({"best_promotion_rank": None}),
            (-1.0,) * 8,
        )

    def test_sync_best_checkpoint_state_mirrors_into_run_state_metadata(self):
        """R5: the best-checkpoint selection is mirrored into the trainer's
        run_state_metadata, which build_run_state persists verbatim."""
        class _FakeTrainer:
            def __init__(self):
                self.run_state_metadata = {}

        trainer = _FakeTrainer()
        _sync_best_checkpoint_state(
            trainer,
            best_promotion_rank=(0.75, -0.01, 0.8, 0.9, -0.02, 0.6, 0.7, 0.9),
            best_geometry_metric=0.31,
            best_checkpoint_path=r"C:\out\best_geometry_model.pt",
        )
        self.assertEqual(
            trainer.run_state_metadata["best_promotion_rank"],
            [0.75, -0.01, 0.8, 0.9, -0.02, 0.6, 0.7, 0.9],
        )
        self.assertEqual(trainer.run_state_metadata["best_geometry_metric"], 0.31)
        self.assertEqual(
            trainer.run_state_metadata["best_checkpoint_path"],
            r"C:\out\best_geometry_model.pt",
        )

    def test_geometry_non_regression_rejects_boundary_and_diversity_collapse(self):
        baseline = {
            "generated_aircraft_valid_fraction": 0.75,
            "generated_unique_fraction": 1.0,
            "generated_mean_largest_component_fraction": 0.9,
            "generated_mean_normalization_boundary_fraction": 0.01,
            "generated_worst_topk_recall": 0.2,
        }
        candidate = {
            "generated_aircraft_valid_fraction": 0.75,
            "generated_unique_fraction": 0.5,
            "generated_mean_largest_component_fraction": 0.9,
            "generated_mean_normalization_boundary_fraction": 0.5,
            "generated_worst_topk_recall": 0.2,
        }

        decision = _geometry_non_regression(candidate, baseline)

        self.assertEqual(decision["status"], "fail")
        self.assertEqual(
            decision["failed_checks"],
            [
                "generated_unique_fraction",
                "generated_mean_normalization_boundary_fraction",
            ],
        )

    def test_geometry_non_regression_rejects_aircraft_validity_drop(self):
        baseline = {
            "generated_aircraft_valid_fraction": 0.5,
            "generated_unique_fraction": 1.0,
            "generated_mean_largest_component_fraction": 0.9,
            "generated_mean_normalization_boundary_fraction": 0.01,
            "generated_worst_topk_recall": 0.2,
        }
        candidate = {
            **baseline,
            "generated_aircraft_valid_fraction": 0.25,
        }

        decision = _geometry_non_regression(candidate, baseline)

        self.assertEqual(decision["status"], "fail")
        self.assertEqual(
            decision["failed_checks"],
            ["generated_aircraft_valid_fraction"],
        )

    def test_geometry_non_regression_rejects_worse_occupancy_collapse(self):
        baseline = {
            "generated_aircraft_valid_fraction": 0.0,
            "generated_unique_fraction": 1.0,
            "generated_mean_largest_component_fraction": 1.0,
            "generated_mean_normalization_boundary_fraction": 0.5,
            "generated_worst_recall": 1.0,
            "generated_mean_occupied_fraction": 0.75,
            "target_mean_occupied_fraction": 0.01,
        }
        candidate = {
            **baseline,
            "generated_mean_occupied_fraction": 0.90,
        }

        decision = _geometry_non_regression(candidate, baseline)

        self.assertEqual(decision["status"], "fail")
        self.assertEqual(
            decision["failed_checks"],
            ["generated_occupancy_error"],
        )

    def test_build_epoch_dataset_uses_deterministic_subset(self):
        dataset = TensorDataset(torch.arange(10))

        subset_a = _build_epoch_dataset(dataset, max_samples_per_epoch=4, subset_seed=7)
        subset_b = _build_epoch_dataset(dataset, max_samples_per_epoch=4, subset_seed=7)

        self.assertEqual(len(subset_a), 4)
        self.assertEqual(subset_a.indices, subset_b.indices)

    def test_build_epoch_dataset_returns_full_dataset_when_limit_disabled(self):
        dataset = TensorDataset(torch.arange(5))

        full_dataset = _build_epoch_dataset(dataset, max_samples_per_epoch=0, subset_seed=0)

        self.assertIs(full_dataset, dataset)

    def test_build_epoch_dataset_honors_manifest_train_split(self):
        dataset = TensorDataset(torch.arange(6))
        dataset.metadata = {
            "split_assignments": ["train", "val", "train", "test", "train", "holdout"]
        }

        training = _build_epoch_dataset(
            dataset,
            max_samples_per_epoch=0,
            subset_seed=0,
        )

        self.assertEqual(training.indices, [0, 2, 4])

    def test_compute_core_loss_ignores_aerodynamic_term(self):
        metrics = {
            "mse": 1.0,
            "geometry_reconstruction": 2.0,
            "generation_reconstruction": 3.0,
            "consistency": 4.0,
            "connectivity": 4.0,
            "aerodynamic": 100.0,
        }

        self.assertEqual(compute_core_loss(metrics), 10.0)

    def test_compute_core_loss_prefers_explicit_optimization_loss(self):
        metrics = {
            "optimization_loss": 7.5,
            "mse": 1.0,
            "geometry_reconstruction": 2.0,
            "generation_reconstruction": 3.0,
            "consistency": 4.0,
            "connectivity": 400.0,
            "aerodynamic": 1000.0,
        }

        self.assertEqual(compute_core_loss(metrics), 7.5)

    def test_summarize_stability_marks_converged_window(self):
        history = []
        for epoch in range(1, 25):
            history.append(
                {
                    "epoch": epoch,
                    "loss": 12.0 + 0.02 * (epoch % 2),
                    "aerodynamic": 1.0,
                    "connectivity": 3.0,
                    "core_loss": 12.0 + 0.02 * (epoch % 2),
                }
            )

        report = summarize_stability(
            history,
            metric="core_loss",
            window=20,
            convergence_target=13.0,
            convergence_cv_threshold=0.01,
            convergence_drift_threshold=0.05,
            oscillation_cv_threshold=0.30,
        )

        self.assertTrue(report["converged"])
        self.assertEqual(report["status"], "converged")

    def test_stable_random_guess_geometry_is_not_called_converged(self):
        history = [
            {
                "epoch": epoch,
                "loss": 5.0,
                "optimization_loss": 5.0,
                "clean_geometry_reconstruction": 0.693,
                "generation_reconstruction": 0.693,
                "direct_aero_loss": 0.5,
                "direct_connectivity_loss": 0.4,
            }
            for epoch in range(1, 6)
        ]

        report = summarize_stability(
            history,
            metric="optimization_loss",
            window=5,
            convergence_target=20.0,
            required_geometry_loss_max=0.2,
        )

        self.assertFalse(report["converged"])
        self.assertEqual(report["status"], "stable_not_learned")
        self.assertFalse(report["geometry_learning_ready"])

    def test_summarize_stability_flags_aerodynamic_dominance(self):
        history = []
        for epoch in range(1, 25):
            history.append(
                {
                    "epoch": epoch,
                    "loss": (-1.0) ** epoch * 60.0,
                    "aerodynamic": (-1.0) ** epoch * 55.0,
                    "connectivity": 4.0 + 0.1 * epoch,
                    "core_loss": 5.0 + 0.02 * epoch,
                }
            )

        report = summarize_stability(
            history,
            metric="core_loss",
            window=20,
            convergence_target=4.0,
            convergence_cv_threshold=0.01,
            convergence_drift_threshold=0.01,
            oscillation_cv_threshold=0.30,
        )

        self.assertTrue(report["oscillating"])
        self.assertEqual(report["suspected_root_cause"], "aerodynamic_loss_dominance")

    def test_summarize_stability_flags_aerodynamic_objective_drift(self):
        history = []
        for epoch in range(1, 25):
            history.append(
                {
                    "epoch": epoch,
                    "loss": 100.0 + epoch * 8.0,
                    "aerodynamic": 95.0 + epoch * 8.0,
                    "connectivity": 0.0,
                    "core_loss": 0.95 + 0.02 * (epoch % 3),
                }
            )

        report = summarize_stability(
            history,
            metric="core_loss",
            window=20,
            convergence_target=1.05,
            convergence_cv_threshold=0.10,
            convergence_drift_threshold=0.05,
            oscillation_cv_threshold=0.25,
        )

        self.assertFalse(report["oscillating"])
        self.assertTrue(report["aerodynamic_diverging"])
        self.assertEqual(report["suspected_root_cause"], "aerodynamic_objective_drift")


if __name__ == "__main__":
    unittest.main()
