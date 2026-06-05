import os
import sys
import unittest

import torch
from torch.utils.data import TensorDataset


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from run_monitored_training import _build_epoch_dataset
from training_stability import compute_core_loss, summarize_stability


class TestTrainingStability(unittest.TestCase):
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

    def test_compute_core_loss_ignores_aerodynamic_term(self):
        metrics = {
            "mse": 1.0,
            "geometry_reconstruction": 2.0,
            "consistency": 3.0,
            "connectivity": 4.0,
            "aerodynamic": 100.0,
        }

        self.assertEqual(compute_core_loss(metrics), 10.0)

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
