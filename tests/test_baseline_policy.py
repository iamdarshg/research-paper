import os
import sys
import unittest
from pathlib import Path

import yaml


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from multi_seed_eval import (
    build_baseline_statistics_report,
    build_statistical_summary,
    validate_baseline_policy,
)
from aircraft_diffusion_cfd import build_baseline_family_report


class TestBaselinePolicy(unittest.TestCase):
    def test_statistical_summary_blocks_when_seed_count_is_too_low(self):
        summary = build_statistical_summary(
            [{"seed": 0, "lift_to_drag": 4.0}],
            metric_keys=["lift_to_drag"],
            min_seeds=3,
        )

        self.assertEqual(summary["status"], "blocked")
        self.assertIn("insufficient seeds", summary["blockers"][0])

    def test_statistical_summary_reports_mean_and_std(self):
        summary = build_statistical_summary(
            [
                {"seed": 0, "lift_to_drag": 4.0},
                {"seed": 1, "lift_to_drag": 6.0},
                {"seed": 2, "lift_to_drag": 8.0},
            ],
            metric_keys=["lift_to_drag"],
            min_seeds=3,
        )

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["metrics"]["lift_to_drag"]["mean"], 6.0)
        self.assertGreater(summary["metrics"]["lift_to_drag"]["std"], 0.0)

    def test_statistical_summary_blocks_non_finite_metric_values(self):
        summary = build_statistical_summary(
            [
                {"seed": 0, "lift_to_drag": 4.0},
                {"seed": 1, "lift_to_drag": float("nan")},
                {"seed": 2, "lift_to_drag": 8.0},
            ],
            metric_keys=["lift_to_drag"],
            min_seeds=3,
        )

        self.assertEqual(summary["status"], "blocked")
        self.assertIn(
            "metric lift_to_drag has non-finite values for seeds: 1",
            summary["blockers"],
        )

    def test_baseline_policy_requires_named_claim_bearing_baselines(self):
        report = validate_baseline_policy(
            {
                "baseline_name": "minimal_grounded_aircraft",
                "stl_paths": ["../biplane.stl"],
            },
            required_baselines=["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"],
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("missing baseline_set", report["blockers"][0])

    def test_checked_in_baseline_config_names_required_baseline_families(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((repo_root / "CLI" / "baseline_config.yaml").read_text(encoding="utf-8"))

        report = validate_baseline_policy(config)

        self.assertEqual(report["status"], "pass")

    def test_baseline_statistics_blocks_missing_required_baseline_outputs(self):
        report = build_baseline_statistics_report(
            baseline_config={
                "baseline_name": "claim_bearing_aircraft",
                "baseline_set": ["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"],
            },
            baseline_report={
                "baselines": {
                    "bundled_grounded_stl": {
                        "status": "pass",
                        "results": {"F-18_Hornet.stl": {"lift_to_drag": 5.0}},
                    },
                },
            },
            condition_validation_report={
                "correlations": {"target_speed_vs_measured_drag": {"r": 0.4, "p": 0.1}},
                "raw_data": {
                    "measured_drag": [1.0, 2.0, 3.0],
                    "measured_lift": [4.0, 6.0, 9.0],
                    "occupancy": [0.1, 0.2, 0.3],
                },
            },
            min_seeds=3,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn(
            "missing required baseline report families: retrieval, unconditional_checkpoint",
            report["blockers"],
        )

    def test_baseline_statistics_report_combines_policy_and_multi_seed_summary(self):
        report = build_baseline_statistics_report(
            baseline_config={
                "baseline_name": "claim_bearing_aircraft",
                "baseline_set": ["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"],
            },
            baseline_report={
                "baselines": {
                    "retrieval": {"status": "pass", "results": {"nearest_manifest_record": {"lift_to_drag": 4.0}}},
                    "unconditional_checkpoint": {"status": "pass", "results": {"seed_0": {"lift_to_drag": 4.5}}},
                    "bundled_grounded_stl": {
                        "status": "pass",
                        "results": {
                            "F-18_Hornet.stl": {"lift_to_drag": 5.0},
                            "biplane.stl": {"lift_to_drag": 3.0},
                        },
                    },
                },
            },
            condition_validation_report={
                "correlations": {"target_speed_vs_measured_drag": {"r": 0.4, "p": 0.1}},
                "raw_data": {
                    "measured_drag": [1.0, 2.0, 3.0],
                    "measured_lift": [4.0, 6.0, 9.0],
                    "occupancy": [0.1, 0.2, 0.3],
                },
            },
            min_seeds=3,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["baseline_policy"]["status"], "pass")
        self.assertEqual(report["multi_seed_summary"]["status"], "pass")
        self.assertIn("lift_to_drag", report["multi_seed_summary"]["metrics"])

    def test_baseline_family_report_requires_all_families_to_pass(self):
        report = build_baseline_family_report(
            bundled_grounded_results={"F-18_Hornet.stl": {"lift_to_drag": 5.0}},
            retrieval_results={"record_0": {"lift_to_drag": 4.0}},
            unconditional_results={},
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["baselines"]["bundled_grounded_stl"]["status"], "pass")
        self.assertEqual(report["baselines"]["retrieval"]["status"], "pass")
        self.assertEqual(report["baselines"]["unconditional_checkpoint"]["status"], "blocked")

    def test_baseline_family_report_passes_when_all_required_families_have_results(self):
        report = build_baseline_family_report(
            bundled_grounded_results={"F-18_Hornet.stl": {"lift_to_drag": 5.0}},
            retrieval_results={"record_0": {"lift_to_drag": 4.0}},
            unconditional_results={"seed_0": {"lift_to_drag": 4.5}},
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            sorted(report["baselines"].keys()),
            ["bundled_grounded_stl", "retrieval", "unconditional_checkpoint"],
        )

    def test_baseline_family_report_blocks_non_finite_metrics(self):
        report = build_baseline_family_report(
            bundled_grounded_results={"F-18_Hornet.stl": {"lift_to_drag": 5.0}},
            retrieval_results={"record_0": {"lift_to_drag": 4.0}},
            unconditional_results={"seed_0": {"lift_to_drag": float("nan")}},
        )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["baselines"]["unconditional_checkpoint"]["status"], "blocked")
        self.assertTrue(
            any(
                "non-finite" in error
                for error in report["baselines"]["unconditional_checkpoint"].get("errors", [])
            )
        )


if __name__ == "__main__":
    unittest.main()
