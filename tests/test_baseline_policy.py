import os
import sys
import json
import subprocess
import tempfile
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from multi_seed_eval import (
    build_baseline_statistics_report,
    build_statistical_summary,
    validate_baseline_policy,
)


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

    def test_baseline_statistics_report_passes_with_required_baselines_and_metrics(self):
        report = build_baseline_statistics_report(
            baseline_config={
                "baseline_name": "minimal_grounded_aircraft",
                "baseline_set": ["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"],
            },
            records=[
                {"seed": 0, "baseline": "retrieval", "lift_to_drag": 4.0},
                {"seed": 1, "baseline": "retrieval", "lift_to_drag": 6.0},
                {"seed": 2, "baseline": "retrieval", "lift_to_drag": 8.0},
                {"seed": 0, "baseline": "unconditional_checkpoint", "lift_to_drag": 3.0},
                {"seed": 1, "baseline": "unconditional_checkpoint", "lift_to_drag": 5.0},
                {"seed": 2, "baseline": "unconditional_checkpoint", "lift_to_drag": 7.0},
                {"seed": 0, "baseline": "bundled_grounded_stl", "lift_to_drag": 2.0},
                {"seed": 1, "baseline": "bundled_grounded_stl", "lift_to_drag": 4.0},
                {"seed": 2, "baseline": "bundled_grounded_stl", "lift_to_drag": 6.0},
            ],
            metric_keys=["lift_to_drag"],
            min_seeds=3,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["baseline_policy"]["status"], "pass")
        self.assertEqual(report["baselines"]["retrieval"]["metrics"]["lift_to_drag"]["mean"], 6.0)
        self.assertEqual(report["baselines"]["retrieval"]["seed_count"], 3)

    def test_baseline_statistics_report_blocks_missing_baseline_seed_and_metric(self):
        report = build_baseline_statistics_report(
            baseline_config={
                "baseline_name": "minimal_grounded_aircraft",
                "baseline_set": ["retrieval", "unconditional_checkpoint"],
            },
            records=[
                {"seed": 0, "baseline": "retrieval", "lift_to_drag": 4.0},
                {"seed": 1, "baseline": "retrieval"},
            ],
            metric_keys=["lift_to_drag"],
            min_seeds=3,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("missing required baselines" in blocker for blocker in report["blockers"]))
        self.assertTrue(any("baseline retrieval: insufficient seeds" in blocker for blocker in report["blockers"]))
        self.assertTrue(any("baseline unconditional_checkpoint has no records" in blocker for blocker in report["blockers"]))

    def test_report_only_cli_writes_baseline_statistics_json(self):
        script_path = os.path.join(CLI_DIR, "multi_seed_eval.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "baseline_config.json")
            records_path = os.path.join(tmpdir, "records.json")
            output_path = os.path.join(tmpdir, "baseline_statistics.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "baseline_name": "minimal_grounded_aircraft",
                        "baseline_set": ["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"],
                    },
                    config_file,
                )
            with open(records_path, "w", encoding="utf-8") as records_file:
                json.dump(
                    [
                        {"seed": 0, "baseline": "retrieval", "lift_to_drag": 4.0},
                        {"seed": 1, "baseline": "retrieval", "lift_to_drag": 6.0},
                        {"seed": 2, "baseline": "retrieval", "lift_to_drag": 8.0},
                        {"seed": 0, "baseline": "unconditional_checkpoint", "lift_to_drag": 3.0},
                        {"seed": 1, "baseline": "unconditional_checkpoint", "lift_to_drag": 5.0},
                        {"seed": 2, "baseline": "unconditional_checkpoint", "lift_to_drag": 7.0},
                        {"seed": 0, "baseline": "bundled_grounded_stl", "lift_to_drag": 2.0},
                        {"seed": 1, "baseline": "bundled_grounded_stl", "lift_to_drag": 4.0},
                        {"seed": 2, "baseline": "bundled_grounded_stl", "lift_to_drag": 6.0},
                    ],
                    records_file,
                )

            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--baseline-config",
                    config_path,
                    "--records-json",
                    records_path,
                    "--metric-key",
                    "lift_to_drag",
                    "--baseline-statistics-output",
                    output_path,
                    "--min-seeds",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_path, "r", encoding="utf-8") as output_file:
                report = json.load(output_file)
            self.assertEqual(report["status"], "pass")

    def test_report_only_cli_fails_closed_and_writes_blocked_report(self):
        script_path = os.path.join(CLI_DIR, "multi_seed_eval.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "baseline_config.json")
            records_path = os.path.join(tmpdir, "records.json")
            output_path = os.path.join(tmpdir, "baseline_statistics.json")
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(
                    {
                        "baseline_name": "minimal_grounded_aircraft",
                        "baseline_set": ["retrieval"],
                    },
                    config_file,
                )
            with open(records_path, "w", encoding="utf-8") as records_file:
                json.dump(
                    [
                        {"seed": 0, "baseline": "retrieval", "lift_to_drag": 4.0},
                    ],
                    records_file,
                )

            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "--baseline-config",
                    config_path,
                    "--records-json",
                    records_path,
                    "--metric-key",
                    "lift_to_drag",
                    "--baseline-statistics-output",
                    output_path,
                    "--min-seeds",
                    "3",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            with open(output_path, "r", encoding="utf-8") as output_file:
                report = json.load(output_file)
            self.assertEqual(report["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
