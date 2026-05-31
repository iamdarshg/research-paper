import os
import sys
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from multi_seed_eval import build_statistical_summary, validate_baseline_policy


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


if __name__ == "__main__":
    unittest.main()
