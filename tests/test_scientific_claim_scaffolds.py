import json
import os
import subprocess
import sys
import tempfile
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from scientific_claim_scaffolds import (
    build_cfd_guided_training_ablation_report,
    build_prior_method_comparison_report,
    build_publication_quality_validation_report,
)


def _complete_ablation_metadata():
    arm = {
        "seeds": [0, 1, 2],
        "checkpoints": ["ckpt-0.pt", "ckpt-1.pt"],
        "training_curves": "training-curves.json",
        "candidate_rankings": "rankings.json",
        "cfd_metrics": "cfd-metrics.json",
    }
    return {
        "arms": {
            "cfd_guided": dict(arm),
            "control": dict(arm),
        },
        "matched_config_fields": ["dataset", "model", "optimizer"],
        "changed_config_fields": ["aero_loss_weight"],
        "statistical_comparison": {"test": "paired-bootstrap", "artifact": "stats.json"},
    }


def _complete_prior_metadata():
    method = {
        "citation": "Synthetic fixture",
        "implementation_source": "local-test-double",
        "version_or_commit": "abc123",
        "evaluation_protocol": "same generated test split",
        "metric_mapping": {"lift_to_drag": "lift_to_drag"},
        "reproduction_status": "reproduced-for-schema-test",
        "sample_set_id": "fixture-samples",
    }
    return {
        "methods": [
            {"method_id": "ours", **method},
            {"method_id": "prior-a", **method},
        ],
        "comparison": {
            "sample_set_id": "fixture-samples",
            "seeds": [0, 1, 2],
            "metrics": ["lift_to_drag"],
            "statistical_tests": {"lift_to_drag": "paired-bootstrap"},
            "result_interpretation": "schema fixture only; no superiority claim",
        },
    }


def _complete_publication_metadata():
    return {
        "solver_settings": {"solver": "synthetic", "mach": 0.1, "reynolds": 100000},
        "convergence_study": {
            "resolution_ladder": [32, 64, 128],
            "metrics": {"drag_coefficient": [0.12, 0.11, 0.105]},
        },
        "sensitivity_study": {
            "parameters": ["mach", "timestep"],
            "metrics": {"drag_coefficient": "sensitivity-table.json"},
        },
        "external_validation": {
            "reference_cases": ["sphere-drag"],
            "agreement_metrics": {"drag_error_pct": 3.0},
        },
        "residuals_or_forces": "forces-and-residuals.json",
    }


class TestScientificClaimScaffolds(unittest.TestCase):
    def test_missing_evidence_blocks_all_scaffolds(self):
        reports = [
            build_cfd_guided_training_ablation_report({}),
            build_prior_method_comparison_report({}),
            build_publication_quality_validation_report({}),
        ]

        for report in reports:
            self.assertEqual(report["status"], "blocked")
            self.assertTrue(report["blockers"])
            self.assertIn("checks evidence presence", report["claim_boundary"])

    def test_complete_synthetic_ablation_metadata_passes_schema_only(self):
        report = build_cfd_guided_training_ablation_report(_complete_ablation_metadata())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["matched_seeds"], [0, 1, 2])
        self.assertIn("does not establish", report["claim_boundary"])

    def test_ablation_blocks_unmatched_seeds_and_extra_changed_fields(self):
        metadata = _complete_ablation_metadata()
        metadata["arms"]["control"]["seeds"] = [0, 1, 3]
        metadata["changed_config_fields"] = ["aero_loss_weight", "batch_size"]

        report = build_cfd_guided_training_ablation_report(metadata)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("ablation arms must use identical matched seeds", report["blockers"])
        self.assertIn("changed_config_fields must be exactly ['aero_loss_weight']", report["blockers"])

    def test_complete_synthetic_prior_method_metadata_passes_schema_only(self):
        report = build_prior_method_comparison_report(_complete_prior_metadata())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["method_count"], 2)
        self.assertIn("does not establish superiority", report["claim_boundary"])

    def test_prior_method_blocks_mismatched_sample_sets(self):
        metadata = _complete_prior_metadata()
        metadata["methods"][1]["sample_set_id"] = "different-samples"

        report = build_prior_method_comparison_report(metadata)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("all methods must use the comparison sample_set_id", report["blockers"])

    def test_complete_synthetic_publication_metadata_passes_schema_only(self):
        report = build_publication_quality_validation_report(_complete_publication_metadata())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["resolution_count"], 3)
        self.assertIn("publication-quality validity", report["claim_boundary"])

    def test_publication_validation_blocks_short_resolution_ladder(self):
        metadata = _complete_publication_metadata()
        metadata["convergence_study"]["resolution_ladder"] = [32, 64]

        report = build_publication_quality_validation_report(metadata)

        self.assertEqual(report["status"], "blocked")
        self.assertIn(
            "convergence_study.resolution_ladder must contain at least three resolutions",
            report["blockers"],
        )

    def test_cli_writes_one_requested_report_type(self):
        script_path = os.path.join(CLI_DIR, "scientific_claim_scaffolds.py")
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = os.path.join(tmpdir, "metadata.json")
            output_path = os.path.join(tmpdir, "prior_method_comparison.json")
            with open(metadata_path, "w", encoding="utf-8") as handle:
                json.dump(_complete_prior_metadata(), handle)

            result = subprocess.run(
                [
                    sys.executable,
                    script_path,
                    "prior-method-comparison",
                    "--metadata",
                    metadata_path,
                    "--output",
                    output_path,
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with open(output_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["gate_id"], "prior_method_comparison")
            self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
