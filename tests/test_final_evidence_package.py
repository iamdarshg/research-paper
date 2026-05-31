import os
import sys
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from final_evidence import evaluate_final_evidence_package


class TestFinalEvidencePackage(unittest.TestCase):
    def test_missing_claim_artifacts_keep_claim_blocked(self):
        report = evaluate_final_evidence_package(
            {
                "aircraft_validity": {"status": "pass"},
                "condition_benchmark": {"status": "blocked"},
            }
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("condition_benchmark", report["blocked_gates"])
        self.assertIn("baseline_statistics", report["blocked_gates"])

    def test_all_required_claim_artifacts_pass(self):
        report = evaluate_final_evidence_package(
            {
                "aircraft_validity": {"status": "pass"},
                "condition_benchmark": {"status": "pass"},
                "manufacturing_constraints": {"status": "pass"},
                "baseline_statistics": {"status": "pass"},
                "manifest_validation": {"status": "pass"},
            }
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["blocked_gates"], [])

    def test_strict_package_blocks_mixed_run_identifiers(self):
        reports = {
            "aircraft_validity": {
                "status": "pass",
                "run_id": "run-a",
                "checkpoint_hash": "ckpt-a",
                "manifest_hash": "manifest-a",
                "protocol_hash": "protocol-a",
            },
            "condition_benchmark": {
                "status": "pass",
                "run_id": "run-a",
                "checkpoint_hash": "ckpt-b",
                "manifest_hash": "manifest-a",
                "protocol_hash": "protocol-a",
            },
            "manufacturing_constraints": {
                "status": "pass",
                "run_id": "run-a",
                "checkpoint_hash": "ckpt-a",
                "manifest_hash": "manifest-a",
                "protocol_hash": "protocol-a",
            },
            "baseline_statistics": {
                "status": "pass",
                "run_id": "run-a",
                "checkpoint_hash": "ckpt-a",
                "manifest_hash": "manifest-a",
                "protocol_hash": "protocol-a",
            },
            "manifest_validation": {
                "status": "pass",
                "run_id": "run-a",
                "checkpoint_hash": "ckpt-a",
                "manifest_hash": "manifest-a",
                "protocol_hash": "protocol-a",
            },
        }

        report = evaluate_final_evidence_package(reports, require_run_consistency=True)

        self.assertEqual(report["status"], "blocked")
        self.assertIn("run_consistency", report["blocked_gates"])
        self.assertTrue(report["run_consistency"]["errors"])


if __name__ == "__main__":
    unittest.main()
