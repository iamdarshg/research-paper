import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from build_reference_evidence import build_reference_evidence_bundle
from final_evidence import evaluate_final_evidence_package
from validate_manifest import validate_manifest_file


class TestReferenceEvidenceBundle(unittest.TestCase):
    def test_builder_writes_claim_bearing_manifest_checkpoint_and_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            report = build_reference_evidence_bundle(root, sample_count=20)

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["sample_count"], 20)
            manifest_path = Path(report["artifacts"]["manifest"])
            checkpoint_path = Path(report["artifacts"]["checkpoint"])
            records_path = Path(report["artifacts"]["baseline_records"])
            metadata_path = Path(report["artifacts"]["run_metadata"])

            self.assertTrue(manifest_path.exists())
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(records_path.exists())
            self.assertTrue(metadata_path.exists())

            manifest_report = validate_manifest_file(str(manifest_path), level="claim-bearing")
            self.assertEqual(manifest_report["status"], "pass")
            self.assertEqual(manifest_report["record_count"], 20)

            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["generator_type"], "deterministic_reference_fixture")
            self.assertFalse(checkpoint["claim_bearing_trained_model"])
            self.assertIn("Turbulence Modeling Resource", checkpoint["reference_basis"][0]["title"])

    def test_run_metadata_can_supply_strict_final_evidence_consistency(self):
        run_metadata = {
            "run_id": "reference-run",
            "checkpoint_hash": "sha256:checkpoint",
            "manifest_hash": "sha256:manifest",
            "protocol_hash": "sha256:protocol",
        }
        reports = {
            "manifest_validation": {"status": "pass"},
            "aircraft_validity": {"status": "pass"},
            "condition_benchmark": {"status": "pass"},
            "manufacturing_constraints": {"status": "pass"},
            "baseline_statistics": {"status": "pass"},
        }

        report = evaluate_final_evidence_package(
            reports,
            require_run_consistency=True,
            run_metadata=run_metadata,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["blocked_gates"], [])
        self.assertEqual(report["run_consistency"]["values"]["run_id"], "reference-run")


if __name__ == "__main__":
    unittest.main()
