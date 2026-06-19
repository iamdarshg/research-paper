import json
import os
import sys
import unittest
from pathlib import Path


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import validate_manifest


class TestManifestContract(unittest.TestCase):
    def test_schema_example_lists_claim_bearing_required_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "docs" / "dataset" / "manifest_schema.example.json"

        payload = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertIn("required_record_fields", payload)
        self.assertIn("required_design_spec_fields", payload)
        self.assertIn("geometry_provenance", payload["required_record_fields"])
        self.assertIn("preprocessing_version", payload["required_record_fields"])
        self.assertIn("payload_mass_max_g", payload["required_design_spec_fields"])
        self.assertIn("manufacturing_method", payload["required_design_spec_fields"])

    def test_validator_required_design_fields_match_schema_example(self):
        repo_root = Path(__file__).resolve().parents[1]
        schema_path = repo_root / "docs" / "dataset" / "manifest_schema.example.json"
        payload = json.loads(schema_path.read_text(encoding="utf-8"))

        validator_fields = validate_manifest.validate_manifest_records(
            [],
            manifest_path=str(repo_root / "docs" / "dataset" / "empty.jsonl"),
            level="claim-bearing",
        )["required_design_spec_fields"]

        self.assertEqual(payload["required_design_spec_fields"], validator_fields)


if __name__ == "__main__":
    unittest.main()
