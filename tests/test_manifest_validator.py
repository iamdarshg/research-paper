import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import validate_manifest


class TestManifestValidator(unittest.TestCase):
    def test_minimal_repo_manifest_passes_basic_but_blocks_claim_bearing(self):
        repo_root = Path(__file__).resolve().parents[1]
        manifest_path = repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl"

        basic_report = validate_manifest.validate_manifest_file(str(manifest_path), level="basic")
        claim_report = validate_manifest.validate_manifest_file(str(manifest_path), level="claim-bearing")

        self.assertEqual(basic_report["status"], "pass")
        self.assertEqual(basic_report["record_count"], 2)
        self.assertEqual(claim_report["status"], "blocked")
        self.assertGreater(len(claim_report["errors"]), 0)
        self.assertTrue(
            any("provenance" in error.lower() or "preprocessing_version" in error.lower() for error in claim_report["errors"])
        )

    def test_claim_bearing_validation_accepts_complete_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_path = root / "sample.npy"
            geometry_path.write_bytes(b"fake")
            manifest_path = root / "manifest.jsonl"
            record = {
                "geometry_path": "sample.npy",
                "split": "train",
                "source_id": "demo-aircraft-001",
                "geometry_provenance": "internal-demo",
                "preprocessing_version": "voxelizer-v1",
                "units": "m",
                "design_family": "uav",
                "design_spec": {
                    "target_speed_mps": 42.0,
                    "wingspan_limit_m": 1.8,
                    "thrust_to_weight_min": 0.42,
                    "turn_rate_min_deg_s": 16.0,
                    "required_static_thrust_n": 160.0,
                    "engine_diameter_mm": 120,
                    "engine_length_mm": 240,
                    "engine_count_min": 1,
                    "engine_count_max": 1,
                    "payload_mass_min_g": 400,
                    "payload_mass_max_g": 900,
                    "takeoff_distance_min_m": 90,
                    "takeoff_distance_max_m": 180,
                    "wall_thickness_min_mm": 1,
                    "wall_thickness_max_mm": 2,
                    "part_count_min": 1,
                    "part_count_max": 6,
                    "manufacturing_method": "sheet_balsa_tabbed",
                },
            }
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = validate_manifest.validate_manifest_file(str(manifest_path), level="claim-bearing")

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["errors"], [])

    def test_claim_bearing_validation_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            geometry_path = root / "sample.npy"
            geometry_path.write_bytes(b"fake")
            manifest_path = root / "manifest.jsonl"
            record = {
                "geometry_path": "sample.npy",
                "split": "train",
                "design_spec": {
                    "target_speed_mps": 42.0,
                    "manufacturing_method": "sheet_balsa_tabbed",
                },
            }
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = validate_manifest.validate_manifest_file(str(manifest_path), level="claim-bearing")

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(any("source_id" in error for error in report["errors"]))
        self.assertTrue(any("geometry_provenance" in error for error in report["errors"]))
        self.assertTrue(any("preprocessing_version" in error for error in report["errors"]))
        self.assertTrue(any("units" in error for error in report["errors"]))
        self.assertTrue(any("wingspan_limit_m" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
