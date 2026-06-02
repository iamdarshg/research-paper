import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from write_run_metadata import build_run_metadata


class TestRunMetadata(unittest.TestCase):
    def test_metadata_hashes_trained_checkpoint_not_reference_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / "final_optimized_model.pt"
            reference_checkpoint = root / "reference_checkpoint.json"
            manifest = root / "manifest.jsonl"
            protocol = root / "first_training.yaml"
            checkpoint.write_bytes(b"trained-checkpoint-bytes")
            reference_checkpoint.write_text('{"generator_type":"deterministic_reference_fixture"}', encoding="utf-8")
            manifest.write_text('{"sample_id":"a"}\n', encoding="utf-8")
            protocol.write_text("train:\n  enabled: true\n", encoding="utf-8")

            report = build_run_metadata(
                checkpoint_path=checkpoint,
                manifest_path=manifest,
                protocol_path=protocol,
                output_path=root / "run_metadata.json",
            )

            self.assertEqual(report["checkpoint_path"], str(checkpoint.resolve()))
            self.assertNotEqual(report["checkpoint_hash"], report.get("reference_checkpoint_hash"))
            self.assertTrue(report["checkpoint_hash"].startswith("sha256:"))
            self.assertTrue((root / "run_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
