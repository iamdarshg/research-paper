import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "CLI"))

from run_condition_benchmark import (
    FIXED_SWEEPS,
    build_condition_benchmark_report,
    load_manifest_records,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(payload, thrust, turn_rate, wall, metrics, split="test"):
    return {
        "sample_id": f"sample-{payload}-{thrust}-{turn_rate}-{wall}",
        "geometry_path": "shape.stl",
        "split": split,
        "design_spec": {
            "payload_mass_max_g": payload,
            "required_static_thrust_n": thrust,
            "turn_rate_min_deg_s": turn_rate,
            "wall_thickness_min_mm": wall,
        },
        "response_metrics": metrics,
    }


class ConditionBenchmarkTests(unittest.TestCase):
    def test_minimal_manifest_blocks_before_checkpoint_is_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = tmp_path / "minimal.jsonl"
            _write_jsonl(
                manifest,
                [
                    {
                        "sample_id": "minimal-0",
                        "geometry_path": "shape.stl",
                        "split": "test",
                    }
                ],
            )

            report = build_condition_benchmark_report(
                manifest_path=manifest,
                checkpoint_path=tmp_path / "missing.pt",
                seeds=[0, 1],
                min_grounded_records=4,
            )

            self.assertEqual(report["status"], "blocked")
            self.assertEqual(report["record_count"], 1)
            self.assertFalse(report["checkpoint_checked"])
            self.assertIn("insufficient grounded records", report["blockers"][0])

    def test_report_uses_fixed_sweeps_and_deterministic_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = tmp_path / "grounded.jsonl"
            checkpoint = tmp_path / "checkpoint.pt"
            checkpoint.write_text("placeholder", encoding="utf-8")
            _write_jsonl(
                manifest,
                [
                    _record(10, 100, 12, 1.0, {
                        "payload_response": 1.0,
                        "thrust_response": 1.0,
                        "maneuverability_response": 1.0,
                        "structural_response": 1.0,
                    }),
                    _record(20, 200, 24, 2.0, {
                        "payload_response": 2.0,
                        "thrust_response": 2.0,
                        "maneuverability_response": 2.0,
                        "structural_response": 2.0,
                    }),
                ],
            )

            report = build_condition_benchmark_report(
                manifest_path=manifest,
                checkpoint_path=checkpoint,
                seeds=[7, 3, 7],
                min_grounded_records=2,
            )

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["seeds"], [3, 7])
            self.assertEqual(
                [sweep["condition_field"] for sweep in FIXED_SWEEPS],
                [
                    "payload_mass_max_g",
                    "required_static_thrust_n",
                    "turn_rate_min_deg_s",
                    "wall_thickness_min_mm",
                ],
            )
            self.assertEqual(
                [sweep["id"] for sweep in report["sweeps"]],
                [sweep["id"] for sweep in FIXED_SWEEPS],
            )
            self.assertTrue(all(sweep["status"] == "pass" for sweep in report["sweeps"]))

    def test_directional_mismatch_fails_the_benchmark(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest = tmp_path / "grounded.jsonl"
            checkpoint = tmp_path / "checkpoint.pt"
            checkpoint.write_text("placeholder", encoding="utf-8")
            _write_jsonl(
                manifest,
                [
                    _record(10, 100, 12, 1.0, {
                        "payload_response": 2.0,
                        "thrust_response": 1.0,
                        "maneuverability_response": 1.0,
                        "structural_response": 1.0,
                    }),
                    _record(20, 200, 24, 2.0, {
                        "payload_response": 1.0,
                        "thrust_response": 2.0,
                        "maneuverability_response": 2.0,
                        "structural_response": 2.0,
                    }),
                ],
            )

            report = build_condition_benchmark_report(
                manifest_path=manifest,
                checkpoint_path=checkpoint,
                seeds=[0],
                min_grounded_records=2,
            )

            payload_sweep = next(sweep for sweep in report["sweeps"] if sweep["id"] == "payload_increase")
            self.assertEqual(report["status"], "fail")
            self.assertEqual(payload_sweep["status"], "fail")
            self.assertLess(payload_sweep["observed_delta"], 0)

    def test_load_manifest_records_rejects_non_object_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "bad.jsonl"
            manifest.write_text("[1, 2, 3]\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain JSON objects"):
                load_manifest_records(manifest)

    def test_load_manifest_records_accepts_utf8_bom_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "bom.jsonl"
            manifest.write_text('{"sample_id":"bom-ok"}\n', encoding="utf-8-sig")

            records = load_manifest_records(manifest)

            self.assertEqual(records, [{"sample_id": "bom-ok"}])


if __name__ == "__main__":
    unittest.main()
