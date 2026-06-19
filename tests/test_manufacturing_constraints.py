import os
import sys
import unittest
import json
import tempfile


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from condition_feasibility import build_manufacturing_constraints_report, main, validate_condition_feasibility
from aircraft_diffusion_cfd import DesignSpec


class TestManufacturingConstraints(unittest.TestCase):
    def test_rejects_impossible_engine_and_thrust_payload_combo(self):
        report = validate_condition_feasibility(
            {
                "engine_count_min": 0,
                "engine_count_max": 0,
                "required_static_thrust_n": 800.0,
                "payload_mass_min_g": 4000,
                "payload_mass_max_g": 2000,
                "wall_thickness_min_mm": 0.1,
                "wall_thickness_max_mm": 0.2,
                "part_count_min": 12,
                "part_count_max": 4,
                "manufacturing_method": "fdm_pla_0p4mm",
            }
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("engine_count", report["failed_checks"])
        self.assertIn("payload_bounds", report["failed_checks"])
        self.assertIn("part_count_bounds", report["failed_checks"])
        self.assertIn("wall_thickness", report["failed_checks"])

    def test_accepts_plausible_distinct_condition_payload(self):
        report = validate_condition_feasibility(
            {
                "engine_count_min": 1,
                "engine_count_max": 2,
                "required_static_thrust_n": 220.0,
                "payload_mass_min_g": 500,
                "payload_mass_max_g": 1500,
                "wall_thickness_min_mm": 1.0,
                "wall_thickness_max_mm": 2.0,
                "part_count_min": 2,
                "part_count_max": 8,
                "manufacturing_method": "fdm_pla_0p4mm",
                "thrust_to_weight_min": 0.55,
                "target_speed_mps": 55.0,
                "turn_rate_min_deg_s": 18.0,
            }
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])

    def test_design_spec_rejects_impossible_payload_before_generation(self):
        with self.assertRaisesRegex(ValueError, "payload_mass_min_g"):
            DesignSpec(
                payload_mass_min_g=5000,
                payload_mass_max_g=1000,
                wall_thickness_min_mm=1,
                wall_thickness_max_mm=2,
                part_count_min=1,
                part_count_max=4,
            )

    def test_design_spec_rejects_method_specific_wall_thickness(self):
        with self.assertRaisesRegex(ValueError, "wall_thickness_min_mm"):
            DesignSpec(
                wall_thickness_min_mm=0.1,
                wall_thickness_max_mm=0.2,
                manufacturing_method="fdm_pla_0p4mm",
            )

    def test_manifest_report_accepts_schema_target_speed_mps(self):
        report = build_manufacturing_constraints_report(
            [
                {
                    "target_speed_mps": 55.0,
                    "engine_count_min": 1,
                    "engine_count_max": 2,
                    "required_static_thrust_n": 220.0,
                    "payload_mass_min_g": 500,
                    "payload_mass_max_g": 1500,
                    "wall_thickness_min_mm": 1.0,
                    "wall_thickness_max_mm": 2.0,
                    "part_count_min": 2,
                    "part_count_max": 8,
                    "manufacturing_method": "fdm_pla_0p4mm",
                    "thrust_to_weight_min": 0.55,
                    "turn_rate_min_deg_s": 18.0,
                }
            ]
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["sample_count"], 1)

    def test_manifest_cli_writes_blocked_report_for_impossible_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = os.path.join(tmp, "manifest.jsonl")
            output_path = os.path.join(tmp, "manufacturing_constraints.json")
            record = {
                "design_spec": {
                    "target_speed_mps": -1.0,
                    "engine_count_min": 1,
                    "engine_count_max": 1,
                    "required_static_thrust_n": 100.0,
                    "payload_mass_min_g": 0,
                    "payload_mass_max_g": 10,
                    "wall_thickness_min_mm": 1.0,
                    "wall_thickness_max_mm": 2.0,
                    "part_count_min": 1,
                    "part_count_max": 1,
                    "manufacturing_method": "fdm_pla_0p4mm",
                }
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            with unittest.mock.patch.object(sys, "argv", ["condition_feasibility.py", "--manifest", manifest_path, "--output", output_path]):
                exit_code = main()

            self.assertEqual(exit_code, 2)
            with open(output_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["status"], "blocked")
            self.assertIn(0, report["blocked_sample_indices"])


if __name__ == "__main__":
    unittest.main()
