import os
import sys
import unittest


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from gate_readiness import build_gate_readiness_report


class TestGateReadiness(unittest.TestCase):
    def test_gate_implementation_readiness_is_at_least_ninety_percent(self):
        report = build_gate_readiness_report()

        self.assertEqual(report["gate_count"], 13)
        self.assertGreaterEqual(report["implementation_readiness"]["completed_ratio"], 0.90)
        self.assertEqual(report["implementation_readiness"]["completed_count"], 13)
        self.assertEqual(report["implementation_readiness"]["status"], "pass")

    def test_claim_bearing_evidence_stays_blocked_without_real_artifacts(self):
        report = build_gate_readiness_report()

        self.assertEqual(report["claim_bearing_evidence"]["passed_count"], 0)
        self.assertEqual(report["claim_bearing_evidence"]["status"], "blocked")
        blocked_gate_ids = {
            gate["id"]
            for gate in report["gates"]
            if gate["claim_bearing_evidence_status"] != "pass"
        }
        self.assertEqual(len(blocked_gate_ids), 13)

    def test_reference_final_evidence_marks_only_supported_gate_scopes(self):
        final_evidence = {
            "status": "pass",
            "gates": {
                "manifest_validation": {"status": "pass"},
                "aircraft_validity": {"status": "pass"},
                "condition_benchmark": {"status": "pass"},
                "manufacturing_constraints": {"status": "pass"},
                "baseline_statistics": {"status": "pass"},
            },
        }

        report = build_gate_readiness_report(final_evidence=final_evidence)
        passed_gate_ids = {
            gate["id"]
            for gate in report["gates"]
            if gate["claim_bearing_evidence_status"] == "pass"
        }

        self.assertEqual(report["claim_bearing_evidence"]["passed_count"], 8)
        self.assertIn("manifest_validation", passed_gate_ids)
        self.assertIn("generates_aircraft_structures", passed_gate_ids)
        self.assertIn("conditioned_flight_profile_manufacturing", passed_gate_ids)
        self.assertNotIn("aerodynamically_optimized", passed_gate_ids)
        self.assertNotIn("cfd_guided_training", passed_gate_ids)

    def test_each_completed_gate_has_documentation_and_machine_readable_artifact(self):
        report = build_gate_readiness_report()

        for gate in report["gates"]:
            self.assertEqual(gate["implementation_status"], "complete", gate)
            self.assertTrue(gate["documentation_artifacts"], gate)
            self.assertTrue(gate["machine_readable_artifacts"], gate)
            self.assertTrue(gate["tests_or_verification"], gate)


if __name__ == "__main__":
    unittest.main()
