import os
import sys
import unittest
import tempfile

import torch
import numpy as np


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_validity import evaluate_aircraft_validity, evaluate_aircraft_validity_batch


def minimally_plausible_aircraft(res=32):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    mid_y = res // 2

    voxels[mid_z - 2:mid_z + 3, mid_y - 3:mid_y + 3, 6:26] = 1.0
    voxels[mid_z - 1:mid_z + 2, 5:27, 13:19] = 1.0
    voxels[mid_z:mid_z + 2, 10:22, 5:9] = 1.0
    return voxels


class TestAircraftValidity(unittest.TestCase):
    def test_minimally_plausible_control_passes_first_pass_checks(self):
        report = evaluate_aircraft_validity(minimally_plausible_aircraft())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])
        self.assertGreaterEqual(report["metrics"]["symmetry_score"], 0.95)

    def test_asymmetric_blob_fails_aircraft_specific_checks(self):
        voxels = torch.zeros((32, 32, 32))
        voxels[3:8, 2:7, 3:8] = 1.0

        report = evaluate_aircraft_validity(voxels)

        self.assertEqual(report["status"], "fail")
        self.assertIn("symmetry", report["failed_checks"])
        self.assertIn("span_sanity", report["failed_checks"])
        self.assertIn("wing_body_balance", report["failed_checks"])

    def test_batch_report_aggregates_multiple_voxel_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = os.path.join(tmp, "valid.npy")
            invalid_path = os.path.join(tmp, "invalid.npy")
            np.save(valid_path, minimally_plausible_aircraft().numpy())
            blob = torch.zeros((32, 32, 32))
            blob[3:8, 2:7, 3:8] = 1.0
            np.save(invalid_path, blob.numpy())

            report = evaluate_aircraft_validity_batch([valid_path, invalid_path])

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["passed_sample_count"], 1)
        self.assertEqual(report["failed_sample_indices"], [1])


if __name__ == "__main__":
    unittest.main()
