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


def wing_only_extrusion(res=32):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    voxels[mid_z - 1:mid_z + 2, 4:28, 8:24] = 1.0
    return voxels


def dense_swept_slab(res=32):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    mid_y = res // 2
    for x in range(10, 23):
        taper = abs(x - 16) / 6
        half_span = max(4, int(15 * (1 - taper * 0.30)))
        half_thickness = max(1, int(3 * (1 - taper * 0.25)))
        y0 = max(0, mid_y - half_span)
        y1 = min(res, mid_y + half_span + 1)
        voxels[
            mid_z - half_thickness:mid_z + half_thickness + 1,
            y0:y1,
            x,
        ] = 1.0
        voxels[
            mid_z - half_thickness:mid_z + half_thickness + 1,
            mid_y - 3:mid_y + 3,
            x,
        ] = 1.0
    return voxels


def ellipsoid_blob(res=32):
    z, y, x = torch.meshgrid(
        torch.arange(res),
        torch.arange(res),
        torch.arange(res),
        indexing="ij",
    )
    center = (res - 1) / 2
    return (
        ((z - center) / (0.14 * res)) ** 2
        + ((y - center) / (0.38 * res)) ** 2
        + ((x - center) / (0.28 * res)) ** 2
        <= 1
    ).float()


def sparse_transport_aircraft(res=64):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    mid_y = res // 2
    voxels[mid_z - 1:mid_z + 2, mid_y - 1:mid_y + 2, 10:54] = 1.0
    voxels[mid_z:mid_z + 1, 6:58, 24:35] = 1.0
    voxels[mid_z:mid_z + 2, mid_y - 2:mid_y + 3, 8:14] = 1.0
    voxels[mid_z:mid_z + 2, mid_y - 2:mid_y + 3, 50:56] = 1.0
    return voxels


class TestAircraftValidity(unittest.TestCase):
    def test_minimally_plausible_control_passes_first_pass_checks(self):
        report = evaluate_aircraft_validity(minimally_plausible_aircraft())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])
        self.assertGreaterEqual(report["metrics"]["symmetry_score"], 0.80)

    def test_rotated_aircraft_passes_after_canonical_alignment(self):
        rotated = minimally_plausible_aircraft().permute(1, 2, 0).contiguous()

        report = evaluate_aircraft_validity(rotated)

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])
        self.assertIn(report["canonicalization"]["permutation"], ([2, 0, 1], [2, 1, 0]))

    def test_asymmetric_blob_fails_aircraft_specific_checks(self):
        voxels = torch.zeros((32, 32, 32))
        voxels[12:17, 2:7, 5:10] = 1.0
        voxels[12:17, 2:5, 17:24] = 1.0

        report = evaluate_aircraft_validity(voxels)

        self.assertEqual(report["status"], "fail")
        self.assertIn("symmetry", report["failed_checks"])
        self.assertIn("span_sanity", report["failed_checks"])
        self.assertIn("wing_body_balance", report["failed_checks"])

    def test_wing_only_extrusion_fails_centerline_dominance(self):
        report = evaluate_aircraft_validity(wing_only_extrusion())

        self.assertEqual(report["status"], "fail")
        self.assertIn("longitudinal_profile_variation", report["failed_checks"])

    def test_dense_swept_slab_fails_blocky_planform_checks(self):
        report = evaluate_aircraft_validity(dense_swept_slab())

        self.assertEqual(report["status"], "fail")
        self.assertIn("planform_sparsity", report["failed_checks"])
        self.assertIn("fuselage_end_presence", report["failed_checks"])

    def test_ellipsoid_blob_fails_aircraft_definition_checks(self):
        report = evaluate_aircraft_validity(ellipsoid_blob())

        self.assertEqual(report["status"], "fail")
        self.assertIn("planform_sparsity", report["failed_checks"])

    def test_sparse_transport_aircraft_passes_low_occupancy_floor(self):
        report = evaluate_aircraft_validity(sparse_transport_aircraft())

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])

    def test_batch_report_aggregates_multiple_voxel_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid_path = os.path.join(tmp, "valid.npy")
            invalid_path = os.path.join(tmp, "invalid.npy")
            np.save(valid_path, minimally_plausible_aircraft().numpy())
            blob = torch.zeros((32, 32, 32))
            blob[12:17, 2:7, 5:10] = 1.0
            blob[12:17, 2:5, 17:24] = 1.0
            np.save(invalid_path, blob.numpy())

            report = evaluate_aircraft_validity_batch([valid_path, invalid_path])

        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["passed_sample_count"], 1)
        self.assertEqual(report["failed_sample_indices"], [1])


if __name__ == "__main__":
    unittest.main()
