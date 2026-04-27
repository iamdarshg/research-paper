
import unittest
import torch
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from geometry import AircraftPart, TypedAircraftGeometry
from constraints import ConstraintProjector
from config import MissionProfile

class TestAircraftConstraints(unittest.TestCase):
    def setUp(self):
        self.res = 32
        self.device = torch.device('cpu')
        self.mission = MissionProfile(
            aircraft_class="uav",
            manufacturing_method="3d_print"
        )

    def test_typed_geometry_channels(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        self.assertEqual(geom.tensor.shape[0], len(AircraftPart))

        # Set a part
        mask = torch.zeros((self.res, self.res, self.res))
        mask[10:20, 10:20, 10:20] = 1.0
        geom.set_part_mask(AircraftPart.FUSELAGE, mask)

        combined = geom.get_combined_occupancy()
        self.assertEqual(torch.sum(combined), 10*10*10)

    def test_constraint_projection_symmetry(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        # Asymmetric wing on one side only (Y > mid)
        # For res=32, mid=16. set voxel at y=20
        wing_mask = torch.zeros((self.res, self.res, self.res))
        wing_mask[16, 20, 16] = 1.0
        geom.set_part_mask(AircraftPart.WING, wing_mask)

        projector = ConstraintProjector(self.res, device=self.device)
        projected = projector.project(geom, self.mission)

        # Check symmetry on Y axis
        mask = projected.get_part_mask(AircraftPart.WING)
        # For even res=32, flip is exact.
        flipped = torch.flip(mask, [1])

        # Debugging sum
        print(f"Original sum: {torch.sum(wing_mask)}, Projected sum: {torch.sum(mask)}")

        self.assertTrue(torch.allclose(mask, flipped))
        self.assertGreater(torch.sum(mask), 0)

    def test_prop_clearance_violation(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        # Block the nose area (prop disk is at cx=int(res*0.1))
        res = self.res
        # prop_mask center is at (cx, cy, cz), uses z,y,x indexing in meshgrid
        cx, cy, cz = int(res * 0.1), res // 2, res // 2
        nose_block = torch.zeros((res, res, res))
        # Match indexing ij: (z, y, x)
        nose_block[cz, cy, cx] = 1.0
        geom.set_part_mask(AircraftPart.SKIN, nose_block)

        projector = ConstraintProjector(self.res, device=self.device)
        projected = projector.project(geom, self.mission)

        report = projector.get_report(projected)
        # Should have detected prop clearance violation
        has_prop_violation = any(v['type'] == 'prop_clearance' for v in report['violations'])
        self.assertTrue(has_prop_violation)

        # Skin at prop disk should be cleared
        self.assertEqual(projected.get_part_mask(AircraftPart.SKIN)[cz, cy, cx], 0.0)

    def test_report_generation(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        projector = ConstraintProjector(self.res, device=self.device)
        projected = projector.project(geom, self.mission)
        report = projector.get_report(projected)

        self.assertIn('valid', report)
        self.assertIn('violations', report)
        self.assertIn('metrics', report)
        self.assertIn('parts_breakdown', report['metrics'])

    def test_feasibility_lift_weight(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        # Add tiny wing
        wing = torch.zeros((self.res, self.res, self.res))
        wing[16, 16, 16] = 1.0
        geom.set_part_mask(AircraftPart.WING, wing)

        projector = ConstraintProjector(self.res, device=self.device)
        # Mock high weight results
        mock_cfd = {'force_z': 1.0, 'force_x': 0.1}
        feasibility = projector.check_feasibility(geom, mock_cfd, self.mission)

        report = projector.get_report(geom)
        # Should report insufficient lift if weight > 1N
        if feasibility['weight_n'] > 1.0:
            self.assertTrue(any(v['type'] == 'insufficient_lift' for v in report['violations']))

if __name__ == '__main__':
    unittest.main()
