
import unittest
import torch
import sys
import os
import torch.nn.functional as F

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from geometry import AircraftPart, TypedAircraftGeometry
from constraints import ConstraintProjector, ConstraintReport
from config import MissionProfile, CFDConfig
from cfd_simulator import AdvancedCFDSimulator
from generator import OptimizedAircraftGenerator

class TestAircraftConstraintsAdvanced(unittest.TestCase):
    def setUp(self):
        self.res = 32
        self.device = torch.device('cpu')

    def test_typed_geometry_amr_path(self):
        """Verify that typed geometry correctly propagates through the AMR path."""
        res = self.res
        geom = TypedAircraftGeometry(res, device=self.device)
        # Add some solid volume
        fuselage = torch.zeros((res, res, res))
        fuselage[10:20, 10:20, 10:20] = 1.0
        geom.set_part_mask(AircraftPart.FUSELAGE, fuselage)

        config = CFDConfig(base_grid_resolution=res, use_amr=True)
        simulator = AdvancedCFDSimulator(config, device=self.device)

        # This should not crash
        results = simulator.simulate_aerodynamics(geom, steps=1)
        self.assertIn('drag_coefficient', results)

    def test_typed_geometry_external_validation_path(self):
        """Verify that typed geometry correctly propagates through external validation path."""
        res = self.res
        geom = TypedAircraftGeometry(res, device=self.device)
        fuselage = torch.zeros((res, res, res))
        fuselage[10:20, 10:20, 10:20] = 1.0
        geom.set_part_mask(AircraftPart.FUSELAGE, fuselage)

        config = CFDConfig(base_grid_resolution=res)
        # Probability 1.0 to force external validation path
        config.validation_probability = 1.0
        simulator = AdvancedCFDSimulator(config, device=self.device)

        # This should not crash even if external validation returns None
        results = simulator.simulate_aerodynamics(geom, steps=1)
        self.assertIn('drag_coefficient', results)

    def test_failed_stl_repair_rejects_export(self):
        """Verify that if mesh repair fails, export is truly rejected."""
        from mesh_utils import voxels_to_stl_checked
        res = 8
        # Create a non-manifold, weird voxel pattern that might fail marching cubes or repair
        # A single floating voxel might be hard to make non-watertight but let's try
        voxels = torch.zeros((res, res, res))
        voxels[1, 1, 1] = 1.0

        report = ConstraintReport()
        # We'll mock trimesh.Trimesh.is_watertight to return False and fill_holes to do nothing
        import unittest.mock as mock
        with mock.patch('trimesh.Trimesh.is_watertight', new_callable=mock.PropertyMock) as mock_watertight:
            mock_watertight.return_value = False
            with mock.patch('trimesh.Trimesh.fill_holes') as mock_fill:
                # fill_holes doesn't fix it
                success = voxels_to_stl_checked(voxels, "fail.stl", resolution=res, report=report)
                self.assertFalse(success)
                self.assertEqual(report.export_status, "rejected")
                self.assertFalse(os.path.exists("fail.stl"))

    def test_empty_voxel_export_sets_rejected(self):
        """Verify that empty voxel grid results in rejected export status."""
        from mesh_utils import voxels_to_stl_checked
        res = 8
        voxels = torch.zeros((res, res, res))
        report = ConstraintReport()
        success = voxels_to_stl_checked(voxels, "empty.stl", resolution=res, report=report)
        self.assertFalse(success)
        self.assertEqual(report.export_status, "rejected")
        self.assertTrue(any(v.type == "empty_or_invalid_mesh" for v in report.violations))

    def test_valid_fixture_no_manufacturing_violations(self):
        """Verify that a valid fixture doesn't get bogus manufacturing violations."""
        res = 32
        geom = TypedAircraftGeometry(res, device=self.device)
        # Create a reasonably thick skin (3x3x3 shell)
        skin = torch.zeros((res, res, res))
        skin[10:20, 10:20, 10:20] = 1.0
        # Make it a shell
        eroded = torch.zeros((res, res, res))
        eroded[11:19, 11:19, 11:19] = 1.0
        skin = skin - eroded
        geom.set_part_mask(AircraftPart.SKIN, skin)

        report = ConstraintReport()
        projector = ConstraintProjector(res, device=self.device, existing_report=report)
        # Test with metal_sheet which checks for shell
        mission = MissionProfile(manufacturing_method="metal_sheet")
        projector.project(geom, mission)

        self.assertFalse(any(v.type == "metal_sheet_shell" for v in report.violations))

    def test_spar_repair_connectivity(self):
        """Verify that spar repair ensures real connectivity, not just voxel count."""
        res = 32
        geom = TypedAircraftGeometry(res, device=self.device)

        # Fuselage at front, Spar at back
        fuselage = torch.zeros((res, res, res))
        fuselage[:, :, :5] = 1.0
        geom.set_part_mask(AircraftPart.FUSELAGE, fuselage)

        spar = torch.zeros((res, res, res))
        spar[:, :, 25:] = 1.0
        geom.set_part_mask(AircraftPart.SPAR, spar)

        # Initially not connected
        projector = ConstraintProjector(res, device=self.device)
        self.assertFalse(projector._check_connectivity(spar, fuselage))

        # Run project
        report = ConstraintReport()
        projector = ConstraintProjector(res, device=self.device, existing_report=report)
        projected = projector.project(geom, MissionProfile())

        new_spar = projected.get_part_mask(AircraftPart.SPAR)
        # Now should be connected
        self.assertTrue(projector._check_connectivity(new_spar, fuselage))
        self.assertTrue(any(v.type == "spar_continuity" for v in report.violations))

if __name__ == '__main__':
    unittest.main()
