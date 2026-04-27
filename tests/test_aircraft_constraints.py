
import unittest
import torch
import sys
import os
import torch.nn.functional as F

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from geometry import AircraftPart, TypedAircraftGeometry
from constraints import ConstraintProjector, ConstraintReport
from config import MissionProfile

class TestAircraftConstraints(unittest.TestCase):
    def setUp(self):
        self.res = 32
        self.device = torch.device('cpu')

    def test_typed_geometry_channels(self):
        geom = TypedAircraftGeometry(self.res, device=self.device)
        self.assertEqual(geom.tensor.shape[0], len(AircraftPart))

        mask = torch.zeros((self.res, self.res, self.res))
        mask[10:20, 10:20, 10:20] = 1.0
        geom.set_part_mask(AircraftPart.FUSELAGE, mask)

        combined = geom.get_combined_occupancy()
        self.assertEqual(torch.sum(combined), 10*10*10)

    def test_constraint_projection_symmetry(self):
        mission = MissionProfile(aircraft_class="uav", manufacturing_method="3d_print")
        geom = TypedAircraftGeometry(self.res, device=self.device)
        wing_mask = torch.zeros((self.res, self.res, self.res))
        wing_mask[16, 20, 16] = 1.0
        geom.set_part_mask(AircraftPart.WING, wing_mask)

        projector = ConstraintProjector(self.res, device=self.device)
        projected = projector.project(geom, mission)

        mask = projected.get_part_mask(AircraftPart.WING)
        flipped = torch.flip(mask, [1])
        self.assertTrue(torch.allclose(mask, flipped))

    def test_manufacturing_method_specifics(self):
        """Verify that manufacturing method changes active constraints."""
        res = self.res
        skin = torch.zeros((res, res, res))
        skin[10:22, 10:22, 10:22] = 1.0 # Solid block

        # 1. Composite (Curvature smoothing)
        mission_comp = MissionProfile(manufacturing_method="composite")
        geom_comp = TypedAircraftGeometry(res, device=self.device)
        geom_comp.set_part_mask(AircraftPart.SKIN, skin.clone())
        report_comp = ConstraintReport()
        projector_comp = ConstraintProjector(res, device=self.device, existing_report=report_comp)
        projector_comp.project(geom_comp, mission_comp)
        self.assertTrue(any(v.type == "composite_radius" for v in report_comp.violations))

        # 2. Metal Sheet (Shell enforcement)
        mission_metal = MissionProfile(manufacturing_method="metal_sheet")
        geom_metal = TypedAircraftGeometry(res, device=self.device)
        geom_metal.set_part_mask(AircraftPart.SKIN, skin.clone())
        report_metal = ConstraintReport()
        projector_metal = ConstraintProjector(res, device=self.device, existing_report=report_metal)
        projector_metal.project(geom_metal, mission_metal)
        self.assertTrue(any(v.type == "metal_sheet_shell" for v in report_metal.violations))

        # Verify metal shell is thinner than original solid block
        metal_skin = geom_metal.get_part_mask(AircraftPart.SKIN)
        self.assertLess(torch.sum(metal_skin), torch.sum(skin))

    def test_prop_clearance_and_mounts(self):
        mission = MissionProfile(propulsion_type="electric")
        geom = TypedAircraftGeometry(self.res, device=self.device)
        res = self.res
        cx, cy, cz = int(res * 0.1), res // 2, res // 2

        # Obstruction in prop disk
        skin = torch.zeros((res, res, res))
        skin[cz, cy, cx] = 1.0
        geom.set_part_mask(AircraftPart.SKIN, skin)

        report = ConstraintReport()
        projector = ConstraintProjector(self.res, device=self.device, existing_report=report)
        projected = projector.project(geom, mission)

        self.assertTrue(any(v.type == "prop_clearance" for v in report.violations))
        self.assertEqual(projected.get_part_mask(AircraftPart.SKIN)[cz, cy, cx], 0.0)

        # Hardpoint should have been added
        self.assertGreater(torch.sum(projected.get_part_mask(AircraftPart.HARDPOINT)), 0)

    def test_spar_continuity_repair(self):
        mission = MissionProfile()
        geom = TypedAircraftGeometry(self.res, device=self.device)
        res = self.res

        # Disconnected spar (Wing only)
        spar = torch.zeros((res, res, res))
        spar[res//2, res-5:res, res//2] = 1.0
        geom.set_part_mask(AircraftPart.SPAR, spar)

        report = ConstraintReport()
        projector = ConstraintProjector(self.res, device=self.device, existing_report=report)
        projected = projector.project(geom, mission)

        self.assertTrue(any(v.type == "spar_continuity" for v in report.violations))
        self.assertTrue(report.repaired)
        # Spar should now have more voxels (the bridge)
        self.assertGreater(torch.sum(projected.get_part_mask(AircraftPart.SPAR)), 5)

    def test_report_accumulation_pipeline(self):
        """Test that reports accumulate violations through generate -> simulate -> export."""
        from generator import OptimizedAircraftGenerator
        from cfd_simulator import AdvancedCFDSimulator
        from config import CFDConfig

        # Use a real generator with a report
        checkpoint_path = "tests/mock_checkpoint.pt"
        if not os.path.exists(checkpoint_path):
             # Create a mock checkpoint for testing
             model_config = {"latent_dim": 16, "condition_dim": 32, "encoder_channels": [8, 16], "decoder_channels": [16, 8]}
             diff_config = {"timesteps": 100}
             checkpoint = {
                 "model_config": model_config,
                 "diffusion_config": diff_config,
                 "mission_encoder": {
                     "class_emb.weight": torch.randn(7, 8),
                     "prop_emb.weight": torch.randn(5, 4),
                     "mfg_emb.weight": torch.randn(4, 4),
                     "numeric_mlp.0.weight": torch.randn(32, 10),
                     "numeric_mlp.0.bias": torch.randn(32),
                     "numeric_mlp.2.weight": torch.randn(16, 32),
                     "numeric_mlp.2.bias": torch.randn(16),
                     "final_mlp.0.weight": torch.randn(32, 32),
                     "final_mlp.0.bias": torch.randn(32),
                     "final_mlp.2.weight": torch.randn(32, 32),
                     "final_mlp.2.bias": torch.randn(32)
                 },
                 "consistency_model": {"student_model": {}},
                 "diffusion_model": {},
                 "converter": {"decoder.4.weight": torch.randn(32*32*32, 2048), "decoder.4.bias": torch.randn(32*32*32)}
             }
             # Minimal state dict for loading
             torch.save(checkpoint, checkpoint_path)

        generator = OptimizedAircraftGenerator(checkpoint_path, device=self.device)
        mission = MissionProfile(manufacturing_method="metal_sheet")
        report = ConstraintReport()

        # 1. Generation & Projection
        typed_geom = generator.generate(mission, return_typed=True, existing_report=report)
        self.assertGreater(len(report.violations), 0) # Heuristic projection violations

        # 2. Simulation & Feasibility
        sim_config = CFDConfig(base_grid_resolution=self.res)
        simulator = AdvancedCFDSimulator(sim_config, device=self.device)
        results = simulator.simulate_aerodynamics(typed_geom, steps=1, mission=mission, existing_report=report)
        # Re-attach report if simulator creates a new one (it shouldn't if we handle it)
        # Actually simulate_aerodynamics in current implementation takes mission and runs projector.
        # We need to ensure it uses the SAME report.

        # 3. Export
        export_path = "test_accum.stl"
        generator.save_stl(typed_geom, export_path, report=report)

        self.assertIn(report.export_status, ["success", "repaired", "rejected"])
        self.assertGreater(len(report.violations), 0)

        if os.path.exists(export_path): os.remove(export_path)

if __name__ == '__main__':
    unittest.main()
