
import unittest
import torch
import os
import json
import shutil
from pathlib import Path
import sys

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import MissionProfile, CFDConfig, LabelTier
from generator import OptimizedAircraftGenerator
from cfd_simulator import AdvancedCFDSimulator
from data_utils import GroundTruthExporter

class TestIssue15Pipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_issue_15_pipeline")
        self.test_dir.mkdir(exist_ok=True)
        self.device = torch.device('cpu')

        # Create a mock checkpoint
        self.checkpoint_path = self.test_dir / "mock_model.pt"
        mock_checkpoint = {
            'model_config': {'latent_dim': 16, 'condition_dim': 32, 'grid_resolution': 16, 'base_grid_resolution': 16},
            'diffusion_config': {'timesteps': 100, 'beta_start': 0.0001, 'beta_end': 0.02, 'student_steps': 4, 'teacher_steps': 1000},
            'consistency_model': {},
            'diffusion_model': {},
            'converter': {},
            'mission_encoder': {},
            'aero_surrogate': {}
        }
        torch.save(mock_checkpoint, self.checkpoint_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if Path("./ground_truth").exists():
            shutil.rmtree("./ground_truth")

    def test_label_tier_serialization(self):
        """Verify label tiers survive export/load."""
        exporter = GroundTruthExporter()
        geom = torch.zeros((16, 16, 16))
        meta = {
            'label_tier': LabelTier.EXTERNAL_PDE,
            'drag_coefficient': 0.5,
            'lbm_converged': True,
            'pinn_ready': True
        }
        exporter.export_sample("tier_test", geom, metadata=meta)

        with open("./ground_truth/cfd_labels.json", 'r') as f:
            labels = json.load(f)

        self.assertEqual(labels[0]['tier'], "external_pde")

    def test_candidate_ranking_flow(self):
        """Verify the Sample -> Rank -> Validate flow works."""
        from unittest.mock import MagicMock, patch

        with patch('torch.load', return_value=torch.load(self.checkpoint_path)):
            with patch('models.LatentDiffusionUNet.load_state_dict'):
                with patch('models.LatentTo3DConverter.load_state_dict'):
                    with patch('models.ConsistencyModel.load_state_dict'):
                        with patch('models.MissionEncoder.load_state_dict'):
                             with patch('models.AeroSurrogate.load_state_dict'):
                                 generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        # Mock dependencies for ranking
        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(5, 16))
        generator.converter = MagicMock(return_value=torch.randn(5, 16, 16, 16))
        generator.surrogate.rank = MagicMock(return_value=torch.tensor([0.1, 0.5, 0.2, 0.9, 0.3]))

        mission = MissionProfile(cruise_speed_mps=50, stall_speed_mps=20)

        # This calls generate_candidates
        with patch('cfd_simulator.AdvancedCFDSimulator.simulate_aerodynamics') as mock_sim:
            mock_sim.return_value = {
                'drag_coefficient': 0.1,
                'lift_coefficient': 0.5,
                'velocity_fields': (torch.zeros(16,16,16), torch.zeros(16,16,16), torch.zeros(16,16,16)),
                'pressure_field': torch.zeros(16,16,16),
                'constraints': {'violations': [], 'repaired': False},
                'pinn_ready': False,
                'label_tier': 'lbm_raw'
            }

            # Request top 2 candidates from 5 samples
            best_geom = generator.generate(mission, num_candidates=5, top_k=2)

            # Verify D3Q27 was called for top-k (2 times)
            self.assertEqual(mock_sim.call_count, 2)

            # Verify labels were exported
            self.assertTrue(Path("./ground_truth/cfd_labels.json").exists())

    def test_cache_invalidation_per_geometry(self):
        """Verify BFL q-cache is not reused across different geometries (Fix A)."""
        from advanced_lbm_solver import D3Q27Solver
        solver = D3Q27Solver(resolution=8, device=self.device)

        geom1 = torch.zeros((8, 8, 8))
        geom1[2:4, :, :] = 1.0

        geom2 = torch.zeros((8, 8, 8))
        geom2[:, 2:4, :] = 1.0

        q1 = solver._get_q(geom1)
        q1_ptr = q1.data_ptr()

        q2 = solver._get_q(geom2)
        q2_ptr = q2.data_ptr()

        # Should be different pointers/tensors
        self.assertNotEqual(q1_ptr, q2_ptr)

        # geom1 again should hit cache if data_ptr same (though in practice with new tensors it might not)
        # But definitely geom1 and geom2 must be different.
        self.assertFalse(torch.allclose(q1, q2))

if __name__ == '__main__':
    unittest.main()
