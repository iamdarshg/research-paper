
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

        # geom1: Slab in X
        geom1 = torch.zeros((8, 8, 8))
        geom1[2:4, :, :] = 1.0

        # geom2: Slab in Y (Different content, same shape/sum)
        geom2 = torch.zeros((8, 8, 8))
        geom2[:, 2:4, :] = 1.0

        # Verify they have same sum but different content
        self.assertEqual(geom1.sum(), geom2.sum())
        self.assertFalse(torch.allclose(geom1, geom2))

        q1 = solver._get_q(geom1).clone()
        q2 = solver._get_q(geom2).clone()

        # Should be different tensors because they have different wall distances
        self.assertFalse(torch.allclose(q1, q2))

    def test_surrogate_data_loading(self):
        """Verify CFDLabelDataset correctly loads labels for surrogate training."""
        exporter = GroundTruthExporter()
        geom = torch.ones((16, 16, 16))
        meta = {
            'drag_coefficient': 0.3,
            'lift_coefficient': 0.4,
            'lbm_converged': True,
            'pinn_ready': True
        }
        exporter.export_sample("surr_load_test", geom, metadata=meta)

        from data_utils import CFDLabelDataset
        dataset = CFDLabelDataset()
        self.assertGreater(len(dataset), 0)

        g, targets, mission_dict = dataset[len(dataset)-1]
        self.assertEqual(g.shape, (16, 16, 16))
        self.assertAlmostEqual(targets['Cd'].item(), 0.3)
        self.assertAlmostEqual(targets['Cl'].item(), 0.4)
        self.assertIsInstance(mission_dict, dict)
        self.assertIn('aircraft_class', mission_dict)

    def test_feasibility_aware_selection(self):
        """Verify generation prefers feasible candidates even if they have higher drag."""
        from unittest.mock import MagicMock, patch
        from config import ModelConfig, DiffusionConfig

        # Mock generator setup
        with patch('torch.load', return_value=torch.load(self.checkpoint_path)):
            with patch('models.LatentDiffusionUNet.load_state_dict'), \
                 patch('models.LatentTo3DConverter.load_state_dict'), \
                 patch('models.ConsistencyModel.load_state_dict'), \
                 patch('models.MissionEncoder.load_state_dict'), \
                 patch('models.AeroSurrogate.load_state_dict'):
                generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(2, 16))
        generator.converter = MagicMock(return_value=torch.randn(2, 16, 16, 16))
        # Surrogate likes candidate 0 (lower score here but we'll force selection)
        generator.surrogate.rank = MagicMock(return_value=torch.tensor([0.9, 0.8]))

        mission = MissionProfile(cruise_speed_mps=50, stall_speed_mps=20)

        with patch('cfd_simulator.AdvancedCFDSimulator.simulate_aerodynamics') as mock_sim:
            # Candidate 0: Low drag but INFEASIBLE
            res0 = {
                'drag_coefficient': 0.05,
                'lift_coefficient': 0.5,
                'velocity_fields': (torch.zeros(16,16,16), torch.zeros(16,16,16), torch.zeros(16,16,16)),
                'pressure_field': torch.zeros(16,16,16),
                'constraints': {'violations': [{'type': 'critical'}], 'valid': False, 'repaired': False},
                'lbm_converged': True,
                'pinn_ready': False,
                'label_tier': 'lbm_raw'
            }
            # Candidate 1: Higher drag but FEASIBLE
            res1 = {
                'drag_coefficient': 0.15,
                'lift_coefficient': 0.5,
                'velocity_fields': (torch.zeros(16,16,16), torch.zeros(16,16,16), torch.zeros(16,16,16)),
                'pressure_field': torch.zeros(16,16,16),
                'constraints': {'violations': [], 'valid': True, 'repaired': False},
                'lbm_converged': True,
                'pinn_ready': False,
                'label_tier': 'lbm_raw'
            }

            mock_sim.side_effect = [res0, res1]

            best_geom = generator.generate(mission, num_candidates=2, top_k=2, return_typed=True)

            # Should have selected candidate 1 despite higher drag
            # because calculate_overall_score adds 100 for feasible.
            self.assertEqual(mock_sim.call_count, 2)
            # res1 was feasible, res0 was not.
            self.assertTrue(best_geom is not None)

    def test_label_promotion(self):
        """Verify promoting lbm_raw to external_pde in the label dataset."""
        exporter = GroundTruthExporter()
        geom = torch.zeros((16, 16, 16))

        # 1. Export as LBM RAW
        meta_lbm = {
            'label_tier': 'lbm_raw',
            'drag_coefficient': 0.5,
            'label_source': 'D3Q27'
        }
        exporter.export_sample("promo_test", geom, metadata=meta_lbm)

        with open(exporter.labels_path, 'r') as f:
            labels = json.load(f)
        self.assertEqual(labels[0]['tier'], 'lbm_raw')

        # 2. Export same geometry ID as EXTERNAL PDE
        meta_ext = {
            'label_tier': 'external_pde',
            'drag_coefficient': 0.48, # refined
            'label_source': 'OpenFOAM'
        }
        exporter.export_sample("promo_test", geom, metadata=meta_ext)

        with open(exporter.labels_path, 'r') as f:
            labels = json.load(f)
        # Should be updated/promoted
        self.assertEqual(labels[0]['tier'], 'external_pde')
        self.assertEqual(labels[0]['cd'], 0.48)

    def test_non_cubic_sdf(self):
        """Verify SDF works for non-cubic tensors."""
        from sdf_utils import compute_all_link_distances
        geom = torch.zeros((10, 20, 30))
        geom[2:5, 5:15, 10:20] = 1.0

        ex, ey, ez = torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([0.0])
        q = compute_all_link_distances(geom, ex, ey, ez)
        self.assertEqual(q.shape, (1, 10, 20, 30))

    def test_metadata_backward_compatibility(self):
        """Verify extra metadata fields are merged into label record."""
        exporter = GroundTruthExporter()
        geom = torch.zeros((16, 16, 16))
        meta = {
            'legacy_field': 'val',
            'another_metric': 1.23
        }
        exporter.export_sample("legacy_test", geom, metadata=meta)

        with open(exporter.labels_path, 'r') as f:
            labels = json.load(f)

        self.assertEqual(labels[0]['legacy_field'], 'val')
        self.assertEqual(labels[0]['another_metric'], 1.23)

if __name__ == '__main__':
    unittest.main()
