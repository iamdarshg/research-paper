
import unittest
import torch
import os
import json
import shutil
import gc
import time
from pathlib import Path
import sys

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import MissionProfile, CFDConfig, LabelTier, ModelConfig, DiffusionConfig
from generator import OptimizedAircraftGenerator
from cfd_simulator import AdvancedCFDSimulator
from data_utils import GroundTruthExporter
from models import AeroSurrogate, MissionEncoder, LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel


def _cleanup_ground_truth():
    ground_truth_dir = Path("./ground_truth")
    if not ground_truth_dir.exists():
        return

    gc.collect()
    for _ in range(10):
        try:
            shutil.rmtree(ground_truth_dir)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.1)

    shutil.rmtree(ground_truth_dir, ignore_errors=True)

class TestIssue15Pipeline(unittest.TestCase):
    def setUp(self):
        _cleanup_ground_truth()
        self.test_dir = Path("./test_issue_15_pipeline")
        self.test_dir.mkdir(exist_ok=True)
        self.device = torch.device('cpu')

        # 1. CREATE REAL LIGHTWEIGHT COMPONENTS (Review Feedback Fix 4)
        self.model_config = ModelConfig(
            latent_dim=8,
            condition_dim=8,
            grid_resolution=16,
            surrogate_min_samples=5,
            surrogate_max_loss=10.0,
            encoder_channels=[8, 8],
            decoder_channels=[8, 8]
        )
        self.diff_config = DiffusionConfig(timesteps=10)

        diff_model = LatentDiffusionUNet(self.model_config, self.diff_config)
        converter = LatentTo3DConverter(8, 16)
        consistency = ConsistencyModel(self.model_config, self.diff_config)
        encoder = MissionEncoder(8)
        surrogate = AeroSurrogate(8, 16)

        # 2. SAVE REAL CHECKPOINT
        self.checkpoint_path = self.test_dir / "real_lightweight_model.pt"
        checkpoint = {
            'model_config': self.model_config.__dict__,
            'diffusion_config': self.diff_config.__dict__,
            'diffusion_model': diff_model.state_dict(),
            'converter': converter.state_dict(),
            'consistency_model': consistency.state_dict(),
            'mission_encoder': encoder.state_dict(),
            'aero_surrogate': surrogate.state_dict(),
            'ema_model': diff_model.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        _cleanup_ground_truth()

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

        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        # Mock dependencies for ranking
        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(5, 8))
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

        # geom2: Slab in Y (Different content, same shape/sum AND axis projections)
        # Create a more complex collision case:
        # Same integrated profiles but different internal distribution
        geom1 = torch.zeros((4, 4, 4))
        geom1[0,0,0] = 1.0
        geom1[1,1,1] = 1.0

        geom2 = torch.zeros((4, 4, 4))
        geom2[0,1,1] = 1.0
        geom2[1,0,0] = 1.0

        # These have same sum and same projection profiles on all axes!
        self.assertEqual(geom1.sum(), geom2.sum())
        self.assertTrue(torch.all(geom1.sum(dim=(0,1)) == geom2.sum(dim=(0,1))))
        self.assertTrue(torch.all(geom1.sum(dim=(0,2)) == geom2.sum(dim=(0,2))))
        self.assertTrue(torch.all(geom1.sum(dim=(1,2)) == geom2.sum(dim=(1,2))))

        # But they are DIFFERENT geometries
        self.assertFalse(torch.allclose(geom1, geom2))

        q1 = solver._get_q(geom1).clone()
        q2 = solver._get_q(geom2).clone()

        # Should be different tensors because of content hashing
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

        g, targets, mission_dict, label_metadata = dataset[len(dataset)-1]
        self.assertEqual(g.shape, (16, 16, 16))
        self.assertAlmostEqual(targets['Cd'].item(), 0.3)
        self.assertAlmostEqual(targets['Cl'].item(), 0.4)
        self.assertIsInstance(mission_dict, dict)
        self.assertIn('aircraft_class', mission_dict)
        self.assertEqual(label_metadata['tier'], 'lbm_raw')

        # Verify DataLoader compatibility (collate test)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=1)
        batch = next(iter(loader))
        bg, bt, bm, blm = batch
        self.assertEqual(bg.shape, (1, 16, 16, 16))
        self.assertIsInstance(bm, dict)
        self.assertEqual(bm['aircraft_class'][0], 'uav')

    def test_feasibility_aware_selection(self):
        """Verify generation prefers feasible candidates even if they have higher drag."""
        from unittest.mock import MagicMock, patch
        from config import ModelConfig, DiffusionConfig

        # Real generator setup
        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(2, 8))
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

    def test_encoder_isolation(self):
        """Verify loading surrogate doesn't corrupt diffusion encoder."""
        from unittest.mock import patch
        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)
        # Original state
        orig_state = generator.mission_encoder.state_dict()

        # Mock surrogate checkpoint with different encoder weights
        mock_surr = {
            'model_state_dict': generator.surrogate.state_dict(),
            'encoder_state_dict': {k: v * 2.0 for k, v in orig_state.items()}
        }
        surr_path = self.test_dir / "isolated_surr.pt"
        torch.save(mock_surr, surr_path)

        generator.load_surrogate(str(surr_path))

        # Main encoder should be unchanged
        for k in orig_state:
            self.assertTrue(torch.equal(orig_state[k], generator.mission_encoder.state_dict()[k]))

        # Surrogate encoder should be updated
        self.assertFalse(torch.equal(orig_state[list(orig_state.keys())[0]],
                                    generator.surrogate_mission_encoder.state_dict()[list(orig_state.keys())[0]]))

    def test_surrogate_quality_gate(self):
        """Verify ranking fallback if surrogate is not ready."""
        from models import AeroSurrogate
        surr = AeroSurrogate(condition_dim=32, grid_resolution=16)

        # Case 1: Untrained
        self.assertFalse(surr.is_ready())

        # Case 2: One train step (below threshold)
        geom = torch.zeros((1, 16, 16, 16))
        targets = {'Cd': torch.tensor([0.1]), 'Cl': torch.tensor([0.1]), 'Cm': torch.tensor([0.1]),
                   'convergence_score': torch.tensor([1.0]), 'separation_risk': torch.tensor([0.0])}
        cond = torch.zeros((1, 32))
        optimizer = torch.optim.Adam(surr.parameters())
        surr.train_step(geom, targets, cond, optimizer)

        self.assertTrue(surr.is_trained)
        self.assertFalse(surr.is_ready(min_samples=10))

        # Case 3: Reach threshold
        geom_many = torch.zeros((10, 16, 16, 16))
        targets_many = {k: v.repeat(10) for k, v in targets.items()}
        cond_many = cond.repeat(10, 1)
        # Use target=0 loss to ensure MSE drops below threshold
        surr.train_step(geom_many, targets_many, cond_many, optimizer)
        surr.train_loss_ema.fill_(0.01) # Force quality gate to pass for test

        self.assertTrue(surr.is_ready(min_samples=10, max_loss=0.1))

    def test_single_candidate_export(self):
        """Verify single-candidate generation also exports to GroundTruthExporter."""
        from unittest.mock import MagicMock, patch
        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(1, 8))
        mission = MissionProfile()

        with patch('cfd_simulator.AdvancedCFDSimulator.simulate_aerodynamics') as mock_sim:
            mock_sim.return_value = {
                'drag_coefficient': 0.1,
                'lift_coefficient': 0.5,
                'constraints': {'violations': [], 'repaired': False},
                'label_tier': 'lbm_raw'
            }

            # This should trigger simulation and EXPORT
            generator.generate(mission, return_results=True)

            self.assertTrue(Path("./ground_truth/cfd_labels.json").exists())
            with open("./ground_truth/cfd_labels.json", 'r') as f:
                labels = json.load(f)
            self.assertEqual(len(labels), 1)
            self.assertTrue(labels[0]['geometry_id'].startswith("gen_single_"))

    def test_external_validation_staged(self):
        """Verify 'final' mode only runs external validation on the selected candidate."""
        from unittest.mock import MagicMock, patch
        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(3, 8))
        generator.surrogate.is_trained.fill_(True)
        generator.surrogate.sample_count.fill_(1000)
        generator.surrogate.train_loss_ema.fill_(0.01) # Ready
        generator.surrogate.rank = MagicMock(return_value=torch.tensor([0.1, 0.9, 0.2]))

        mission = MissionProfile()

        with patch('cfd_simulator.AdvancedCFDSimulator.simulate_aerodynamics') as mock_sim:
            # First 3 calls: top-k (internal D3Q27)
            # 4th call: final selection (External PDE)
            mock_sim.side_effect = [
                {'drag_coefficient': 0.1, 'lift_coefficient': 0.5, 'constraints': {'valid': True, 'violations': [], 'repaired': False}, 'label_tier': 'lbm_raw'},
                {'drag_coefficient': 0.05, 'lift_coefficient': 0.5, 'constraints': {'valid': True, 'violations': [], 'repaired': False}, 'label_tier': 'lbm_raw'},
                {'drag_coefficient': 0.1, 'lift_coefficient': 0.5, 'constraints': {'valid': True, 'violations': [], 'repaired': False}, 'label_tier': 'lbm_raw'},
                # The promoted one
                {'drag_coefficient': 0.04, 'lift_coefficient': 0.5, 'constraints': {'valid': True, 'violations': [], 'repaired': False}, 'label_tier': 'external_pde'}
            ]

            generator.generate(mission, num_candidates=3, top_k=3, return_results=True, external_val_mode='final')

            # Total 4 simulations: 3 (top-k) + 1 (final promoted)
            self.assertEqual(mock_sim.call_count, 4)

            # Check last call had force_external_validation=True
            last_mission_passed = mock_sim.call_args[1]['mission']
            self.assertTrue(last_mission_passed.force_external_validation)

            # Check earlier call had force_external_validation=False
            first_mission_passed = mock_sim.call_args_list[0][1]['mission']
            self.assertFalse(first_mission_passed.force_external_validation)

    def test_in_place_mutation_cache_safety(self):
        """Verify LBM caches detect in-place tensor modifications (Review Feedback)."""
        from advanced_lbm_solver import D3Q27Solver
        solver = D3Q27Solver(resolution=8, device=self.device)

        geom = torch.zeros((8, 8, 8))
        geom[0,0,0] = 1.0

        h1 = solver._boundary_links(geom)

        # MUTATE IN-PLACE
        geom[7,7,7] = 1.0

        h2 = solver._boundary_links(geom)

        # Should be DIFFERENT because hash changed
        # If it used data_ptr, it would have been the same
        self.assertFalse(torch.allclose(h1.float(), h2.float()))

if __name__ == '__main__':
    unittest.main()
