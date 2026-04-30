
import unittest
import torch
import os
import json
import shutil
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import MissionProfile, CFDConfig, LabelTier, ModelConfig, DiffusionConfig
from generator import OptimizedAircraftGenerator
from data_utils import GroundTruthExporter, CFDLabelDataset
from models import AeroSurrogate, MissionEncoder

class TestIssue15Integration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_issue_15_integration")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device('cpu')

        # Create a mock checkpoint
        self.checkpoint_path = self.test_dir / "gen_model.pt"
        self.model_config = ModelConfig(latent_dim=16, condition_dim=32, grid_resolution=16, surrogate_min_samples=5)
        mock_checkpoint = {
            'model_config': self.model_config.__dict__,
            'diffusion_config': DiffusionConfig().__dict__,
            'consistency_model': {},
            'diffusion_model': {},
            'converter': {},
            'mission_encoder': MissionEncoder(32).state_dict(),
            'aero_surrogate': AeroSurrogate(32, 16).state_dict()
        }
        torch.save(mock_checkpoint, self.checkpoint_path)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        if Path("./ground_truth").exists():
            shutil.rmtree("./ground_truth")

    def test_full_multi_fidelity_loop(self):
        """Verify the full label -> train -> load -> rank -> promote loop."""

        # 1. GENERATE LABELS (Mocked)
        exporter = GroundTruthExporter()
        geom = torch.zeros((16, 16, 16))
        for i in range(10):
            meta = {
                'cd': 0.1 + i*0.01,
                'cl': 0.5,
                'tier': 'lbm_raw',
                'geometry_ref': f"sample_loop_{i}/geometry.npy",
                'mission_profile': {'aircraft_class': 'uav'}
            }
            sample_dir = Path("./ground_truth") / f"sample_loop_{i}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            import numpy as np
            np.save(sample_dir / "geometry.npy", geom.numpy())
            exporter.export_sample(f"loop_{i}", geom, metadata=meta)

        self.assertTrue(Path("./ground_truth/cfd_labels.json").exists())

        # 2. TRAIN SURROGATE (Integration level)
        dataset = CFDLabelDataset()
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=5)

        surr = AeroSurrogate(32, 16)
        enc = MissionEncoder(32)
        opt = torch.optim.Adam(list(surr.parameters()) + list(enc.parameters()))

        for geoms, targets, missions in loader:
            # Reconstruct profiles from collated dict
            profiles = []
            for i in range(geoms.shape[0]):
                kwargs = {k: (v[i].item() if torch.is_tensor(v) else v[i]) for k, v in missions.items()}
                profiles.append(MissionProfile(**kwargs))
            cond = enc(profiles)
            surr.train_step(geoms, targets, cond, opt)

        # Verify surrogate is ready (met threshold of 5 in ModelConfig)
        # Use high max_mse because model is random in test
        self.assertTrue(surr.is_ready(min_samples=5, max_mse=10.0))

        surr_path = self.test_dir / "trained_surr.pt"
        torch.save({'model_state_dict': surr.state_dict(), 'encoder_state_dict': enc.state_dict()}, surr_path)

        # 3. LOAD AND RANK
        with patch('torch.load', return_value=torch.load(self.checkpoint_path)):
            with patch('models.LatentDiffusionUNet.load_state_dict'), \
                 patch('models.LatentTo3DConverter.load_state_dict'), \
                 patch('models.ConsistencyModel.load_state_dict'), \
                 patch('models.MissionEncoder.load_state_dict'), \
                 patch('models.AeroSurrogate.load_state_dict'):
                generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.load_surrogate(str(surr_path))
        # Override config thresholds for the test to ensure it uses the surrogate path
        generator.model_config.surrogate_min_samples = 5
        generator.model_config.surrogate_max_mse = 10.0

        self.assertTrue(generator.surrogate.is_ready(min_samples=5, max_mse=10.0))

        # Mock generation components
        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(2, 16))
        generator.converter = MagicMock(return_value=torch.randn(2, 16, 16, 16))

        # 4. GENERATE WITH PROMOTION
        mission = MissionProfile()
        with patch('cfd_simulator.AdvancedCFDSimulator.simulate_aerodynamics') as mock_sim:
            # 2 top-k calls (raw) + 1 final call (promoted)
            mock_sim.side_effect = [
                {'drag_coefficient': 0.1, 'lift_coefficient': 0.5, 'label_tier': 'lbm_raw', 'constraints': {'valid': True, 'violations': [], 'repaired': False}},
                {'drag_coefficient': 0.12, 'lift_coefficient': 0.5, 'label_tier': 'lbm_raw', 'constraints': {'valid': True, 'violations': [], 'repaired': False}},
                {'drag_coefficient': 0.08, 'lift_coefficient': 0.5, 'label_tier': 'external_pde', 'constraints': {'valid': True, 'violations': [], 'repaired': False}}
            ]

            # This should use surrogate ranking because it's ready
            generator.generate(mission, num_candidates=2, top_k=2, return_results=True, external_val_mode='final')

            # Verify promotion used the SAME ID (Fix 4: history preservation)
            with open("./ground_truth/cfd_labels.json", 'r') as f:
                final_labels = json.load(f)

            # Find the promoted label
            promoted = [l for l in final_labels if l['tier'] == 'external_pde']
            self.assertEqual(len(promoted), 1)

            # It should have history (from the raw run of the same ID)
            self.assertGreater(len(promoted[0].get('fidelity_history', [])), 0)
            self.assertEqual(promoted[0]['fidelity_history'][0]['tier'], 'lbm_raw')

if __name__ == '__main__':
    unittest.main()
