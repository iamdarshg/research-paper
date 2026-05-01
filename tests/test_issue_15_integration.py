
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
from models import AeroSurrogate, MissionEncoder, LatentDiffusionUNet, LatentTo3DConverter, ConsistencyModel

class TestIssue15Integration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("./test_issue_15_integration")
        self.test_dir.mkdir(exist_ok=True, parents=True)
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

        surr = AeroSurrogate(8, 16)
        enc = MissionEncoder(8)
        opt = torch.optim.Adam(list(surr.parameters()) + list(enc.parameters()))

        for geoms, targets, missions, meta in loader:
            # Reconstruct profiles from collated dict
            profiles = []
            for i in range(geoms.shape[0]):
                kwargs = {k: (v[i].item() if torch.is_tensor(v) else v[i]) for k, v in missions.items()}
                profiles.append(MissionProfile(**kwargs))
            cond = enc(profiles)
            surr.train_step(geoms, targets, cond, opt)

        # Verify surrogate is ready (met threshold of 5 in ModelConfig)
        # Use high max_loss because model is random in test
        self.assertTrue(surr.is_ready(min_samples=5, max_loss=10.0))

        surr_path = self.test_dir / "trained_surr.pt"
        torch.save({'model_state_dict': surr.state_dict(), 'encoder_state_dict': enc.state_dict()}, surr_path)

        # 3. LOAD AND RANK (No patching of load_state_dict!)
        generator = OptimizedAircraftGenerator(str(self.checkpoint_path), device=self.device)

        generator.load_surrogate(str(surr_path))
        # Override config thresholds for the test to ensure it uses the surrogate path
        generator.model_config.surrogate_min_samples = 5
        generator.model_config.surrogate_max_loss = 10.0

        self.assertTrue(generator.surrogate.is_ready(min_samples=5, max_loss=10.0))

        # We only mock the expensive diffusion sampling, not the wiring
        generator.consistency_model.fast_inference = MagicMock(return_value=torch.randn(2, 8))

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
