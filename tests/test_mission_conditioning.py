
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import ModelConfig, DiffusionConfig, MissionProfile, DesignSpec
from models import MissionEncoder, LatentDiffusionUNet, ConsistencyModel, ResidualBlock3D

class TestMissionConditioning(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.model_config = ModelConfig(latent_dim=16, condition_dim=32)
        self.diffusion_config = DiffusionConfig()
        self.encoder = MissionEncoder(condition_dim=32).to(self.device)
        self.unet = LatentDiffusionUNet(self.model_config, self.diffusion_config).to(self.device)

    def test_mission_profile_validation(self):
        """Verify MissionProfile validates inputs"""
        with self.assertRaises(ValueError):
            MissionProfile(target_speed=-10.0)
        with self.assertRaises(ValueError):
            MissionProfile(aircraft_type="ufo")

        mp = MissionProfile(target_speed=100.0, aircraft_type="military")
        self.assertEqual(mp.target_speed, 100.0)

    def test_mission_encoder_output(self):
        """Verify MissionEncoder returns correct shape"""
        mp = MissionProfile(target_speed=60.0, aircraft_type="cargo")
        condition = self.encoder(mp)
        self.assertEqual(condition.shape, (1, 32))

        mps = [mp, MissionProfile(target_speed=40.0)]
        condition_batch = self.encoder(mps)
        self.assertEqual(condition_batch.shape, (2, 32))

    def test_unet_conditioning_path(self):
        """Verify UNet residual blocks receive and use condition"""
        x = torch.randn(2, 16)
        t = torch.tensor([1, 5])
        condition = torch.randn(2, 32)

        # Should pass with correct shape
        out = self.unet(x, t, condition=condition)
        self.assertEqual(out.shape, (2, 16))

        # Should fail with wrong shape
        with self.assertRaises(ValueError):
            self.unet(x, t, condition=torch.randn(2, 16))

    def test_consistency_model_propagation(self):
        """Verify ConsistencyModel propagates condition"""
        cm = ConsistencyModel(self.model_config, self.diffusion_config)
        x = torch.randn(2, 16)
        t_s = torch.tensor([1, 2])
        t_t = torch.tensor([10, 20])
        condition = torch.randn(2, 32)

        # Verify loss computation doesn't crash
        loss = cm.consistency_loss(x, t_s, t_t, condition=condition)
        self.assertTrue(torch.isfinite(loss))

        # Verify inference doesn't crash
        out = cm.fast_inference((1, 16), num_steps=2, condition=condition[:1])
        self.assertEqual(out.shape, (1, 16))

    def test_deterministic_ab_difference(self):
        """Verify that different mission profiles yield different outputs under fixed noise"""
        # We need a small trained-like state for measurable difference,
        # but even random weights should show SOME difference if propagation works.
        torch.manual_seed(42)
        cm = ConsistencyModel(self.model_config, self.diffusion_config)
        initial_noise = torch.randn(1, 16)

        mp1 = MissionProfile(target_speed=20.0, aircraft_type="civilian")
        mp2 = MissionProfile(target_speed=150.0, aircraft_type="military")

        c1 = self.encoder(mp1)
        c2 = self.encoder(mp2)

        out1 = cm.fast_inference((1, 16), num_steps=4, condition=c1, initial_noise=initial_noise)
        out2 = cm.fast_inference((1, 16), num_steps=4, condition=c2, initial_noise=initial_noise)

        # Outputs should be different because conditions are different
        self.assertFalse(torch.allclose(out1, out2, atol=1e-5))

        # Same condition should yield same output
        out1_again = cm.fast_inference((1, 16), num_steps=4, condition=c1, initial_noise=initial_noise)
        self.assertTrue(torch.allclose(out1, out1_again))

if __name__ == '__main__':
    unittest.main()
