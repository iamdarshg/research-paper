
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import ModelConfig, DiffusionConfig, MissionProfile, DesignSpec
from models import MissionEncoder, LatentDiffusionUNet, ConsistencyModel, ResidualBlock3D, LatentTo3DConverter

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
            MissionProfile(cruise_speed_mps=-10.0)
        with self.assertRaises(ValueError):
            MissionProfile(aircraft_class="ufo")
        with self.assertRaises(ValueError):
            # stall >= cruise
            MissionProfile(cruise_speed_mps=30, stall_speed_mps=30)

        mp = MissionProfile(cruise_speed_mps=100.0, aircraft_class="fighter", stall_speed_mps=40)
        self.assertEqual(mp.cruise_speed_mps, 100.0)

    def test_design_spec_adapter_validation(self):
        """Verify DesignSpec adapter handles default and large speeds correctly"""
        # Default
        ds = DesignSpec()
        mp = ds.to_mission_profile()
        self.assertLess(mp.stall_speed_mps, mp.cruise_speed_mps)
        self.assertEqual(mp.cruise_speed_mps, 7.0)

        # Large speed
        ds2 = DesignSpec(target_speed=200.0)
        mp2 = ds2.to_mission_profile()
        self.assertLess(mp2.stall_speed_mps, mp2.cruise_speed_mps)

    def test_mission_encoder_output(self):
        """Verify MissionEncoder returns correct shape"""
        mp = MissionProfile(cruise_speed_mps=60.0, manufacturing_method="composite", stall_speed_mps=20)
        condition = self.encoder(mp)
        self.assertEqual(condition.shape, (1, 32))

        mps = [mp, MissionProfile(cruise_speed_mps=40.0, stall_speed_mps=10)]
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
        """Verify that different mission profiles yield different geometry under fixed noise"""
        from models import LatentTo3DConverter
        torch.manual_seed(42)
        cm = ConsistencyModel(self.model_config, self.diffusion_config)

        # Manually perturb conditioning weights so un-trained model shows difference
        for m in cm.student_model.modules():
            if isinstance(m, ResidualBlock3D) and m.condition_mlp is not None:
                nn.init.normal_(m.condition_mlp[-1].weight, std=10.0)
                nn.init.normal_(m.condition_mlp[-1].bias, std=10.0)

        conv = LatentTo3DConverter(latent_dim=16, grid_resolution=16)
        initial_noise = torch.randn(1, 16)

        mp1 = MissionProfile(cruise_speed_mps=20.0, aircraft_class="uav", stall_speed_mps=10)
        mp2 = MissionProfile(cruise_speed_mps=150.0, aircraft_class="fighter", stall_speed_mps=40)

        c1 = self.encoder(mp1)
        c2 = self.encoder(mp2)

        # Latents
        l1 = cm.fast_inference((1, 16), num_steps=4, condition=c1, initial_noise=initial_noise)
        l2 = cm.fast_inference((1, 16), num_steps=4, condition=c2, initial_noise=initial_noise)

        # Voxel grids
        v1 = torch.sigmoid(conv(l1))
        v2 = torch.sigmoid(conv(l2))

        # Acceptance-style test using thresholded occupancy
        occ1 = (v1 > 0.5).float().sum().item()
        occ2 = (v2 > 0.5).float().sum().item()
        self.assertNotEqual(occ1, occ2)

        # Same condition should yield same geometry
        l1_again = cm.fast_inference((1, 16), num_steps=4, condition=c1, initial_noise=initial_noise)
        v1_again = torch.sigmoid(conv(l1_again))
        self.assertTrue(torch.allclose(v1, v1_again))

    def test_trainer_normalization_formats(self):
        """Verify trainer correctly normalizes diverse mission batch formats"""
        from trainer import OptimizedDiffusionTrainer
        from config import CFDConfig, TrainingConfig

        trainer = OptimizedDiffusionTrainer(self.model_config, self.diffusion_config, TrainingConfig(), CFDConfig())

        # Single MissionProfile
        mp = MissionProfile(cruise_speed_mps=50, stall_speed_mps=20)
        batch1 = {'mission_profile': mp}
        res1 = trainer._normalize_mission_batch(batch1, 2)
        self.assertEqual(len(res1), 2)
        self.assertEqual(res1[0].cruise_speed_mps, 50)

        # List of MissionProfile
        batch2 = {'mission_profile': [mp, mp]}
        res2 = trainer._normalize_mission_batch(batch2, 2)
        self.assertEqual(res2[0].cruise_speed_mps, 50)

        # List of dicts
        batch3 = {'mission_profile': [{'cruise_speed_mps': 60, 'stall_speed_mps': 30}, {'cruise_speed_mps': 40, 'stall_speed_mps': 10}]}
        res3 = trainer._normalize_mission_batch(batch3, 2)
        self.assertEqual(res3[0].cruise_speed_mps, 60)
        self.assertEqual(res3[1].cruise_speed_mps, 40)

        # Dict of tensors
        batch4 = {'mission_profile': {
            'cruise_speed_mps': torch.tensor([80.0, 90.0]),
            'stall_speed_mps': torch.tensor([40.0, 45.0]),
            'aircraft_class': ['uav', 'fighter']
        }}
        res4 = trainer._normalize_mission_batch(batch4, 2)
        self.assertEqual(res4[0].cruise_speed_mps, 80.0)
        self.assertEqual(res4[1].aircraft_class, 'fighter')

    def test_generator_propagation(self):
        """Verify generator correctly encodes mission and propagates to inference"""
        from generator import OptimizedAircraftGenerator
        from unittest.mock import MagicMock, patch

        # Use a mock checkpoint
        mock_checkpoint = {
            'model_config': {'latent_dim': 16, 'condition_dim': 32, 'grid_resolution': 16},
            'diffusion_config': {'timesteps': 100, 'beta_start': 0.0001, 'beta_end': 0.02, 'student_steps': 4, 'teacher_steps': 1000},
            'consistency_model': {},
            'diffusion_model': {},
            'converter': {},
            'mission_encoder': {}
        }

        with patch('torch.load', return_value=mock_checkpoint):
            with patch.object(LatentDiffusionUNet, 'load_state_dict'):
                with patch.object(LatentTo3DConverter, 'load_state_dict'):
                    with patch.object(ConsistencyModel, 'load_state_dict'):
                        with patch.object(MissionEncoder, 'load_state_dict'):
                            gen = OptimizedAircraftGenerator("fake.pt")

        gen.mission_encoder.forward = MagicMock(return_value=torch.randn(1, 32))
        gen.consistency_model.fast_inference = MagicMock(return_value=torch.randn(1, 16))

        mp = MissionProfile(cruise_speed_mps=50, stall_speed_mps=20)
        gen.generate(mp)

        # Verify encoder received the profile
        gen.mission_encoder.forward.assert_called_with(mp)
        # Verify inference received the condition
        self.assertIn('condition', gen.consistency_model.fast_inference.call_args.kwargs)
        self.assertIsNotNone(gen.consistency_model.fast_inference.call_args.kwargs['condition'])

    def test_propagation_spies(self):
        """Verify condition propagation using mock/spy techniques"""
        from unittest.mock import MagicMock

        # Mock student model to capture calls
        cm = ConsistencyModel(self.model_config, self.diffusion_config)
        cm.student_model.forward = MagicMock(side_effect=cm.student_model.forward)
        cm.teacher_model.forward = MagicMock(side_effect=cm.teacher_model.forward)

        condition = torch.randn(1, 32)

        # Test inference propagation
        cm.fast_inference((1, 16), num_steps=2, condition=condition)
        # Check that student was called with condition at every step
        self.assertEqual(cm.student_model.forward.call_count, 2)
        for call in cm.student_model.forward.call_args_list:
            self.assertTrue(torch.allclose(call.kwargs['condition'], condition))

        # Test loss propagation
        cm.student_model.forward.reset_mock()
        x = torch.randn(1, 16)
        t_s = torch.tensor([1])
        t_t = torch.tensor([10])
        cm.consistency_loss(x, t_s, t_t, condition=condition)

        # Both teacher and student should have received condition
        cm.student_model.forward.assert_called()
        self.assertTrue(torch.allclose(cm.student_model.forward.call_args.kwargs['condition'], condition))
        cm.teacher_model.forward.assert_called()
        self.assertTrue(torch.allclose(cm.teacher_model.forward.call_args.kwargs['condition'], condition))

if __name__ == '__main__':
    unittest.main()
