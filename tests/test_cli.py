
import unittest
import torch
import sys
import os
import numpy as np

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import ModelConfig, DiffusionConfig, TrainingConfig, CFDConfig, DesignSpec
from models import LatentDiffusionUNet, ConsistencyModel, LatentTo3DConverter
from trainer import OptimizedDiffusionTrainer
from data_utils import AircraftDesignDataset
from cfd_simulator import AdvancedCFDSimulator

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.m_config = ModelConfig(latent_dim=16, base_grid_resolution=8)
        self.d_config = DiffusionConfig(timesteps=10)
        self.t_config = TrainingConfig(num_epochs=1, batch_size=2)
        self.cfd_config = CFDConfig(base_grid_resolution=8)

    def test_model_init(self):
        model = LatentDiffusionUNet(self.m_config, self.d_config)
        self.assertIsInstance(model, LatentDiffusionUNet)

        x = torch.randn(2, 16)
        t = torch.zeros(2, dtype=torch.long)
        out = model(x, t)
        self.assertEqual(out.shape, (2, 16))

    def test_consistency_model(self):
        model = ConsistencyModel(self.m_config, self.d_config)
        self.assertIsInstance(model, ConsistencyModel)

        x = torch.randn(2, 16)
        out = model.fast_inference((2, 16), num_steps=2)
        self.assertEqual(out.shape, (2, 16))

    def test_converter(self):
        conv = LatentTo3DConverter(latent_dim=16, grid_resolution=8)
        latent = torch.randn(2, 16)
        voxels = conv(latent)
        self.assertEqual(voxels.shape, (2, 8, 8, 8))

    def test_simulator_base(self):
        simulator = AdvancedCFDSimulator(self.cfd_config, self.device)
        geometry = torch.zeros((8, 8, 8))
        geometry[3:5, 3:5, 3:5] = 1.0

        results = simulator.simulate_aerodynamics(geometry, steps=1)
        self.assertIn('drag_coefficient', results)
        self.assertIn('lift_coefficient', results)

if __name__ == '__main__':
    unittest.main()
