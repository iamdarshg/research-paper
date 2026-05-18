
import unittest
import torch
import sys
import os
import numpy as np

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from aircraft_diffusion_cfd import (
    ModelConfig, DiffusionConfig, TrainingConfig, CFDConfig,
    LatentDiffusionUNet, ConsistencyModel, LatentTo3DConverter,
    OptimizedDiffusionTrainer, AircraftDesignDataset, DesignSpec, aircraft_collate_fn,
    AdvancedCFDSimulator
)

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

    def test_aircraft_collate_fn_preserves_design_spec_objects(self):
        batch = [
            {
                "latent": torch.zeros(16),
                "geometry": torch.zeros((8, 8, 8)),
                "condition_vector": torch.ones(4),
                "design_spec": DesignSpec(target_speed=42.0),
            },
            {
                "latent": torch.ones(16),
                "geometry": torch.ones((8, 8, 8)),
                "condition_vector": torch.zeros(4),
                "design_spec": DesignSpec(target_speed=55.0),
            },
        ]

        collated = aircraft_collate_fn(batch)

        self.assertEqual(collated["latent"].shape, (2, 16))
        self.assertEqual(collated["geometry"].shape, (2, 8, 8, 8))
        self.assertEqual(collated["condition_vector"].shape, (2, 4))
        self.assertEqual(len(collated["design_spec"]), 2)
        self.assertIsInstance(collated["design_spec"][0], DesignSpec)
        self.assertEqual(collated["design_spec"][1].target_speed, 55.0)

if __name__ == '__main__':
    unittest.main()
