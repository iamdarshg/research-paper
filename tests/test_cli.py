
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
from run_airshow_flight_path_tests import _binarize_voxel, _blend_lateral_symmetry

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

    def test_high_resolution_converter_uses_coordinate_decoder(self):
        conv = LatentTo3DConverter(latent_dim=16, grid_resolution=96, coordinate_chunk_size=200000)
        latent = torch.randn(1, 16)
        voxels = conv(latent)
        self.assertEqual(conv.decoder_mode, "coordinate")
        self.assertEqual(voxels.shape, (1, 96, 96, 96))
        max_parameter_count = max(param.numel() for param in conv.parameters())
        self.assertLess(max_parameter_count, 1000000)

    def test_high_resolution_converter_decodes_selected_indices(self):
        conv = LatentTo3DConverter(latent_dim=16, grid_resolution=96, coordinate_chunk_size=5)
        latent = torch.randn(2, 16)
        indices = torch.tensor([0, 17, 96 * 96 * 96 - 1], dtype=torch.long)
        logits = conv.forward_flat_indices(latent, indices)
        self.assertEqual(logits.shape, (2, 3))

    def test_target_occupancy_binarization_uses_topk(self):
        grid = torch.tensor(
            [
                [[0.1, 0.7], [0.2, 0.9]],
                [[0.3, 0.8], [0.4, 0.6]],
            ],
            dtype=torch.float32,
        )

        fixed = _binarize_voxel(grid, threshold=0.5)
        self.assertEqual(int(fixed.sum()), 4)

        topk = _binarize_voxel(grid, threshold=0.5, target_occupancy=0.25)
        self.assertEqual(int(topk.sum()), 2)
        self.assertEqual(float(topk[0, 1, 1]), 1.0)
        self.assertEqual(float(topk[1, 0, 1]), 1.0)

    def test_export_symmetry_blend_mirrors_lateral_axis(self):
        grid = torch.zeros((2, 4, 2), dtype=torch.float32)
        grid[:, 0, :] = 1.0

        symmetric = _blend_lateral_symmetry(grid, 1.0)

        self.assertTrue(torch.allclose(symmetric[:, 0, :], symmetric[:, -1, :]))
        self.assertTrue(torch.allclose(symmetric[:, 1, :], symmetric[:, -2, :]))
        self.assertGreater(float(symmetric[:, -1, :].sum()), 0.0)

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
