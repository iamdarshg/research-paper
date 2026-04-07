
import unittest
import torch
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from advanced_lbm_solver import D3Q27CascadedSolver, GPULBMSolver, D3Q27Lattice
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig

class TestLBMSolvers(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.config = CFDConfig(base_grid_resolution=8, mach_number=0.1, reynolds_number=100)
        self.config.lbm_config = LBMPhysicsConfig()
        self.config.lbm_config.grid_spacing = 1.0 / 8.0

    def test_d3q27_init(self):
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        self.assertEqual(solver.f.shape, (27, 8, 8, 8))
        self.assertFalse(torch.isnan(solver.f).any())

    def test_d3q19_init(self):
        solver = GPULBMSolver(self.config, self.device, LBMPhysicsConfig)
        self.assertEqual(solver.f.shape, (19, 8, 8, 8))
        self.assertFalse(torch.isnan(solver.f).any())

    def test_d3q27_step(self):
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        geometry_mask = torch.zeros((8, 8, 8), device=self.device)
        # Add a small obstacle
        geometry_mask[3:5, 3:5, 3:5] = 1.0

        initial_f = solver.f.clone()
        solver.collide_stream(geometry_mask, steps=1)

        self.assertFalse(torch.equal(initial_f, solver.f))
        self.assertFalse(torch.isnan(solver.f).any())

    def test_d3q27_aero_coeffs(self):
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        geometry_mask = torch.zeros((8, 8, 8), device=self.device)
        geometry_mask[3:5, 3:5, 3:5] = 1.0

        coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
        self.assertIn('drag_coefficient', coeffs)
        self.assertIn('lift_coefficient', coeffs)
        self.assertIsInstance(coeffs['drag_coefficient'], float)

if __name__ == '__main__':
    unittest.main()
