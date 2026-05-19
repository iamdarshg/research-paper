
import unittest
import torch
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from advanced_lbm_solver import D3Q27CascadedSolver
from mixed_precision_solver import wrap_solver_mixed_precision
from config import CFDConfig, LBMPhysicsConfig

class TestMixedPrecision(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu') # Use CPU for CI, but MixedPrecision normally wants CUDA
        # Note: wrap_solver_mixed_precision will only enable FP16 if CUDA is available.
        self.config = CFDConfig(base_grid_resolution=8, mach_number=0.1, reynolds_number=100)
        self.config.lbm_config = LBMPhysicsConfig()
        self.config.lbm_config.grid_spacing = 1.0 / 8.0

    def test_mixed_precision_proxy(self):
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        # Even if CUDA is not available, the wrapper should still work (in FP32)
        wrapped = wrap_solver_mixed_precision(solver, enable_fp16=True)

        geometry_mask = torch.zeros((8, 8, 8), device=self.device)
        geometry_mask[3:5, 3:5, 3:5] = 1.0

        # This should call the core solver's native logic through the wrapper
        wrapped.collide_stream(geometry_mask, steps=1)

        # Verify that macroscopic fields were updated
        self.assertFalse(torch.allclose(wrapped.velocity_x, torch.zeros_like(wrapped.velocity_x)))

        # Verify we can compute macroscopic
        rho, u = wrapped.compute_macroscopic()
        self.assertEqual(rho.shape, (8, 8, 8))
        self.assertEqual(u.shape, (3, 8, 8, 8))

if __name__ == '__main__':
    unittest.main()
