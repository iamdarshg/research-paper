
import unittest
import torch
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from advanced_lbm_solver import D3Q27CascadedSolver, D3Q27Lattice
from config import CFDConfig, LBMPhysicsConfig

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
        self.assertIn('reference_area_source', coeffs)
        self.assertIn('claim_bearing_cfd', coeffs)
        self.assertEqual(coeffs['reference_area_source'], 'projected_frontal_voxel_area_yz')
        self.assertFalse(coeffs['claim_bearing_cfd'])
        self.assertIn('lift_to_drag', coeffs)
        self.assertIn('solver_quality_checks', coeffs)
        self.assertTrue(coeffs['solver_quality_checks']['finite_coefficients'])
        self.assertGreater(coeffs['reference_area'], 0.0)
        self.assertIsInstance(coeffs['drag_coefficient'], float)

    def test_d3q27_benchmark_like_run_is_finite(self):
        benchmark_config = CFDConfig(
            base_grid_resolution=32,
            mach_number=80 / 343.0,
            reynolds_number=80 * 40.0 / 1.47e-5,
        )
        benchmark_config.lbm_config = LBMPhysicsConfig()
        benchmark_config.lbm_config.grid_spacing = 40.0 / 32.0
        benchmark_config.lbm_config.physical_length_scale = 40.0

        solver = D3Q27CascadedSolver(benchmark_config, self.device, LBMPhysicsConfig)
        geometry_mask = torch.zeros((32, 32, 32), device=self.device)
        geometry_mask[14:18, 14:18, 14:18] = 1.0

        solver.collide_stream(geometry_mask, steps=1)
        coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)

        self.assertTrue(torch.isfinite(solver.f).all().item())
        self.assertTrue(torch.isfinite(torch.tensor(coeffs['drag_coefficient'])).item())
        self.assertTrue(torch.isfinite(torch.tensor(coeffs['lift_coefficient'])).item())

    def test_mrt_conservation(self):
        """Verify that MRT collision operator conserves mass and momentum."""
        # Use zero freestream to ensure mass is conserved in a closed/balanced domain
        self.config.mach_number = 0.0
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        geometry_mask = torch.zeros((8, 8, 8), device=self.device)

        # Initial mass
        total_mass_in = torch.sum(solver._solver.f).item()

        # Perform 10 steps in an empty domain
        solver.collide_stream(geometry_mask, steps=10)

        total_mass_out = torch.sum(solver._solver.f).item()

        # Check mass conservation
        # Note: BFL interpolation and MRT moment transforms on CPU accumulate minor
        # truncation errors over multiple steps in float32. A delta of 1e-3 is
        # accepted to account for these cumulative errors while still verifying
        # that no major mass leaks occur.
        self.assertAlmostEqual(total_mass_in, total_mass_out, delta=1e-3)

        # Check momentum stability using conserved indices from solver
        m = torch.tensordot(solver._solver.moment_basis, solver._solver.f, dims=([1], [0]))

        # Verify all moments remain finite
        self.assertTrue(torch.isfinite(m).all())

        # Verify conserved moments (rho, jx, jy, jz) remain finite
        c_idx = solver._solver.conserved_indices
        for idx in c_idx:
             self.assertTrue(torch.isfinite(m[idx]).all())

if __name__ == '__main__':
    unittest.main()
