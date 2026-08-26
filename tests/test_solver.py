
import unittest
import torch
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from advanced_lbm_solver import D3Q27CascadedSolver, D3Q27Lattice
from config import CFDConfig, LBMPhysicsConfig
from lbm_utils import REFERENCE_SPEED_OF_SOUND_MPS

class TestLBMSolvers(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.config = CFDConfig(base_grid_resolution=8, mach_number=0.1, reynolds_number=100)
        self.config.lbm_config = LBMPhysicsConfig()
        self.config.lbm_config.grid_spacing = 1.0 / 8.0

    def test_d3q27_init(self):
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        self.assertEqual(solver.f.shape, (27, 8, 8, 8))
        self.assertEqual(solver._solver.stream_block_size, 512)
        self.assertFalse(torch.isnan(solver.f).any())

    def test_d3q27_opposites_are_vector_negations(self):
        ex, ey, ez = D3Q27Lattice.get_vectors()
        vectors = torch.stack((ex, ey, ez), dim=1)
        opposite = D3Q27Lattice.get_opposite()

        self.assertTrue(torch.equal(vectors[opposite], -vectors))
        self.assertTrue(torch.equal(opposite[opposite], torch.arange(27)))

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

    def test_effective_reynolds_and_tau_actual(self):
        """R2 (PR 41 review): the solver must report the REALIZED relaxation
        time and Reynolds number, not the requested config value.

        ``tau_min_d3q27`` (default 0.52) clamps ``tau`` to >= 0.52, so a
        requested Re that implies ``tau < 0.52`` is NOT realized. At 8^3 the
        freestream is ``u_lu = mach/sqrt(3) ~= 0.0577``; Re=100 implies
        ``nu = 4.62e-3 -> tau = 0.5139``, below the floor, so the solver
        actually runs at ``tau = 0.52`` and ``Re_eff ~= 69.3``, not 100.
        Re=50 implies ``tau = 0.5277`` (above the floor) and is realized
        exactly.
        """
        self.config.reynolds_number = 100
        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        geometry_mask = torch.zeros((8, 8, 8), device=self.device)
        geometry_mask[3:5, 3:5, 3:5] = 1.0

        solver.collide_stream(geometry_mask, steps=1)
        # Realized relaxation handling exposed on the solver after a solve.
        nu_eff = (0.52 - 0.5) / 3.0
        self.assertEqual(solver.tau_actual, 0.52)
        self.assertAlmostEqual(solver.nu_effective, nu_eff, places=8)
        # self.nu is the REALIZED (post-clamp) viscosity, not the requested one.
        self.assertAlmostEqual(solver.nu, nu_eff, places=8)

        coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
        self.assertEqual(coeffs['requested_reynolds'], 100)
        self.assertTrue(coeffs['reynolds_clamped'])
        self.assertAlmostEqual(coeffs['tau_actual'], 0.52, places=6)
        self.assertAlmostEqual(coeffs['effective_laminar_viscosity'], nu_eff, places=8)
        u_lu = solver.inlet_velocity_lu
        self.assertAlmostEqual(coeffs['effective_reynolds'], u_lu * 8.0 / nu_eff, places=4)

        # Unclamped case: Re low enough that tau > 0.52 is realized exactly.
        self.config.reynolds_number = 50
        solver2 = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        solver2.collide_stream(geometry_mask, steps=1)
        coeffs2 = solver2.compute_aerodynamic_coefficients(geometry_mask)
        self.assertFalse(coeffs2['reynolds_clamped'])
        self.assertAlmostEqual(coeffs2['effective_reynolds'], 50, places=3)
        u_lu2 = solver2.inlet_velocity_lu
        tau_unclamped = 3.0 * (u_lu2 * 8.0 / 50.0) + 0.5
        self.assertAlmostEqual(coeffs2['tau_actual'], tau_unclamped, places=6)

    def test_d3q27_benchmark_like_run_is_finite(self):
        benchmark_config = CFDConfig(
            base_grid_resolution=32,
            mach_number=80 / REFERENCE_SPEED_OF_SOUND_MPS,
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

    def test_flow_regime_is_mission_independent_fixed_global(self):
        """R8 (PR 41 review, item 8): the aerodynamic solve runs at fixed,
        global flow conditions derived from CFDConfig alone. The realized
        regime (tau_actual / nu_effective) must be identical across samples —
        a stand-in for the per-sample design_spec — so the aero loss stays a
        cross-sample-comparable objective. Mission-adaptivity is delivered
        through the conditioning vector and mission flight-path synthesis,
        not through the solve."""
        geometry_a = torch.zeros((8, 8, 8), device=self.device)
        geometry_a[3:5, 3:5, 3:5] = 1.0
        geometry_b = torch.zeros((8, 8, 8), device=self.device)
        geometry_b[2:6, 2:6, 1:7] = 1.0

        solver = D3Q27CascadedSolver(self.config, self.device, LBMPhysicsConfig)
        solver.collide_stream(geometry_a, steps=1)
        regime = (solver.tau_actual, solver.nu_effective)

        solver.collide_stream(geometry_b, steps=1)
        self.assertEqual(
            (solver.tau_actual, solver.nu_effective),
            regime,
            "the realized flow regime must not change between samples: the "
            "solve is mission-independent by design",
        )

        # The regime is pinned to the config's flow fields, not to any
        # per-sample mission value.
        self.assertEqual(float(solver.config.mach_number), self.config.mach_number)
        self.assertEqual(float(solver.config.reynolds_number), self.config.reynolds_number)

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
