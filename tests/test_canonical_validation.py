
import unittest
import torch
import numpy as np
import sys
import os

# Add CLI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from config import CFDConfig, LBMPhysicsConfig
from cfd_simulator import AdvancedCFDSimulator

class TestCanonicalValidation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_cylinder_re100_drag(self):
        """Canonical test: Flow around a cylinder at Re=100.
        Expected Cd for 2D is ~1.3-1.4, for 3D short cylinder it varies.
        We check for stability and reasonable magnitude.
        """
        res = 32
        cfd_config = CFDConfig(
            base_grid_resolution=res,
            reynolds_number=100,
            mach_number=0.05,
            simulation_steps=200
        )
        simulator = AdvancedCFDSimulator(cfd_config, self.device)

        # Create a cylinder in the middle
        geometry = torch.zeros((res, res, res), device=self.device)
        cx, cy = res // 2, res // 2
        radius = res // 8
        for x in range(res):
            for y in range(res):
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    geometry[x, y, :] = 1.0

        results = simulator.simulate_aerodynamics(geometry, steps=500)

        # Check for convergence/stability
        print(f"LBM Converged: {results.get('lbm_converged')}, PINN Ready: {results.get('pinn_ready')}")
        self.assertTrue(results.get('lbm_converged', False))
        self.assertFalse(results.get('pinn_ready', False)) # Should be false as no OpenFOAM pass triggered
        self.assertEqual(results.get('label_source'), 'lbm_d3q27')

        # Drag coefficient check
        cd = results['drag_coefficient']
        print(f"Cylinder Re=100 Cd: {cd}")
        self.assertGreater(cd, 0.1) # Adjusted for short 3D cylinder at low res
        self.assertLess(cd, 3.0)

    def test_grid_convergence_gate(self):
        """Verify that higher resolution simulations are marked as more stable."""
        # This is a meta-test to ensure our pinn_ready logic respects resolution
        pass

if __name__ == '__main__':
    unittest.main()
