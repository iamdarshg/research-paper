
import unittest
import torch
import sys
import os
import numpy as np

# Add GUI to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GUI'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'CLI'))

from cfd_solver_integration import CFDSolverWorker
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig

class TestGUIWorker(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.cfd_domain_params = {
            'domain_size': [1.0, 1.0, 1.0],
            'body_size': 1.0,
            'offset': [0, 0, 0]
        }

    def test_worker_init(self):
        worker = CFDSolverWorker(
            "dummy.stl", 10000, 0.05, 100,
            solver_type="d3q27_cascaded",
            grid_resolution=8,
            cfd_domain_params=self.cfd_domain_params
        )
        self.assertEqual(worker.stl_path, "dummy.stl")
        self.assertEqual(worker.grid_resolution, 8)
        self.assertFalse(worker._is_interrupted)

    def test_interruption(self):
        worker = CFDSolverWorker(
            "dummy.stl", 10000, 0.05, 100,
            solver_type="d3q27_cascaded",
            grid_resolution=8,
            cfd_domain_params=self.cfd_domain_params
        )
        self.assertFalse(worker._is_interrupted)
        worker.requestInterruption()
        self.assertTrue(worker._is_interrupted)

if __name__ == '__main__':
    unittest.main()
