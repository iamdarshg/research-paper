import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from advanced_lbm_solver import D3Q27CascadedSolver as AdvancedD3Q27CascadedSolver
from cascaded_lbm import D3Q27CascadedSolver as LegacyD3Q27CascadedSolver
from lbm_utils import D3Q27_LATTICE_SOUND_SPEED, mach_to_lattice_velocity


class TestPhysicsConfig:
    max_mach = 1.0
    target_lattice_velocity = 1.0
    tau_min_d3q27 = 0.52
    s_e_d3q27 = 1.2
    s_h_d3q27 = 1.6
    drag_link_metric_exponent = None
    use_triton_streaming = False
    convergence_tolerance = 1e-5
    check_convergence_every = 10
    smagorinsky_constant = 0.17
    q_threshold = 0.0
    use_shape_drag_correction = False


def make_config(mach_number: float):
    lbm_config = SimpleNamespace(grid_spacing=0.125, time_step=0.001, physical_length_scale=1.0)
    return SimpleNamespace(
        base_grid_resolution=8,
        resolution=8,
        mach_number=mach_number,
        reynolds_number=100.0,
        simulation_steps=1,
        lbm_config=lbm_config,
    )


class TestLBMMachMapping(unittest.TestCase):
    def test_helper_maps_mach_to_d3q27_lattice_sound_speed(self):
        self.assertAlmostEqual(mach_to_lattice_velocity(0.3), 0.3 / math.sqrt(3.0), places=12)
        self.assertAlmostEqual(D3Q27_LATTICE_SOUND_SPEED, 1.0 / math.sqrt(3.0), places=12)

    def test_advanced_solver_uses_mach_over_sqrt_three_when_not_clipped(self):
        cfg = make_config(0.12)
        solver = AdvancedD3Q27CascadedSolver(cfg, torch.device("cpu"), TestPhysicsConfig)
        self.assertAlmostEqual(solver.inlet_velocity_lu, 0.12 / math.sqrt(3.0), places=7)

    def test_legacy_solver_uses_same_lattice_mapping(self):
        cfg = make_config(0.12)
        solver = LegacyD3Q27CascadedSolver(cfg, torch.device("cpu"), TestPhysicsConfig)
        self.assertAlmostEqual(solver.inlet_velocity_lu, 0.12 / math.sqrt(3.0), places=7)


if __name__ == "__main__":
    unittest.main()
