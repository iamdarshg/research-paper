import math
import os
import sys
import unittest
from types import SimpleNamespace

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from advanced_lbm_solver import D3Q27CascadedSolver as AdvancedD3Q27CascadedSolver
from cascaded_lbm import D3Q27CascadedSolver as LegacyD3Q27CascadedSolver
from config import CFDConfig, LBMPhysicsConfig
from lbm_utils import (
    D3Q27_LATTICE_SOUND_SPEED,
    d3q27_lattice_freestream_velocity_from_mach,
    mach_to_lattice_velocity,
)


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


def make_legacy_config(mach_number: float):
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
    def test_helpers_map_mach_to_d3q27_lattice_sound_speed(self):
        mach = 0.3
        expected = mach / math.sqrt(3.0)
        self.assertAlmostEqual(mach_to_lattice_velocity(mach), expected, places=12)
        self.assertAlmostEqual(d3q27_lattice_freestream_velocity_from_mach(mach), expected, places=12)
        self.assertAlmostEqual(D3Q27_LATTICE_SOUND_SPEED, 1.0 / math.sqrt(3.0), places=12)

    def test_advanced_solver_uses_same_lattice_mach_mapping(self):
        config = CFDConfig(base_grid_resolution=16, mach_number=0.05, reynolds_number=100)
        config.lbm_config = LBMPhysicsConfig()
        config.lbm_config.grid_spacing = 1.0 / 16.0
        solver = AdvancedD3Q27CascadedSolver(config, torch.device("cpu"), LBMPhysicsConfig)
        expected = d3q27_lattice_freestream_velocity_from_mach(config.mach_number)
        self.assertAlmostEqual(solver.inlet_velocity_lu, expected, places=12)

    def test_legacy_solver_uses_same_lattice_mapping(self):
        config = make_legacy_config(0.12)
        solver = LegacyD3Q27CascadedSolver(config, torch.device("cpu"), TestPhysicsConfig)
        expected = d3q27_lattice_freestream_velocity_from_mach(config.mach_number)
        self.assertAlmostEqual(solver.inlet_velocity_lu, expected, places=7)


if __name__ == "__main__":
    unittest.main()
