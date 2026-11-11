"""Reproduce the evaluate-baselines device-mismatch crash (Agent D finding F5).

The canonical smoke protocol (CLI/run_protocols/smoke_8gb.yaml) trains fine
but crashes in the `evaluate-baselines` step at
advanced_lbm_solver.py:2069 (compute_aerodynamic_coefficients raw_scalars
torch.stack) with a CUDA/CPU device mismatch. This script reproduces the exact
call sequence to identify the off-device tensor.
"""
import os
import sys

import numpy as np
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from advanced_lbm_solver import D3Q27CascadedSolver  # noqa: E402
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_voxel_grid(shape=(32, 32, 32)):
    """A small airplane-like blob near the center of the lattice."""
    grid = torch.zeros(shape, dtype=torch.float32)
    c = shape[0] // 2
    grid[c - 3:c + 3, c - 3:c + 3, c - 3:c + 3] = 1.0
    return grid


def main():
    cfg = CFDConfig(base_grid_resolution=32, mach_number=0.025,
                    reynolds_number=1e5, simulation_steps=50)
    solver = D3Q27CascadedSolver(cfg, device, LBMPhysicsConfig)
    grid = build_voxel_grid().to(device)
    solver.collide_stream(grid, steps=50)
    coeffs = solver.compute_aerodynamic_coefficients(grid)
    print("coefficients:", {k: coeffs[k] for k in
                            ("drag_coefficient", "lift_coefficient",
                             "reference_area", "lbm_converged")})


if __name__ == "__main__":
    main()