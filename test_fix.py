#!/usr/bin/env python3
"""
Test script to verify the tensor dimension alignment fix.
"""

import os
import sys

import torch

sys.path.append('.')

from aircraft_diffusion_cfd import AdvancedCFDSimulator, CFDConfig, LatentTo3DConverter, ModelConfig


def test_tensor_dimension_alignment():
    """Test that model and CFD solver resolutions are aligned."""
    print("Testing tensor dimension alignment fix")
    print("=" * 50)

    test_cases = [
        {"solver": "D3Q27", "expected_res": 16},
    ]

    for case in test_cases:
        print(f"\nTesting {case['solver']} solver...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        cfd_config = CFDConfig(
            solver_type=case['solver'],
            base_grid_resolution=case['expected_res'],
        )
        print(f"CFD config resolution: {cfd_config.base_grid_resolution}")

        cfd_simulator = AdvancedCFDSimulator(cfd_config, device)
        target_resolution = cfd_simulator.resolution
        print(f"CFD simulator resolution: {target_resolution}")

        model_config = ModelConfig(
            latent_dim=16,
            grid_resolution=target_resolution,
        )
        print(f"Model config grid resolution: {model_config.grid_resolution}")

        converter = LatentTo3DConverter(
            latent_dim=model_config.latent_dim,
            grid_resolution=model_config.grid_resolution,
        )

        test_latent = torch.randn(1, model_config.latent_dim)
        voxel_grid = converter(test_latent)
        print(f"Input latent shape: {test_latent.shape}")
        print(f"Output voxel grid shape: {voxel_grid.shape}")

        expected_shape = (target_resolution, target_resolution, target_resolution)
        assert voxel_grid.shape[1:] == expected_shape, "Dimension mismatch still exists"

        geometry_mask = (voxel_grid[0] > 0.5).float()
        print(f"Geometry mask shape: {geometry_mask.shape}")
        assert geometry_mask.shape == expected_shape, "Geometry mask shape is incorrect"

    print("\nTensor dimension alignment fix is working correctly.")


def test_cfd_solver_compatibility():
    """Test that the CFD solver can handle the aligned tensor dimensions."""
    print("\nTesting CFD solver compatibility...")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfd_config = CFDConfig(solver_type="D3Q27", base_grid_resolution=16)
    cfd_simulator = AdvancedCFDSimulator(cfd_config, device)

    test_geometry = torch.ones(16, 16, 16) * 0.8
    test_geometry[8, 8, 8] = 0.2

    print(f"Test geometry shape: {test_geometry.shape}")
    print(
        "CFD solver expects: "
        f"({cfd_simulator.resolution}, {cfd_simulator.resolution}, {cfd_simulator.resolution})"
    )

    try:
        print("Running CFD simulation...")
        results = cfd_simulator.simulate_aerodynamics(test_geometry, steps=10)
    except Exception as exc:
        raise AssertionError(f"CFD simulation failed: {exc}") from exc

    print("CFD simulation completed successfully")
    print(f"Results: {results}")


if __name__ == "__main__":
    print("Starting tensor dimension alignment tests")
    print("=" * 60)
    test_tensor_dimension_alignment()
    test_cfd_solver_compatibility()
    print("\nAll tests passed.")
