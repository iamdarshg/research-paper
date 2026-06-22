import os
import sys

import numpy as np
import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import build_nasa_crm_whole_aircraft_context as builder


def test_run_local_analysis_uses_requested_grid_and_steps(monkeypatch, tmp_path):
    calls = {}

    class DummyDataset:
        def __init__(self, num_samples, grid_size):
            calls["dataset_grid_size"] = grid_size

        def _voxelize_stl(self, stl_path, grid_size):
            calls["voxelize_grid_size"] = grid_size
            return torch.ones((grid_size, grid_size, grid_size), dtype=torch.float32)

    class DummySimulator:
        def __init__(self, config, device):
            calls["solver_grid_size"] = config.base_grid_resolution
            calls["solver_device"] = str(device)

        def simulate_aerodynamics(self, voxels, steps):
            calls["solver_steps"] = steps
            return {"drag_coefficient": 1.25, "ignored": "not numeric"}

    monkeypatch.setattr(builder, "AircraftDesignDataset", DummyDataset)
    monkeypatch.setattr(builder, "AdvancedCFDSimulator", DummySimulator)
    monkeypatch.setattr(
        builder,
        "evaluate_aircraft_validity",
        lambda voxels: {"status": "pass", "metrics": {"occupancy_ratio": 1.0}},
    )

    voxel_path = tmp_path / "voxels" / "sample.npy"
    analysis = builder.run_local_analysis(
        tmp_path / "sample.stl",
        voxel_path,
        grid_size=12,
        simulation_steps=7,
        analysis_device="cpu",
    )

    assert calls == {
        "dataset_grid_size": 12,
        "voxelize_grid_size": 12,
        "solver_grid_size": 12,
        "solver_device": "cpu",
        "solver_steps": 7,
    }
    assert analysis["cfd"] == {"drag_coefficient": 1.25}
    assert np.load(voxel_path).shape == (12, 12, 12)
