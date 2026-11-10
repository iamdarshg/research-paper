# ---------------------------------------------------------------------------
# F7: AdvancedCFDSimulator.simulate_aerodynamics (aircraft_diffusion_cfd.py)
# feeds the caller's geometry straight into collide_stream / BFL without
# resizing to the solver resolution. The smoke protocol's validate-conditions
# step builds the simulator at grid_size=32 but generates voxels at the
# checkpoint's 16^3, raising "The shape of the mask [16,16,16] does not
# match the shape of the indexed tensor [32,32,32]".
# ---------------------------------------------------------------------------
import os
import sys

import pytest
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

class TestSimulatorGeometryResolutionContract:
    def test_simulator_resizes_smaller_geometry_to_solver_resolution(self):
        from aircraft_diffusion_cfd import AdvancedCFDSimulator, CFDConfig

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        simulator = AdvancedCFDSimulator(
            CFDConfig(base_grid_resolution=32, simulation_steps=20),
            torch.device("cuda"),
        )
        # 16^3 geometry into a 32^3 solver (the validate-conditions path).
        grid = torch.zeros(16, 16, 16)
        grid[7:9, 7:9, 7:9] = 1.0
        results = simulator.simulate_aerodynamics(grid, steps=20)
        assert "drag_coefficient" in results
        assert torch.isfinite(
            torch.tensor(results["drag_coefficient"])
        ).item()

    def test_simulator_rejects_non_cubic_geometry_with_clear_error(self):
        from aircraft_diffusion_cfd import AdvancedCFDSimulator, CFDConfig

        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        simulator = AdvancedCFDSimulator(
            CFDConfig(base_grid_resolution=32, simulation_steps=10),
            torch.device("cuda"),
        )
        grid = torch.zeros(16, 16, 17)  # non-cubic
        with pytest.raises(ValueError, match="[Cc]ubic"):
            simulator.simulate_aerodynamics(grid, steps=10)