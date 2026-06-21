import os
import sys
from dataclasses import dataclass

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from sequential_diagnostic_optimizer import (  # noqa: E402
    SequentialDiagnosticOptimizationConfig,
    SequentialDiagnosticOptimizer,
)


@dataclass
class _Spec:
    space_weight: float = 0.2
    drag_weight: float = 1.0
    lift_weight: float = 0.5


class _FakeCFD:
    def __init__(self):
        self.calls = []

    def simulate_aerodynamics(self, geometry, steps=100):
        occupied = float(geometry.sum().item())
        self.calls.append({"occupied": occupied, "steps": steps})
        return {
            "training_drag_coefficient": 1.0 / max(occupied, 1.0),
            "lift_coefficient": min(1.0, occupied / 64.0),
        }


def _two_component_grid(size=8):
    grid = torch.zeros((size, size, size), dtype=torch.float32)
    grid[1:3, 1:3, 1:3] = 0.95
    grid[-3:-1, -3:-1, -3:-1] = 0.95
    return grid


def test_evaluator_uses_measured_connectivity_and_cfd_terms():
    cfd = _FakeCFD()
    config = SequentialDiagnosticOptimizationConfig(
        enable_aerodynamic=True,
        cfd_steps=7,
        connectivity_weight=10.0,
        aerodynamic_weight=1.0,
        validity_weight=1.0,
    )
    optimizer = SequentialDiagnosticOptimizer(cfd, config=config)

    result = optimizer.evaluate(_two_component_grid(), _Spec())

    assert cfd.calls == [{"occupied": 16.0, "steps": 7}]
    assert result["metrics"]["connected_fraction"] == 0.5
    assert result["terms"]["connectivity_loss"] == 0.5
    assert result["terms"]["aerodynamic_loss"] > 0.0
    assert result["total_loss"] > 0.0


def test_genetic_optimizer_evaluates_candidates_sequentially_and_keeps_best():
    cfd = _FakeCFD()
    config = SequentialDiagnosticOptimizationConfig(
        method="genetic",
        population_size=3,
        generations=1,
        mutation_rate=0.2,
        mutation_sigma=0.3,
        enable_aerodynamic=True,
        cfd_steps=5,
        seed=123,
    )
    optimizer = SequentialDiagnosticOptimizer(cfd, config=config)

    result = optimizer.optimize(_two_component_grid(), _Spec())

    assert result["method"] == "genetic"
    assert result["candidates_evaluated"] == 6
    assert len(cfd.calls) == result["candidates_evaluated"]
    assert [call["steps"] for call in cfd.calls] == [5] * result["candidates_evaluated"]
    assert len(result["history"]) == 2
    assert result["best"]["total_loss"] <= result["initial"]["total_loss"]
    assert result["voxel_grid"].shape == (8, 8, 8)
    assert result["binary_grid"].shape == (8, 8, 8)


def test_spsa_optimizer_records_two_point_measurements():
    config = SequentialDiagnosticOptimizationConfig(
        method="spsa",
        spsa_steps=2,
        enable_aerodynamic=False,
        seed=321,
    )
    optimizer = SequentialDiagnosticOptimizer(None, config=config)

    result = optimizer.optimize(_two_component_grid(), _Spec())

    assert result["method"] == "spsa"
    assert result["candidates_evaluated"] == 7
    assert len(result["history"]) == 3
    assert "spsa_gradient_scale" in result["history"][1]
    assert result["best"]["total_loss"] <= result["initial"]["total_loss"]


def test_topk_binarization_scores_sparse_probability_grid():
    cfd = _FakeCFD()
    config = SequentialDiagnosticOptimizationConfig(
        enable_aerodynamic=True,
        threshold=0.5,
        binarization_target_occupancy=0.125,
        cfd_steps=3,
    )
    optimizer = SequentialDiagnosticOptimizer(cfd, config=config)
    grid = torch.linspace(0.01, 0.49, 8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)

    result = optimizer.evaluate(grid, _Spec())

    assert result["metrics"]["occupancy_ratio"] == 0.125
    assert cfd.calls == [{"occupied": 64.0, "steps": 3}]
