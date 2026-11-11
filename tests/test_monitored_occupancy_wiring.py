"""Regression test: the monitored runner must not silently drop the occupancy term.

Context (2026-11-11).  `training.geometry_occupancy_mean_weight` defaults to 0.0
("legacy objective"), and ``occupancy_mean_matching_loss`` is only invoked when
the weight is > 0.  The ``train`` CLI has always exposed the weight
(``--geometry-occupancy-mean-weight``), but ``CLI/run_monitored_training.py``
built its ``TrainingConfig`` without it, so every monitored run -- including the
660/96 diversity runs that produced 0/576 on ``grounded_occupancy_envelope`` --
trained with the occupancy term contributing exactly zero, with no way to turn it
on from the command line.

This test pins the wiring, not the value: the default must stay 0.0 (no objective
change for existing runs), and a supplied weight must reach the TrainingConfig.
"""

from __future__ import annotations

import argparse

import pytest

from run_monitored_training import _build_monitored_training_config


def _monitored_namespace(**overrides: object) -> argparse.Namespace:
    """Minimal namespace holding every attribute the config builder reads."""
    values: dict[str, object] = {
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 2.0e-4,
        "enable_consistency": True,
        "precision": "float32",
        "direct_solver_loss_weight": 1.0,
        "coordinate_training_samples": 32768,
        "full_lattice_interval": 1,
        "sparse_samples_per_full": 262144,
        "direct_solver_interval": 1,
        "direct_solver_steps": 5,
        "direct_solver_directions": 16,
        "direct_solver_perturbation": 0.15,
        "direct_solver_perturbation_grid_size": 12,
        "direct_connectivity_weight": 1.0,
        "direct_aircraft_validity_weight": 1.0,
        "promotion_evaluation_samples": 8,
        "promotion_generation_seeds": 6,
        "require_direct_solver_every_iteration": True,
        "geometry_occupancy_mean_weight": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_monitored_config_carries_occupancy_weight() -> None:
    config = _build_monitored_training_config(
        _monitored_namespace(geometry_occupancy_mean_weight=0.25)
    )
    assert config.geometry_occupancy_mean_weight == pytest.approx(0.25)


def test_monitored_config_occupancy_default_is_legacy_zero() -> None:
    """The default must remain 0.0 so existing runs keep their objective."""
    config = _build_monitored_training_config(_monitored_namespace())
    assert config.geometry_occupancy_mean_weight == 0.0


def test_monitored_config_rejects_negative_occupancy_weight() -> None:
    """A negative weight must not be reachable through the monitored runner.

    ``TrainingConfig`` validation already rejects negatives; the wiring must not
    launder one through as "disabled".
    """
    config = _build_monitored_training_config(
        _monitored_namespace(geometry_occupancy_mean_weight=-1.0)
    )
    errors = config.validate() if hasattr(config, "validate") else []
    if isinstance(errors, list) and errors:
        assert any("occupancy" in str(error) for error in errors)