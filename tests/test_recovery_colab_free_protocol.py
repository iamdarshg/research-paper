"""Reference-protocol schema checks for the issue-#39 free-cloud recovery probe."""

from pathlib import Path

import pytest
import yaml

PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "CLI" / "run_protocols" / "recovery_colab_free.yaml"


@pytest.fixture(scope="module")
def protocol() -> dict:
    assert PROTOCOL_PATH.exists(), f"missing {PROTOCOL_PATH}"
    with open(PROTOCOL_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict) and "protocol" in cfg
    return cfg["protocol"]


def test_protocol_fixed_settings(protocol: dict) -> None:
    assert protocol["grid_size"] == 96
    assert protocol["batch_size"] == 1
    assert protocol["solver"] == "D3Q27"
    assert protocol["direct_solver_steps"] == 50
    assert protocol["checkpoint_every_updates"] == 1


def test_protocol_free_session_budget(protocol: dict) -> None:
    # 40 cumulative updates across 8 conservative 5-update free sessions.
    assert protocol["total_updates"] == 40
    assert protocol["updates_per_session"] == 5
    assert 40 % protocol["updates_per_session"] == 0


def test_protocol_spsa_directions_justified(protocol: dict) -> None:
    # 16 directions (CLI default, not config.yaml's 32): fewer solver calls per
    # update so a bounded update fits inside a free-session wall-clock budget.
    assert protocol["direct_solver_directions"] == 16


def test_protocol_fixed_seeds(protocol: dict) -> None:
    assert protocol["fixed_validation_seeds"] == [0, 1, 2, 3, 4, 5]
