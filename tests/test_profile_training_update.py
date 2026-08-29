import pytest

from CLI.profile_training_update import _profile_grid_size


class _Config:
    def __init__(self, grid_resolution):
        self.grid_resolution = grid_resolution


class _Simulator:
    def __init__(self, resolution):
        self.resolution = resolution


class _Trainer:
    def __init__(self, model_size, solver_size):
        self.model_config = _Config(model_size)
        self.cfd_simulator = _Simulator(solver_size)


def test_profile_grid_size_uses_agreeing_model_and_solver_resolution():
    assert _profile_grid_size(_Trainer(128, 128)) == 128


def test_profile_grid_size_rejects_model_solver_mismatch_before_cfd():
    with pytest.raises(RuntimeError, match="trainer model/solver grid mismatch"):
        _profile_grid_size(_Trainer(96, 128))


def test_profile_grid_size_requires_a_trainer_resolution():
    with pytest.raises(RuntimeError, match="cannot determine profiler grid size"):
        _profile_grid_size(type("EmptyTrainer", (), {})())
