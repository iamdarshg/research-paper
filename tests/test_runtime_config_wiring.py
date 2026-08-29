from types import SimpleNamespace

import CLI.aircraft_diffusion_cfd as adc
from CLI.aircraft_diffusion_cfd import config_value
import CLI.run_monitored_training as monitored_training
from CLI.run_monitored_training import _build_monitored_training_config


def _command_default(command, name):
    return next(param.default for param in command.params if param.name == name)


def test_core_dataclasses_follow_checked_in_config():
    diffusion = adc.DiffusionConfig()
    training = adc.TrainingConfig()

    assert diffusion.teacher_steps == int(config_value("diffusion", "timesteps", 1000))
    assert diffusion.student_steps == int(config_value("diffusion", "student_steps", 4))
    assert training.precision == str(config_value("training", "precision", "float32"))
    assert training.disconnection_penalty == float(
        config_value("training", "disconnection_penalty", 30.0)
    )
    assert training.val_interval == int(config_value("training", "val_interval", 2))
    assert training.enable_pipeline_parallelism is bool(
        config_value("training", "enable_pipeline_parallelism", False)
    )
    assert training.num_pipeline_stages == int(
        config_value("training", "num_pipeline_stages", 8)
    )


def test_monitored_training_config_does_not_override_sparse_runtime_schedule():
    args = SimpleNamespace(
        num_epochs=1,
        batch_size=1,
        learning_rate=2e-5,
        precision="bfloat16",
        coordinate_training_samples=65536,
        full_lattice_interval=64,
        sparse_samples_per_full=262144,
        direct_solver_loss_weight=1.0,
        direct_solver_interval=32,
        direct_solver_steps=1,
        direct_solver_directions=8,
        direct_solver_batch_chunk=4,
        direct_solver_perturbation=0.15,
        direct_solver_perturbation_grid_size=4,
        direct_connectivity_weight=1.0,
        direct_aircraft_validity_weight=1.0,
        promotion_evaluation_samples=2,
        promotion_generation_seeds=1,
        require_direct_solver_every_iteration=False,
    )

    training = _build_monitored_training_config(args)

    assert training.precision == "bfloat16"
    assert training.coordinate_training_samples == 65536
    assert training.direct_solver_interval == 32
    assert training.full_lattice_interval == 64
    assert training.sparse_samples_per_full == 262144
    assert training.direct_solver_directions == 8
    assert training.require_direct_solver_every_iteration is False
    assert training.enable_pipeline_parallelism is bool(
        config_value("training", "enable_pipeline_parallelism", False)
    )


def test_monitored_cfd_config_uses_explicit_backend_and_stream_block_size():
    args = SimpleNamespace(
        solver="D3Q27",
        lbm_stream_bfl_backend="fused_stream_bfl",
        stream_block_size=512,
    )

    assert hasattr(monitored_training, "_build_monitored_cfd_config")
    cfd = monitored_training._build_monitored_cfd_config(args, 128)

    assert cfd.solver_type == "D3Q27"
    assert cfd.use_fused_stream_bfl is True
    assert cfd.lbm_config.stream_block_size == 512


def test_train_command_defaults_follow_config():
    from CLI.aircraft_diffusion_cfd import train

    expected = {
        "precision": str(config_value("training", "precision", "float32")),
        "disconnection_penalty": float(
            config_value("training", "disconnection_penalty", 30.0)
        ),
        "enable_pipeline": bool(
            config_value("training", "enable_pipeline_parallelism", False)
        ),
        "enable_checkpointing": bool(
            config_value("model", "enable_gradient_checkpointing", True)
        ),
        "solver": str(config_value("cfd", "solver", "D3Q27")),
        "direct_solver_gradient_clip": float(
            config_value("training", "direct_solver_gradient_clip", 1.0)
        ),
        "overfit_geometry_gate_samples": int(
            config_value("training", "overfit_geometry_gate_samples", 8)
        ),
        "overfit_min_generated_aircraft_valid_fraction": float(
            config_value("training", "overfit_min_generated_aircraft_valid_fraction", 0.125)
        ),
    }
    for name, value in expected.items():
        assert _command_default(train, name) == value
