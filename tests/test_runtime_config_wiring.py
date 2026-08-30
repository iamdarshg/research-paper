import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

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
        enable_consistency=False,
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
    assert training.enable_consistency is False
    assert training.coordinate_training_samples == 65536
    assert training.direct_solver_interval == 32
    assert training.full_lattice_interval == 64
    assert training.sparse_samples_per_full == 262144
    assert training.direct_solver_directions == 8
    assert training.require_direct_solver_every_iteration is False
    assert training.enable_pipeline_parallelism is bool(
        config_value("training", "enable_pipeline_parallelism", False)
    )


def test_disabled_consistency_schedule_never_runs_progressive_distillation():
    training = adc.TrainingConfig(enable_consistency=False, consistency_interval=1)

    assert not adc.should_run_consistency_update(training, batch_idx=0)
    assert not adc.should_run_consistency_update(training, batch_idx=1)


def test_enabled_consistency_schedule_obeys_interval():
    training = adc.TrainingConfig(enable_consistency=True, consistency_interval=3)

    assert adc.should_run_consistency_update(training, batch_idx=0)
    assert not adc.should_run_consistency_update(training, batch_idx=1)
    assert adc.should_run_consistency_update(training, batch_idx=3)


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
    assert cfd.lbm_config.grid_spacing == 1.0 / 128


def test_monitored_model_config_uses_explicit_compile_and_checkpoint_flags():
    args = SimpleNamespace(
        enable_compile=True,
        enable_gradient_checkpointing=True,
    )
    model = adc.ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        use_torch_compile=False,
        enable_gradient_checkpointing=False,
    )

    assert hasattr(monitored_training, "_apply_monitored_model_runtime_config")
    configured = monitored_training._apply_monitored_model_runtime_config(model, args)

    assert configured is model
    assert configured.use_torch_compile is True
    assert configured.enable_gradient_checkpointing is True


def test_gpu_exact_runtime_skips_unused_host_edt_workspace(monkeypatch):
    prepared = []
    monkeypatch.setattr(monitored_training, "gpu_exact_available", lambda _device: True)
    monkeypatch.setattr(
        monitored_training,
        "prepare_edt_workspace",
        lambda shape: prepared.append(shape),
    )

    reserved = monitored_training._prepare_host_edt_workspace_for_runtime(
        torch.device("cuda"), 128
    )

    assert reserved is False
    assert prepared == []


def test_monitored_runtime_requires_valid_requested_gpu_edt_attestation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        monitored_training,
        "approve_gpu_exact_attestation",
        lambda path, device: calls.append((path, device)) or True,
    )

    assert monitored_training._approve_gpu_exact_runtime(
        "attestation.json", torch.device("cuda")
    ) is True
    assert calls == [("attestation.json", torch.device("cuda"))]

    monkeypatch.setattr(
        monitored_training,
        "approve_gpu_exact_attestation",
        lambda _path, _device: False,
    )
    with pytest.raises(RuntimeError, match="attestation"):
        monitored_training._approve_gpu_exact_runtime(
            "stale.json", torch.device("cuda")
        )


def test_monitored_runtime_disables_tensorboard_by_default(monkeypatch):
    monkeypatch.delenv("RESEARCH_DISABLE_TENSORBOARD", raising=False)

    monitored_training._configure_tensorboard_runtime(False)

    assert os.environ["RESEARCH_DISABLE_TENSORBOARD"] == "1"


def test_monitored_runtime_can_explicitly_enable_tensorboard(monkeypatch):
    monkeypatch.setenv("RESEARCH_DISABLE_TENSORBOARD", "1")

    monitored_training._configure_tensorboard_runtime(True)

    assert "RESEARCH_DISABLE_TENSORBOARD" not in os.environ


def test_training_input_identity_binds_manifest_and_sidecar_bytes(tmp_path):
    geometry_path = tmp_path / "geometry.npy"
    latent_path = tmp_path / "latents.npy"
    np.save(geometry_path, np.zeros((2, 2, 2), dtype=np.uint8), allow_pickle=False)
    np.save(latent_path, np.zeros((1, 4), dtype=np.float32), allow_pickle=False)
    geometry_sha256 = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "geometry_path": geometry_path.name,
                "voxel_sha256": geometry_sha256,
                "latent_path": latent_path.name,
                "latent_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    initial = monitored_training._training_inputs_identity(str(manifest_path))
    np.save(latent_path, np.ones((1, 4), dtype=np.float32), allow_pickle=False)
    changed_latent = monitored_training._training_inputs_identity(str(manifest_path))
    np.save(latent_path, np.zeros((1, 4), dtype=np.float32), allow_pickle=False)
    changed_geometry = np.ones((2, 2, 2), dtype=np.uint8)
    np.save(geometry_path, changed_geometry, allow_pickle=False)

    assert changed_latent != initial
    with pytest.raises(ValueError, match="geometry_path"):
        monitored_training._training_inputs_identity(str(manifest_path))


def test_cpu_runtime_keeps_reference_edt_workspace_prewarm(monkeypatch):
    prepared = []
    monkeypatch.setattr(
        monitored_training,
        "prepare_edt_workspace",
        lambda shape: prepared.append(shape),
    )

    reserved = monitored_training._prepare_host_edt_workspace_for_runtime(
        torch.device("cpu"), 128
    )

    assert reserved is True
    assert prepared == [(128, 128, 128)]


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
