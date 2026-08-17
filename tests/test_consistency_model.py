import os
import random
import sys

import numpy as np
import pytest
import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (
    CFDConfig,
    ConsistencyModel,
    DesignSpec,
    DiffusionConfig,
    ModelConfig,
    OptimizedDiffusionTrainer,
    TrainingConfig,
    balanced_voxel_bce_with_logits,
    bound_latent_to_corpus_support,
    soft_dice_loss_with_logits,
    sparse_voxel_reconstruction_loss,
    apply_configured_optimizer_learning_rates,
    select_training_timesteps,
    restore_resume_learning_rate_if_zero,
    LatentDiffusionUNet,
    LatentTo3DConverter,
    GroupedQueryAttention,
    load_width_expanded_state_dict,
    move_optimizer_state,
    atomic_save_run_state,
    capture_rng_state,
    restore_rng_state,
    validate_run_state_compatibility,
    grounded_threshold_margin_loss,
    validate_solver_integrated_training_config,
)


def test_run_state_round_trip_restores_rng_and_is_atomic(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.tensor([0.5, -0.25])
    optimizer.step()
    state = {
        "model": {"value": parameter.detach().clone()},
        "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(),
        "completed_in_epoch": 2,
    }
    path = tmp_path / "latest_run_state.pt"
    atomic_save_run_state(path, state)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    parameter.data.add_(10.0)
    random.random()
    np.random.random()
    torch.rand(3)

    restored = torch.load(path, map_location="cpu", weights_only=False)
    parameter.data.copy_(restored["model"]["value"])
    restore_rng_state(restored["rng"])

    assert torch.equal(parameter, state["model"]["value"])
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_run_state_compatibility_configuration_uses_intersection_semantics():
    """R4 (PR 41 review, item 4): the ``configuration`` sub-dict is compared
    over the INTERSECTION of keys, so a fingerprint key added in a newer code
    version (here ``experiment_flags``) does not block resuming an older
    run-state that predates it. Keys present on both sides are still compared
    strictly: a tf32 flip must block the resume.
    """
    base = {
        "manifest_identity": "manifest-a",
        "grid_size": 96,
        "latent_dim": 192,
        "split": "train",
        "sample_count": 32,
    }
    # Old run-state: no experiment_flags key. New code adds it. Resume is fine.
    old_state = {
        **base,
        "configuration": {
            "training_config": {"learning_rate": 0.00002},
            "cfd_config": {"solver": "D3Q27"},
        },
    }
    new_expected = {
        **base,
        "configuration": {
            "training_config": {"learning_rate": 0.00002},
            "cfd_config": {"solver": "D3Q27"},
            "experiment_flags": {
                "graph_decode_mlp": False,
                "batch_guard_dot_reads": True,
                "deferred_solver_reads": True,
                "tf32_gemm_math": True,
            },
        },
    }
    assert validate_run_state_compatibility(old_state, new_expected) == []

    # Both sides carry experiment_flags -> strict comparison. tf32 flipped.
    same_code = {
        **base,
        "configuration": {
            "training_config": {"learning_rate": 0.00002},
            "experiment_flags": {
                "graph_decode_mlp": False,
                "batch_guard_dot_reads": True,
                "deferred_solver_reads": True,
                "tf32_gemm_math": False,
            },
        },
    }
    assert validate_run_state_compatibility(same_code, new_expected) == [
        "configuration.experiment_flags"
    ]

    # A shared configuration key with a different value is still caught.
    drifted = {
        **base,
        "configuration": {
            "training_config": {"learning_rate": 0.001},
            "experiment_flags": {
                "graph_decode_mlp": False,
                "batch_guard_dot_reads": True,
                "deferred_solver_reads": True,
                "tf32_gemm_math": True,
            },
        },
    }
    assert validate_run_state_compatibility(drifted, new_expected) == [
        "configuration.training_config",
    ]


def test_run_state_compatibility_reports_all_immutable_mismatches():
    expected = {
        "manifest_identity": "manifest-a",
        "grid_size": 96,
        "latent_dim": 192,
        "split": "train",
        "sample_count": 32,
    }
    actual = {
        "manifest_identity": "manifest-b",
        "grid_size": 32,
        "latent_dim": 64,
        "split": "val",
        "sample_count": 16,
    }

    errors = validate_run_state_compatibility(actual, expected)

    assert errors == [
        "manifest_identity",
        "grid_size",
        "latent_dim",
        "split",
        "sample_count",
    ]


def test_threshold_margin_loss_is_zero_when_both_classes_clear_fixed_threshold():
    probabilities = torch.tensor([0.9, 0.8, 0.1, 0.2], requires_grad=True)
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])

    loss = grounded_threshold_margin_loss(
        probabilities,
        target,
        threshold=0.5,
        positive_margin=0.1,
        negative_margin=0.1,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()


def test_threshold_margin_loss_penalizes_disappearing_and_false_solid_voxels():
    probabilities = torch.tensor([0.51, 0.99], requires_grad=True)
    target = torch.tensor([1.0, 0.0])

    loss = grounded_threshold_margin_loss(
        probabilities,
        target,
        threshold=0.7,
        positive_margin=0.1,
        negative_margin=0.1,
    )

    assert loss.item() > 0.0
    loss.backward()
    assert torch.isfinite(probabilities.grad).all()
    assert probabilities.grad[0] < 0
    assert probabilities.grad[1] > 0


def test_threshold_margin_loss_normalizes_positive_and_negative_regions_independently():
    balanced = grounded_threshold_margin_loss(
        torch.tensor([0.7, 0.7, 0.7, 0.7]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        threshold=0.5,
        positive_margin=0.2,
        negative_margin=0.2,
    )
    sparse = grounded_threshold_margin_loss(
        torch.tensor([0.7, 0.7, 0.7, 0.7]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        threshold=0.5,
        positive_margin=0.2,
        negative_margin=0.2,
    )

    assert balanced.item() == pytest.approx(sparse.item())


def test_threshold_margin_loss_uses_configured_threshold_not_batch_topk():
    loss = grounded_threshold_margin_loss(
        torch.tensor([0.55, 0.54, 0.01, 0.0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        threshold=0.5,
        positive_margin=0.1,
        negative_margin=0.1,
    )

    assert loss.item() > 0.0


def test_threshold_margin_configuration_rejects_invalid_values():
    with pytest.raises(ValueError, match="threshold_positive_margin"):
        validate_solver_integrated_training_config(
            TrainingConfig(threshold_positive_margin=-0.1)
        )
    with pytest.raises(ValueError, match="threshold_positive_margin"):
        validate_solver_integrated_training_config(
            TrainingConfig(
                geometry_materialization_threshold=0.9,
                threshold_positive_margin=0.1,
            )
        )


def test_grouped_query_attention_shares_key_value_heads_and_preserves_shape():
    attention = GroupedQueryAttention(channels=32, num_groups=8, num_kv_groups=4)
    value = torch.randn(2, 32, 2, 2, 2)

    output = attention(value)

    assert output.shape == value.shape
    assert attention.to_k.out_channels == 16
    assert attention.to_v.out_channels == 16


def test_consistency_add_noise_preserves_latent_shape():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(config, DiffusionConfig())
    latent = torch.zeros((2, 4))
    noise = torch.ones_like(latent)
    timesteps = torch.tensor([0, 10])

    noised = model._add_noise(latent, timesteps, noise)

    assert noised.shape == latent.shape


def test_consistency_add_noise_matches_primary_noise_schedule():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    diffusion = DiffusionConfig(timesteps=8)
    model = ConsistencyModel(config, diffusion)
    latent = torch.tensor([[0.0, 0.25, 0.5, 0.75]])
    noise = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
    timesteps = torch.tensor([6])

    consistency_noised = model._add_noise(latent, timesteps, noise)
    primary_noised = model.noise_schedule.q_sample(latent, timesteps, noise)

    assert torch.allclose(consistency_noised, primary_noised)


def test_corpus_support_bound_uses_bounded_forward_and_straight_through_gradient():
    latent = torch.tensor([[-2.0, 0.25, 3.0]], requires_grad=True)

    bounded = bound_latent_to_corpus_support(latent, 0.0, 1.0)
    bounded.sum().backward()

    assert torch.equal(bounded.detach(), torch.tensor([[0.0, 0.25, 1.0]]))
    assert torch.equal(latent.grad, torch.ones_like(latent))


def test_fast_inference_respects_corpus_latent_support():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(config, DiffusionConfig(timesteps=8, student_steps=2))

    generated = model.fast_inference((2, 4), num_steps=2)

    assert torch.isfinite(generated).all()
    assert float(generated.detach().min()) >= 0.0
    assert float(generated.detach().max()) <= 1.0


def test_fast_inference_replays_explicit_initial_noise_with_gradients():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(
        config,
        DiffusionConfig(timesteps=8, student_steps=4),
    )
    model.eval()
    initial_noise = torch.randn((1, 4))

    torch.manual_seed(1)
    first = model.fast_inference(
        (1, 4),
        num_steps=4,
        initial_noise=initial_noise,
    )
    torch.manual_seed(999)
    second = model.fast_inference(
        (1, 4),
        num_steps=4,
        initial_noise=initial_noise,
    )
    second.sum().backward()

    assert torch.equal(first.detach(), second.detach())
    assert any(
        parameter.grad is not None
        and float(parameter.grad.detach().abs().sum()) > 0.0
        for parameter in model.student_model.parameters()
    )


def test_training_timesteps_cycle_over_exact_inference_schedule():
    observed = [
        int(
            select_training_timesteps(
                global_step=step,
                batch_size=1,
                diffusion_timesteps=1000,
                inference_steps=4,
                device=torch.device("cpu"),
                mode="inference_stratified",
            ).item()
        )
        for step in range(8)
    ]

    assert observed == [999, 666, 333, 0, 999, 666, 333, 0]




def test_consistency_rejects_mismatched_teacher_student_timesteps():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(config, DiffusionConfig(timesteps=8))

    try:
        model.consistency_loss(
            torch.zeros((1, 4)),
            torch.tensor([2]),
            torch.tensor([3]),
        )
    except ValueError as exc:
        assert "same diffusion timestep" in str(exc)
    else:
        raise AssertionError("mismatched consistency timesteps must fail closed")


def test_training_config_weights_generation_reconstruction_by_default():
    config = TrainingConfig()

    assert config.generation_reconstruction_weight == 1.0


def test_balanced_voxel_bce_penalizes_sparse_empty_prior():
    target = torch.zeros((1, 1000), dtype=torch.float32)
    target[0, :5] = 1.0
    empty_prior_logits = torch.full_like(target, -5.3)
    perfect_logits = torch.where(target > 0.5, torch.full_like(target, 8.0), torch.full_like(target, -8.0))

    empty_prior_loss = balanced_voxel_bce_with_logits(empty_prior_logits, target)
    perfect_loss = balanced_voxel_bce_with_logits(perfect_logits, target)

    assert float(empty_prior_loss) > 2.0
    assert float(perfect_loss) < 0.001


def test_sparse_reconstruction_loss_penalizes_wrong_shape_overlap():
    target = torch.zeros((1, 64), dtype=torch.float32)
    target[0, 8:16] = 1.0
    perfect_logits = torch.where(
        target > 0.5,
        torch.full_like(target, 8.0),
        torch.full_like(target, -8.0),
    )
    shifted_target = torch.zeros_like(target)
    shifted_target[0, 40:48] = 1.0
    shifted_logits = torch.where(
        shifted_target > 0.5,
        torch.full_like(target, 8.0),
        torch.full_like(target, -8.0),
    )

    perfect_dice = soft_dice_loss_with_logits(perfect_logits, target)
    shifted_dice = soft_dice_loss_with_logits(shifted_logits, target)
    perfect_total = sparse_voxel_reconstruction_loss(
        perfect_logits, target, dice_weight=1.0
    )
    shifted_total = sparse_voxel_reconstruction_loss(
        shifted_logits, target, dice_weight=1.0
    )

    assert float(perfect_dice) < 0.01
    assert float(shifted_dice) > 0.9
    assert float(perfect_total) < float(shifted_total)


def test_population_weighted_sampled_dice_preserves_sparse_lattice_prevalence():
    target = torch.tensor([[1.0, 0.0]])
    logits = torch.tensor([[2.1972246, 0.0]])  # probabilities 0.9 and 0.5

    balanced_sample_loss = sparse_voxel_reconstruction_loss(
        logits,
        target,
        dice_weight=1.0,
    )
    sparse_population_loss = sparse_voxel_reconstruction_loss(
        logits,
        target,
        dice_weight=1.0,
        population_positive_counts=torch.tensor([1.0]),
        population_negative_counts=torch.tensor([999.0]),
    )

    assert float(sparse_population_loss) > float(balanced_sample_loss) + 0.7


def test_population_weighted_sampled_dice_matches_full_loss_when_sample_is_full_population():
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    logits = torch.tensor([[3.0, -2.0, 1.0, -4.0]])

    full_loss = sparse_voxel_reconstruction_loss(logits, target, dice_weight=1.0)
    weighted_loss = sparse_voxel_reconstruction_loss(
        logits,
        target,
        dice_weight=1.0,
        population_positive_counts=torch.tensor([2.0]),
        population_negative_counts=torch.tensor([2.0]),
    )

    assert torch.allclose(weighted_loss, full_loss, atol=1.0e-6)


def test_chunked_full_grounding_matches_direct_full_lattice_gradient():
    model_config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        coordinate_chunk_size=7,
        coordinate_decoder_width=16,
        coordinate_decoder_depth=2,
        coordinate_fourier_bands=2,
        use_torch_compile=False,
    )
    training_config = TrainingConfig(
        num_epochs=1,
        coordinate_decoder_threshold=1,
        clean_geometry_reconstruction_weight=1.7,
        geometry_dice_weight=0.8,
    )
    trainer = OptimizedDiffusionTrainer(
        model_config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=2),
        training_config,
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    latent = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    target = torch.zeros((1, 4, 4, 4))
    target[:, 1:3, 1:3, :] = 1.0

    direct_logits = trainer.converter(latent)
    direct_loss = sparse_voxel_reconstruction_loss(
        direct_logits,
        target,
        dice_weight=training_config.geometry_dice_weight,
    )
    (training_config.clean_geometry_reconstruction_weight * direct_loss).backward()
    expected = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in trainer.converter.parameters()
    ]
    for parameter in trainer.converter.parameters():
        parameter.grad = None

    chunked_loss = trainer._backward_full_grounded_coordinate_loss(latent, target)
    actual = [parameter.grad for parameter in trainer.converter.parameters()]

    assert torch.allclose(chunked_loss, direct_loss.detach(), atol=1.0e-6)
    for expected_gradient, actual_gradient in zip(expected, actual):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            assert actual_gradient is not None
            assert torch.allclose(actual_gradient, expected_gradient, atol=2.0e-5, rtol=2.0e-4)


def test_width_expansion_preserves_overlap_and_initializes_new_channels_softly():
    source = torch.nn.Linear(3, 4)
    target = torch.nn.Linear(3, 6)
    with torch.no_grad():
        source.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(4, 3))
        source.bias.copy_(torch.arange(4, dtype=torch.float32))

    report = load_width_expanded_state_dict(
        target,
        source.state_dict(),
        expansion_scale=0.01,
    )

    assert report == {"exact": 0, "expanded": 2, "skipped": 0}
    assert torch.equal(target.weight[:4], source.weight)
    assert torch.equal(target.bias[:4], source.bias)
    assert float(target.weight[4:].detach().abs().max()) < 0.01
    assert float(target.bias[4:].detach().abs().max()) < 0.01


def test_optimizer_state_offload_moves_adam_moments():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    moved_bytes = move_optimizer_state(optimizer, "cpu")

    assert moved_bytes == 0
    assert all(
        not isinstance(value, torch.Tensor) or value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
    )


def test_trainer_syncs_consistency_teacher_from_diffusion_ema():
    model_config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    trainer = OptimizedDiffusionTrainer(
        model_config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=2),
        TrainingConfig(num_epochs=1),
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    assert trainer.val_cfd_simulator is None
    assert [group["name"] for group in trainer.optimizer.param_groups] == [
        "diffusion",
        "coordinate_converter",
        "consistency_student",
    ]
    assert (
        trainer.optimizer.param_groups[1]["lr"]
        == trainer.training_config.converter_learning_rate
    )
    with torch.no_grad():
        for parameter in trainer.diffusion_model.parameters():
            parameter.fill_(0.01)
        for parameter in trainer.ema_model.parameters():
            parameter.fill_(0.02)
        for parameter in trainer.consistency_model.teacher_model.parameters():
            parameter.zero_()

    trainer._sync_consistency_teacher()

    for teacher_parameter, ema_parameter, diffusion_parameter in zip(
        trainer.consistency_model.teacher_model.parameters(),
        trainer.ema_model.parameters(),
        trainer.diffusion_model.parameters(),
    ):
        assert torch.equal(teacher_parameter, ema_parameter)
        assert not torch.equal(teacher_parameter, diffusion_parameter)


def test_consistency_huber_retains_raw_mse_but_bounds_extreme_gradient():
    class ConstantTeacher(torch.nn.Module):
        def forward(self, value, timestep, condition=None):
            return torch.zeros_like(value)

    class ScaledStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(100.0))

        def forward(self, value, timestep, condition=None):
            return torch.ones_like(value) * self.scale

    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(config, DiffusionConfig(timesteps=8))
    model.teacher_model = ConstantTeacher()
    model.student_model = ScaledStudent()

    loss = model.consistency_loss(
        torch.zeros((1, 4)),
        torch.tensor([7]),
        torch.tensor([7]),
        loss_type="huber",
        huber_delta=1.0,
    )
    loss.backward()

    assert model.last_consistency_metrics["raw_mse"] == 10_000.0
    assert float(loss.detach()) == 99.5
    assert torch.allclose(model.student_model.scale.grad, torch.tensor(1.0))


def test_sparse_consistency_updates_cycle_over_every_inference_timestep():
    model_config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    trainer = OptimizedDiffusionTrainer(
        model_config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=4),
        TrainingConfig(num_epochs=1, consistency_interval=20),
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    observed = []

    def fake_consistency_loss(
        latent,
        t_student,
        t_teacher,
        condition=None,
        *,
        loss_type,
        huber_delta,
    ):
        observed.append(int(t_student.item()))
        trainer.consistency_model.last_consistency_metrics = {
            "timestep_mean": float(t_student.item()),
            "raw_mse": 0.0,
        }
        return latent.sum() * 0.0

    trainer.consistency_model.consistency_loss = fake_consistency_loss
    for _ in range(4):
        trainer._compute_consistency_loss(torch.zeros((1, 4)))

    assert observed == [7, 5, 2, 0]
    assert trainer.consistency_update_step == 4


def test_train_epoch_recombines_data_consistency_and_direct_student_gradients():
    class DifferentiableDirectLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.last_components = {}

        def forward(
            self,
            logits,
            design_spec,
            simulator,
            seed=None,
            reference_occupancy=None,
        ):
            self.calls += 1
            value = torch.sigmoid(logits).mean()
            self.last_components = {
                "aero_loss": float(value.detach()),
                "connectivity_loss": 0.1,
                "aircraft_validity_loss": 0.2,
                "spsa_gradient_norm": 0.3,
                "spsa_gradient_norm_unclipped": 0.4,
                "aero_spsa_gradient_norm": 0.1,
                "aero_spsa_gradient_norm_unclipped": 0.2,
                "connectivity_spsa_gradient_norm": 0.1,
                "connectivity_spsa_gradient_norm_unclipped": 0.2,
                "aircraft_validity_spsa_gradient_norm": 0.1,
                "aircraft_validity_spsa_gradient_norm_unclipped": 0.2,
            }
            return value

    model_config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    trainer = OptimizedDiffusionTrainer(
        model_config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=4),
        TrainingConfig(
            num_epochs=1,
            consistency_interval=1,
            direct_solver_steps=1,
            direct_solver_directions=1,
            direct_solver_interval=1,
            offload_optimizer_state_between_steps=False,
        ),
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    fake_direct = DifferentiableDirectLoss()
    trainer.direct_solver_loss = fake_direct
    batch = {
        "latent": torch.zeros((1, 4)),
        "geometry": torch.zeros((1, 4, 4, 4)),
        "design_spec": [DesignSpec()],
    }

    metrics = trainer.train_epoch([batch], grid_size=4)

    assert fake_direct.calls == 1
    assert metrics["direct_solver_eval_count"] == 1
    assert metrics["direct_solver_call_count"] == 3
    assert metrics["direct_solver_iteration_coverage"] == 1.0
    assert metrics["consistency_eval_count"] == 1
    assert metrics["student_data_gradient_norm_applied"] > 0.0
    assert metrics["student_consistency_gradient_norm_applied"] > 0.0
    assert metrics["student_direct_gradient_norm_applied"] > 0.0
    assert trainer.global_step == 1


def test_restore_resume_learning_rate_if_zero_resets_completed_checkpoint_lr():
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.0)

    restored = restore_resume_learning_rate_if_zero(optimizer, 2e-4)

    assert restored is True
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4]


def test_restore_resume_learning_rate_if_zero_preserves_active_lr():
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    restored = restore_resume_learning_rate_if_zero(optimizer, 2e-4)

    assert restored is False
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4]


def test_resume_reapplies_global_optimizer_group_learning_rates():
    modules = [torch.nn.Linear(2, 2) for _ in range(3)]
    optimizer = torch.optim.AdamW(
        [
            {"params": modules[0].parameters(), "lr": 9e-3, "name": "diffusion"},
            {"params": modules[1].parameters(), "lr": 9e-3, "name": "coordinate_converter"},
            {"params": modules[2].parameters(), "lr": 9e-3, "name": "consistency_student"},
        ]
    )
    config = TrainingConfig(
        learning_rate=2e-4,
        converter_learning_rate=1e-3,
        consistency_student_learning_rate=5e-5,
    )

    applied = apply_configured_optimizer_learning_rates(optimizer, config)

    assert applied == {
        "diffusion": 2e-4,
        "coordinate_converter": 1e-3,
        "consistency_student": 5e-5,
    }
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4, 1e-3, 5e-5]


def test_corpus_scaling_law_increases_capacity_only_with_distinct_geometries():
    small = ModelConfig.scaled_for_corpus(185, 96)
    target = ModelConfig.scaled_for_corpus(600, 96)
    large = ModelConfig.scaled_for_corpus(5000, 96)

    assert small.latent_dim == target.latent_dim == large.latent_dim == 192
    assert small.coordinate_decoder_width < target.coordinate_decoder_width <= large.coordinate_decoder_width
    assert target.coordinate_fourier_bands == 6
    assert target.grid_resolution == 96


def test_six_hundred_geometry_configuration_uses_global_latent_width():
    config = ModelConfig.scaled_for_corpus(600, 96)
    diffusion_config = DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=2)
    diffusion = LatentDiffusionUNet(config, diffusion_config)
    student = ConsistencyModel(config, diffusion_config).student_model
    converter = LatentTo3DConverter(
        config.latent_dim,
        config.grid_resolution,
        coordinate_decoder_threshold=96,
        coordinate_chunk_size=config.coordinate_chunk_size,
        coordinate_decoder_width=config.coordinate_decoder_width,
        coordinate_decoder_depth=config.coordinate_decoder_depth,
        coordinate_fourier_bands=config.coordinate_fourier_bands,
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in list(diffusion.parameters()) + list(student.parameters()) + list(converter.parameters())
    )

    assert parameter_count > 7_000_000
    assert config.latent_dim == 192


def test_coordinate_decoder_uses_fourier_positions_at_96_cubed_without_full_grid_allocation():
    config = ModelConfig.scaled_for_corpus(600, 96)
    converter = LatentTo3DConverter(
        config.latent_dim,
        96,
        coordinate_decoder_threshold=96,
        coordinate_chunk_size=32,
        coordinate_decoder_width=config.coordinate_decoder_width,
        coordinate_decoder_depth=config.coordinate_decoder_depth,
        coordinate_fourier_bands=config.coordinate_fourier_bands,
    )
    latent = torch.zeros((1, config.latent_dim))
    logits = converter.forward_flat_indices(latent, torch.tensor([0, 42, 1_024, 10_000]))

    assert logits.shape == (1, 4)
    assert converter.decoder_mode == "coordinate"


def test_coordinate_decoder_checkpointed_chunks_backpropagate_to_latent():
    converter = LatentTo3DConverter(
        latent_dim=8,
        grid_resolution=4,
        coordinate_decoder_threshold=1,
        coordinate_chunk_size=7,
        coordinate_decoder_width=16,
        coordinate_decoder_depth=2,
        coordinate_fourier_bands=2,
        enable_coordinate_gradient_checkpointing=True,
    )
    converter.train()
    latent = torch.randn((1, 8), requires_grad=True)

    converter(latent).mean().backward()

    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert float(latent.grad.abs().sum()) > 0.0


def test_train_epoch_bounded_interruption_validates_only_processed_updates():
    class DifferentiableDirectLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.last_components = {}

        def forward(
            self,
            logits,
            design_spec,
            simulator,
            seed=None,
            reference_occupancy=None,
        ):
            self.calls += 1
            value = torch.sigmoid(logits).mean()
            self.last_components = {
                "aero_loss": float(value.detach()),
                "connectivity_loss": 0.1,
                "aircraft_validity_loss": 0.2,
                "spsa_gradient_norm": 0.3,
                "spsa_gradient_norm_unclipped": 0.4,
                "aero_spsa_gradient_norm": 0.1,
                "aero_spsa_gradient_norm_unclipped": 0.2,
                "connectivity_spsa_gradient_norm": 0.1,
                "connectivity_spsa_gradient_norm_unclipped": 0.2,
                "aircraft_validity_spsa_gradient_norm": 0.1,
                "aircraft_validity_spsa_gradient_norm_unclipped": 0.2,
            }
            return value

    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    trainer = OptimizedDiffusionTrainer(
        config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=4),
        TrainingConfig(
            num_epochs=1,
            consistency_interval=1,
            direct_solver_steps=1,
            direct_solver_directions=1,
            direct_solver_interval=1,
            offload_optimizer_state_between_steps=False,
        ),
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    fake_direct = DifferentiableDirectLoss()
    trainer.direct_solver_loss = fake_direct
    trainer.stop_after_updates = 1
    batch = {
        "latent": torch.zeros((1, 4)),
        "geometry": torch.zeros((1, 4, 4, 4)),
        "design_spec": [DesignSpec()],
    }

    metrics = trainer.train_epoch([batch, batch], grid_size=4)

    assert fake_direct.calls == 1
    assert metrics["direct_solver_eval_count"] == 1
    assert metrics["direct_solver_iteration_coverage"] == 1.0
    assert trainer.global_step == 1
