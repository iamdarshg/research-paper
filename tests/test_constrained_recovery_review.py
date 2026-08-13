import json
import os
import random
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    atomic_save_run_state,
    capture_data_anchor_gradients,
    grounded_threshold_margin_loss,
    resolve_run_state_path,
    validate_run_state_compatibility,
    validate_direct_solver_iteration_coverage,
)
from multiobjective_gradients import capture_gradients, clear_gradients, combine_gradient_branches
from run_monitored_training import (
    _append_jsonl,
    _build_split_dataset,
    _iter_loader_without_rng_advance,
    _build_objective_configuration_fingerprint,
    _prepare_geometry_threshold_for_run,
    _reset_epoch_checkpoint_segment,
    _resume_epoch_position,
    _run_state_checkpoint_due,
    _reconcile_updates_log,
    restore_promotion_baseline,
)


def test_resumed_suffix_coverage_uses_processed_suffix():
    config = TrainingConfig(require_direct_solver_every_iteration=True)

    validate_direct_solver_iteration_coverage(24, 24, config)


def test_eight_plus_twenty_four_log_reconciliation_has_no_duplicate_steps(tmp_path):
    path = tmp_path / "updates.jsonl"
    checkpoint = None
    for step in range(1, 9):
        checkpoint = _append_jsonl(
            path,
            {"global_step": step, "kind": "optimizer_update"},
        )
    _reconcile_updates_log(path, checkpoint)
    for step in range(9, 33):
        _append_jsonl(path, {"global_step": step, "kind": "optimizer_update"})

    steps = [json.loads(line)["global_step"] for line in path.read_text().splitlines()]
    assert steps == list(range(1, 33))
    assert len(steps) == len(set(steps)) == 32


def test_loader_iterator_does_not_advance_torch_rng():
    loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=1, shuffle=False)
    torch.manual_seed(123)
    expected = torch.get_rng_state()

    iterator = _iter_loader_without_rng_advance(loader)

    assert torch.equal(torch.get_rng_state(), expected)
    assert next(iterator)[0].item() == 0


def test_final_parameter_gradient_is_not_uphill_on_anchor():
    parameter = torch.nn.Parameter(torch.zeros(2))

    combine_gradient_branches(
        [parameter],
        {
            "data": (torch.tensor([1.0, 0.0]),),
            "consistency": (torch.tensor([-3.0, 0.0]),),
            "direct": (torch.tensor([1.0, 0.0]),),
        },
        {"data": 10.0, "consistency": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
    )

    assert float(torch.dot(parameter.grad, torch.tensor([1.0, 0.0]))) >= -1.0e-10


def test_final_parameter_update_respects_each_active_topology_guard():
    parameter = torch.nn.Parameter(torch.zeros(2))

    combine_gradient_branches(
        [parameter],
        {
            "data": (torch.tensor([1.0, 0.0]),),
            "consistency": (torch.tensor([0.0, 0.0]),),
            "direct": (torch.tensor([0.0, -3.0]),),
        },
        {"data": 10.0, "consistency": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
        final_guard_branches={
            "data": (torch.tensor([1.0, 0.0]),),
            "connectivity": (torch.tensor([0.0, 1.0]),),
            "validity": (torch.tensor([1.0, 0.0]),),
        },
    )

    for guard in (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
    ):
        assert float(torch.dot(parameter.grad, guard)) >= -1.0e-10


def test_log_ahead_is_reconciled_to_checkpoint_boundary(tmp_path):
    path = tmp_path / "updates.jsonl"
    first = _append_jsonl(path, {"global_step": 1, "kind": "optimizer_update"})
    second = _append_jsonl(path, {"global_step": 2, "kind": "optimizer_update"})

    result = _reconcile_updates_log(path, first)

    assert result["truncated_records"] == 1
    assert [json.loads(line)["global_step"] for line in path.read_text().splitlines()] == [1]
    assert second["offset"] > first["offset"]


def test_checkpoint_ahead_of_log_fails_closed(tmp_path):
    path = tmp_path / "updates.jsonl"
    checkpoint = {
        "offset": 100,
        "sha256": "not-the-log",
        "global_step": 2,
    }
    with pytest.raises(ValueError, match="checkpoint is ahead"):
        _reconcile_updates_log(path, checkpoint)


def test_interrupted_replacement_uses_previous_run_state(tmp_path):
    target = tmp_path / "latest_run_state.pt"
    atomic_save_run_state(target, {"step": 1})
    previous = target.with_name(target.name + ".previous")
    atomic_save_run_state(target, {"step": 2})
    target.unlink()

    assert resolve_run_state_path(target) == previous


def test_objective_and_epoch_configuration_are_resume_immutable():
    expected = {
        "manifest_identity": "manifest",
        "grid_size": 4,
        "latent_dim": 4,
        "split": "train",
        "sample_count": 4,
        "configuration": {"num_epochs": 4, "direct_solver_steps": 5},
    }
    actual = {**expected, "configuration": {"num_epochs": 5, "direct_solver_steps": 6}}

    assert validate_run_state_compatibility(actual, expected) == [
        "configuration.direct_solver_steps",
        "configuration.num_epochs",
    ]


def test_runner_restores_saved_threshold_before_resume_fingerprint(tmp_path):
    class ThresholdProbe:
        device = torch.device("cpu")

        def __init__(self):
            self.geometry_probability_threshold = 0.91
            self.geometry_threshold_calibrated = False
            self.calls = []

        def _set_geometry_probability_threshold(self, threshold, *, calibrated, calibration):
            self.calls.append(("restore", threshold, calibrated))
            self.geometry_probability_threshold = float(threshold)
            self.geometry_threshold_calibrated = bool(calibrated)

        def calibrate_geometry_materialization_threshold(self, loader):
            self.calls.append(("calibrate",))
            raise AssertionError("exact resume must not recalibrate the threshold")

    state_path = tmp_path / "latest_run_state.pt"
    atomic_save_run_state(
        state_path,
        {
            "geometry_probability_threshold": 0.37,
            "geometry_threshold_calibrated": True,
            "geometry_threshold_calibration": {"source": "saved"},
        },
    )
    trainer = ThresholdProbe()

    _prepare_geometry_threshold_for_run(
        trainer,
        calibration_loader=None,
        resume_run_state=state_path,
    )

    assert trainer.geometry_probability_threshold == pytest.approx(0.37)
    assert trainer.calls == [("restore", 0.37, True)]


def test_resume_fingerprint_contains_live_training_behavior():
    training = TrainingConfig(
        consistency_interval=3,
        consistency_loss_type="mse",
        consistency_huber_delta=0.7,
        gradient_clip=0.4,
        ema_decay=0.97,
        project_conflicting_direct_gradient=False,
        geometry_reconstruction_weight=1.2,
        generation_reconstruction_weight=0.8,
        threshold_positive_margin=0.04,
        threshold_negative_margin=0.03,
    )
    model = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    diffusion = DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=4)
    cfd = CFDConfig(base_grid_resolution=4)
    configuration = _build_objective_configuration_fingerprint(
        args=type(
            "Args",
            (),
            {
                "num_epochs": 2,
                "planned_optimizer_updates": 8,
                "batch_size": 1,
                "subset_seed": 4,
                "promotion_split": "val",
                "promotion_evaluation_samples": 2,
                "promotion_generation_seeds": 3,
                "solver": "D3Q27",
                "lbm_stream_bfl_backend": "pytorch_reference",
            },
        )(),
        training_config=training,
        model_config=model,
        diffusion_config=diffusion,
        cfd_config=cfd,
        geometry_probability_threshold=0.37,
        sample_order=[0, 1],
        promotion_sample_order=[2],
    )

    assert configuration["training_config"]["gradient_clip"] == 0.4
    assert configuration["training_config"]["ema_decay"] == 0.97
    assert configuration["training_config"]["consistency_interval"] == 3
    assert configuration["training_config"]["consistency_loss_type"] == "mse"
    assert configuration["training_config"]["geometry_reconstruction_weight"] == 1.2
    assert configuration["training_config"]["generation_reconstruction_weight"] == 0.8
    assert configuration["geometry_materialization_threshold"] == 0.37
    assert configuration["training_config"]["project_conflicting_direct_gradient"] is False


def test_next_epoch_checkpoint_cadence_resets_to_zero_segment():
    resume_state = {"epoch_index": 2, "completed_in_epoch": 8}

    _reset_epoch_checkpoint_segment(resume_state, next_epoch=3)

    assert resume_state == {"epoch_index": 3, "completed_in_epoch": 0}
    assert not _run_state_checkpoint_due(4, 8, 4)
    assert _run_state_checkpoint_due(4, 0, 4)


def test_completed_multi_epoch_state_advances_outer_loop():
    assert _resume_epoch_position(0, 32, 32) == (1, 0)
    assert _resume_epoch_position(1, 8, 32) == (1, 8)


def test_exact_resume_restores_original_promotion_baseline_identity():
    state = {
        "run_state_metadata": {
            "promotion_baseline": {"generated_unique_fraction": 0.91},
            "promotion_baseline_identity": {
                "split": "val",
                "sample_order": [4, 7],
                "evaluation_samples": 16,
                "generation_seeds": 6,
            },
        }
    }

    assert restore_promotion_baseline(
        state,
        promotion_split="val",
        promotion_sample_order=[4, 7],
        evaluation_samples=16,
        generation_seeds=6,
    ) == {"generated_unique_fraction": 0.91}

    with pytest.raises(ValueError, match="baseline identity"):
        restore_promotion_baseline(
            state,
            promotion_split="train",
            promotion_sample_order=[4, 7],
            evaluation_samples=16,
            generation_seeds=6,
        )


def test_missing_promotion_split_fails_closed():
    dataset = TensorDataset(torch.arange(3))
    dataset.metadata = {"split_assignments": ["train", "train", "holdout"]}

    with pytest.raises(ValueError, match="no records"):
        _build_split_dataset(dataset, "val")


def test_calibrated_threshold_margin_clamps_dense_and_is_finite():
    threshold = 0.9752044081687927
    probabilities = torch.tensor([0.99, 0.10], requires_grad=True)
    target = torch.tensor([1.0, 0.0])

    components = grounded_threshold_margin_loss(
        probabilities,
        target,
        threshold=threshold,
        positive_margin=0.05,
        negative_margin=0.05,
        return_components=True,
    )
    components["loss"].backward()

    assert torch.isfinite(components["loss"])
    assert torch.isfinite(probabilities.grad).all()
    assert components["threshold_positive_margin_loss"] > 0.0


def test_captured_student_data_anchor_includes_exact_margin_and_blocks_direct_conflict():
    class DenseConverter(torch.nn.Module):
        decoder_mode = "dense"

        def forward(self, latent):
            return latent.expand(1, 1, 1, 2)

    trainer = object.__new__(OptimizedDiffusionTrainer)
    trainer.converter = DenseConverter()
    trainer.device = torch.device("cpu")
    trainer.geometry_threshold_calibrated = True
    trainer.geometry_probability_threshold = 0.5
    trainer.training_config = type(
        "MarginConfig",
        (),
        {
            "threshold_positive_margin": 0.2,
            "threshold_negative_margin": 0.2,
            "threshold_positive_margin_weight": 1.0,
            "threshold_negative_margin_weight": 1.0,
        },
    )()
    parameter = torch.nn.Parameter(torch.tensor([0.0]))
    latent = parameter.reshape(1, 1, 1, 1)
    target = torch.ones((1, 1, 1, 2))

    margin_value = trainer._backward_full_grounded_threshold_margin(
        latent,
        target,
        loss_scale=1.0,
    )
    captured = capture_gradients([parameter])
    clear_gradients([parameter])

    assert margin_value.item() > 0.0
    assert captured[0].item() < 0.0

    captured = capture_data_anchor_gradients(
        [parameter],
        parameter.detach().mul(0.0),
        exact_margin_loss=parameter.detach().mul(0.0) + captured[0].detach() * parameter,
    )

    combine_gradient_branches(
        [parameter],
        {"data": captured, "direct": (torch.tensor([1.0]),)},
        {"data": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
        final_guard_branches={"data": captured},
    )
    assert float(torch.dot(parameter.grad, captured[0])) >= -1.0e-10


def test_interrupted_two_plus_two_resume_is_trajectory_equivalent(tmp_path):
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    diffusion = DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=4)
    training = TrainingConfig(
        num_epochs=1,
        consistency_interval=1,
        direct_solver_steps=1,
        direct_solver_directions=1,
        direct_solver_interval=1,
        offload_optimizer_state_between_steps=False,
    )
    cfd = CFDConfig(base_grid_resolution=4)
    batches = [
        {
            "latent": torch.full((1, 4), float(index) / 10.0),
            "geometry": torch.zeros((1, 4, 4, 4)),
            "design_spec": [DesignSpec()],
        }
        for index in range(4)
    ]

    class FakeMeasuredObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_components = {}

        def forward(self, logits, design_spec, simulator, seed=None, reference_occupancy=None):
            value = torch.sigmoid(logits).mean()
            self.last_components = {
                "aero_loss": float(value.detach()),
                "connectivity_loss": 0.1,
                "aircraft_validity_loss": 0.2,
                "spsa_gradient_norm": 0.3,
                "spsa_gradient_norm_unclipped": 0.3,
                "aero_spsa_gradient_norm": 0.1,
                "aero_spsa_gradient_norm_unclipped": 0.1,
                "connectivity_spsa_gradient_norm": 0.1,
                "connectivity_spsa_gradient_norm_unclipped": 0.1,
                "aircraft_validity_spsa_gradient_norm": 0.1,
                "aircraft_validity_spsa_gradient_norm_unclipped": 0.1,
            }
            return value

    def make_trainer():
        trainer = OptimizedDiffusionTrainer(
            config,
            diffusion,
            training,
            cfd,
            device=torch.device("cpu"),
        )
        trainer.direct_solver_loss = FakeMeasuredObjective()
        return trainer

    def set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def compare_state(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                compare_state(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for first, second in zip(left, right):
                compare_state(first, second)
        else:
            assert left == right

    set_seed(2026)
    uninterrupted = make_trainer()
    uninterrupted_records = []
    uninterrupted.update_metrics_callback = uninterrupted_records.append
    uninterrupted.train_epoch(batches, grid_size=4)
    uninterrupted_rng = torch.get_rng_state().clone()
    uninterrupted_python_rng = random.getstate()

    run_state_path = tmp_path / "latest_run_state.pt"
    updates_path = tmp_path / "updates.jsonl"
    compatibility = {
        "manifest_identity": "toy",
        "grid_size": 4,
        "latent_dim": 4,
        "split": "train",
        "sample_count": 4,
        "configuration": {"num_epochs": 1, "direct_solver_steps": 1},
    }
    set_seed(2026)
    interrupted = make_trainer()

    def interrupted_update(record):
        interrupted.run_state_log_metadata = _append_jsonl(updates_path, record)
        if record["completed_in_epoch"] == 2:
            interrupted.save_run_state(
                run_state_path,
                epoch_index=0,
                completed_in_epoch=2,
                sample_order=[0, 1, 2, 3],
                compatibility=compatibility,
            )

    interrupted.update_metrics_callback = interrupted_update
    interrupted.stop_after_updates = 2
    interrupted.train_epoch(batches, grid_size=4)
    saved_state = torch.load(run_state_path, map_location="cpu", weights_only=False)

    set_seed(9999)
    resumed = make_trainer()
    resume_info = resumed.load_run_state(
        run_state_path,
        expected_compatibility=compatibility,
    )
    _reconcile_updates_log(updates_path, resume_info["log_reconciliation"])
    resumed_records = []
    def resumed_update(record):
        resumed_records.append(record)
        resumed.run_state_log_metadata = _append_jsonl(updates_path, record)
    resumed.update_metrics_callback = resumed_update
    resumed.train_epoch(batches, grid_size=4, start_batch=2)

    assert [record["global_step"] for record in uninterrupted_records] == [1, 2, 3, 4]
    assert [record["global_step"] for record in resumed_records] == [3, 4]
    assert [
        json.loads(line)["global_step"]
        for line in updates_path.read_text().splitlines()
    ] == [1, 2, 3, 4]
    compare_state(uninterrupted.diffusion_model.state_dict(), resumed.diffusion_model.state_dict())
    compare_state(uninterrupted.converter.state_dict(), resumed.converter.state_dict())
    compare_state(uninterrupted.optimizer.state_dict(), resumed.optimizer.state_dict())
    compare_state(uninterrupted.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert uninterrupted.global_step == resumed.global_step == 4
    assert torch.equal(torch.get_rng_state(), uninterrupted_rng)
    assert random.getstate() == uninterrupted_python_rng
    assert saved_state["completed_in_epoch"] == 2
