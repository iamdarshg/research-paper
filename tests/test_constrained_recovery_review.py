import json
import os
import random
import sys
from dataclasses import fields

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
    DirectSolverSPSALoss,
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
import aircraft_diffusion_cfd as recovery
from multiobjective_gradients import combine_gradient_branches
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
    _updates_log_reconciliation_metadata,
    restore_promotion_baseline,
)
import run_monitored_training as monitored_training


@pytest.fixture(autouse=True)
def _force_sequential_direct_solver():
    """Pin the direct-solver forward loop to the sequential per-sample path.

    This trainer-integration suite predates Task 10 and drives the forward loop
    with a mocked ``_direct_measured_objective_for_single`` so it can assert
    per-sample design-spec / guard semantics. That mock is only consulted by the
    sequential branch; the batched branch calls ``_direct_measured_objectives_batch``
    instead. Two cases need chunk=1 here:

    - Direct-loss tests pass a stub simulator (``object()``) with no
      ``lbm_solver``. The Task 10 capability fallback already routes these to
      the sequential branch, so the pin is harmless for them.
    - ``train_epoch``-level tests use the trainer's real
      ``AdvancedCFDSimulator`` (batch-capable), so the capability fallback does
      NOT apply: without the pin the probe loop would take the batched branch,
      bypass the mocked objective, and the per-sample call-count assertions
      (e.g. ``len(calls) == 6``) would fail.

    Forcing chunk=1 keeps both cases on the exact sequential path they were
    written to exercise.
    """
    old = recovery._DIRECT_SOLVER_BATCH_CHUNK
    recovery._DIRECT_SOLVER_BATCH_CHUNK = 1
    yield
    recovery._DIRECT_SOLVER_BATCH_CHUNK = old


def _controlled_spsa_objective(active_guards_by_sample):
    calls = []
    slopes = {
        "occupancy_loss": 0.05,
        "aero_loss": 0.07,
        "connectivity_loss": 0.09,
        "aircraft_validity_loss": 0.11,
    }

    def objective(
        probabilities,
        design_spec,
        simulator,
        cfd_steps,
        connectivity_weight,
        aircraft_validity_weight,
        threshold,
        target_occupancy,
        return_components=False,
    ):
        call_index = len(calls)
        sample_index = call_index // 3
        phase = call_index % 3
        signed_offset = 0.0 if phase == 0 else (1.0 if phase == 1 else -1.0)
        active_guards = set(active_guards_by_sample[sample_index])
        components = {
            "occupancy_loss": 1.0 + sample_index,
            "aero_loss": 2.0 + sample_index,
            "connectivity_loss": 0.4 + sample_index,
            "aircraft_validity_loss": (
                0.5 + sample_index
                if "aircraft_validity_loss" in active_guards
                else 0.0
            ),
            "connectivity_guard_shortfall": (
                0.2 if "connectivity_loss" in active_guards else 0.0
            ),
        }
        for name, slope in slopes.items():
            components[name] += signed_offset * slope
        # The recovery fix removes occupancy from the SPSA component set and
        # from total_loss (it is reported as telemetry only; the analytic
        # gradient is applied at the replay site). The mock mirrors that: the
        # occupancy_loss key stays in the component dict for probe parity, but
        # total_loss is the sum of the SPSA-probed components only.
        components["total_loss"] = sum(
            components[name]
            for name in (
                "aero_loss",
                "connectivity_loss",
                "aircraft_validity_loss",
            )
        )
        calls.append(
            {
                "sample_index": sample_index,
                "phase": phase,
                "components": dict(components),
            }
        )
        return components if return_components else probabilities.new_tensor(
            components["total_loss"]
        )

    return objective, calls


def _controlled_design_spec_objective():
    calls = []

    def objective(
        probabilities,
        design_spec,
        simulator,
        cfd_steps,
        connectivity_weight,
        aircraft_validity_weight,
        threshold,
        target_occupancy,
        return_components=False,
    ):
        call_index = len(calls)
        sample_index = call_index // 3
        phase = call_index % 3
        signed_offset = 0.0 if phase == 0 else (1.0 if phase == 1 else -1.0)
        occupancy_loss = float(design_spec.space_weight) * (
            1.0 + sample_index + 0.05 * signed_offset
        )
        aero_loss = (
            float(design_spec.drag_weight)
            * (10.0 + sample_index + 0.07 * signed_offset)
            + float(design_spec.lift_weight)
            * (100.0 + sample_index + 0.11 * signed_offset)
        )
        components = {
            "occupancy_loss": occupancy_loss,
            "aero_loss": aero_loss,
            "connectivity_loss": 0.0,
            "aircraft_validity_loss": 0.0,
            "connectivity_guard_shortfall": 0.0,
        }
        # Recovery fix: occupancy is telemetry only, excluded from total_loss
        # (the SPSA component set now probes aero/connectivity/validity).
        components["total_loss"] = sum(
            components[name]
            for name in (
                "aero_loss",
                "connectivity_loss",
                "aircraft_validity_loss",
            )
        )
        calls.append(
            {
                "sample_index": sample_index,
                "phase": phase,
                "design_spec": design_spec,
                "components": dict(components),
            }
        )
        return components if return_components else probabilities.new_tensor(
            components["total_loss"]
        )

    return objective, calls


def _round5_design_specs():
    return (
        DesignSpec(space_weight=0.80, drag_weight=0.15, lift_weight=0.05),
        DesignSpec(space_weight=0.05, drag_weight=0.25, lift_weight=0.70),
    )


def _round4_trainer(*, freeze_decoder_for_generated_paths=True):
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
    training = TrainingConfig(
        num_epochs=1,
        consistency_interval=100,
        direct_solver_steps=1,
        direct_solver_directions=1,
        direct_solver_interval=1,
        direct_solver_perturbation_grid_size=0,
        direct_solver_gradient_clip=100.0,
        direct_aero_gradient_max_norm=100.0,
        direct_occupancy_gradient_max_norm=100.0,
        direct_connectivity_gradient_max_norm=100.0,
        direct_validity_gradient_max_norm=100.0,
        student_data_gradient_max_norm=100.0,
        student_direct_gradient_max_norm=100.0,
        gradient_clip=100.0,
        offload_optimizer_state_between_steps=False,
        freeze_decoder_for_generated_paths=freeze_decoder_for_generated_paths,
    )
    trainer = OptimizedDiffusionTrainer(
        model,
        diffusion,
        training,
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    trainer.geometry_threshold_calibrated = True
    trainer.geometry_probability_threshold = 0.5
    return trainer


def _round5_direct_loss():
    return DirectSolverSPSALoss(
        cfd_steps=1,
        perturbation=0.2,
        perturbation_grid_size=0,
        gradient_clip=100.0,
        aero_gradient_max_norm=100.0,
        occupancy_gradient_max_norm=100.0,
        connectivity_gradient_max_norm=100.0,
        validity_gradient_max_norm=100.0,
        connectivity_weight=0.0,
        aircraft_validity_weight=0.0,
        directions=1,
        seed=29,
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
    _reconcile_updates_log(
        path,
        _updates_log_reconciliation_metadata(path, checkpoint),
    )
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


def test_lbm_shape_drag_configuration_remains_dataclass_and_serializable():
    names = {field.name for field in fields(__import__("aircraft_diffusion_cfd").LBMPhysicsConfig)}
    assert {
        "shape_drag_correction_coefficients",
        "shape_drag_correction_min",
        "shape_drag_correction_max",
    } <= names

    config = __import__("aircraft_diffusion_cfd").LBMPhysicsConfig(
        shape_drag_correction_coefficients=(1.0, 2.0),
        shape_drag_correction_min=0.2,
        shape_drag_correction_max=2.5,
    )
    assert config.shape_drag_correction_coefficients == (1.0, 2.0)
    payload = {
        "shape_drag_correction_coefficients": [-1.0, 0.5],
        "shape_drag_correction_min": 0.3,
        "shape_drag_correction_max": 2.0,
    }
    restored = __import__("aircraft_diffusion_cfd").LBMPhysicsConfig(**payload)
    assert restored.shape_drag_correction_coefficients == (-1.0, 0.5)


def test_direct_spsa_preserves_per_sample_design_specs_and_weighted_scalar(
    monkeypatch,
):
    objective, calls = _controlled_design_spec_objective()
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    specs = _round5_design_specs()
    probabilities = torch.stack(
        (
            torch.full((4, 4, 4), 0.25),
            torch.full((4, 4, 4), 0.75),
        )
    ).requires_grad_(True)

    measured_loss = _round5_direct_loss()(
        probabilities,
        specs,
        object(),
        seed=29,
    )
    measured_loss.backward()

    assert len(calls) == 6
    assert [call["design_spec"] for call in calls[:3]] == [specs[0]] * 3
    assert [call["phase"] for call in calls[:3]] == [0, 1, 2]
    assert [call["design_spec"] for call in calls[3:]] == [specs[1]] * 3
    assert [call["phase"] for call in calls[3:]] == [0, 1, 2]
    expected = sum(
        calls[sample_index * 3]["components"]["total_loss"]
        for sample_index in range(2)
    ) / 2.0
    first_spec_for_both = (
        specs[0].drag_weight * 10.0
        + specs[0].lift_weight * 100.0
        + specs[0].drag_weight * 11.0
        + specs[0].lift_weight * 101.0
    ) / 2.0
    assert measured_loss.item() == pytest.approx(expected)
    assert measured_loss.item() != pytest.approx(first_spec_for_both)
    assert probabilities.grad is not None
    assert torch.isfinite(probabilities.grad).all()


@pytest.mark.parametrize("as_singleton_sequence", [False, True])
def test_direct_spsa_broadcasts_single_design_spec_for_compatibility(
    monkeypatch,
    as_singleton_sequence,
):
    objective, calls = _controlled_design_spec_objective()
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    spec = _round5_design_specs()[0]
    probabilities = torch.stack(
        (
            torch.full((4, 4, 4), 0.25),
            torch.full((4, 4, 4), 0.75),
        )
    ).requires_grad_(True)

    spec_input = (spec,) if as_singleton_sequence else spec
    _round5_direct_loss()(probabilities, spec_input, object(), seed=29).backward()

    assert len(calls) == 6
    assert [call["design_spec"] for call in calls] == [spec] * 6


def test_direct_spsa_rejects_mismatched_design_spec_count(monkeypatch):
    objective, calls = _controlled_design_spec_objective()
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    probabilities = torch.stack(
        (
            torch.full((4, 4, 4), 0.25),
            torch.full((4, 4, 4), 0.75),
        )
    ).requires_grad_(True)

    with pytest.raises(
        ValueError,
        match=r"design_spec sequence must contain one value or one value per batch item.*3.*batch size 2",
    ):
        _round5_direct_loss()(
            probabilities,
            (*_round5_design_specs(), _round5_design_specs()[0]),
            object(),
            seed=29,
        )

    assert calls == []


def test_train_epoch_preserves_per_sample_design_specs_for_all_solver_calls(
    monkeypatch,
):
    objective, calls = _controlled_design_spec_objective()
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    specs = _round5_design_specs()
    torch.manual_seed(37)
    trainer = _round4_trainer()
    trainer.direct_solver_loss = _round5_direct_loss()
    batch = {
        "latent": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "geometry": torch.stack(
            (torch.zeros((4, 4, 4)), torch.ones((4, 4, 4)))
        ),
        "design_spec": list(specs),
    }

    metrics = trainer.train_epoch([batch], grid_size=4)

    assert len(calls) == 6
    assert [call["design_spec"] for call in calls[:3]] == [specs[0]] * 3
    assert [call["design_spec"] for call in calls[3:]] == [specs[1]] * 3
    expected = sum(
        calls[sample_index * 3]["components"]["total_loss"]
        for sample_index in range(2)
    ) / 2.0
    first_spec_for_both = (
        specs[0].drag_weight * 10.0
        + specs[0].lift_weight * 100.0
        + specs[0].drag_weight * 11.0
        + specs[0].lift_weight * 101.0
    ) / 2.0
    assert metrics["direct_solver_loss"] == pytest.approx(expected)
    assert metrics["direct_solver_loss"] != pytest.approx(first_spec_for_both)
    assert metrics["direct_solver_eval_count"] == 1
    assert metrics["direct_solver_call_count"] == 6


@pytest.mark.parametrize(
    ("active_guards_by_sample", "expected_union"),
    [
        (((), ("connectivity_loss",)), ["connectivity_loss"]),
        ((("connectivity_loss",), ()), ["connectivity_loss"]),
        (
            (
                ("connectivity_loss",),
                ("aircraft_validity_loss",),
            ),
            ["connectivity_loss", "aircraft_validity_loss"],
        ),
    ],
    ids=(
        "first-inactive-later-active",
        "first-active-later-inactive",
        "different-guards-per-sample",
    ),
)
def test_mixed_batch_spsa_guards_remain_batch_aligned_and_replay_in_train_epoch(
    monkeypatch,
    active_guards_by_sample,
    expected_union,
):
    objective, calls = _controlled_spsa_objective(active_guards_by_sample)
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    loss_fn = DirectSolverSPSALoss(
        cfd_steps=1,
        perturbation=0.2,
        perturbation_grid_size=0,
        gradient_clip=100.0,
        aero_gradient_max_norm=100.0,
        occupancy_gradient_max_norm=100.0,
        connectivity_gradient_max_norm=100.0,
        validity_gradient_max_norm=100.0,
        connectivity_weight=1.0,
        aircraft_validity_weight=1.0,
        directions=1,
        seed=17,
    )
    probabilities = torch.stack(
        (
            torch.full((4, 4, 4), 0.35),
            torch.full((4, 4, 4), 0.65),
        )
    ).requires_grad_(True)

    measured_loss = loss_fn(probabilities, DesignSpec(), object(), seed=17)
    measured_loss.backward()

    assert len(calls) == 6
    assert [call["sample_index"] for call in calls] == [0, 0, 0, 1, 1, 1]
    expected_base_losses = [
        calls[index * 3]["components"]["total_loss"] for index in range(2)
    ]
    assert measured_loss.item() == pytest.approx(sum(expected_base_losses) / 2.0)
    assert loss_fn.last_components["active_guard_names"] == expected_union
    guard_buffers = loss_fn.last_components["_accepted_guard_gradients"]
    assert set(guard_buffers) == set(expected_union)
    for guard_name in expected_union:
        guard = guard_buffers[guard_name]
        assert guard.shape == probabilities.shape
        for sample_index, active_names in enumerate(active_guards_by_sample):
            if guard_name in active_names:
                assert torch.count_nonzero(guard[sample_index]).item() > 0
            else:
                assert torch.equal(
                    guard[sample_index],
                    torch.zeros_like(guard[sample_index]),
                )

    objective, train_calls = _controlled_spsa_objective(active_guards_by_sample)
    monkeypatch.setattr(recovery, "_direct_measured_objective_for_single", objective)
    torch.manual_seed(23)
    trainer = _round4_trainer()
    batch = {
        "latent": torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "geometry": torch.stack(
            (torch.zeros((4, 4, 4)), torch.ones((4, 4, 4)))
        ),
        "design_spec": [DesignSpec(), DesignSpec()],
    }

    metrics = trainer.train_epoch([batch], grid_size=4)

    expected_replayed = [
        replay_name
        for source_name, replay_name in (
            ("connectivity_loss", "connectivity"),
            ("aircraft_validity_loss", "validity"),
        )
        if source_name in expected_union
    ]
    assert len(train_calls) == 6
    assert metrics["direct_solver_eval_count"] == 1
    assert metrics["direct_solver_call_count"] == 6
    assert trainer.last_gradient_lifecycle["replayed_guard_names"] == expected_replayed


def test_generated_path_converter_freeze_filters_captured_branches_before_step():
    class ControlledMeasuredObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_components = {}

        def forward(self, logits, design_spec, simulator, seed=None, reference_occupancy=None):
            self.last_components = {
                "aero_loss": 0.2,
                "occupancy_loss": 0.1,
                "connectivity_loss": 0.3,
                "aircraft_validity_loss": 0.0,
                "active_guard_names": ["connectivity_loss"],
                "_accepted_guard_gradients": {
                    "connectivity_loss": torch.full_like(logits, 0.5),
                },
                "spsa_gradient_norm": 1.0,
                "spsa_gradient_norm_unclipped": 1.0,
            }
            return logits.mean()

    def run_one_update(freeze_enabled):
        torch.manual_seed(101)
        trainer = _round4_trainer(
            freeze_decoder_for_generated_paths=freeze_enabled
        )
        trainer.direct_solver_loss = ControlledMeasuredObjective()
        captured = {}

        def capture_step(*args, **kwargs):
            converter_group = next(
                group
                for group in trainer.optimizer.param_groups
                if group.get("name") == "coordinate_converter"
            )
            captured["converter_gradients"] = tuple(
                None
                if parameter.grad is None
                else parameter.grad.detach().clone()
                for parameter in converter_group["params"]
            )
            captured["lifecycle"] = dict(trainer.last_gradient_lifecycle)

        trainer.optimizer.step = capture_step
        torch.manual_seed(303)
        trainer.train_epoch(
            [
                {
                    "latent": torch.tensor([[0.2, 0.4, 0.6, 0.8]]),
                    "geometry": torch.ones((1, 4, 4, 4)),
                    "design_spec": [DesignSpec()],
                }
            ],
            grid_size=4,
        )
        return captured

    frozen = run_one_update(True)
    unfrozen = run_one_update(False)

    frozen_lifecycle = frozen["lifecycle"]
    unfrozen_lifecycle = unfrozen["lifecycle"]
    assert frozen_lifecycle["clean_grounded_converter_gradient_norm"] > 0.0
    assert unfrozen_lifecycle["clean_grounded_converter_gradient_norm"] > 0.0
    assert frozen_lifecycle["generated_path_converter_gradient_norms_before_freeze"]["data"] > 0.0
    assert frozen_lifecycle["generated_path_converter_gradient_norms_before_freeze"]["direct"] > 0.0
    assert frozen_lifecycle["generated_path_converter_gradient_norms_before_freeze"]["connectivity"] > 0.0
    assert all(
        value == 0.0
        for value in frozen_lifecycle[
            "generated_path_converter_gradient_norms_after_freeze"
        ].values()
    )
    assert unfrozen_lifecycle[
        "generated_path_converter_gradient_norms_after_freeze"
    ] == pytest.approx(
        unfrozen_lifecycle[
            "generated_path_converter_gradient_norms_before_freeze"
        ]
    )

    frozen_converter = tuple(
        gradient for gradient in frozen["converter_gradients"] if gradient is not None
    )
    unfrozen_converter = tuple(
        gradient for gradient in unfrozen["converter_gradients"] if gradient is not None
    )
    assert frozen_converter
    assert sum(float(gradient.norm().item()) for gradient in frozen_converter) > 0.0
    assert any(
        not torch.allclose(frozen_gradient, unfrozen_gradient)
        for frozen_gradient, unfrozen_gradient in zip(
            frozen_converter,
            unfrozen_converter,
        )
    )


def test_train_epoch_preserves_all_group_gradients_and_active_guard_invariant():
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
        consistency_interval=100,
        direct_solver_steps=1,
        direct_solver_directions=1,
        direct_solver_interval=1,
        offload_optimizer_state_between_steps=False,
    )
    trainer = OptimizedDiffusionTrainer(
        config,
        diffusion,
        training,
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    trainer.geometry_threshold_calibrated = True
    trainer.geometry_probability_threshold = 0.5

    class ControlledMeasuredObjective(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_components = {}

        def forward(self, logits, design_spec, simulator, seed=None, reference_occupancy=None):
            self.last_components = {
                "aero_loss": 0.2,
                "occupancy_loss": 0.1,
                "connectivity_loss": 0.3,
                "aircraft_validity_loss": 0.0,
                "active_guard_names": ["connectivity_loss"],
                "_accepted_guard_gradients": {
                    "connectivity_loss": torch.ones_like(logits),
                    "aircraft_validity_loss": -torch.ones_like(logits),
                },
                "spsa_gradient_norm": 1.0,
                "spsa_gradient_norm_unclipped": 1.0,
                "aero_spsa_gradient_norm": 0.1,
                "aero_spsa_gradient_norm_unclipped": 0.1,
                "connectivity_spsa_gradient_norm": 0.1,
                "connectivity_spsa_gradient_norm_unclipped": 0.1,
                "aircraft_validity_spsa_gradient_norm": 0.0,
                "aircraft_validity_spsa_gradient_norm_unclipped": 0.0,
            }
            return logits.mean()

    trainer.direct_solver_loss = ControlledMeasuredObjective()
    batch = {
        "latent": torch.full((1, 4), 0.2),
        "geometry": torch.zeros((1, 4, 4, 4)),
        "design_spec": [DesignSpec()],
    }
    captured_steps = []
    original_step = trainer.optimizer.step

    def capture_step(*args, **kwargs):
        captured_steps.append(
            tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for group in trainer.optimizer.param_groups
                for parameter in group["params"]
            )
        )
        return original_step(*args, **kwargs)

    trainer.optimizer.step = capture_step
    trainer.train_epoch([batch], grid_size=4)

    lifecycle = trainer.last_gradient_lifecycle
    assert lifecycle["replayed_guard_names"] == ["connectivity"]
    assert lifecycle["replay_isolated"] is True
    assert captured_steps and all(gradient is not None for gradient in captured_steps[0])
    assert lifecycle["data_group_norms"]["diffusion"] > 0.0
    assert lifecycle["data_group_norms"]["coordinate_converter"] > 0.0
    assert lifecycle["exact_margin_loss"] > 0.0
    assert lifecycle["data_margin_gradient_norm"] > 0.0

    actual = captured_steps[0]
    for guard_name, guard in lifecycle["active_guard_gradients"].items():
        dot = sum(
            float((gradient * guard_value).sum().item())
            for gradient, guard_value in zip(actual, guard)
            if gradient is not None and guard_value is not None
        )
        assert dot >= -1.0e-8, (guard_name, dot)


def test_runner_main_restores_saved_threshold_and_resets_cadence(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    run_state = tmp_path / "latest_run_state.pt"
    atomic_save_run_state(
        run_state,
        {
            "geometry_probability_threshold": 0.37,
            "geometry_threshold_calibrated": True,
            "geometry_threshold_calibration": {"source": "saved"},
        },
    )
    updates = tmp_path / "updates.jsonl"
    updates.write_text("", encoding="utf-8")

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, **kwargs):
            self.grid_size = 4
            self.latent_dim = 4
            self.metadata = {
                "split_assignments": ["train", "train", "val"],
                "unique_geometry_count": 3,
            }
            self.records = [
                {
                    "latent": torch.zeros(4),
                    "geometry": torch.zeros((4, 4, 4)),
                    "design_spec": DesignSpec(),
                }
                for _ in range(3)
            ]

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            return self.records[index]

    class FakeTrainer:
        instances = []

        def __init__(self, *args, device=None, **kwargs):
            self.device = device or torch.device("cpu")
            # This test pins the "resume restores the saved threshold" path, so
            # calibration must remain enabled here (the config-fixed override is
            # tested separately below).
            self.training_config = TrainingConfig(
                calibrate_geometry_materialization_threshold=True
            )
            self.geometry_probability_threshold = 0.91
            self.geometry_threshold_calibrated = False
            self.threshold_calls = []
            self.parameter = torch.nn.Parameter(torch.zeros(()))
            self.optimizer = torch.optim.SGD([self.parameter], lr=1.0)
            self.scheduler = None
            self.scheduler_step_per_update = True
            self.global_step = 1
            self.run_state_metadata = {}
            self.stop_after_updates = None
            self.starts = []
            self.saved_states = []
            FakeTrainer.instances.append(self)

        def _set_geometry_probability_threshold(self, threshold, *, calibrated, calibration):
            self.threshold_calls.append(float(threshold))
            self.geometry_probability_threshold = float(threshold)
            self.geometry_threshold_calibrated = bool(calibrated)

        def calibrate_geometry_materialization_threshold(self, loader):
            raise AssertionError("resume-run-state must use the saved threshold")

        def load_run_state(self, path, *, expected_compatibility):
            assert expected_compatibility["configuration"][
                "geometry_materialization_threshold"
            ] == pytest.approx(0.37)
            assert "training_config" in expected_compatibility["configuration"]
            return {
                "epoch_index": 0,
                "completed_in_epoch": 1,
                "sample_order": [0, 1],
                "global_step": 1,
                "run_state_metadata": {
                    "promotion_baseline": {"generated_unique_fraction": 0.5},
                    "promotion_baseline_report": {},
                    "promotion_baseline_metrics": {},
                    "promotion_baseline_identity": {
                        "split": "val",
                        "sample_order": [2],
                        "evaluation_samples": 16,
                        "generation_seeds": 6,
                    },
                },
                "log_reconciliation": {
                    "offset": 0,
                    "sha256": __import__("hashlib").sha256(b"").hexdigest(),
                },
            }

        def train_epoch(self, train_loader, *, grid_size, start_batch):
            self.starts.append(int(start_batch))
            if self.run_state_checkpoint_callback is not None:
                completed = 2 if len(self.starts) == 1 else 1
                self.run_state_checkpoint_callback(completed, 2)
            return {
                "loss": 1.0,
                "optimization_loss": 1.0,
                "mse": 0.1,
                "geometry_reconstruction": 0.1,
                "generation_reconstruction": 0.1,
                "clean_geometry_reconstruction": 0.1,
                "consistency": 0.1,
                "direct_solver_loss": 0.1,
            }

        def evaluate_geometry_promotion_gate(self, loader):
            return {
                "status": "fail",
                "reconstruction_recall": 0.5,
                "generated_recall": 0.5,
                "generated_worst_recall": 0.5,
                "generated_mean_occupied_fraction": 0.1,
                "target_mean_occupied_fraction": 0.1,
                "generated_aircraft_valid_fraction": 0.5,
                "generated_unique_fraction": 0.5,
                "generated_mean_largest_component_fraction": 0.7,
                "generated_mean_normalization_boundary_fraction": 0.0,
            }

        def save_run_state(self, path, **kwargs):
            self.saved_states.append(dict(kwargs))

        def save_checkpoint(self, path):
            return None

    monkeypatch.setattr(monitored_training, "AircraftDesignDataset", FakeDataset)
    monkeypatch.setattr(monitored_training, "OptimizedDiffusionTrainer", FakeTrainer)
    monkeypatch.setattr(monitored_training, "prepare_edt_workspace", lambda shape: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_monitored_training.py",
            "--manifest", str(manifest),
            "--num-epochs", "2",
            "--batch-size", "1",
            "--latent-dim", "4",
            "--grid-size", "4",
            "--resume-run-state", str(run_state),
            "--history-output", str(tmp_path / "history.json"),
            "--updates-output", str(updates),
            "--checkpoint-every-updates", "1",
            "--save-every", "0",
            "--no-save-final-checkpoint",
            "--no-stop-on-promotion-pass",
        ],
    )

    assert monitored_training.main() == 0
    trainer = FakeTrainer.instances[-1]
    assert trainer.threshold_calls == [0.37]
    assert trainer.starts == [1, 0]
    assert 2 in [state["completed_in_epoch"] for state in trainer.saved_states]
    assert 1 in [state["completed_in_epoch"] for state in trainer.saved_states]


def test_log_ahead_is_reconciled_to_checkpoint_boundary(tmp_path):
    path = tmp_path / "updates.jsonl"
    first = _append_jsonl(path, {"global_step": 1, "kind": "optimizer_update"})
    second = _append_jsonl(path, {"global_step": 2, "kind": "optimizer_update"})

    result = _reconcile_updates_log(
        path,
        _updates_log_reconciliation_metadata(path, first),
    )

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


class ThresholdProbe:
    """Lightweight trainer stand-in for _prepare_geometry_threshold_for_run tests."""

    device = torch.device("cpu")

    def __init__(self, *, calibrate=True):
        self.training_config = TrainingConfig(
            calibrate_geometry_materialization_threshold=bool(calibrate)
        )
        self.geometry_probability_threshold = 0.91
        self.geometry_threshold_calibrated = False
        self.calls = []

    def _set_geometry_probability_threshold(self, threshold, *, calibrated, calibration):
        self.calls.append(("restore", threshold, calibrated, dict(calibration or {})))
        self.geometry_probability_threshold = float(threshold)
        self.geometry_threshold_calibrated = bool(calibrated)
        self.geometry_threshold_calibration = dict(calibration or {})

    def calibrate_geometry_materialization_threshold(self, loader):
        self.calls.append(("calibrate",))
        raise AssertionError("exact resume must not recalibrate the threshold")


def test_runner_restores_saved_threshold_before_resume_fingerprint(tmp_path):
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
    assert trainer.calls == [("restore", 0.37, True, {"source": "saved"})]


def test_config_fixed_threshold_overrides_saved_threshold_when_calibration_disabled(
    tmp_path,
):
    """With calibration disabled the config value is authoritative at resume time.

    This is the recovery fix: the failed run's calibrated 0.9752 threshold sat in
    the free-running distribution tail, so even an exact resume must re-force the
    config's fixed 0.5 threshold over any saved checkpoint/saved threshold.
    """
    state_path = tmp_path / "latest_run_state.pt"
    atomic_save_run_state(
        state_path,
        {
            "geometry_probability_threshold": 0.9752,
            "geometry_threshold_calibrated": True,
            "geometry_threshold_calibration": {"source": "saved", "threshold": 0.9752},
        },
    )
    trainer = ThresholdProbe(calibrate=False)
    trainer.training_config.geometry_materialization_threshold = 0.5

    result = _prepare_geometry_threshold_for_run(
        trainer,
        calibration_loader=None,
        resume_run_state=state_path,
    )

    assert trainer.geometry_probability_threshold == pytest.approx(0.5)
    assert trainer.geometry_threshold_calibrated is True
    assert result["source"] == "config_fixed"
    assert result["frozen_for_run"] is True
    assert result["threshold"] == pytest.approx(0.5)
    # The saved 0.9752 was restored first, then overridden by the config value.
    assert trainer.calls == [
        ("restore", 0.9752, True, {"source": "saved", "threshold": 0.9752}),
        ("restore", 0.5, True, {"source": "config_fixed", "frozen_for_run": True, "threshold": 0.5}),
    ]


def test_analytic_occupancy_logit_gradient_behavior():
    """Deterministic one-sided brake, soft anchor, clipping, and ref handling."""
    import math as _math

    trainer = _round4_trainer()
    trainer.geometry_probability_threshold = 0.5
    trainer.training_config.occupancy_mean_probability_weight = 0.5
    trainer.training_config.occupancy_soft_temperature = 0.05
    trainer.training_config.occupancy_soft_weight = 0.5
    trainer.training_config.direct_occupancy_gradient_max_norm = 1.0
    trainer.direct_solver_loss.last_components = {}

    def field(probability):
        logit = _math.log(probability / (1.0 - probability))
        return torch.full((1, 8, 8, 8), logit)

    healthy_sparse = field(0.01)  # mean p ~ 0.01 < 0.5
    saturated = field(0.95)  # mean p ~ 0.95 > 0.5

    # (a) Deterministic: identical inputs produce byte-identical gradients.
    first = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, 0.5, DesignSpec()
    )
    second = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, 0.5, DesignSpec()
    )
    assert torch.equal(first, second)

    # (b) One-sided brake: with only the mean-probability term active, a healthy
    # sparse field (mean(p) < threshold) gets an exactly-zero gradient.
    trainer.training_config.occupancy_soft_weight = 0.0
    brake_only = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, 0.5, DesignSpec()
    )
    assert torch.equal(brake_only, torch.zeros_like(brake_only))

    # (c) Nonzero when mean(p) > threshold (saturated field engages the brake).
    brake_only_saturated = trainer._analytic_occupancy_logit_gradient(
        saturated, 0.5, DesignSpec()
    )
    assert float(brake_only_saturated.norm().item()) > 0.0

    # (d) Soft threshold-anchored surrogate: an empty field below the reference
    # gets a NEGATIVE logit gradient, which gradient descent subtracts, moving
    # the field UP toward the reference occupancy.
    trainer.training_config.occupancy_soft_weight = 0.5
    anchored = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, 0.5, DesignSpec()
    )
    assert float(anchored.mean().item()) < 0.0

    # (e) Gradient-norm clipping: a saturated field's raw gradient is clipped to
    # the configured per-sample max norm.
    trainer.training_config.direct_occupancy_gradient_max_norm = 0.05
    clipped = trainer._analytic_occupancy_logit_gradient(
        saturated, 0.5, DesignSpec()
    )
    assert float(clipped.norm().item()) <= 0.05 * (1.0 + 1.0e-6)

    # (f) Scalar vs 1-element tensor reference_occupancy behave identically.
    trainer.training_config.direct_occupancy_gradient_max_norm = 1.0
    scalar_ref = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, 0.5, DesignSpec()
    )
    tensor_ref = trainer._analytic_occupancy_logit_gradient(
        healthy_sparse, torch.tensor([0.5]), DesignSpec()
    )
    assert torch.allclose(scalar_ref, tensor_ref)

    telemetry = trainer.direct_solver_loss.last_components
    assert telemetry.get("occupancy_analytic_gradient_enabled") == 1.0
    assert "occupancy_reference" in telemetry
    assert "occupancy_analytic_gradient_norm" in telemetry
    assert "occupancy_mean_probability" in telemetry
    assert "occupancy_soft_surrogate" in telemetry


def test_resume_fingerprint_contains_live_training_behavior():
    training = TrainingConfig(
        consistency_interval=3,
        consistency_loss_type="mse",
        consistency_huber_delta=0.7,
        gradient_clip=0.4,
        ema_decay=0.97,
        project_conflicting_direct_gradient=False,
        freeze_decoder_for_generated_paths=False,
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
    assert configuration["training_config"]["freeze_decoder_for_generated_paths"] is False


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

    (parameter * 0.0).backward()
    margin_value = trainer._backward_full_grounded_threshold_margin(
        latent,
        target,
        loss_scale=1.0,
    )
    captured = capture_data_anchor_gradients([parameter])

    assert margin_value.item() > 0.0
    assert captured[0].item() < 0.0

    combine_gradient_branches(
        [parameter],
        {"data": captured, "direct": (torch.tensor([1.0]),)},
        {"data": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
        final_guard_branches={"data": captured},
    )
    assert float(torch.dot(parameter.grad, captured[0])) >= -1.0e-10


def test_coordinate_threshold_margin_backpropagates_all_chunks_after_data_backward():
    class CoordinateConverter(torch.nn.Module):
        decoder_mode = "coordinate"

        def forward_flat_indices(self, latent, indices):
            return latent[:, :1] + indices.float().unsqueeze(0) * 0.0

    trainer = object.__new__(OptimizedDiffusionTrainer)
    trainer.converter = CoordinateConverter()
    trainer.device = torch.device("cpu")
    trainer.geometry_threshold_calibrated = True
    trainer.geometry_probability_threshold = 0.5
    trainer.model_config = type("ModelConfig", (), {"coordinate_chunk_size": 2})()
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
    parameter = torch.nn.Parameter(torch.tensor([[0.0]]))
    latent = parameter * 1.0
    target = torch.ones((1, 1, 1, 5))

    (latent.square().sum() * 0.0).backward(retain_graph=True)
    margin_value = trainer._backward_full_grounded_threshold_margin(
        latent,
        target,
        loss_scale=1.0,
    )

    assert margin_value.item() > 0.0
    assert parameter.grad is not None
    assert parameter.grad.item() < 0.0


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
    interrupted.run_state_updates_log_path = str(updates_path)

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
