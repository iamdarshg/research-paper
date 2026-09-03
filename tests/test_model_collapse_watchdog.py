import torch
from torch import nn

from aircraft_diffusion_cfd import (
    OptimizedDiffusionTrainer,
    TrainingConfig,
    evaluate_model_collapse,
    evaluate_promotion_collapse,
    geometry_probability_standard_deviation_loss,
    validate_collapse_watchdog_config,
    validate_decoder_gradient_cadence,
)


def _watchdog_config(**overrides):
    values = {
        "collapse_watchdog_warmup_updates": 2,
        "collapse_watchdog_patience": 2,
        "collapse_watchdog_probability_std_floor": 1.0e-4,
        "collapse_watchdog_probability_span_floor": 1.0e-3,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def test_constant_geometry_is_allowed_during_warmup_then_pauses():
    config = _watchdog_config()
    state = {}
    probabilities = torch.full((32,), 0.499, dtype=torch.float32)

    warmup = evaluate_model_collapse(
        global_step=1,
        training_config=config,
        state=state,
        probabilities=probabilities,
    )
    first_failure = evaluate_model_collapse(
        global_step=2,
        training_config=config,
        state=state,
        probabilities=probabilities,
    )
    second_failure = evaluate_model_collapse(
        global_step=3,
        training_config=config,
        state=state,
        probabilities=probabilities,
    )

    assert warmup["triggered"] is False
    assert first_failure["triggered"] is False
    assert first_failure["consecutive_failures"] == 1
    assert second_failure["triggered"] is True
    assert "geometry_probability_signal_constant" in second_failure["reason_codes"]
    assert "geometry_materialization_blank" in second_failure["reason_codes"]


def test_nonfinite_values_pause_immediately_even_without_geometry_stats():
    config = _watchdog_config(collapse_watchdog_warmup_updates=100)

    decision = evaluate_model_collapse(
        global_step=1,
        training_config=config,
        nonfinite_observed=True,
        losses={"optimization": float("nan")},
    )

    assert decision["triggered"] is True
    assert "nonfinite_forward_or_solver_value" in decision["reason_codes"]
    assert "nonfinite_losses_optimization" in decision["reason_codes"]


def test_converter_optimizer_step_stall_is_a_hard_failure():
    config = _watchdog_config(collapse_watchdog_warmup_updates=0)

    decision = evaluate_model_collapse(
        global_step=6009,
        training_config=config,
        probabilities=torch.linspace(0.1, 0.9, 32),
        gradient_norms={"coordinate_converter": 1.0},
        optimizer_steps={"coordinate_converter": 95},
    )

    assert decision["triggered"] is True
    assert "coordinate_converter_optimizer_step_stalled" in decision["reason_codes"]
    assert decision["coordinate_converter_optimizer_step_lag"] == 5914


def test_diverse_geometry_does_not_trigger_watchdog():
    config = _watchdog_config(collapse_watchdog_warmup_updates=0)

    decision = evaluate_model_collapse(
        global_step=10,
        training_config=config,
        probabilities=torch.linspace(0.1, 0.9, 32),
        gradient_norms={"coordinate_converter": 0.1},
        optimizer_steps={"coordinate_converter": 10},
    )

    assert decision["triggered"] is False
    assert decision["reason_codes"] == []
    assert decision["occupied_fraction"] == 0.5


def test_relative_warm_start_geometry_collapse_requires_persistence():
    config = _watchdog_config(
        collapse_watchdog_warmup_updates=0,
        collapse_watchdog_relative_probability_std_floor=0.5,
        collapse_watchdog_relative_probability_span_floor=0.5,
    )
    state = {}
    healthy = torch.tensor([0.10, 0.30, 0.70, 0.90])
    collapsed = torch.tensor([0.45, 0.46, 0.47, 0.48])

    baseline = evaluate_model_collapse(
        global_step=1,
        training_config=config,
        state=state,
        probabilities=healthy,
    )
    first = evaluate_model_collapse(
        global_step=2,
        training_config=config,
        state=state,
        probabilities=collapsed,
    )
    second = evaluate_model_collapse(
        global_step=3,
        training_config=config,
        state=state,
        probabilities=collapsed,
    )

    assert baseline["triggered"] is False
    assert first["triggered"] is False
    assert second["triggered"] is True
    assert "geometry_probability_signal_collapsed_relative" in second["reason_codes"]
    assert second["relative_probability_std"] < 0.5
    assert second["relative_probability_span"] < 0.5


def test_effective_rank_and_mhc_movement_are_live_hard_policies():
    config = _watchdog_config(
        collapse_watchdog_warmup_updates=0,
        collapse_watchdog_max_mhc_functional_movement=0.1,
    )
    state = {}
    evaluate_model_collapse(
        global_step=1,
        training_config=config,
        state=state,
        probabilities=torch.linspace(0.1, 0.9, 8),
        representation_effective_rank=4.0,
    )
    decision = evaluate_model_collapse(
        global_step=2,
        training_config=config,
        state=state,
        probabilities=torch.linspace(0.1, 0.9, 8),
        representation_effective_rank=1.0,
        mhc_telemetry={
            "converter": {
                "mhc_block": {"routing_functional_movement": 0.2}
            }
        },
    )

    assert decision["triggered"] is True
    assert "representation_effective_rank_collapsed" in decision["reason_codes"]
    assert "mhc_routing_functional_movement_exceeded" in decision["reason_codes"]


def test_module_update_ratio_budget_is_a_transactional_hard_failure():
    config = _watchdog_config(collapse_watchdog_enabled=False)
    decision = evaluate_model_collapse(
        global_step=1,
        training_config=config,
        probabilities=torch.linspace(0.1, 0.9, 8),
        module_update_ratios={
            "diffusion": {"update_parameter_ratio": 0.02},
        },
        module_update_ratio_limits={"diffusion": 0.01},
    )

    assert decision["triggered"] is True
    assert "module_update_ratio_exceeded:diffusion" in decision["reason_codes"]


def test_promotion_nonuniqueness_pauses_after_warmup():
    config = _watchdog_config(
        collapse_watchdog_warmup_updates=2,
        collapse_watchdog_min_unique_fraction=0.5,
    )

    decision = evaluate_promotion_collapse(
        {
            "generated_evaluation_count": 16,
            "generated_unique_fraction": 1.0 / 16.0,
            "generated_aircraft_valid_fraction": 0.0,
            "generated_mean_occupied_fraction": 0.0,
        },
        config,
        global_step=2,
    )

    assert decision["triggered"] is True
    assert "promotion_outputs_non_unique" in decision["reason_codes"]
    assert "promotion_outputs_blank_or_solid" in decision["reason_codes"]


def test_frozen_decoder_rejects_sparse_full_lattice_cadence():
    config = _watchdog_config(
        freeze_decoder_for_generated_paths=True,
        full_lattice_interval=64,
    )

    try:
        validate_decoder_gradient_cadence(config)
    except ValueError as exc:
        assert "full_lattice_interval must be 1" in str(exc)
    else:
        raise AssertionError("sparse frozen-decoder cadence was accepted")


def test_probability_std_loss_increases_low_nonzero_spread():
    logits = torch.tensor([[-0.01, 0.0, 0.01, 0.02]], requires_grad=True)
    initial_std = torch.sigmoid(logits.detach()).std(unbiased=False)
    loss, observed = geometry_probability_standard_deviation_loss(
        logits,
        target_standard_deviation=0.20,
    )

    loss.backward()
    with torch.no_grad():
        updated = logits - 0.5 * logits.grad
        updated_std = torch.sigmoid(updated).std(unbiased=False)

    assert loss.item() > 0.0
    assert torch.isclose(observed, initial_std)
    assert updated_std > initial_std


def test_probability_std_loss_stops_at_target():
    logits = torch.tensor([[-4.0, 4.0]])

    loss, observed = geometry_probability_standard_deviation_loss(
        logits,
        target_standard_deviation=0.20,
    )

    assert observed.item() > 0.20
    assert loss.item() == 0.0


def test_warm_start_functional_anchor_is_frozen_and_weakly_measured():
    class TinyDiffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, latent, timestep, condition=None):
            return latent * self.scale

    trainer = object.__new__(OptimizedDiffusionTrainer)
    trainer.device = torch.device("cpu")
    trainer.diffusion_model = TinyDiffusion()
    trainer.training_config = TrainingConfig(
        functional_anchor_enabled=True,
        functional_anchor_bank_size=2,
        functional_anchor_huber_delta=0.10,
    )
    trainer.recovery_mode = True
    trainer.functional_anchor_bank = None
    trainer.run_state_metadata = {}

    latent = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    trainer._initialize_functional_anchor_bank(latent, None)
    initial_loss = trainer._functional_anchor_loss()
    with torch.no_grad():
        trainer.diffusion_model.scale.add_(0.5)
    drift_loss = trainer._functional_anchor_loss()

    assert trainer.functional_anchor_bank["target"].requires_grad is False
    assert initial_loss.item() == 0.0
    assert drift_loss.item() > 0.0


def test_probability_std_config_rejects_unbounded_target_or_negative_weight():
    for config in (
        _watchdog_config(geometry_probability_std_target=0.51),
        _watchdog_config(geometry_probability_std_weight=-0.01),
    ):
        try:
            validate_collapse_watchdog_config(config)
        except ValueError as exc:
            assert "geometry_probability_std" in str(exc)
        else:
            raise AssertionError("invalid probability standard-deviation config was accepted")
