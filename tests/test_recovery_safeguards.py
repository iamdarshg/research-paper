import random

import numpy as np
import pytest
import torch

from recovery_safeguards import (
    TransactionalOptimizerStep,
    apply_direct_gradient_trust_region,
    effective_rank,
    parameter_update_ratios,
    update_ratio_limit_violations,
    validate_seed_separation,
)


def test_direct_gradient_trust_region_caps_relative_to_data_without_reducing_data():
    data = (torch.tensor([3.0, 4.0]),)
    direct = (torch.tensor([30.0, 40.0]),)

    applied, telemetry = apply_direct_gradient_trust_region(
        data, direct, norm_ratio=0.10
    )

    assert torch.allclose(applied[0], torch.tensor([0.3, 0.4]))
    assert telemetry["data_gradient_norm"] == pytest.approx(5.0)
    assert telemetry["direct_gradient_norm_raw"] == pytest.approx(50.0)
    assert telemetry["direct_gradient_norm_applied"] == pytest.approx(0.5)
    assert telemetry["direct_gradient_scale"] == pytest.approx(0.01)
    assert telemetry["data_gradient_reduced"] == 0.0


def test_direct_gradient_trust_region_zero_direct_and_zero_data_are_safe():
    zero, zero_telemetry = apply_direct_gradient_trust_region(
        (torch.zeros(2),), (torch.zeros(2),), norm_ratio=0.10
    )
    assert torch.equal(zero[0], torch.zeros(2))
    assert zero_telemetry["direct_gradient_scale"] == 1.0

    data_zero, telemetry = apply_direct_gradient_trust_region(
        (torch.zeros(2),), (torch.ones(2),), norm_ratio=0.10
    )
    assert torch.equal(data_zero[0], torch.zeros(2))
    assert telemetry["direct_gradient_norm_applied"] == 0.0


def test_parameter_update_ratios_are_module_specific():
    before = {
        "diffusion": (torch.ones(2),),
        "converter": (torch.ones(2) * 10.0,),
    }
    after = {
        "diffusion": (torch.ones(2) * 1.1,),
        "converter": (torch.ones(2) * 10.2,),
    }
    ratios = parameter_update_ratios(before, after)
    assert ratios["diffusion"]["update_parameter_ratio"] == pytest.approx(0.1)
    assert ratios["converter"]["update_parameter_ratio"] == pytest.approx(0.02)


def test_update_ratio_limit_violations_are_explicit_and_fail_closed():
    ratios = {
        "diffusion": {"update_parameter_ratio": 0.004},
        "converter": {"update_parameter_ratio": 0.020},
        "mhc_routing": {"update_parameter_ratio": float("nan")},
    }

    violations = update_ratio_limit_violations(
        ratios,
        {"diffusion": 0.01, "converter": 0.01, "mhc_routing": 0.005},
    )

    assert set(violations) == {"converter", "mhc_routing"}
    assert violations["converter"]["limit"] == pytest.approx(0.01)
    assert np.isnan(violations["mhc_routing"]["update_parameter_ratio"])


def test_effective_rank_distinguishes_one_dimensional_and_two_dimensional_support():
    one_dimensional = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    two_dimensional = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    assert effective_rank(one_dimensional) == pytest.approx(1.0)
    assert effective_rank(two_dimensional) == pytest.approx(2.0)


def test_seed_separation_rejects_watchdog_sentinel_overlap():
    validate_seed_separation([1, 2], [3, 4], [5])
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_separation([1, 2], [2, 3])


def test_transaction_rolls_back_parameters_optimizer_scheduler_ema_and_rng():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    ema = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ema.weight.fill_(5.0)
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    transaction = TransactionalOptimizerStep(
        optimizer,
        {"diffusion": (parameter,)},
        scheduler=scheduler,
        ema_model=ema,
    )
    transaction.capture()
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    # Restore the RNG point that existed at capture for the actual step.
    transaction.rollback()

    parameter.grad = torch.ones_like(parameter)
    attempts = {"count": 0}

    def step():
        attempts["count"] += 1
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            ema.weight.add_(1.0)
        if attempts["count"] == 1:
            parameter.data.add_(10.0)

    def healthy():
        return attempts["count"] == 2

    def retry():
        parameter.grad = torch.ones_like(parameter)

    assert transaction.run(step, healthy, retry=retry) == 2
    assert attempts["count"] == 2
    # A rejected first step was not allowed to advance state twice.
    assert parameter.item() == pytest.approx(0.899, abs=1.0e-5)
    assert optimizer.state[parameter]["step"].item() == pytest.approx(1.0)
    assert scheduler.last_epoch == 1
    assert ema.weight.item() == pytest.approx(6.0, abs=1.0e-5)

    # The saved RNG state is also restored before the retry.
    assert random.random() == pytest.approx(expected_python)
    assert np.random.rand() == pytest.approx(expected_numpy)
    assert torch.rand(1) == pytest.approx(expected_torch)
