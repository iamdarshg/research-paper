import math
import os
import sys

import pytest
import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from multiobjective_gradients import (
    NonFiniteGradientError,
    capture_gradients,
    clear_gradients,
    combine_gradient_branches,
    gradient_cosine_similarity,
    gradient_l2_norm,
)


def test_capture_clones_current_gradients_and_clear_removes_them():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 4.0])

    captured = capture_gradients([parameter])
    parameter.grad.add_(10.0)
    clear_gradients([parameter])

    assert torch.equal(captured[0], torch.tensor([3.0, 4.0]))
    assert parameter.grad is None
    assert gradient_l2_norm(captured) == pytest.approx(5.0)


def test_l2_norm_stays_finite_for_representable_extreme_values():
    norm = gradient_l2_norm(
        (torch.tensor([1.0e308], dtype=torch.float64),),
        branch_name="extreme",
    )

    assert math.isfinite(norm)
    assert norm == pytest.approx(1.0e308)


def test_all_named_branches_contribute_to_target_gradient():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    branches = {
        "data": (torch.tensor([1.0, 2.0]), None),
        "consistency": (torch.tensor([3.0, 4.0]), torch.tensor([5.0])),
        "direct": (None, torch.tensor([6.0])),
    }

    telemetry = combine_gradient_branches(
        [first, second],
        branches,
        max_norms={"data": 100.0, "consistency": 100.0, "direct": 100.0},
    )

    assert torch.equal(first.grad, torch.tensor([4.0, 6.0]))
    assert torch.equal(second.grad, torch.tensor([11.0]))
    assert set(telemetry) == set(branches)
    assert all(item.present and item.nonzero for item in telemetry.values())
    assert all(item.scale == 1.0 for item in telemetry.values())


def test_extreme_branch_is_clipped_independently():
    parameter = torch.nn.Parameter(torch.zeros(2))
    branches = {
        "data": (torch.tensor([3.0, 4.0]),),
        "consistency": (torch.tensor([300.0, 400.0]),),
        "direct": (torch.tensor([-1.0, 2.0]),),
    }

    telemetry = combine_gradient_branches(
        [parameter],
        branches,
        max_norms={"data": 10.0, "consistency": 10.0, "direct": 10.0},
    )

    assert torch.allclose(parameter.grad, torch.tensor([8.0, 14.0]))
    assert telemetry["data"].scale == 1.0
    assert telemetry["data"].applied_norm == pytest.approx(5.0)
    assert telemetry["consistency"].raw_norm == pytest.approx(500.0)
    assert telemetry["consistency"].applied_norm == pytest.approx(10.0)
    assert telemetry["consistency"].scale == pytest.approx(0.02)
    assert telemetry["direct"].scale == 1.0


def test_tiny_gradient_is_not_amplified_and_remains_nonzero():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    tiny = torch.tensor([1.0e-14, -2.0e-14], dtype=torch.float64)

    telemetry = combine_gradient_branches(
        [parameter],
        {"tiny": (tiny,)},
        max_norms={"tiny": 1.0},
    )

    assert torch.equal(parameter.grad, tiny)
    assert telemetry["tiny"].scale == 1.0
    assert telemetry["tiny"].raw_norm > 0.0
    assert telemetry["tiny"].applied_norm > 0.0
    assert telemetry["tiny"].nonzero


def test_gradient_cosine_similarity_reports_alignment_and_conflict():
    first = (torch.tensor([1.0, 0.0]),)

    assert gradient_cosine_similarity(first, first) == pytest.approx(1.0)
    assert gradient_cosine_similarity(
        first,
        (torch.tensor([-1.0, 0.0]),),
    ) == pytest.approx(-1.0)
    assert gradient_cosine_similarity(first, (None,)) == 0.0


def test_conflicting_direct_component_is_projected_off_data_anchor():
    parameter = torch.nn.Parameter(torch.zeros(2))

    telemetry = combine_gradient_branches(
        [parameter],
        {
            "data": (torch.tensor([1.0, 0.0]),),
            "direct": (torch.tensor([-1.0, 1.0]),),
        },
        max_norms={"data": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
    )

    assert torch.allclose(parameter.grad, torch.tensor([1.0, 1.0]))
    assert telemetry["direct"].conflict_projected
    assert telemetry["direct"].anchor_cosine_before < 0.0
    assert telemetry["direct"].anchor_cosine_after == pytest.approx(0.0)
    assert telemetry["direct"].projection_norm == pytest.approx(1.0)


def test_aligned_direct_component_is_preserved_without_projection():
    parameter = torch.nn.Parameter(torch.zeros(2))

    telemetry = combine_gradient_branches(
        [parameter],
        {
            "data": (torch.tensor([1.0, 0.0]),),
            "direct": (torch.tensor([1.0, 1.0]),),
        },
        max_norms={"data": 10.0, "direct": 10.0},
        conflict_anchor="data",
        project_conflicting_branches=("direct",),
    )

    assert torch.allclose(parameter.grad, torch.tensor([2.0, 1.0]))
    assert not telemetry["direct"].conflict_projected
    assert telemetry["direct"].anchor_cosine_before > 0.0
    assert telemetry["direct"].anchor_cosine_after > 0.0


def test_missing_gradients_remain_missing_and_are_reported():
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    third = torch.nn.Parameter(torch.zeros(1))

    telemetry = combine_gradient_branches(
        [first, second, third],
        {
            "absent": (None, None, None),
            "partial": (None, torch.tensor([2.0]), None),
            "zero": (torch.tensor([0.0]), None, None),
        },
        max_norms={"absent": 1.0, "partial": 1.0, "zero": 1.0},
    )

    assert torch.equal(first.grad, torch.tensor([0.0]))
    assert torch.equal(second.grad, torch.tensor([1.0]))
    assert third.grad is None
    assert not telemetry["absent"].present
    assert not telemetry["absent"].nonzero
    assert telemetry["absent"].scale == 1.0
    assert telemetry["zero"].present
    assert not telemetry["zero"].nonzero
    assert telemetry["partial"].present
    assert telemetry["partial"].nonzero


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_branch_fails_without_replacing_target_gradient(bad_value):
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([7.0])

    with pytest.raises(NonFiniteGradientError, match="nonfinite"):
        combine_gradient_branches(
            [parameter],
            {
                "finite": (torch.tensor([1.0]),),
                "bad": (torch.tensor([bad_value]),),
            },
            max_norms={"finite": 1.0, "bad": 1.0},
        )

    assert torch.equal(parameter.grad, torch.tensor([7.0]))
