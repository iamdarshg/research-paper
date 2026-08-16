"""Utilities for capturing and balancing independent parameter gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import torch


GradientBuffer = Tuple[Optional[torch.Tensor], ...]


class NonFiniteGradientError(FloatingPointError):
    """Raised when a gradient branch cannot be applied safely."""


@dataclass(frozen=True)
class BranchGradientTelemetry:
    """Norm and presence information for one independently scaled branch."""

    raw_norm: float
    applied_norm: float
    scale: float
    present: bool
    nonzero: bool
    anchor_cosine_before: float = 0.0
    anchor_cosine_after: float = 0.0
    conflict_projected: bool = False
    projection_norm: float = 0.0


def capture_gradients(
    parameters: Iterable[torch.nn.Parameter],
) -> GradientBuffer:
    """Detach and clone the current gradient for each parameter."""

    return tuple(
        None
        if parameter.grad is None
        else parameter.grad.detach().clone(memory_format=torch.preserve_format)
        for parameter in parameters
    )


def clear_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Clear parameter gradients without modifying parameter values."""

    for parameter in parameters:
        parameter.grad = None


def add_gradient_buffers(
    first: Sequence[Optional[torch.Tensor]],
    second: Sequence[Optional[torch.Tensor]],
) -> GradientBuffer:
    """Add two aligned gradient buffers without mutating either input."""
    if len(first) != len(second):
        raise ValueError("gradient branches must have the same buffer count")
    result: list[Optional[torch.Tensor]] = []
    for first_value, second_value in zip(first, second):
        if first_value is None and second_value is None:
            result.append(None)
        elif first_value is None:
            result.append(second_value.detach().clone())
        elif second_value is None:
            result.append(first_value.detach().clone())
        else:
            result.append(first_value.detach().clone() + second_value.detach())
    return tuple(result)


def _validate_gradient_layout(
    gradient: torch.Tensor,
    *,
    branch_name: str,
    parameter_index: int,
) -> None:
    if gradient.layout != torch.strided:
        raise ValueError(
            f"gradient branch {branch_name!r} parameter {parameter_index} "
            f"uses unsupported layout {gradient.layout}"
        )


def gradient_l2_norm(
    gradients: Sequence[Optional[torch.Tensor]],
    *,
    branch_name: str = "<unnamed>",
) -> float:
    """Return a finite L2 norm, rejecting any nonfinite gradient.

    Streaming per-tensor reduction: each present gradient is reduced on its own
    (``torch.linalg.vector_norm``) so no full-model FP64 concatenated vector
    (and no full-size square allocation) is ever materialized.  The per-tensor
    norms are combined with a max-scaled accumulation so a single representable
    extreme value (e.g. 1e308) keeps a finite result instead of overflowing the
    intermediate sum of squares.
    """
    per_tensor_norms: list[float] = []
    for parameter_index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        _validate_gradient_layout(
            gradient,
            branch_name=branch_name,
            parameter_index=parameter_index,
        )
        gradient_f64 = gradient.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(gradient_f64).all().item()):
            raise NonFiniteGradientError(
                f"gradient branch {branch_name!r} contains nonfinite values"
            )
        # Per-tensor norm of the fp64 tensor; a tensor with finite values whose
        # norm still overflows (e.g. two 1e308 entries) reports inf here and is
        # rejected by the scale check below, matching the concatenated behavior.
        per_tensor_norms.append(
            float(torch.linalg.vector_norm(gradient_f64, ord=2).item())
        )
    if not per_tensor_norms:
        return 0.0
    scale = max(per_tensor_norms)
    if not math.isfinite(scale):
        raise NonFiniteGradientError(
            f"gradient branch {branch_name!r} has a nonfinite L2 norm"
        )
    if scale == 0.0:
        return 0.0
    total_norm = scale * math.sqrt(
        sum((norm / scale) ** 2 for norm in per_tensor_norms)
    )
    if not math.isfinite(total_norm):
        raise NonFiniteGradientError(
            f"gradient branch {branch_name!r} has a nonfinite L2 norm"
        )
    return total_norm


def gradient_cosine_similarity(
    first: Sequence[Optional[torch.Tensor]],
    second: Sequence[Optional[torch.Tensor]],
    *,
    first_name: str = "first",
    second_name: str = "second",
) -> float:
    """Return finite cosine similarity, or zero when either branch is zero."""

    if len(first) != len(second):
        raise ValueError("gradient branches must have the same buffer count")
    first_norm = gradient_l2_norm(first, branch_name=first_name)
    second_norm = gradient_l2_norm(second, branch_name=second_name)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    dot = _gradient_dot_product(
        first,
        second,
        first_name=first_name,
        second_name=second_name,
    )
    cosine = dot / (first_norm * second_norm)
    if not math.isfinite(cosine):
        raise NonFiniteGradientError("gradient cosine similarity is nonfinite")
    return float(max(-1.0, min(1.0, cosine)))


def _gradient_dot_product(
    first: Sequence[Optional[torch.Tensor]],
    second: Sequence[Optional[torch.Tensor]],
    *,
    first_name: str,
    second_name: str,
) -> float:
    if len(first) != len(second):
        raise ValueError("gradient branches must have the same buffer count")

    # Streaming per-tensor accumulation: the per-pair fp64 products are summed
    # in a Python float so no full-model FP64 concatenated vectors are built.
    # Summation order differs from the concatenated form; callers compare within
    # the repo's atol/rtol, not bit-exactness.
    dot = 0.0
    for parameter_index, (first_gradient, second_gradient) in enumerate(
        zip(first, second)
    ):
        if first_gradient is None or second_gradient is None:
            continue
        _validate_gradient_layout(
            first_gradient,
            branch_name=first_name,
            parameter_index=parameter_index,
        )
        _validate_gradient_layout(
            second_gradient,
            branch_name=second_name,
            parameter_index=parameter_index,
        )
        first_f64 = first_gradient.detach().to(dtype=torch.float64)
        second_f64 = second_gradient.detach().to(dtype=torch.float64)
        if not bool(
            torch.logical_and(
                torch.isfinite(first_f64),
                torch.isfinite(second_f64),
            ).all().item()
        ):
            raise NonFiniteGradientError(
                f"gradient dot product for {first_name!r} and {second_name!r} "
                "contains nonfinite values"
            )
        dot += float(
            torch.dot(first_f64.reshape(-1), second_f64.reshape(-1)).item()
        )
    if not math.isfinite(dot):
        raise NonFiniteGradientError(
            f"gradient dot product for {first_name!r} and {second_name!r} "
            "is nonfinite"
        )
    return dot


def project_conflicting_gradient(
    gradients: Sequence[Optional[torch.Tensor]],
    anchor: Sequence[Optional[torch.Tensor]],
    *,
    branch_name: str,
    anchor_name: str,
) -> tuple[GradientBuffer, float, float, bool, float]:
    """Remove only the component that points uphill on an anchor objective."""

    if len(gradients) != len(anchor):
        raise ValueError("gradient branches must have the same buffer count")

    cosine_before = gradient_cosine_similarity(
        gradients,
        anchor,
        first_name=branch_name,
        second_name=anchor_name,
    )
    branch_norm = gradient_l2_norm(gradients, branch_name=branch_name)
    anchor_norm = gradient_l2_norm(anchor, branch_name=anchor_name)
    if branch_norm == 0.0 or anchor_norm == 0.0 or cosine_before >= 0.0:
        cloned = tuple(
            None
            if gradient is None
            else gradient.detach().clone(memory_format=torch.preserve_format)
            for gradient in gradients
        )
        return cloned, cosine_before, cosine_before, False, 0.0

    dot = _gradient_dot_product(
        gradients,
        anchor,
        first_name=branch_name,
        second_name=anchor_name,
    )
    anchor_norm_squared = anchor_norm * anchor_norm
    coefficient = dot / max(anchor_norm_squared, 1.0e-300)
    projected: list[Optional[torch.Tensor]] = []
    for gradient, anchor_gradient in zip(gradients, anchor):
        if gradient is None and anchor_gradient is None:
            projected.append(None)
            continue
        if gradient is None:
            projected.append(
                anchor_gradient.detach().clone(
                    memory_format=torch.preserve_format
                ).mul_(-coefficient)
            )
            continue
        value = gradient.detach().clone(memory_format=torch.preserve_format)
        if anchor_gradient is not None:
            value.add_(anchor_gradient, alpha=-coefficient)
        projected.append(value)

    projected_buffer = tuple(projected)
    cosine_after = gradient_cosine_similarity(
        projected_buffer,
        anchor,
        first_name=branch_name,
        second_name=anchor_name,
    )
    if cosine_after < -1.0e-6:
        raise NonFiniteGradientError(
            f"conflict projection left branch {branch_name!r} opposed to "
            f"anchor {anchor_name!r}: cosine={cosine_after:.6g}"
        )
    projection_norm = abs(dot) / max(anchor_norm, 1.0e-300)
    return (
        projected_buffer,
        cosine_before,
        cosine_after,
        True,
        projection_norm,
    )


def project_improvement_gradients_against_guards(
    improvements: Mapping[str, Sequence[Optional[torch.Tensor]]],
    guards: Mapping[str, Sequence[Optional[torch.Tensor]]],
    *,
    guard_order: Sequence[str] = ("reconstruction", "connectivity", "validity"),
    tolerance: float = 1.0e-10,
    max_passes: int = 16,
) -> tuple[dict[str, GradientBuffer], dict[str, dict[str, Any]]]:
    """Project measured improvement directions into all active guard half-spaces."""
    if tolerance < 0.0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite and nonnegative")
    if int(max_passes) <= 0:
        raise ValueError("max_passes must be greater than zero")
    ordered_guards = tuple(
        name for name in guard_order if name in guards
    ) + tuple(sorted(set(guards).difference(guard_order)))
    if not ordered_guards:
        return {
            name: tuple(
                None
                if value is None
                else value.detach().clone(memory_format=torch.preserve_format)
                for value in gradient
            )
            for name, gradient in improvements.items()
        }, {
            name: {
                "pre_cosines": {},
                "post_cosines": {},
                "active_guard_set": [],
                "projected": False,
                "projection_norm": 0.0,
                "accepted_norm": gradient_l2_norm(gradient, branch_name=name),
            }
            for name, gradient in improvements.items()
        }

    accepted: dict[str, GradientBuffer] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    for name, gradient in improvements.items():
        original_dtypes = tuple(
            None if value is None else value.dtype for value in gradient
        )
        current = tuple(
            None
            if value is None
            else value.detach().to(dtype=torch.float64)
            for value in gradient
        )
        pre_cosines = {
            guard_name: gradient_cosine_similarity(
                current,
                guards[guard_name],
                first_name=name,
                second_name=guard_name,
            )
            for guard_name in ordered_guards
        }
        projection_norm = 0.0
        projected = False
        for _ in range(int(max_passes)):
            changed = False
            for guard_name in ordered_guards:
                dot = _gradient_dot_product(
                    current,
                    guards[guard_name],
                    first_name=name,
                    second_name=guard_name,
                )
                if dot >= -float(tolerance):
                    continue
                guard_norm = gradient_l2_norm(
                    guards[guard_name], branch_name=guard_name
                )
                if guard_norm == 0.0:
                    continue
                coefficient = dot / max(guard_norm * guard_norm, 1.0e-300)
                projected_values: list[Optional[torch.Tensor]] = []
                for value, guard_value in zip(current, guards[guard_name]):
                    if value is None and guard_value is None:
                        projected_values.append(None)
                    elif value is None:
                        projected_values.append(
                            guard_value.detach().to(dtype=torch.float64).mul(-coefficient)
                        )
                    elif guard_value is None:
                        projected_values.append(value.detach().clone())
                    else:
                        projected_values.append(
                            value.detach().clone().add(
                                guard_value.detach().to(dtype=torch.float64),
                                alpha=-coefficient,
                            )
                        )
                current = tuple(projected_values)
                projection_norm += abs(dot) / max(guard_norm, 1.0e-300)
                projected = True
                changed = True
            if not changed:
                break
        accepted_current = tuple(
            None
            if value is None
            else value.to(dtype=original_dtypes[index])
            for index, value in enumerate(current)
        )
        projection_fallback_zero = False
        post_cosines = {
            guard_name: gradient_cosine_similarity(
                accepted_current,
                guards[guard_name],
                first_name=name,
                second_name=guard_name,
            )
            for guard_name in ordered_guards
        }
        for guard_name in ordered_guards:
            # Evaluate the residual on the float64 `current` projection, NOT the
            # float32-cast `accepted_current`. After projecting out a guard's
            # direction the true residual is ~1e-16 (float64), but rounding the
            # accepted gradient back to its original dtype leaves a residual dot
            # of magnitude ~1e-7 whose sign is pure accumulation noise. Checking
            # that noise against `tolerance` is a knife-edge that spuriously
            # zeroes the whole improvement (fallback below) for otherwise-valid
            # gradients; the float64 residual preserves the check's intent of
            # catching projections that genuinely failed to remove a guard.
            final_dot = _gradient_dot_product(
                current,
                guards[guard_name],
                first_name=name,
                second_name=guard_name,
            )
            if final_dot < -float(tolerance):
                accepted_current = tuple(
                    None if value is None else torch.zeros_like(value)
                    for value in accepted_current
                )
                projection_fallback_zero = True
                break
        if projection_fallback_zero:
            post_cosines = {
                guard_name: gradient_cosine_similarity(
                    accepted_current,
                    guards[guard_name],
                    first_name=name,
                    second_name=guard_name,
                )
                for guard_name in ordered_guards
            }
        accepted[name] = accepted_current
        telemetry[name] = {
            "pre_cosines": pre_cosines,
            "post_cosines": post_cosines,
            "active_guard_set": list(ordered_guards),
            "projected": projected,
            "projection_fallback_zero": projection_fallback_zero,
            "projection_norm": float(projection_norm),
            "accepted_norm": gradient_l2_norm(accepted_current, branch_name=name),
        }
    return accepted, telemetry


def combine_constrained_measured_gradients(
    components: Mapping[str, Sequence[Optional[torch.Tensor]]],
    *,
    guard_names: Sequence[str] = ("reconstruction", "connectivity", "validity"),
    improvement_names: Sequence[str] = ("occupancy", "aero"),
) -> tuple[GradientBuffer, dict[str, Any]]:
    """Retain every measured component while constraining improvement directions."""
    guards = {
        name: components[name]
        for name in guard_names
        if name in components
    }
    improvements = {
        name: components[name]
        for name in improvement_names
        if name in components
    }
    accepted, telemetry = project_improvement_gradients_against_guards(
        improvements,
        guards,
    )
    all_buffers = list(components.values())
    if not all_buffers:
        return tuple(), {
            "active_guard_set": list(guards),
            "components": telemetry,
            "accepted_norm": 0.0,
        }
    buffer_count = len(all_buffers[0])
    combined: list[Optional[torch.Tensor]] = [None] * buffer_count
    for name, gradient in components.items():
        selected = accepted.get(name, gradient)
        for index, value in enumerate(selected):
            if value is None:
                continue
            combined[index] = (
                value.detach().clone(memory_format=torch.preserve_format)
                if combined[index] is None
                else combined[index] + value
            )
    result = tuple(combined)
    final_accepted, final_telemetry = project_improvement_gradients_against_guards(
        {"combined": result},
        guards,
        guard_order=tuple(guard_names),
    )
    result = final_accepted["combined"]
    return result, {
        "active_guard_set": list(guards),
        "components": telemetry,
        "final_invariant": final_telemetry["combined"],
        "accepted_norm": gradient_l2_norm(result, branch_name="accepted"),
    }


def apply_max_norm(
    gradients: Sequence[Optional[torch.Tensor]],
    max_norm: float,
    *,
    branch_name: str = "<unnamed>",
) -> tuple[GradientBuffer, BranchGradientTelemetry]:
    """Clone and independently limit one branch without amplifying it."""

    limit = float(max_norm)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_norm must be finite and greater than zero")

    raw_norm = gradient_l2_norm(gradients, branch_name=branch_name)
    present = any(gradient is not None for gradient in gradients)
    nonzero = raw_norm > 0.0
    scale = min(1.0, limit / raw_norm) if nonzero else 1.0

    if nonzero and scale <= 0.0:
        raise NonFiniteGradientError(
            f"gradient branch {branch_name!r} trust-region scale underflowed"
        )

    applied = tuple(
        None
        if gradient is None
        else gradient.detach().clone(memory_format=torch.preserve_format).mul_(scale)
        for gradient in gradients
    )
    applied_norm = gradient_l2_norm(applied, branch_name=branch_name)
    if nonzero and applied_norm == 0.0:
        raise NonFiniteGradientError(
            f"finite nonzero gradient branch {branch_name!r} became zero "
            "after trust-region scaling"
        )

    telemetry = BranchGradientTelemetry(
        raw_norm=raw_norm,
        applied_norm=applied_norm,
        scale=scale,
        present=present,
        nonzero=nonzero,
    )
    return applied, telemetry


def _validate_branch_against_parameters(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[Optional[torch.Tensor]],
    *,
    branch_name: str,
) -> None:
    if len(gradients) != len(parameters):
        raise ValueError(
            f"gradient branch {branch_name!r} has {len(gradients)} buffers "
            f"for {len(parameters)} parameters"
        )

    for parameter_index, (parameter, gradient) in enumerate(
        zip(parameters, gradients)
    ):
        if gradient is None:
            continue
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"gradient branch {branch_name!r} parameter {parameter_index} "
                f"has shape {tuple(gradient.shape)}, expected "
                f"{tuple(parameter.shape)}"
            )
        if gradient.device != parameter.device:
            raise ValueError(
                f"gradient branch {branch_name!r} parameter {parameter_index} "
                f"is on {gradient.device}, expected {parameter.device}"
            )
        if gradient.dtype != parameter.dtype:
            raise ValueError(
                f"gradient branch {branch_name!r} parameter {parameter_index} "
                f"has dtype {gradient.dtype}, expected {parameter.dtype}"
            )


def combine_gradient_branches(
    parameters: Iterable[torch.nn.Parameter],
    branches: Mapping[str, Sequence[Optional[torch.Tensor]]],
    max_norms: Mapping[str, float],
    *,
    conflict_anchor: Optional[str] = None,
    project_conflicting_branches: Sequence[str] = (),
    final_guard_branches: Optional[
        Mapping[str, Sequence[Optional[torch.Tensor]]]
    ] = None,
) -> dict[str, BranchGradientTelemetry]:
    """Independently limit branches and enforce final parameter-space guards.

    ``final_guard_branches`` are the already-measured parameter-space guard
    directions.  They remain separate through the final combination so a
    topology guard cannot be lost when voxel-space direct components are
    collapsed into one student branch.
    """

    parameter_list = tuple(parameters)
    unknown_limits = set(max_norms).difference(branches)
    if unknown_limits:
        names = ", ".join(sorted(unknown_limits))
        raise ValueError(f"max_norms contains unknown gradient branches: {names}")

    missing_limits = set(branches).difference(max_norms)
    if missing_limits:
        names = ", ".join(sorted(missing_limits))
        raise ValueError(f"max_norms is missing gradient branches: {names}")

    combined: list[Optional[torch.Tensor]] = [None] * len(parameter_list)
    telemetry: dict[str, BranchGradientTelemetry] = {}
    applied_branches: dict[str, GradientBuffer] = {}

    for branch_name, gradients in branches.items():
        gradient_list = tuple(gradients)
        _validate_branch_against_parameters(
            parameter_list,
            gradient_list,
            branch_name=branch_name,
        )
        applied, branch_telemetry = apply_max_norm(
            gradient_list,
            max_norms[branch_name],
            branch_name=branch_name,
        )
        telemetry[branch_name] = branch_telemetry
        applied_branches[branch_name] = applied

    final_guards = dict(final_guard_branches or {})
    for guard_name, gradients in final_guards.items():
        _validate_branch_against_parameters(
            parameter_list,
            tuple(gradients),
            branch_name=f"final guard {guard_name}",
        )

    projected_names = tuple(project_conflicting_branches)
    if conflict_anchor is None and projected_names:
        raise ValueError(
            "conflict_anchor is required when project_conflicting_branches is set"
        )
    if conflict_anchor is not None:
        if conflict_anchor not in applied_branches:
            raise ValueError(f"unknown conflict anchor branch: {conflict_anchor}")
        unknown_projected = set(projected_names).difference(applied_branches)
        if unknown_projected:
            names = ", ".join(sorted(unknown_projected))
            raise ValueError(f"unknown projected gradient branches: {names}")
        if conflict_anchor in projected_names:
            raise ValueError("conflict anchor cannot be projected against itself")

        anchor_gradients = applied_branches[conflict_anchor]
        for branch_name in projected_names:
            (
                projected,
                cosine_before,
                cosine_after,
                was_projected,
                projection_norm,
            ) = project_conflicting_gradient(
                applied_branches[branch_name],
                anchor_gradients,
                branch_name=branch_name,
                anchor_name=conflict_anchor,
            )
            applied_branches[branch_name] = projected
            telemetry[branch_name] = replace(
                telemetry[branch_name],
                applied_norm=gradient_l2_norm(
                    projected,
                    branch_name=branch_name,
                ),
                anchor_cosine_before=cosine_before,
                anchor_cosine_after=cosine_after,
                conflict_projected=was_projected,
                projection_norm=projection_norm,
            )

    for branch_name in branches:
        applied = applied_branches[branch_name]
        for parameter_index, gradient in enumerate(applied):
            if gradient is None:
                continue
            if combined[parameter_index] is None:
                combined[parameter_index] = gradient
            else:
                combined[parameter_index] = combined[parameter_index] + gradient

    final_anchor_cosine_before = 0.0
    final_anchor_cosine_after = 0.0
    final_projection_norm = 0.0
    if conflict_anchor is not None:
        anchor_gradients = applied_branches[conflict_anchor]
        non_anchor_values: list[Optional[torch.Tensor]] = []
        for index in range(len(parameter_list)):
            value: Optional[torch.Tensor] = None
            for branch_name, branch_values in applied_branches.items():
                if branch_name == conflict_anchor:
                    continue
                branch_value = branch_values[index]
                if branch_value is not None:
                    value = (
                        branch_value.detach().clone()
                        if value is None
                        else value + branch_value
                    )
            non_anchor_values.append(value)
        non_anchor = tuple(non_anchor_values)
        final_before = tuple(
            None
            if anchor_gradients[index] is None and non_anchor[index] is None
            else (
                non_anchor[index]
                if anchor_gradients[index] is None
                else (
                    anchor_gradients[index]
                    if non_anchor[index] is None
                    else anchor_gradients[index] + non_anchor[index]
                )
            )
            for index in range(len(parameter_list))
        )
        final_anchor_cosine_before = gradient_cosine_similarity(
            final_before,
            anchor_gradients,
            first_name="final_update",
            second_name=conflict_anchor,
        )
        projected_non_anchor, _, final_anchor_cosine_after, projected, final_projection_norm = (
            project_conflicting_gradient(
                non_anchor,
                anchor_gradients,
                branch_name="final_update_non_anchor",
                anchor_name=conflict_anchor,
            )
        )
        for index, anchor_value in enumerate(anchor_gradients):
            value = anchor_value
            other = projected_non_anchor[index]
            if other is not None:
                value = other if value is None else value + other
            combined[index] = value
        telemetry["final_invariant"] = BranchGradientTelemetry(
            raw_norm=gradient_l2_norm(non_anchor, branch_name="final_non_anchor"),
            applied_norm=gradient_l2_norm(projected_non_anchor, branch_name="final_non_anchor"),
            scale=1.0,
            present=any(value is not None for value in non_anchor),
            nonzero=gradient_l2_norm(projected_non_anchor, branch_name="final_non_anchor") > 0.0,
            anchor_cosine_before=final_anchor_cosine_before,
            anchor_cosine_after=final_anchor_cosine_after,
            conflict_projected=projected,
            projection_norm=final_projection_norm,
        )

    if final_guards:
        accepted_final, final_guard_telemetry = (
            project_improvement_gradients_against_guards(
                {"final_update": tuple(combined)},
                final_guards,
                guard_order=("data", "reconstruction", "connectivity", "validity"),
            )
        )
        combined = list(accepted_final["final_update"])
        final_update_telemetry = final_guard_telemetry["final_update"]
        telemetry["final_guard_invariant"] = BranchGradientTelemetry(
            raw_norm=gradient_l2_norm(
                tuple(combined), branch_name="final_update"
            ),
            applied_norm=gradient_l2_norm(
                tuple(combined), branch_name="final_update"
            ),
            scale=1.0,
            present=any(value is not None for value in combined),
            nonzero=gradient_l2_norm(
                tuple(combined), branch_name="final_update"
            ) > 0.0,
            anchor_cosine_before=float(
                final_update_telemetry["pre_cosines"].get("data", 0.0)
            ),
            anchor_cosine_after=float(
                final_update_telemetry["post_cosines"].get("data", 0.0)
            ),
            conflict_projected=bool(final_update_telemetry["projected"]),
            projection_norm=float(final_update_telemetry["projection_norm"]),
        )

    gradient_l2_norm(combined, branch_name="<combined>")

    for parameter, gradient in zip(parameter_list, combined):
        parameter.grad = gradient

    return telemetry
