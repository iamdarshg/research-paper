"""Utilities for capturing and balancing independent parameter gradients."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

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


def _validate_gradient_tensor(
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
    if not bool(torch.isfinite(gradient).all().item()):
        raise NonFiniteGradientError(
            f"gradient branch {branch_name!r} parameter {parameter_index} "
            "contains nonfinite values"
        )


def gradient_l2_norm(
    gradients: Sequence[Optional[torch.Tensor]],
    *,
    branch_name: str = "<unnamed>",
) -> float:
    """Return a finite L2 norm, rejecting any nonfinite gradient."""

    total_norm = 0.0
    for parameter_index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        _validate_gradient_tensor(
            gradient,
            branch_name=branch_name,
            parameter_index=parameter_index,
        )
        magnitudes = gradient.detach().abs()
        max_magnitude = (
            float(magnitudes.max().item()) if magnitudes.numel() else 0.0
        )
        if max_magnitude == 0.0:
            tensor_norm = 0.0
        else:
            normalized_norm = float(
                torch.linalg.vector_norm(
                    magnitudes / max_magnitude,
                    ord=2,
                    dtype=torch.float64,
                ).item()
            )
            tensor_norm = max_magnitude * normalized_norm
        if not math.isfinite(tensor_norm):
            raise NonFiniteGradientError(
                f"gradient branch {branch_name!r} has a nonfinite L2 norm"
            )
        total_norm = math.hypot(total_norm, tensor_norm)

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

    dot = 0.0
    for parameter_index, (first_gradient, second_gradient) in enumerate(
        zip(first, second)
    ):
        if first_gradient is None or second_gradient is None:
            continue
        _validate_gradient_tensor(
            first_gradient,
            branch_name=first_name,
            parameter_index=parameter_index,
        )
        _validate_gradient_tensor(
            second_gradient,
            branch_name=second_name,
            parameter_index=parameter_index,
        )
        dot += float(
            torch.sum(
                first_gradient.detach().double()
                * second_gradient.detach().double()
            ).item()
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

    dot = 0.0
    for parameter_index, (first_gradient, second_gradient) in enumerate(
        zip(first, second)
    ):
        if first_gradient is None or second_gradient is None:
            continue
        _validate_gradient_tensor(
            first_gradient,
            branch_name=first_name,
            parameter_index=parameter_index,
        )
        _validate_gradient_tensor(
            second_gradient,
            branch_name=second_name,
            parameter_index=parameter_index,
        )
        dot += float(
            torch.sum(
                first_gradient.detach().double()
                * second_gradient.detach().double()
            ).item()
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
) -> dict[str, BranchGradientTelemetry]:
    """Independently limit named branches and replace target ``.grad`` buffers."""

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

    gradient_l2_norm(combined, branch_name="<combined>")

    for parameter, gradient in zip(parameter_list, combined):
        parameter.grad = gradient

    return telemetry
