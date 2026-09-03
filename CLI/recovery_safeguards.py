"""Small, deterministic safety primitives for constrained recovery training.

The training loop is intentionally large and GPU-specific.  These helpers keep
the contracts that must be independently testable (and reusable by a canary
runner) out of that loop: relative direct-gradient budgeting, exact optimizer
transactions, update-ratio measurements, effective rank, and seed separation.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


GradientBuffer = Sequence[Optional[torch.Tensor]]


def _finite_norm(values: GradientBuffer, *, name: str) -> float:
    """Compute a bounded-memory FP64-reduced L2 norm."""

    total = 0.0
    for value in values:
        if value is None:
            continue
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"{name} contains non-finite values")
        contribution = float(
            torch.linalg.vector_norm(value.detach().to(dtype=torch.float64)).item()
        )
        total = math.hypot(total, contribution)
    return total


def apply_direct_gradient_trust_region(
    data_gradients: GradientBuffer,
    direct_gradients: GradientBuffer,
    *,
    norm_ratio: float = 0.10,
    epsilon: float = 1.0e-12,
) -> tuple[tuple[Optional[torch.Tensor], ...], dict[str, float]]:
    """Cap CFD/direct gradients relative to the grounded-data gradient.

    The grounded branch is never scaled.  The direct branch is scaled by

    ``min(1, norm_ratio * ||g_data|| / (||g_direct|| + epsilon))``.

    Norms use FP64 reductions while the returned tensors retain their source
    dtypes.  This function does not mutate either input buffer.
    """

    if len(data_gradients) != len(direct_gradients):
        raise ValueError("data and direct gradient buffers must have equal length")
    if not math.isfinite(float(norm_ratio)) or float(norm_ratio) < 0.0:
        raise ValueError("norm_ratio must be finite and nonnegative")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    data_norm = _finite_norm(data_gradients, name="data gradient")
    direct_norm = _finite_norm(direct_gradients, name="direct gradient")
    alpha = (
        1.0
        if direct_norm == 0.0
        else min(
            1.0,
            float(norm_ratio) * data_norm / (direct_norm + float(epsilon)),
        )
    )
    applied_norm = direct_norm * alpha
    scaled = tuple(
        None if value is None else value.detach().clone().mul(alpha)
        for value in direct_gradients
    )
    return scaled, {
        "data_gradient_norm": data_norm,
        "direct_gradient_norm_raw": direct_norm,
        "direct_gradient_norm_applied": applied_norm,
        "direct_gradient_norm_ratio": float(norm_ratio),
        "direct_gradient_scale": alpha,
        "direct_gradient_clipping_fraction": 1.0 - alpha,
        "data_gradient_reduced": 0.0,
    }


def parameter_update_ratios(
    before: Mapping[str, Sequence[torch.Tensor]],
    after: Mapping[str, Iterable[torch.Tensor]],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, dict[str, float]]:
    """Measure per-module ``||delta theta|| / (||theta|| + epsilon)``."""

    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    result: dict[str, dict[str, float]] = {}
    for module_name, before_values in before.items():
        after_values = tuple(after.get(module_name, ()))
        before_values = tuple(before_values)
        if len(before_values) != len(after_values):
            raise ValueError(f"parameter count changed for module {module_name!r}")
        parameter_norm_sq = 0.0
        update_norm_sq = 0.0
        for old, new in zip(before_values, after_values):
            old64 = old.detach().to(dtype=torch.float64)
            new64 = new.detach().to(dtype=torch.float64)
            if not bool(torch.isfinite(old64).all().item()) or not bool(
                torch.isfinite(new64).all().item()
            ):
                raise FloatingPointError(
                    f"non-finite parameter in module {module_name!r}"
                )
            parameter_norm_sq += float(torch.sum(old64 * old64).item())
            delta = new64 - old64
            update_norm_sq += float(torch.sum(delta * delta).item())
        parameter_norm = math.sqrt(max(parameter_norm_sq, 0.0))
        update_norm = math.sqrt(max(update_norm_sq, 0.0))
        result[module_name] = {
            "module_parameter_norm": parameter_norm,
            "proposed_update_norm": update_norm,
            "update_parameter_ratio": update_norm / (parameter_norm + epsilon),
        }
    return result


def effective_rank(values: torch.Tensor, *, epsilon: float = 1.0e-12) -> float:
    """Return the entropy effective rank of row-wise representations.

    Rows are observations and columns are features.  The covariance spectrum
    is computed in FP64; negative round-off eigenvalues are discarded.  A
    single observation or a zero-variance representation has rank zero.
    """

    if not torch.is_tensor(values) or values.ndim < 2:
        raise ValueError("effective_rank expects a tensor with observations and features")
    if epsilon <= 0.0 or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite and positive")
    matrix = values.detach().reshape(values.shape[0], -1).to(dtype=torch.float64)
    if matrix.shape[0] < 2 or not bool(torch.isfinite(matrix).all().item()):
        return 0.0 if matrix.shape[0] < 2 else float("nan")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).matmul(centered) / float(matrix.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = float(eigenvalues.sum().item())
    if total <= epsilon:
        return 0.0
    probabilities = eigenvalues / total
    probabilities = probabilities[probabilities > epsilon]
    entropy = float((-probabilities * probabilities.log()).sum().item())
    return float(math.exp(entropy))


def validate_seed_separation(
    watchdog_seeds: Sequence[int],
    sentinel_seeds: Sequence[int],
    promotion_seeds: Sequence[int] = (),
) -> None:
    """Reject threshold-tuning and evaluation seed overlap."""

    groups = {
        "watchdog": set(int(seed) for seed in watchdog_seeds),
        "sentinel": set(int(seed) for seed in sentinel_seeds),
        "promotion": set(int(seed) for seed in promotion_seeds),
    }
    if len(groups["watchdog"]) != len(tuple(watchdog_seeds)):
        raise ValueError("watchdog seeds must be unique")
    for first_name, first in groups.items():
        for second_name, second in groups.items():
            if first_name >= second_name:
                continue
            overlap = sorted(first & second)
            if overlap:
                raise ValueError(
                    f"{first_name} and {second_name} seeds overlap: {overlap}"
                )


def _clone_state(value: Any, *, device: torch.device | str | None = None) -> Any:
    if torch.is_tensor(value):
        if device is None:
            return value.detach().clone()
        return value.detach().to(device=device).clone()
    if isinstance(value, dict):
        return {key: _clone_state(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item, device=device) for item in value)
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and available CUDA RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class TransactionalOptimizerStep:
    """Exact rollback boundary for one risky optimizer update.

    ``parameter_modules`` maps stable module names to parameter iterables.  A
    transaction captures all parameters, optimizer state, optional scheduler
    and EMA state, plus RNG state.  ``run`` permits at most one retry and calls
    ``retry`` after restoring the complete pre-step state.
    """

    optimizer: torch.optim.Optimizer
    parameter_modules: Mapping[str, Iterable[torch.nn.Parameter]]
    scheduler: Optional[Any] = None
    ema_model: Optional[torch.nn.Module] = None
    snapshot_device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self._parameters = {
            name: tuple(parameters)
            for name, parameters in self.parameter_modules.items()
        }
        self._captured = False
        self._committed = False

    def capture(self) -> None:
        self._parameter_state = {
            name: tuple(
                parameter.detach().to(device=self.snapshot_device).clone()
                for parameter in parameters
            )
            for name, parameters in self._parameters.items()
        }
        self._optimizer_state = _clone_state(
            self.optimizer.state_dict(), device=self.snapshot_device
        )
        self._scheduler_state = (
            _clone_state(self.scheduler.state_dict())
            if self.scheduler is not None
            else None
        )
        self._ema_state = (
            {
                key: value.detach().to(device=self.snapshot_device).clone()
                if torch.is_tensor(value)
                else _clone_state(value)
                for key, value in self.ema_model.state_dict().items()
            }
            if self.ema_model is not None
            else None
        )
        self._rng_state = capture_rng_state()
        self._captured = True
        self._committed = False

    def _require_capture(self) -> None:
        if not self._captured:
            raise RuntimeError("transaction must be captured before use")

    def rollback(self) -> None:
        self._require_capture()
        for name, parameters in self._parameters.items():
            for parameter, previous in zip(parameters, self._parameter_state[name]):
                parameter.data.copy_(previous)
        self.optimizer.load_state_dict(_clone_state(self._optimizer_state))
        if self.scheduler is not None and self._scheduler_state is not None:
            self.scheduler.load_state_dict(_clone_state(self._scheduler_state))
        if self.ema_model is not None and self._ema_state is not None:
            self.ema_model.load_state_dict(_clone_state(self._ema_state))
        restore_rng_state(self._rng_state)
        self._committed = False

    def commit(self) -> None:
        self._require_capture()
        self._committed = True

    def run(
        self,
        step: Callable[[], None],
        healthy: Callable[[], bool],
        *,
        retry: Optional[Callable[[], None]] = None,
    ) -> int:
        """Run the step and return attempt count; retry at most once."""

        self._require_capture()
        for attempt in range(2):
            if attempt:
                self.rollback()
                if retry is None:
                    raise RuntimeError("transaction failed and no retry was provided")
                retry()
            try:
                step()
            except Exception:
                self.rollback()
                raise
            if healthy():
                self.commit()
                return attempt + 1
        self.rollback()
        raise RuntimeError("transaction rejected after exactly one retry")
