"""Helpers for monitoring training convergence and instability."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


def compute_core_loss(epoch_metrics: Dict[str, Any]) -> float:
    return float(
        epoch_metrics.get("mse", 0.0)
        + epoch_metrics.get("geometry_reconstruction", 0.0)
        + epoch_metrics.get("consistency", 0.0)
        + epoch_metrics.get("connectivity", 0.0)
    )


def _rolling_stats(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "drift_per_epoch": 0.0, "min": 0.0, "max": 0.0}

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    cv = float(std / max(abs(mean), 1e-6))
    drift = float(abs(values[-1] - values[0]) / max(len(values) - 1, 1))
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "drift_per_epoch": drift,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_stability(
    history: List[Dict[str, Any]],
    *,
    metric: str = "core_loss",
    window: int = 20,
    convergence_target: float | None = None,
    convergence_cv_threshold: float = 0.08,
    convergence_drift_threshold: float = 0.35,
    oscillation_cv_threshold: float = 0.30,
) -> Dict[str, Any]:
    if not history:
        return {
            "status": "warming_up",
            "metric": metric,
            "window": window,
            "converged": False,
            "oscillating": False,
            "reason": "no history",
        }

    recent = history[-window:]
    if len(recent) < window:
        return {
            "status": "warming_up",
            "metric": metric,
            "window": window,
            "converged": False,
            "oscillating": False,
            "reason": f"need {window} epochs, have {len(recent)}",
        }

    metric_stats = _rolling_stats(item.get(metric, 0.0) for item in recent)
    total_stats = _rolling_stats(item.get("loss", 0.0) for item in recent)
    aero_stats = _rolling_stats(item.get("aerodynamic", 0.0) for item in recent)
    connectivity_stats = _rolling_stats(item.get("connectivity", 0.0) for item in recent)

    converged = (
        metric_stats["cv"] <= convergence_cv_threshold
        and metric_stats["drift_per_epoch"] <= convergence_drift_threshold
        and (
            convergence_target is None
            or metric_stats["mean"] <= convergence_target
        )
    )
    oscillating = total_stats["cv"] >= oscillation_cv_threshold
    aerodynamic_diverging = (
        aero_stats["drift_per_epoch"] > max(1.0, metric_stats["drift_per_epoch"] * 20.0)
        and recent[-1].get("aerodynamic", 0.0) > recent[0].get("aerodynamic", 0.0)
    )

    suspected_root_cause = None
    if oscillating:
        aero_span = aero_stats["max"] - aero_stats["min"]
        core_span = metric_stats["max"] - metric_stats["min"]
        if (
            aero_stats["std"] > max(connectivity_stats["std"], metric_stats["std"])
            and (
                aero_span > max(1.0, core_span * 3.0)
                or aero_stats["cv"] >= max(connectivity_stats["cv"], metric_stats["cv"])
            )
        ):
            suspected_root_cause = "aerodynamic_loss_dominance"
        elif connectivity_stats["cv"] > aero_stats["cv"] and connectivity_stats["cv"] > metric_stats["cv"]:
            suspected_root_cause = "connectivity_penalty_oscillation"
        else:
            suspected_root_cause = "mixed_loss_instability"
    elif aerodynamic_diverging and metric_stats["mean"] <= (
        convergence_target if convergence_target is not None else metric_stats["mean"] * 1.2
    ):
        suspected_root_cause = "aerodynamic_objective_drift"

    return {
        "status": "converged" if converged else ("oscillating" if oscillating else "stable"),
        "metric": metric,
        "window": window,
        "converged": converged,
        "oscillating": oscillating,
        "aerodynamic_diverging": aerodynamic_diverging,
        "convergence_target": convergence_target,
        "convergence_cv_threshold": convergence_cv_threshold,
        "convergence_drift_threshold": convergence_drift_threshold,
        "oscillation_cv_threshold": oscillation_cv_threshold,
        "metric_stats": metric_stats,
        "total_loss_stats": total_stats,
        "aerodynamic_stats": aero_stats,
        "connectivity_stats": connectivity_stats,
        "suspected_root_cause": suspected_root_cause,
    }
