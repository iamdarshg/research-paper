"""Frozen non-trivial evaluation gate for voxel generator evidence.

This module is deliberately independent of model outputs.  The gate combines the
repository's existing validity heuristic with pre-declared novelty, diversity, and
structural-articulation checks.  It is an evidence gate, not a physical or
 aerodynamic validator.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterable, Sequence

import numpy as np

from aircraft_validity import evaluate_aircraft_validity, canonicalize_aircraft_voxels

GATE_SPEC = {
    "version": "diversity-novelty-v1",
    "novelty_min_fraction": 0.01,
    "diversity_min_distinct": 5,
    "diversity_nontrivial_iou_max": 0.98,
    "condition_min_iou_spread": 0.01,
    # At least two independent, geometric articulation signals are required.
    # These are declared from aircraft semantics, not fitted to generated samples.
    "articulation_min_signals": 2,
    "articulation_signal_thresholds": {
        "fuselage_taper_voxels": 2,
        "wing_chord_variation_voxels": 2,
        "dihedral_variation_voxels": 1,
    },
}
GATE_SPEC_HASH = hashlib.sha256(
    json.dumps(GATE_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _binary(x: Any) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 3:
        raise ValueError(f"expected 3-D voxel array, got {a.shape}")
    return (a > 0.5).astype(np.uint8)


def array_hash(x: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(_binary(x)).tobytes()).hexdigest()


def iou(a: Any, b: Any) -> float:
    aa, bb = _binary(a).astype(bool), _binary(b).astype(bool)
    inter = np.logical_and(aa, bb).sum()
    union = np.logical_or(aa, bb).sum()
    return float(inter / union) if union else 1.0


def novelty_fraction(sample: Any, training_union: Any) -> float:
    s, u = _binary(sample), _binary(training_union)
    occupied = s.astype(bool)
    return float(np.logical_and(occupied, ~u.astype(bool)).sum() / max(occupied.sum(), 1))


def pairwise_ious(samples: Sequence[Any]) -> list[float]:
    return [iou(a, b) for a, b in itertools.combinations(samples, 2)]


def diversity_metrics(samples: Sequence[Any]) -> dict[str, Any]:
    values = list(samples)
    pairwise = pairwise_ious(values)
    nontrivial = sum(1 for value in pairwise if value <= GATE_SPEC["diversity_nontrivial_iou_max"])
    return {
        "sample_count": len(values),
        "distinct_array_count": len({array_hash(v) for v in values}),
        "nontrivial_pair_count": nontrivial,
        "pairwise_iou_mean": float(np.mean(pairwise)) if pairwise else 1.0,
        "pairwise_iou_min": float(np.min(pairwise)) if pairwise else 1.0,
        "pairwise_iou_max": float(np.max(pairwise)) if pairwise else 1.0,
        "pairwise_iou_spread": float(np.max(pairwise) - np.min(pairwise)) if pairwise else 0.0,
    }


def _articulation_signals(grid: np.ndarray) -> dict[str, float]:
    """Measure taper/sweep/dihedral variation after canonical orientation.

    These low-resolution shape signals reject a cross and a stack of axis-aligned
    boxes while retaining tapered, swept, or dihedral aircraft silhouettes.
    """
    canonical, _ = canonicalize_aircraft_voxels(_binary(grid))
    a = canonical.numpy().astype(bool)
    coords = np.argwhere(a)
    if not len(coords):
        return {"fuselage_taper_voxels": 0.0, "wing_chord_variation_voxels": 0.0, "dihedral_variation_voxels": 0.0}
    z, y, x = a.shape
    cy0, cy1 = int(y * 0.42), max(int(y * 0.42) + 1, int(y * 0.58))
    xs = np.where(a.any(axis=(0, 1)))[0]
    center_widths = []
    for xi in xs:
        section = a[:, cy0:cy1, xi]
        center_widths.append(int(section.sum()))
    fuselage_taper = float(max(center_widths) - min(center_widths)) if center_widths else 0.0

    ys = np.where(a.any(axis=(0, 2)))[0]
    outer = [yi for yi in ys if yi < int(y * 0.35) or yi >= int(y * 0.65)]
    chords = []
    z_centers = []
    for yi in outer:
        section = np.argwhere(a[:, yi, :])
        if len(section):
            chords.append(int(section[:, 1].max() - section[:, 1].min() + 1))
            z_centers.append(float(section[:, 0].mean()))
    chord_variation = float(max(chords) - min(chords)) if chords else 0.0
    dihedral = float(max(z_centers) - min(z_centers)) if z_centers else 0.0
    return {
        "fuselage_taper_voxels": fuselage_taper,
        "wing_chord_variation_voxels": chord_variation,
        "dihedral_variation_voxels": dihedral,
    }


def structural_report(sample: Any) -> dict[str, Any]:
    signals = _articulation_signals(sample)
    thresholds = GATE_SPEC["articulation_signal_thresholds"]
    passed = [name for name, value in signals.items() if value >= thresholds[name]]
    return {"signals": signals, "passed_signal_names": passed,
            "pass": len(passed) >= GATE_SPEC["articulation_min_signals"]}


def sample_report(sample: Any, training_union: Any | None = None) -> dict[str, Any]:
    validity = evaluate_aircraft_validity(_binary(sample), canonicalize=True)
    report = {
        "array_sha256": array_hash(sample),
        "old_gate_pass": validity["status"] == "pass",
        "old_gate_failed_checks": validity["failed_checks"],
        "structural": structural_report(sample),
    }
    if training_union is not None:
        report["novel_voxel_fraction"] = novelty_fraction(sample, training_union)
    report["new_gate_pass"] = bool(report["old_gate_pass"] and report["structural"]["pass"] and
                                    (training_union is None or report["novel_voxel_fraction"] >= GATE_SPEC["novelty_min_fraction"]))
    return report


def corpus_report(samples: Sequence[Any], training_union: Any, conditions: Sequence[Any] | None = None) -> dict[str, Any]:
    values = list(samples)
    novelty = [novelty_fraction(v, training_union) for v in values]
    diversity = diversity_metrics(values)
    per_condition: dict[str, Any] = {}
    if conditions is not None:
        for condition, sample in zip(conditions, values):
            key = str(condition)
            per_condition.setdefault(key, []).append(sample)
        per_condition = {key: diversity_metrics(group) for key, group in per_condition.items()}
    old = [sample_report(v)["old_gate_pass"] for v in values]
    structural = [sample_report(v)["structural"]["pass"] for v in values]
    return {
        "gate_spec": GATE_SPEC,
        "gate_spec_sha256": GATE_SPEC_HASH,
        "sample_count": len(values),
        "old_gate_pass_count": int(sum(old)),
        "old_gate_pass_rate": float(np.mean(old)) if old else 0.0,
        "structural_pass_count": int(sum(structural)),
        "novel_voxel_fraction_mean": float(np.mean(novelty)) if novelty else 0.0,
        "novel_voxel_fraction_min": float(np.min(novelty)) if novelty else 0.0,
        "diversity": diversity,
        "per_condition": per_condition,
        "new_gate_pass": bool(values and all(old) and all(structural) and
                               np.mean(novelty) >= GATE_SPEC["novelty_min_fraction"] and
                               diversity["distinct_array_count"] >= GATE_SPEC["diversity_min_distinct"] and
                               diversity["nontrivial_pair_count"] > 0),
    }
