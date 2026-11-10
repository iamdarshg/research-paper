"""Regression tests for the frozen non-trivial diversity/novelty gate."""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
for directory in (os.path.join(ROOT, "CLI"), os.path.join(ROOT, "scripts")):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from diversity_novelty_gate import (  # noqa: E402
    GATE_SPEC,
    GATE_SPEC_HASH,
    diversity_metrics,
    novelty_fraction,
    sample_report,
)


N = 32
H = N // 2


def _plus_cross() -> np.ndarray:
    """The old-screen-passing equal-arm cross used as a frozen negative."""
    a = np.zeros((N, N, N), dtype=np.float32)
    # Two 3-D rods, centred on the bilateral plane, with no taper/fin/wing.
    a[H - 1 : H + 2, H - 10 : H + 11, H - 1 : H + 2] = 1.0
    a[H - 1 : H + 2, H - 1 : H + 2, H - 10 : H + 11] = 1.0
    return a


def _three_box_primitive() -> np.ndarray:
    """Axis-aligned body + slab wing + rectangular fin negative."""
    a = np.zeros((N, N, N), dtype=np.float32)
    a[15:17, 14:18, 4:28] = 1.0
    a[15:17, 3:29, 10:18] = 1.0
    a[16:23, 15:17, 23:28] = 1.0
    return a


def _crm_records():
    manifest = os.path.join(ROOT, "docs", "dataset", "nasa_crm_whole_aircraft_manifest.jsonl")
    if not os.path.exists(manifest):
        return []
    base = os.path.dirname(manifest)
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8") if line.strip()]
    return [np.load(os.path.join(base, row["geometry_path"])) for row in rows]


def test_gate_spec_is_frozen_and_content_addressed():
    expected = hashlib.sha256(
        json.dumps(GATE_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert GATE_SPEC_HASH == expected
    assert GATE_SPEC["version"] == "diversity-novelty-v1"
    assert GATE_SPEC["novelty_min_fraction"] > 0.0
    assert GATE_SPEC["diversity_min_distinct"] >= 5


def test_plus_and_three_box_fail_new_gate_even_when_old_gate_passes():
    for negative in (_plus_cross(), _three_box_primitive()):
        report = sample_report(negative)
        assert report["old_gate_pass"] is True
        assert report["structural"]["pass"] is False
        assert report["new_gate_pass"] is False


def test_constant_training_union_fails_novelty_gate():
    records = _crm_records()
    assert records
    union = np.any(np.stack([record > 0.5 for record in records]), axis=0).astype(np.float32)
    report = sample_report(union, union)
    assert report["old_gate_pass"] is True
    assert report["novel_voxel_fraction"] == 0.0
    assert report["new_gate_pass"] is False


def test_novelty_and_diversity_metrics_are_not_one_voxel_screening():
    a = np.zeros((32, 32, 32), dtype=np.float32)
    b = a.copy()
    # Aircraft-scale occupancy: ~380 voxels. One voxel difference is ~0.3% IoU.
    a[14:18, 14:18, 2:30] = 1.0
    b[14:18, 14:18, 2:30] = 1.0
    b[15, 15, 3] = 0.0
    metrics = diversity_metrics([a, b])
    assert metrics["distinct_array_count"] == 2
    assert metrics["nontrivial_pair_count"] == 0  # one-voxel change is not diversity
    assert novelty_fraction(a, a) == 0.0


def test_crm_ground_truth_remains_old_and_structural_valid():
    records = _crm_records()
    assert len(records) == 15
    for record in records:
        report = sample_report(record)
        assert report["old_gate_pass"], report
        assert report["structural"]["pass"], report
