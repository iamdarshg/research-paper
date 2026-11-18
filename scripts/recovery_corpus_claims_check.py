#!/usr/bin/env python3
"""Re-derive the corpus-only figures quoted in the generator-recovery subsection.

The recovery checkpoint and its generated arrays are not in the repository, so
figures that depend on them cannot be re-derived. The figures below depend only
on the committed NASA CRM corpus and the unmodified validity screen:

* coverage of each held-out record by the voxel-wise union of the six training
  records (fraction of the record's occupied voxels inside the union);
* number of distinct voxel arrays among the nine held-out records;
* the screen verdict for the model-free training-union constant and the two
  non-aircraft primitives (same arrays as scripts/diversity_novelty_eval.py).

Usage: python scripts/recovery_corpus_claims_check.py [--output report.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CLI"))

from aircraft_validity import evaluate_aircraft_validity  # noqa: E402

MANIFEST = ROOT / "docs" / "dataset" / "nasa_crm_whole_aircraft_manifest.jsonl"


def _load(record: dict) -> np.ndarray:
    array = np.load(MANIFEST.parent / record["geometry_path"])
    if array.shape != (32, 32, 32):
        raise ValueError(f"{record['sample_id']}: expected 32^3, got {array.shape}")
    return array > 0.5


def _primitives(n: int = 32) -> dict[str, np.ndarray]:
    h = n // 2
    plus = np.zeros((n, n, n), dtype=np.float32)
    plus[h - 1 : h + 2, h - 10 : h + 11, h - 1 : h + 2] = 1.0
    plus[h - 1 : h + 2, h - 1 : h + 2, h - 10 : h + 11] = 1.0
    three_box = np.zeros((n, n, n), dtype=np.float32)
    three_box[15:17, 14:18, 4:28] = 1.0
    three_box[15:17, 3:29, 10:18] = 1.0
    three_box[16:23, 15:17, 23:28] = 1.0
    return {"plus_cross": plus, "three_box": three_box}


def _screen(array: np.ndarray) -> dict:
    report = evaluate_aircraft_validity(array, canonicalize=True)
    checks = report.get("checks", {})
    failed = [
        name
        for name, value in checks.items()
        if not (value.get("passed") if isinstance(value, dict) else bool(value))
    ]
    return {"status": report.get("status"), "checks": len(checks), "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    records = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    train = [r for r in records if r["split"] in ("test", "val")]
    holdout = [r for r in records if r["split"] == "holdout"]
    union = np.zeros((32, 32, 32), dtype=bool)
    for record in train:
        union |= _load(record)

    rows = []
    distinct = set()
    for record in holdout:
        array = _load(record)
        distinct.add(array.tobytes())
        occupied = int(array.sum())
        rows.append({
            "record": record["sample_id"],
            "occupied_voxels": occupied,
            "coverage_by_train_union": int((array & union).sum()) / occupied,
        })
    coverage = [row["coverage_by_train_union"] for row in rows]
    controls = {"constant_union(train)": _screen(union.astype(np.float32))}
    controls.update({name: _screen(array) for name, array in _primitives().items()})

    report = {
        "manifest": os.path.relpath(MANIFEST, ROOT),
        "train_records": len(train),
        "holdout_records": len(holdout),
        "holdout_distinct_arrays": len(distinct),
        "coverage_min": min(coverage),
        "coverage_max": max(coverage),
        "holdout": rows,
        "negative_controls": controls,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
