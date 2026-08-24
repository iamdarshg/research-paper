#!/usr/bin/env python3
"""Self-expand the grounded corpus via physically meaningful voxel perturbations.

Transforms (all preserve physical aircraft meaning):
  - yaw90 / yaw180 / yaw270: rotation about the vertical axis
  - mirror_x: left-right flip along the fuselage axis

Each candidate is validated through canonicalize_aircraft_voxels + the
aircraft-validity heuristic gates. Only variants that pass all checks and have
a novel canonical SHA-256 hash are accepted.

Usage:
    python CLI/self_expand_corpus.py         --manifest build/grounded_combined_1k_20260716/manifest.jsonl         --output-dir build/self_expanded_corpus_TIMESTAMP         --transforms yaw90,yaw180,yaw270,mirror_x
"""

from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aircraft_validity import canonicalize_aircraft_voxels


TRANSFORMS = ("yaw90", "yaw180", "yaw270", "mirror_x")


def apply_transform(voxels: np.ndarray, transform: str) -> np.ndarray:
    if transform == "yaw90":
        return np.rot90(voxels, k=1, axes=(0, 2))  # rotate in xz plane
    if transform == "yaw180":
        return np.rot90(voxels, k=2, axes=(0, 2))
    if transform == "yaw270":
        return np.rot90(voxels, k=3, axes=(0, 2))
    if transform == "mirror_x":
        return np.flip(voxels, axis=2).copy()  # left-right flip
    raise ValueError(f"Unknown transform: {transform}")


def canonical_hash(voxels: np.ndarray) -> str:
    canonical = (voxels > 0.5).astype(np.uint8)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_geometry(voxels: np.ndarray) -> bool:
    """Return True if the geometry passes the canonical validity gates."""
    try:
        tensor = torch.from_numpy((voxels > 0.5).astype(np.float32))
        _, canon = canonicalize_aircraft_voxels(tensor)
        metrics = canon.get("metrics", {})
        # Core quality gates (same thresholds as the corpus filter)
        if float(metrics.get("occupancy_ratio", 0)) < 0.01:
            return False
        if float(metrics.get("largest_component_fraction", 0)) < 0.5:
            return False
        if float(metrics.get("symmetry_score", 0)) < 0.3:
            return False
        if float(metrics.get("span_fraction_y", 0)) < 0.05:
            return False
        if float(metrics.get("length_fraction_x", 0)) < 0.10:
            return False
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--transforms", default=",".join(TRANSFORMS),
                    help=f"Comma-separated transforms from: {', '.join(TRANSFORMS)}")
    ap.add_argument("--max-variants-per-source", type=int, default=4,
                    help="Maximum perturbation variants per source geometry.")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxels_dir = output_dir / "voxels"
    voxels_dir.mkdir(exist_ok=True)

    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    for t in transforms:
        if t not in TRANSFORMS:
            print(f"ERROR: unknown transform {t!r}. Valid: {TRANSFORMS}")
            return 1

    # Load source manifest
    source_records = []
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                source_records.append(json.loads(line))

    # Collect existing hashes to avoid duplicates
    existing_hashes: Set[str] = set()
    for rec in source_records:
        h = rec.get("canonical_content_sha256") or rec.get("voxel_sha256")
        if h:
            existing_hashes.add(h)

    expanded_records: List[Dict[str, Any]] = []
    stats = {"source": len(source_records), "candidates": 0, "accepted": 0,
             "rejected_invalid": 0, "rejected_duplicate": 0}

    base_dir = Path(args.manifest).parent

    for idx, rec in enumerate(source_records):
        source_id = rec.get("source_id", f"unknown_{idx}")
        gp = rec.get("geometry_path")
        if not gp:
            continue
        gpath = Path(gp)
        if not gpath.is_absolute():
            gpath = base_dir / gpath
        if not gpath.exists():
            continue

        voxels = np.load(str(gpath))
        variants_made = 0

        for tf_name in transforms:
            if variants_made >= args.max_variants_per_source:
                break

            transformed = apply_transform(voxels, tf_name)
            chash = canonical_hash(transformed)
            stats["candidates"] += 1

            if chash in existing_hashes:
                stats["rejected_duplicate"] += 1
                continue

            if not validate_geometry(transformed):
                stats["rejected_invalid"] += 1
                continue

            # Accept: save voxel file and create record
            variant_id = f"selfexp:{tf_name}:{source_id}"
            voxel_filename = f"{variant_id.replace(':', '_')}.npy"
            np.save(str(voxels_dir / voxel_filename), (transformed > 0.5).astype(np.uint8))

            new_rec = {
                "source_id": variant_id,
                "source_type": "self_expanded",
                "parent_source_id": source_id,
                "transform": tf_name,
                "canonical_content_sha256": chash,
                "voxel_sha256": chash,
                "geometry_path": str(Path("voxels") / voxel_filename).replace("\\", "/"),
                "conditioning_mode": rec.get("conditioning_mode", "unconditioned_source_metadata_only"),
                "split": "train",
                "provenance": {
                    "parent_manifest": str(Path(args.manifest).resolve()),
                    "parent_record_index": idx,
                    "expansion_timestamp": datetime.now(timezone.utc).isoformat(),
                    "transform": tf_name,
                },
            }
            expanded_records.append(new_rec)
            existing_hashes.add(chash)
            stats["accepted"] += 1
            variants_made += 1

    # Write manifest and report atomically
    manifest_path = output_dir / "manifest.jsonl"
    report_path = output_dir / "report.json"

    tmp_manifest = manifest_path.with_suffix(".tmp")
    with tmp_manifest.open("w", encoding="utf-8") as f:
        for record in expanded_records:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(tmp_manifest, manifest_path)

    stats["expanded_total"] = len(expanded_records)
    stats["created_at"] = datetime.now(timezone.utc).isoformat()
    stats["transforms_applied"] = transforms
    stats["claim_boundary"] = (
        "Self-expanded geometries are deterministic spatial transforms of "
        "validated parent shapes. They are NOT independent aircraft designs."
    )
    report_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
