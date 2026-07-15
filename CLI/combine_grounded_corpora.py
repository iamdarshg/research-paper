#!/usr/bin/env python3
"""Combine canonical grounded manifests without inventing new training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from validate_manifest import DEFAULT_UNIQUE_GEOMETRY_TARGET, validate_manifest_file


CONDITIONING_FIELDS = (
    "target_speed_mps",
    "wingspan_limit_m",
    "thrust_to_weight_min",
    "turn_rate_min_deg_s",
    "required_static_thrust_n",
    "engine_diameter_mm",
    "engine_length_mm",
    "engine_count_min",
    "engine_count_max",
    "payload_mass_min_g",
    "payload_mass_max_g",
    "takeoff_distance_min_m",
    "takeoff_distance_max_m",
    "wall_thickness_min_mm",
    "wall_thickness_max_mm",
    "part_count_min",
    "part_count_max",
    "manufacturing_method",
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def _resolve_geometry(record: Dict[str, Any], manifest_path: Path) -> Path:
    geometry_path = record.get("geometry_path")
    if not geometry_path:
        raise ValueError(f"Record {record.get('source_id')} has no geometry_path")
    candidate = Path(str(geometry_path))
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = Path(os.path.abspath(candidate))
    if not candidate.exists():
        raise FileNotFoundError(f"Record {record.get('source_id')} geometry does not exist: {candidate}")
    return candidate


def _canonical_content_hash(path: Path) -> str:
    voxels = np.load(path, mmap_mode="r")
    if voxels.ndim != 3:
        raise ValueError(f"Canonical voxel artifact must be 3D, got {tuple(voxels.shape)} at {path}")
    canonical = (np.asarray(voxels) > 0.5).astype(np.uint8, copy=False)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _atomic_write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        temporary_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def combine_manifests(
    manifests: Sequence[Path],
    *,
    output_manifest: Path,
    output_report: Path,
    unique_geometry_target: int = DEFAULT_UNIQUE_GEOMETRY_TARGET,
) -> Dict[str, Any]:
    output_manifest = Path(output_manifest)
    source_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    kept: List[Dict[str, Any]] = []
    unique_content_hashes: set[str] = set()
    source_ids: set[str] = set()
    grid_shapes: Counter[str] = Counter()

    for raw_manifest in manifests:
        manifest_path = Path(raw_manifest)
        for index, raw_record in enumerate(_load_jsonl(manifest_path)):
            source_counts[str(manifest_path)] += 1
            if not isinstance(raw_record.get("canonicalization"), dict):
                raise ValueError(
                    f"{manifest_path}:{index + 1} lacks canonicalization metadata; "
                    "run the canonical corpus filter before combining."
                )
            geometry_path = _resolve_geometry(raw_record, manifest_path)
            content_hash = _canonical_content_hash(geometry_path)
            source_id = str(raw_record.get("source_id") or raw_record.get("sample_id") or "")
            if not source_id:
                raise ValueError(f"{manifest_path}:{index + 1} lacks source_id")
            if source_id in source_ids:
                dropped_counts["duplicate_source_id"] += 1
                continue
            if content_hash in unique_content_hashes:
                dropped_counts["duplicate_canonical_geometry"] += 1
                continue

            output_record = dict(raw_record)
            output_record["geometry_path"] = os.path.relpath(
                os.path.abspath(geometry_path),
                os.path.abspath(output_manifest.parent),
            ).replace("\\", "/")
            output_record["canonical_content_sha256"] = content_hash
            output_record["source_manifest_path"] = str(manifest_path.resolve())
            output_record["source_manifest_record_index"] = index
            # Geometry is admitted from the source record.  Conditioning labels
            # without field-level source evidence are deliberately removed so a
            # mixed-source corpus cannot silently train on inferred missions.
            output_record["source_manifest_design_spec"] = output_record.get("design_spec")
            output_record["design_spec"] = {
                field_name: None for field_name in CONDITIONING_FIELDS
            }
            output_record["design_spec_availability"] = {
                field_name: False for field_name in CONDITIONING_FIELDS
            }
            output_record["design_spec_provenance"] = {
                field_name: "not_used_without_field_level_source_evidence"
                for field_name in CONDITIONING_FIELDS
            }
            output_record["conditioning_mode"] = "unconditioned_source_metadata_only"
            kept.append(output_record)
            source_ids.add(source_id)
            unique_content_hashes.add(content_hash)
            grid_shapes[str(tuple(np.load(geometry_path, mmap_mode="r").shape))] += 1

    if len(grid_shapes) != 1:
        raise ValueError(f"Combined corpus mixes voxel grid shapes: {dict(grid_shapes)}")
    kept.sort(key=lambda record: str(record["source_id"]))
    _atomic_write_jsonl(output_manifest, kept)
    basic_validation = validate_manifest_file(str(output_manifest), level="basic")
    claim_validation = validate_manifest_file(
        str(output_manifest),
        level="claim-bearing",
        unique_geometry_target=unique_geometry_target,
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_manifest": str(output_manifest.resolve()),
        "source_counts": dict(source_counts),
        "dropped_counts": dict(dropped_counts),
        "record_count": len(kept),
        "unique_canonical_geometry_count": len(unique_content_hashes),
        "grid_shapes": dict(grid_shapes),
        "basic_validation": basic_validation,
        "claim_validation": claim_validation,
        "claim_boundary": "The combined set contains canonical CAD-derived voxel geometry. Source conditioning remains limited to each record's declared availability metadata.",
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Canonical input manifest; repeatable.")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--unique-geometry-target", type=int, default=DEFAULT_UNIQUE_GEOMETRY_TARGET)
    args = parser.parse_args(argv)
    report = combine_manifests(
        [Path(value) for value in args.manifest],
        output_manifest=Path(args.output_manifest),
        output_report=Path(args.output_report),
        unique_geometry_target=args.unique_geometry_target,
    )
    print(json.dumps({key: report[key] for key in ("record_count", "unique_canonical_geometry_count", "claim_validation")}, indent=2, sort_keys=True))
    return 0 if report["claim_validation"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
