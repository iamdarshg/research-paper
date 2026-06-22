#!/usr/bin/env python3
"""Merge grounded aircraft manifests and add deterministic flight-path profiles."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def _resolve_geometry_path(record: Dict[str, Any], source_manifest: Path) -> Path:
    geometry_path = record.get("geometry_path")
    if not geometry_path:
        artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
        geometry_path = artifacts.get("voxel_path")
    if not geometry_path:
        raise ValueError(f"Record {record.get('source_id') or record.get('sample_id')} has no geometry_path")
    path = Path(str(geometry_path))
    if not path.is_absolute():
        path = source_manifest.parent / path
    return path.resolve()


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def _design_spec_value(design_spec: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        value = design_spec.get(name)
        if value is not None:
            return float(value)
    return float(default)


def _build_flight_path(record: Dict[str, Any]) -> Dict[str, Any]:
    design_spec = record.get("design_spec") if isinstance(record.get("design_spec"), dict) else {}
    target_speed = _design_spec_value(design_spec, "target_speed_mps", "target_speed", default=30.0)
    takeoff_max = _design_spec_value(design_spec, "takeoff_distance_max_m", default=500.0)
    takeoff_min = _design_spec_value(design_spec, "takeoff_distance_min_m", default=max(25.0, takeoff_max * 0.45))
    turn_rate = _design_spec_value(design_spec, "turn_rate_min_deg_s", default=6.0)
    payload_max = _design_spec_value(design_spec, "payload_mass_max_g", default=0.0)
    configuration = str(record.get("configuration", "")).lower()
    source_id = str(record.get("sample_id") or record.get("source_id") or "unknown")

    if "landing" in configuration:
        terminal_segment = "approach_and_landing"
        cruise_factor = 0.78
    elif "takeoff" in configuration:
        terminal_segment = "takeoff_climbout"
        cruise_factor = 0.86
    elif "cruise" in configuration:
        terminal_segment = "cruise_trim"
        cruise_factor = 1.00
    else:
        terminal_segment = "mission_segment"
        cruise_factor = 0.92

    climb_speed = max(1.0, target_speed * 0.72)
    cruise_speed = max(1.0, target_speed * cruise_factor)
    maneuver_speed = max(1.0, min(cruise_speed, target_speed * 0.82))
    descent_speed = max(1.0, target_speed * 0.64)

    return {
        "profile_id": source_id,
        "segments": [
            {
                "name": "ground_roll",
                "duration_s": round(max(4.0, takeoff_max / max(climb_speed, 1.0)), 3),
                "start_speed_mps": 0.0,
                "end_speed_mps": round(climb_speed, 3),
                "distance_m": round(max(takeoff_min, takeoff_max), 3),
            },
            {
                "name": "initial_climb",
                "duration_s": 120.0,
                "target_speed_mps": round(climb_speed, 3),
                "target_climb_gradient": 0.06,
            },
            {
                "name": terminal_segment,
                "duration_s": 900.0 if cruise_speed > 60.0 else 300.0,
                "target_speed_mps": round(cruise_speed, 3),
                "payload_mass_g": round(payload_max, 3),
            },
            {
                "name": "coordinated_turn_check",
                "duration_s": 45.0,
                "target_speed_mps": round(maneuver_speed, 3),
                "min_turn_rate_deg_s": round(turn_rate, 3),
            },
            {
                "name": "descent_or_recovery",
                "duration_s": 120.0,
                "target_speed_mps": round(descent_speed, 3),
            },
        ],
        "provenance": (
            "deterministic conditioning profile derived from manifest design_spec fields; "
            "not measured trajectory, telemetry, or flight-test ground truth"
        ),
    }


def _geometry_shape(path: Path) -> List[int] | None:
    try:
        return [int(value) for value in np.load(path, mmap_mode="r").shape]
    except Exception:
        return None


def build_flight_path_manifest(
    manifests: Iterable[Path],
    *,
    output_manifest: Path,
    report_path: Path,
    run_id: str,
) -> Dict[str, Any]:
    manifest_paths = [Path(path).resolve() for path in manifests]
    output_manifest = Path(output_manifest).resolve()
    report_path = Path(report_path).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    output_records: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    missing_geometry: List[str] = []
    grid_shapes: Dict[str, int] = {}

    for source_manifest in manifest_paths:
        records = _load_jsonl(source_manifest)
        source_counts[str(source_manifest)] = len(records)
        for index, record in enumerate(records):
            geometry_path = _resolve_geometry_path(record, source_manifest)
            if not geometry_path.exists():
                missing_geometry.append(str(geometry_path))
            shape = _geometry_shape(geometry_path)
            if shape is not None:
                grid_shapes[str(shape)] = grid_shapes.get(str(shape), 0) + 1

            merged = dict(record)
            merged["geometry_path"] = _relative_path(geometry_path, output_manifest.parent)
            merged["source_manifest_path"] = str(source_manifest)
            merged["source_manifest_record_index"] = index
            merged["flight_path"] = _build_flight_path(record)
            output_records.append(merged)

    rendered = "".join(json.dumps(record, sort_keys=True) + "\n" for record in output_records)
    output_manifest.write_text(rendered, encoding="utf-8")

    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(output_records),
        "source_counts": source_counts,
        "grid_shapes": grid_shapes,
        "missing_geometry": missing_geometry,
        "output_manifest": str(output_manifest),
        "claim_boundary": (
            "Combined manifest for training and smoke evidence. Flight paths are deterministic conditioning "
            "profiles derived from source design_spec fields, not measured flight trajectories."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, help="Input JSONL manifest. Repeatable.")
    parser.add_argument("--output-manifest", required=True, help="Output merged JSONL manifest.")
    parser.add_argument("--report", required=True, help="Output JSON report.")
    parser.add_argument("--run-id", default="aircraft-flight-path-manifest")
    args = parser.parse_args()

    report = build_flight_path_manifest(
        [Path(value) for value in args.manifest],
        output_manifest=Path(args.output_manifest),
        report_path=Path(args.report),
        run_id=args.run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not report["missing_geometry"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
