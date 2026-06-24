#!/usr/bin/env python3
"""Attach whole-aircraft geometry proxies to FAA/OpenSky flight-case records."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


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


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")


def _resolve_record_path(record: Mapping[str, Any], source_manifest: Path) -> Path | None:
    ref = record.get("geometry_path") or record.get("stl_path")
    if not ref and isinstance(record.get("artifacts"), dict):
        ref = record["artifacts"].get("voxel_path") or record["artifacts"].get("stl_path")
    if not ref:
        return None
    path = Path(str(ref))
    if not path.is_absolute():
        path = source_manifest.parent / path
    return path.resolve()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _record_speed(record: Mapping[str, Any]) -> float:
    design_spec = record.get("design_spec") if isinstance(record.get("design_spec"), dict) else {}
    regime = record.get("flight_regime") if isinstance(record.get("flight_regime"), dict) else {}
    return _number(
        design_spec.get("target_speed_mps")
        or regime.get("observed_groundspeed_mps")
        or regime.get("route_average_speed_mps")
        or regime.get("approach_speed_mps"),
        0.0,
    )


def _record_span(record: Mapping[str, Any]) -> float:
    design_spec = record.get("design_spec") if isinstance(record.get("design_spec"), dict) else {}
    characteristics = record.get("faa_characteristics") if isinstance(record.get("faa_characteristics"), dict) else {}
    return _number(design_spec.get("wingspan_limit_m") or characteristics.get("wingspan_m"), 0.0)


def _is_airfoil_record(record: Mapping[str, Any]) -> bool:
    family = str(record.get("design_family", "")).lower()
    source_id = str(record.get("source_id") or record.get("sample_id") or "").lower()
    provenance = str(record.get("geometry_provenance", "")).lower()
    return (
        "airfoil_section" in family
        or family.startswith("airfoil")
        or source_id.startswith("naca_")
        or "airfoil-section" in provenance
        or "airfoil section" in provenance
    )


def _eligible_geometries(geometry_records: Sequence[Mapping[str, Any]], geometry_manifest: Path) -> tuple[List[Dict[str, Any]], int]:
    eligible: List[Dict[str, Any]] = []
    excluded_count = 0
    for index, record in enumerate(geometry_records):
        path = _resolve_record_path(record, geometry_manifest)
        if _is_airfoil_record(record):
            excluded_count += 1
            continue
        if path is None or not path.exists():
            excluded_count += 1
            continue
        eligible.append(
            {
                "record": dict(record),
                "record_index": index,
                "path": path,
                "source_id": str(record.get("source_id") or record.get("sample_id") or f"geometry-{index}"),
                "speed": _record_speed(record),
                "span": _record_span(record),
            }
        )
    return eligible, excluded_count


def _score_geometry(case: Mapping[str, Any], geometry: Mapping[str, Any]) -> float:
    case_speed = _record_speed(case)
    case_span = _record_span(case)
    speed = _number(geometry.get("speed"), 0.0)
    span = _number(geometry.get("span"), 0.0)
    speed_score = abs(math.log1p(case_speed) - math.log1p(speed)) if case_speed and speed else 1.0
    span_score = abs(math.log1p(case_span) - math.log1p(span)) if case_span and span else 1.0
    return speed_score + 0.5 * span_score


def _select_geometry(case: Mapping[str, Any], geometries: Sequence[Mapping[str, Any]], case_index: int) -> Mapping[str, Any]:
    scored = sorted(
        ((_score_geometry(case, geometry), str(geometry.get("source_id", "")), geometry) for geometry in geometries),
        key=lambda item: (item[0], item[1]),
    )
    return scored[case_index % len(scored)][2]


def _split_for_index(index: int) -> str:
    remainder = index % 10
    if remainder == 0:
        return "holdout"
    if remainder == 1:
        return "validation"
    return "train"


def _merge_record(
    case_record: Mapping[str, Any],
    geometry: Mapping[str, Any],
    *,
    output_dir: Path,
    case_index: int,
) -> Dict[str, Any]:
    base_record = geometry["record"]
    merged = dict(case_record)
    merged["source_id"] = f"faa-geometry-case-{case_index:06d}"
    merged["sample_id"] = merged["source_id"]
    merged["geometry_path"] = _relative_path(Path(geometry["path"]), output_dir)
    merged["geometry_provenance"] = (
        f"Whole-aircraft geometry proxy assigned from {geometry['source_id']}. "
        f"Base provenance: {base_record.get('geometry_provenance', 'unspecified')}"
    )
    merged["geometry_association"] = {
        "method": "deterministic_diversified_proxy",
        "base_geometry_source_id": geometry["source_id"],
        "base_geometry_record_index": geometry["record_index"],
        "claim_boundary": (
            "Geometry is a reusable whole-aircraft proxy paired with an FAA/OpenSky flight-regime case. "
            "It is not exact CAD for the observed aircraft registration or route."
        ),
    }
    merged["preprocessing_version"] = str(base_record.get("preprocessing_version") or "whole-aircraft-geometry-proxy-v1")
    merged["units"] = str(base_record.get("units") or "m")
    merged["design_family"] = "faa_opensky_geometry_proxy_whole_aircraft"
    merged["split"] = _split_for_index(case_index)
    return merged


def build_geometry_case_manifest(
    *,
    flight_case_manifest: Path,
    geometry_manifest: Path,
    output_manifest: Path,
    report_path: Path,
    target_records: int,
    run_id: str,
) -> Dict[str, Any]:
    flight_case_manifest = Path(flight_case_manifest).resolve()
    geometry_manifest = Path(geometry_manifest).resolve()
    output_manifest = Path(output_manifest).resolve()
    report_path = Path(report_path).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    case_records = _load_jsonl(flight_case_manifest)
    geometry_records = _load_jsonl(geometry_manifest)
    geometries, excluded_count = _eligible_geometries(geometry_records, geometry_manifest)
    if not geometries:
        raise ValueError("No eligible non-airfoil geometry records were found.")
    if target_records <= 0:
        raise ValueError("target_records must be positive")

    selected_cases = case_records[:target_records]
    output_records = [
        _merge_record(
            case_record,
            _select_geometry(case_record, geometries, index),
            output_dir=output_manifest.parent,
            case_index=index,
        )
        for index, case_record in enumerate(selected_cases)
    ]
    rendered = "".join(json.dumps(record, sort_keys=True) + "\n" for record in output_records)
    output_manifest.write_text(rendered, encoding="utf-8")

    split_counts: Dict[str, int] = {}
    association_counts: Dict[str, int] = {}
    for record in output_records:
        split = str(record.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
        base_id = str(record["geometry_association"]["base_geometry_source_id"])
        association_counts[base_id] = association_counts.get(base_id, 0) + 1

    report = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "flight_case_manifest": str(flight_case_manifest),
        "geometry_manifest": str(geometry_manifest),
        "output_manifest": str(output_manifest),
        "record_count": len(output_records),
        "target_records": target_records,
        "input_flight_case_count": len(case_records),
        "input_geometry_count": len(geometry_records),
        "eligible_geometry_count": len(geometries),
        "excluded_geometry_count": excluded_count,
        "unique_geometry_associations": len(association_counts),
        "split_counts": dict(sorted(split_counts.items())),
        "claim_boundary": (
            "Every output row has a geometry_path, but the attached geometry is a deterministic whole-aircraft proxy. "
            "The manifest removes airfoil-section sources and does not claim exact aircraft-type CAD."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight-case-manifest", required=True)
    parser.add_argument("--geometry-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--target-records", type=int, default=5000)
    parser.add_argument("--run-id", default="faa-geometry-case-manifest")
    args = parser.parse_args()

    report = build_geometry_case_manifest(
        flight_case_manifest=Path(args.flight_case_manifest),
        geometry_manifest=Path(args.geometry_manifest),
        output_manifest=Path(args.output_manifest),
        report_path=Path(args.report),
        target_records=args.target_records,
        run_id=args.run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
