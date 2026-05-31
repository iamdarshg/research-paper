#!/usr/bin/env python3
"""Fail-fast feasibility checks for structured condition payloads.

These constraints are pre-generation guards, not certification or structural
proof. NASA-STD-7009B frames credible model use around verification,
validation, sensitivity analysis, and uncertainty qualification:
https://standards.nasa.gov/standard/nasa/nasa-std-7009
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


MIN_WALL_BY_METHOD_MM = {
    "foam_core_hotwire": 1.0,
    "fdm_pla_0p4mm": 0.8,
    "fdm_pla_0p6mm": 1.0,
    "sheet_balsa_tabbed": 1.0,
    "composite_wet_layup": 1.0,
}


def _number(payload: Dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(payload.get(key, default))


def _condition_payload_from_design_spec(design_spec: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(design_spec)
    if "target_speed_mps" not in payload and "target_speed" in payload:
        payload["target_speed_mps"] = payload["target_speed"]
    return payload


def validate_condition_feasibility(payload: Dict[str, Any]) -> Dict[str, Any]:
    failed: List[str] = []
    errors: List[str] = []

    engine_min = int(payload.get("engine_count_min", 1))
    engine_max = int(payload.get("engine_count_max", engine_min))
    required_thrust = _number(payload, "required_static_thrust_n")
    if engine_min <= 0 or engine_max <= 0 or engine_min > engine_max:
        failed.append("engine_count")
        errors.append("engine_count_min and engine_count_max must be positive and ordered")
    if required_thrust > 0 and engine_max <= 0:
        failed.append("engine_thrust_consistency")
        errors.append("nonzero required_static_thrust_n requires at least one engine")

    payload_min = _number(payload, "payload_mass_min_g")
    payload_max = _number(payload, "payload_mass_max_g")
    if payload_min < 0 or payload_max < 0 or payload_min > payload_max:
        failed.append("payload_bounds")
        errors.append("payload mass bounds must be non-negative and ordered")

    part_min = int(payload.get("part_count_min", 1))
    part_max = int(payload.get("part_count_max", part_min))
    if part_min <= 0 or part_max <= 0 or part_min > part_max:
        failed.append("part_count_bounds")
        errors.append("part count bounds must be positive and ordered")

    wall_min = _number(payload, "wall_thickness_min_mm")
    wall_max = _number(payload, "wall_thickness_max_mm", wall_min)
    method = str(payload.get("manufacturing_method", "fdm_pla_0p4mm"))
    method_min_wall = MIN_WALL_BY_METHOD_MM.get(method)
    if wall_min <= 0 or wall_max <= 0 or wall_min > wall_max:
        failed.append("wall_thickness")
        errors.append("wall thickness bounds must be positive and ordered")
    elif method_min_wall is not None and wall_min < method_min_wall:
        failed.append("wall_thickness")
        errors.append(f"{method} requires wall_thickness_min_mm >= {method_min_wall}")

    twr = _number(payload, "thrust_to_weight_min", 0.0)
    if twr < 0:
        failed.append("thrust_to_weight")
        errors.append("thrust_to_weight_min must be non-negative")

    speed = _number(payload, "target_speed_mps", 1.0)
    turn_rate = _number(payload, "turn_rate_min_deg_s", 0.0)
    if speed <= 0:
        failed.append("target_speed")
        errors.append("target_speed_mps must be positive")
    if turn_rate < 0:
        failed.append("maneuverability")
        errors.append("turn_rate_min_deg_s must be non-negative")

    return {
        "status": "pass" if not failed else "blocked",
        "failed_checks": sorted(set(failed)),
        "errors": errors,
        "heuristic_checks": [
            "method-specific minimum wall thickness",
            "engine count and thrust consistency",
            "ordered payload, wall, and part-count bounds",
        ],
    }


def _load_manifest_records(manifest_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{manifest_path}:{line_number} must contain a JSON object")
            records.append(record)
    return records


def build_manufacturing_constraints_report(
    payloads: Iterable[Dict[str, Any]],
    *,
    manifest_path: str | None = None,
) -> Dict[str, Any]:
    sample_reports = []
    for idx, payload in enumerate(payloads):
        if payload.get("_invalid_record") is not None:
            sample_report = {
                "status": "blocked",
                "failed_checks": ["design_spec"],
                "errors": ["manifest record is missing a design_spec object"],
                "heuristic_checks": [],
            }
        else:
            sample_report = validate_condition_feasibility(payload)
        sample_report["sample_index"] = idx
        sample_reports.append(sample_report)

    blocked = [
        report["sample_index"]
        for report in sample_reports
        if report.get("status") != "pass"
    ]
    if not sample_reports:
        status = "blocked"
    else:
        status = "pass" if not blocked else "blocked"

    return {
        "status": status,
        "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
        "sample_count": len(sample_reports),
        "blocked_sample_indices": blocked,
        "samples": sample_reports,
        "claim_boundary": (
            "Payload feasibility and manufacturing-parameter screening only; "
            "geometry-aware structural validation is still required for structural claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate condition payload manufacturing-feasibility constraints.")
    parser.add_argument("--manifest", default=None, help="JSONL manifest with design_spec objects.")
    parser.add_argument("--payload-json", default=None, help="Single condition payload JSON string.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    payloads: List[Dict[str, Any]] = []
    if args.manifest:
        records = _load_manifest_records(Path(args.manifest))
        for idx, record in enumerate(records):
            design_spec = record.get("design_spec")
            if not isinstance(design_spec, dict):
                payloads.append({"_invalid_record": idx})
                continue
            payloads.append(_condition_payload_from_design_spec(design_spec))
    if args.payload_json:
        payload = json.loads(args.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("--payload-json must decode to an object")
        payloads.append(_condition_payload_from_design_spec(payload))

    report = build_manufacturing_constraints_report(payloads, manifest_path=args.manifest)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
