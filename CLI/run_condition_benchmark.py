#!/usr/bin/env python3
"""Run a fail-closed grounded condition-response benchmark.

This first benchmark layer is intentionally metadata-driven. It only reports
pass/fail when the manifest already contains grounded response metrics; otherwise
it returns blocked instead of implying that checkpoint plumbing is evidence.

CFD and simulation credibility require verification, validation, and uncertainty
quantification rather than isolated directional checks. See NASA's CFD V&V
overview and ASME V&V 20:
https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html
https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer/2009/print-book/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


FIXED_SWEEPS: List[Dict[str, str]] = [
    {
        "id": "payload_increase",
        "condition_field": "payload_mass_max_g",
        "metric": "payload_response",
        "expectation": "higher payload condition should have higher payload response metric",
    },
    {
        "id": "thrust_increase",
        "condition_field": "required_static_thrust_n",
        "metric": "thrust_response",
        "expectation": "higher thrust condition should have higher thrust response metric",
    },
    {
        "id": "maneuverability_increase",
        "condition_field": "turn_rate_min_deg_s",
        "metric": "maneuverability_response",
        "expectation": "higher turn-rate condition should have higher maneuverability response metric",
    },
    {
        "id": "wall_thickness_increase",
        "condition_field": "wall_thickness_min_mm",
        "metric": "structural_response",
        "expectation": "higher wall-thickness condition should have higher structural response metric",
    },
]


def load_manifest_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must contain JSON objects")
                records.append(payload)
        return records

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("records", payload))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of sample records")

    records = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{path} record {idx} must be an object")
        records.append(record)
    return records


def parse_seeds(raw: str) -> List[int]:
    seeds = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise ValueError("seed ranges must be increasing")
            seeds.update(range(start, end + 1))
        else:
            seeds.add(int(chunk))
    return sorted(seeds)


def _manifest_blockers(records: Sequence[Dict[str, Any]], min_grounded_records: int) -> List[str]:
    blockers: List[str] = []
    if len(records) < min_grounded_records:
        blockers.append(
            f"insufficient grounded records: found {len(records)}, require at least {min_grounded_records}"
        )

    for idx, record in enumerate(records):
        if not record.get("split"):
            blockers.append(f"record {idx}: missing split")
        design_spec = record.get("design_spec")
        if not isinstance(design_spec, dict):
            blockers.append(f"record {idx}: missing design_spec object")
            continue
        response_metrics = record.get("response_metrics")
        if not isinstance(response_metrics, dict):
            blockers.append(f"record {idx}: missing response_metrics object")
            continue
        for sweep in FIXED_SWEEPS:
            if sweep["condition_field"] not in design_spec:
                blockers.append(f"record {idx}: design_spec missing {sweep['condition_field']}")
            if sweep["metric"] not in response_metrics:
                blockers.append(f"record {idx}: response_metrics missing {sweep['metric']}")
    return blockers


def _numeric_value(record: Dict[str, Any], sweep: Dict[str, str], source: str) -> float:
    if source == "condition":
        value = record["design_spec"][sweep["condition_field"]]
    else:
        value = record["response_metrics"][sweep["metric"]]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{sweep['id']} requires numeric {source} values")
    return float(value)


def _evaluate_sweep(records: Sequence[Dict[str, Any]], sweep: Dict[str, str], min_effect: float) -> Dict[str, Any]:
    ordered = sorted(records, key=lambda record: _numeric_value(record, sweep, "condition"))
    midpoint = len(ordered) // 2
    low_records = ordered[:midpoint]
    high_records = ordered[midpoint:]

    if not low_records or not high_records:
        return {
            **sweep,
            "status": "blocked",
            "observed_delta": None,
            "low_record_count": len(low_records),
            "high_record_count": len(high_records),
            "blockers": ["not enough records to form low/high condition groups"],
        }

    low_mean = mean(_numeric_value(record, sweep, "metric") for record in low_records)
    high_mean = mean(_numeric_value(record, sweep, "metric") for record in high_records)
    observed_delta = high_mean - low_mean
    status = "pass" if observed_delta > min_effect else "fail"
    return {
        **sweep,
        "status": status,
        "observed_delta": observed_delta,
        "low_record_count": len(low_records),
        "high_record_count": len(high_records),
        "low_metric_mean": low_mean,
        "high_metric_mean": high_mean,
        "blockers": [],
    }


def build_condition_benchmark_report(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    seeds: Iterable[int],
    min_grounded_records: int = 20,
    min_effect: float = 0.0,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)
    records = load_manifest_records(manifest_path)
    normalized_seeds = sorted(set(int(seed) for seed in seeds))
    blockers = _manifest_blockers(records, min_grounded_records)

    report: Dict[str, Any] = {
        "benchmark": "grounded_condition_response",
        "schema_version": 1,
        "status": "blocked" if blockers else "pending",
        "manifest_path": str(manifest_path.resolve()),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_checked": False,
        "record_count": len(records),
        "seeds": normalized_seeds,
        "min_grounded_records": min_grounded_records,
        "min_effect": min_effect,
        "blockers": blockers,
        "sweeps": [],
        "claim_boundary": (
            "This benchmark is grounded-response evidence only when the manifest "
            "contains grounded response_metrics for each fixed sweep."
        ),
    }
    if blockers:
        return report

    report["checkpoint_checked"] = True
    if not checkpoint_path.exists():
        report["status"] = "blocked"
        report["blockers"] = [f"checkpoint not found: {checkpoint_path}"]
        return report

    sweep_reports = [_evaluate_sweep(records, sweep, min_effect) for sweep in FIXED_SWEEPS]
    report["sweeps"] = sweep_reports
    if any(sweep["status"] == "blocked" for sweep in sweep_reports):
        report["status"] = "blocked"
        report["blockers"] = [
            blocker
            for sweep in sweep_reports
            for blocker in sweep.get("blockers", [])
        ]
    elif all(sweep["status"] == "pass" for sweep in sweep_reports):
        report["status"] = "pass"
    else:
        report["status"] = "fail"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the grounded condition-response benchmark.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint under evaluation.")
    parser.add_argument("--manifest", required=True, help="Grounded manifest with response metrics.")
    parser.add_argument("--output", default="condition_benchmark_report.json", help="JSON report path.")
    parser.add_argument("--seeds", default="0-4", help="Comma-separated seeds or ranges, e.g. 0,1,4-6.")
    parser.add_argument("--min-grounded-records", type=int, default=20)
    parser.add_argument("--min-effect", type=float, default=0.0)
    args = parser.parse_args()

    report = build_condition_benchmark_report(
        manifest_path=Path(args.manifest),
        checkpoint_path=Path(args.checkpoint),
        seeds=parse_seeds(args.seeds),
        min_grounded_records=args.min_grounded_records,
        min_effect=args.min_effect,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")

    if report["status"] == "pass":
        return 0
    if report["status"] == "fail":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
