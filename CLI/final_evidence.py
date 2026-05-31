#!/usr/bin/env python3
"""Evaluate whether claim-bearing final evidence artifacts are present and passing.

The package-level gate follows NASA-STD-7009B's credibility-product posture:
claim support depends on the assembled evidence lifecycle, not on any single
run status. Source: https://standards.nasa.gov/standard/nasa/nasa-std-7009
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FINAL_GATES = {
    "manifest_validation": "claim-bearing manifest validation",
    "aircraft_validity": "aircraft-specific validity suite",
    "condition_benchmark": "grounded condition-response benchmark",
    "manufacturing_constraints": "structural/manufacturing feasibility gate",
    "baseline_statistics": "baseline and multi-seed statistical report",
}

CONSISTENCY_FIELDS = ("run_id", "checkpoint_hash", "manifest_hash", "protocol_hash")


def evaluate_final_evidence_package(
    reports: Dict[str, Dict[str, Any]],
    *,
    require_run_consistency: bool = False,
) -> Dict[str, Any]:
    blocked = []
    gate_results = {}
    for gate_id, description in REQUIRED_FINAL_GATES.items():
        report = reports.get(gate_id)
        status = report.get("status") if isinstance(report, dict) else "missing"
        gate_results[gate_id] = {
            "description": description,
            "status": status,
        }
        if status != "pass":
            blocked.append(gate_id)
    consistency_errors: List[str] = []
    consistency_values: Dict[str, Any] = {}
    if require_run_consistency:
        passing_reports = {
            gate_id: report
            for gate_id, report in reports.items()
            if gate_id in REQUIRED_FINAL_GATES and isinstance(report, dict) and report.get("status") == "pass"
        }
        for field_name in CONSISTENCY_FIELDS:
            values = {
                gate_id: report.get(field_name)
                for gate_id, report in passing_reports.items()
            }
            missing = [gate_id for gate_id, value in values.items() if not value]
            if missing:
                consistency_errors.append(
                    f"{field_name} missing from passing reports: {', '.join(sorted(missing))}"
                )
                continue
            unique_values = sorted({str(value) for value in values.values()})
            if len(unique_values) > 1:
                consistency_errors.append(
                    f"{field_name} mismatch across passing reports: {values}"
                )
            elif unique_values:
                consistency_values[field_name] = unique_values[0]
        if consistency_errors:
            blocked.append("run_consistency")
    return {
        "status": "pass" if not blocked else "blocked",
        "blocked_gates": blocked,
        "gates": gate_results,
        "run_consistency_required": require_run_consistency,
        "run_consistency": {
            "fields": list(CONSISTENCY_FIELDS),
            "values": consistency_values,
            "errors": consistency_errors,
        },
        "claim_boundary": "All required gates must pass before strengthening paper or README claims.",
    }


def _read_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate final claim-bearing evidence package status.")
    for gate_id in REQUIRED_FINAL_GATES:
        parser.add_argument(f"--{gate_id.replace('_', '-')}", default=None, help=f"JSON report for {gate_id}.")
    parser.add_argument(
        "--require-run-consistency",
        action="store_true",
        help="Require common run/checkpoint/manifest/protocol identifiers across passing reports.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    args = parser.parse_args()

    reports = {}
    for gate_id in REQUIRED_FINAL_GATES:
        value = getattr(args, gate_id)
        if value:
            reports[gate_id] = _read_report(Path(value))
    report = evaluate_final_evidence_package(reports, require_run_consistency=args.require_run_consistency)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
