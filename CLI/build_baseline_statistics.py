#!/usr/bin/env python3
"""Produce baseline_statistics.json from deterministic reference records.

The first-training protocol's ``baseline_statistics`` section consumes the
deterministic ``baseline_records.json`` written by ``build_reference_evidence``
and reduces it into the same report shape the final-evidence gate expects
(status + lineage fields). It reuses the statistics helpers from
``multi_seed_eval`` so the reduced-bundle report and the heavier multi-seed
report agree on the pass/blocked semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from multi_seed_eval import (
    build_statistical_summary,
    validate_baseline_policy,
)
from report_metadata import apply_report_metadata


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_baseline_statistics_from_records(
    records: List[Dict[str, Any]],
    *,
    baseline_config: Dict[str, Any],
    metric_keys: List[str],
    min_seeds: int,
) -> Dict[str, Any]:
    """Reduce deterministic reference records into a claim-boundary report."""
    policy = validate_baseline_policy(baseline_config)
    summary = build_statistical_summary(records, metric_keys, min_seeds=min_seeds)

    blockers: List[str] = []
    if policy["status"] != "pass":
        blockers.extend(policy["blockers"])
    if summary["status"] != "pass":
        blockers.extend(summary["blockers"])

    report = {
        "status": "pass" if not blockers else "blocked",
        "baseline_policy": policy,
        "multi_seed_summary": summary,
        "metric_keys": metric_keys,
        "claim_boundary": (
            "Deterministic reference baseline statistics for the reduced "
            "evidence bundle; this does not establish superiority over "
            "published methods."
        ),
    }
    if blockers:
        report["blockers"] = blockers
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build baseline_statistics.json from deterministic reference records."
    )
    parser.add_argument("--records-json", required=True, help="baseline_records.json path")
    parser.add_argument("--baseline-config", required=True, help="baseline_config.json/yaml path")
    parser.add_argument("--metric-keys", nargs="+", default=["lift_to_drag"])
    parser.add_argument("--min-seeds", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--protocol-config", default=None)
    args = parser.parse_args()

    payload = _load_json(args.records_json)
    records = payload.get("records", [])
    if not isinstance(records, list):
        parser.error("records_json must contain a 'records' list")

    config_path = Path(args.baseline_config)
    if config_path.suffix in {".yaml", ".yml"}:
        baseline_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        baseline_config = _load_json(config_path)
    if not isinstance(baseline_config, dict):
        parser.error("baseline_config must decode to a mapping")

    report = build_baseline_statistics_from_records(
        records,
        baseline_config=baseline_config,
        metric_keys=args.metric_keys,
        min_seeds=args.min_seeds,
    )
    apply_report_metadata(
        report,
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        protocol_path=args.protocol_config,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())