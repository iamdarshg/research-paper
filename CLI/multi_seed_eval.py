#!/usr/bin/env python3
"""
Multi-Seed Evaluation Script for Aircraft Diffusion CFD.
Automates aggregated performance studies across multiple seeds (Issue #32).

Mean/std summaries are uncertainty descriptors, not proof of superiority.
NIST guidance treats standard uncertainty as a standard-deviation expression of
incomplete knowledge: https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

import yaml

from report_metadata import apply_report_metadata


def build_statistical_summary(records, metric_keys, min_seeds=3):
    # Do not collapse claim-bearing comparisons to a single lucky seed. This
    # helper reports sample mean/std and blocks when the seed count is too low.
    seeds = sorted({int(record["seed"]) for record in records if "seed" in record})
    blockers = []
    if len(seeds) < min_seeds:
        blockers.append(f"insufficient seeds: found {len(seeds)}, require at least {min_seeds}")

    metrics = {}
    for key in metric_keys:
        values = []
        non_finite_seeds = []
        for idx, record in enumerate(records):
            if key not in record:
                continue
            seed = record.get("seed", idx)
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                non_finite_seeds.append(str(seed))
                continue
            if not np.isfinite(value):
                non_finite_seeds.append(str(seed))
                continue
            values.append(value)
        if non_finite_seeds:
            blockers.append(
                f"metric {key} has non-finite values for seeds: {', '.join(non_finite_seeds)}"
            )
        if len(values) < min_seeds:
            blockers.append(f"metric {key} has insufficient values: found {len(values)}, require at least {min_seeds}")
            continue
        if non_finite_seeds:
            continue
        metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    return {
        "status": "pass" if not blockers else "blocked",
        "seed_count": len(seeds),
        "seeds": seeds,
        "metrics": metrics,
        "blockers": blockers,
    }


def validate_baseline_policy(config, required_baselines=None):
    required_baselines = required_baselines or [
        "retrieval",
        "unconditional_checkpoint",
        "bundled_grounded_stl",
    ]
    blockers = []
    baseline_set = config.get("baseline_set")
    if not isinstance(baseline_set, list):
        blockers.append("missing baseline_set")
        baseline_set = []
    missing = [name for name in required_baselines if name not in baseline_set]
    if missing:
        blockers.append("missing required baselines: " + ", ".join(missing))
    if not config.get("baseline_name"):
        blockers.append("missing baseline_name")
    return {
        "status": "pass" if not blockers else "blocked",
        "required_baselines": required_baselines,
        "baseline_set": baseline_set,
        "blockers": blockers,
    }


def reported_baseline_families(baseline_report: Dict[str, Any]) -> List[str]:
    """Return claim-bearing baseline families that have concrete report entries."""
    if not baseline_report:
        return []

    baselines = baseline_report.get("baselines")
    if isinstance(baselines, dict):
        families = []
        for name, payload in baselines.items():
            if not isinstance(payload, dict):
                continue
            results = payload.get("results")
            if payload.get("status", "pass") == "pass" and results:
                families.append(str(name))
        return sorted(families)

    # Legacy evaluate-baselines reports were flat STL-result maps. Treat them as
    # only the bundled grounded STL family so they cannot satisfy retrieval or
    # checkpoint-baseline gates by accident.
    return ["bundled_grounded_stl"]


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must decode to a mapping")
    return payload


def _records_from_validation_report(report: Dict[str, Any]) -> List[Dict[str, float]]:
    raw_data = report.get("raw_data") or {}
    keys = sorted(raw_data.keys())
    if not keys:
        return []

    value_count = max((len(raw_data.get(key, [])) for key in keys), default=0)
    records: List[Dict[str, float]] = []
    for idx in range(value_count):
        record: Dict[str, float] = {"seed": idx}
        for key in keys:
            values = raw_data.get(key, [])
            if idx < len(values):
                record[key] = float(values[idx])
        drag = record.get("measured_drag")
        lift = record.get("measured_lift")
        if drag is not None and lift is not None:
            record["lift_to_drag"] = float(lift / max(drag, 1e-6))
        records.append(record)
    return records


def build_baseline_statistics_report(
    *,
    baseline_config: Dict[str, Any],
    baseline_report: Dict[str, Any],
    condition_validation_report: Dict[str, Any],
    min_seeds: int = 3,
) -> Dict[str, Any]:
    baseline_policy = validate_baseline_policy(baseline_config)
    baseline_families = reported_baseline_families(baseline_report)
    records = _records_from_validation_report(condition_validation_report)
    statistical_summary = build_statistical_summary(
        records,
        metric_keys=["measured_drag", "measured_lift", "occupancy", "lift_to_drag"],
        min_seeds=min_seeds,
    )

    blockers = []
    if baseline_policy["status"] != "pass":
        blockers.extend(baseline_policy["blockers"])
    if statistical_summary["status"] != "pass":
        blockers.extend(statistical_summary["blockers"])
    if not baseline_report:
        blockers.append("missing grounded baseline report")
    missing_report_families = [
        name for name in baseline_policy["baseline_set"]
        if name not in baseline_families
    ]
    if missing_report_families:
        blockers.append(
            "missing required baseline report families: " + ", ".join(missing_report_families)
        )

    report = {
        "status": "pass" if not blockers else "blocked",
        "baseline_policy": baseline_policy,
        "baseline_report_families": baseline_families,
        "multi_seed_summary": statistical_summary,
        "grounded_baseline_results": baseline_report,
        "condition_validation_correlations": condition_validation_report.get("correlations", {}),
        "claim_boundary": (
            "Baseline statistics require named baselines plus sufficient repeated runs; "
            "they do not by themselves establish superiority."
        ),
    }
    if blockers:
        report["blockers"] = blockers
    return report


def run_eval(
    checkpoint,
    num_seeds,
    grid_size,
    output_dir,
    *,
    baseline_config_path=None,
    baseline_report_path=None,
    validation_report_path=None,
    output_report_path=None,
    manifest_path=None,
    protocol_config_path=None,
    run_id=None,
    min_seeds=3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    script_path = Path(__file__).resolve().parent / "aircraft_diffusion_cfd.py"

    validation_output = Path(validation_report_path) if validation_report_path else output_dir / "condition_validation.json"
    if not validation_output.exists():
        print(f"Running condition validation study with {num_seeds} seeds...")
        subprocess.run([
            python_exe, str(script_path), "validate-conditions",
            "--checkpoint", checkpoint,
            "--num-seeds", str(num_seeds),
            "--grid-size", str(grid_size),
            "--output", str(validation_output)
        ], check=True)

    val_data = _load_json(validation_output)
    baseline_data = _load_json(baseline_report_path) if baseline_report_path and Path(baseline_report_path).exists() else {}
    baseline_config = _load_yaml(baseline_config_path) if baseline_config_path and Path(baseline_config_path).exists() else {}
    report = build_baseline_statistics_report(
        baseline_config=baseline_config,
        baseline_report=baseline_data,
        condition_validation_report=val_data,
        min_seeds=min_seeds,
    )
    apply_report_metadata(
        report,
        run_id=run_id,
        checkpoint_path=checkpoint,
        manifest_path=manifest_path,
        protocol_path=protocol_config_path,
    )

    output_report = Path(output_report_path) if output_report_path else output_dir / "baseline_statistics.json"
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    correlations = val_data["correlations"]

    print("\n" + "="*60)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Seeds: {num_seeds}")
    print(f"Statistics report: {output_report}")
    print("\nKey Correlations:")
    for key, stats in correlations.items():
        print(f"  {key}: r={stats['r']:.4f}, p={stats['p']:.4f}")
    print("="*60 + "\n")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed scientific evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--grid-size", type=int, default=32, help="Evaluation resolution")
    parser.add_argument("--output-dir", default="./multi_seed_eval_results", help="Output directory")
    parser.add_argument("--baseline-config", default=None, help="Baseline policy YAML used for claim-bearing evaluation")
    parser.add_argument("--baseline-report", default=None, help="JSON report from evaluate-baselines")
    parser.add_argument("--validation-report", default=None, help="Optional precomputed validate-conditions JSON report")
    parser.add_argument("--output-report", default=None, help="Optional baseline statistics JSON report path")
    parser.add_argument("--manifest", default=None, help="Optional manifest path for evidence lineage metadata")
    parser.add_argument("--protocol-config", default=None, help="Optional protocol config path for evidence lineage metadata")
    parser.add_argument("--run-id", default=None, help="Optional run identifier shared across report artifacts")
    parser.add_argument("--min-seeds", type=int, default=3, help="Minimum seeds required before the statistics gate can pass")

    args = parser.parse_args()
    run_eval(
        args.checkpoint,
        args.num_seeds,
        args.grid_size,
        args.output_dir,
        baseline_config_path=args.baseline_config,
        baseline_report_path=args.baseline_report,
        validation_report_path=args.validation_report,
        output_report_path=args.output_report,
        manifest_path=args.manifest,
        protocol_config_path=args.protocol_config,
        run_id=args.run_id,
        min_seeds=args.min_seeds,
    )
