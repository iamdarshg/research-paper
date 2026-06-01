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


def build_statistical_summary(records, metric_keys, min_seeds=3):
    # Do not collapse claim-bearing comparisons to a single lucky seed. This
    # helper reports sample mean/std and blocks when the seed count is too low.
    seeds = sorted({int(record["seed"]) for record in records if "seed" in record})
    blockers = []
    if len(seeds) < min_seeds:
        blockers.append(f"insufficient seeds: found {len(seeds)}, require at least {min_seeds}")

    metrics = {}
    for key in metric_keys:
        values = [float(record[key]) for record in records if key in record]
        metric_seeds = sorted({int(record["seed"]) for record in records if "seed" in record and key in record})
        if len(metric_seeds) < min_seeds:
            blockers.append(f"metric {key} has insufficient seed values: found {len(metric_seeds)}, require at least {min_seeds}")
            continue
        metrics[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "count": len(values),
            "seed_count": len(metric_seeds),
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


def build_baseline_statistics_report(*, baseline_config, records, metric_keys, min_seeds=3):
    required_baselines = [
        "retrieval",
        "unconditional_checkpoint",
        "bundled_grounded_stl",
    ]
    baseline_policy = validate_baseline_policy(
        baseline_config,
        required_baselines=required_baselines,
    )
    blockers = list(baseline_policy["blockers"])

    records_by_baseline = {}
    for record in records:
        baseline = record.get("baseline")
        if baseline:
            records_by_baseline.setdefault(baseline, []).append(record)

    baseline_reports = {}
    for baseline in required_baselines:
        baseline_records = records_by_baseline.get(baseline, [])
        if not baseline_records:
            blockers.append(f"baseline {baseline} has no records")
            baseline_reports[baseline] = {
                "status": "blocked",
                "seed_count": 0,
                "seeds": [],
                "metrics": {},
                "blockers": [f"baseline {baseline} has no records"],
            }
            continue

        summary = build_statistical_summary(
            baseline_records,
            metric_keys=metric_keys,
            min_seeds=min_seeds,
        )
        baseline_blockers = [
            f"baseline {baseline}: {blocker}" for blocker in summary["blockers"]
        ]
        blockers.extend(baseline_blockers)
        baseline_reports[baseline] = {
            **summary,
            "blockers": baseline_blockers,
        }

    return {
        "status": "pass" if not blockers else "blocked",
        "baseline_policy": baseline_policy,
        "baselines": baseline_reports,
        "metric_keys": list(metric_keys),
        "min_seeds": min_seeds,
        "blockers": blockers,
    }


def write_baseline_statistics_report(
    *,
    baseline_config_path,
    records_json_path,
    metric_keys,
    output_path,
    min_seeds=3,
):
    with open(baseline_config_path, "r", encoding="utf-8") as config_file:
        baseline_config = json.load(config_file)
    with open(records_json_path, "r", encoding="utf-8") as records_file:
        records = json.load(records_file)
    if isinstance(records, dict) and "records" in records:
        records = records["records"]
    if not isinstance(records, list):
        raise ValueError("records JSON must be a list or an object with a records list")

    report = build_baseline_statistics_report(
        baseline_config=baseline_config,
        records=records,
        metric_keys=metric_keys,
        min_seeds=min_seeds,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return report


def run_eval(checkpoint, num_seeds, grid_size, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable
    script_path = Path(__file__).resolve().parent / "aircraft_diffusion_cfd.py"

    # 1. Run condition validation
    print(f"Running condition validation study with {num_seeds} seeds...")
    validation_output = output_dir / "condition_validation.json"
    subprocess.run([
        python_exe, str(script_path), "validate-conditions",
        "--checkpoint", checkpoint,
        "--num-seeds", str(num_seeds),
        "--grid-size", str(grid_size),
        "--output", str(validation_output)
    ], check=True)

    # 2. Run batch generation for diversity check
    print(f"Running batch generation study for diversity analysis...")
    batch_dir = output_dir / "batch_study"
    subprocess.run([
        python_exe, str(script_path), "batch-generate",
        "--checkpoint", checkpoint,
        "--output-dir", str(batch_dir),
        "--num-designs", str(num_seeds),
        "--vary-conditions"
    ], check=True)

    # Aggregate results
    with open(validation_output, 'r') as f:
        val_data = json.load(f)

    correlations = val_data["correlations"]

    print("\n" + "="*60)
    print("MULTI-SEED EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Seeds: {num_seeds}")
    print("\nKey Correlations:")
    for key, stats in correlations.items():
        print(f"  {key}: r={stats['r']:.4f}, p={stats['p']:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-seed scientific evaluation.")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--grid-size", type=int, default=32, help="Evaluation resolution")
    parser.add_argument("--output-dir", default="./multi_seed_eval_results", help="Output directory")
    parser.add_argument("--baseline-config", help="Path to baseline configuration JSON")
    parser.add_argument("--records-json", help="Path to baseline metric records JSON")
    parser.add_argument("--metric-key", action="append", dest="metric_keys", default=[], help="Metric key to include")
    parser.add_argument("--baseline-statistics-output", help="Path for baseline_statistics.json")
    parser.add_argument("--min-seeds", type=int, default=3, help="Minimum seeds required per baseline")

    args = parser.parse_args()
    report_args = [
        args.baseline_config,
        args.records_json,
        args.baseline_statistics_output,
    ]
    if any(report_args) or args.metric_keys:
        missing_args = []
        if not args.baseline_config:
            missing_args.append("--baseline-config")
        if not args.records_json:
            missing_args.append("--records-json")
        if not args.metric_keys:
            missing_args.append("--metric-key")
        if not args.baseline_statistics_output:
            missing_args.append("--baseline-statistics-output")
        if missing_args:
            parser.error("report-only mode requires " + ", ".join(missing_args))

        report = write_baseline_statistics_report(
            baseline_config_path=args.baseline_config,
            records_json_path=args.records_json,
            metric_keys=args.metric_keys,
            output_path=args.baseline_statistics_output,
            min_seeds=args.min_seeds,
        )
        if report["status"] != "pass":
            print("Baseline statistics report blocked:")
            for blocker in report["blockers"]:
                print(f"  - {blocker}")
            sys.exit(1)
        print(f"Wrote baseline statistics report to {args.baseline_statistics_output}")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless report-only mode is used")
        run_eval(args.checkpoint, args.num_seeds, args.grid_size, args.output_dir)
