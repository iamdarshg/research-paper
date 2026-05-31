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
        if len(values) < min_seeds:
            blockers.append(f"metric {key} has insufficient values: found {len(values)}, require at least {min_seeds}")
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
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--grid-size", type=int, default=32, help="Evaluation resolution")
    parser.add_argument("--output-dir", default="./multi_seed_eval_results", help="Output directory")

    args = parser.parse_args()
    run_eval(args.checkpoint, args.num_seeds, args.grid_size, args.output_dir)
