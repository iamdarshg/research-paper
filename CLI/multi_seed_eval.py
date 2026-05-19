#!/usr/bin/env python3
"""
Multi-Seed Evaluation Script for Aircraft Diffusion CFD.
Automates aggregated performance studies across multiple seeds (Issue #32).
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path

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
