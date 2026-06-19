#!/usr/bin/env python3
"""Build deterministic whole-aircraft voxel inputs for the aircraft-validity gate.

This builder is intentionally bounded to local, reproducible procedural aircraft
generation. It is useful for gate execution and regression testing, but it is
not a substitute for a claim-bearing public whole-aircraft corpus.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aircraft_diffusion_cfd import _procedural_aircraft_geometry, sample_design_spec
from aircraft_validity import evaluate_aircraft_validity


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_inputs(
    *,
    output_dir: Path,
    metadata_path: Path,
    num_samples: int,
    grid_size: int,
    seed_start: int,
    max_attempts: int,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for seed in range(seed_start, seed_start + max_attempts):
        rng = random.Random(seed)
        design_spec = sample_design_spec(rng)
        voxels = _procedural_aircraft_geometry(design_spec, grid_size)
        report = evaluate_aircraft_validity(voxels)

        item = {
            "seed": seed,
            "sample_id": f"procedural_aircraft_{seed:03d}",
            "design_spec": dataclasses.asdict(design_spec),
            "validity_report": report,
        }
        if report["status"] == "pass":
            np.save(output_dir / f"{item['sample_id']}.npy", voxels.detach().cpu().numpy())
            accepted.append(item)
            if len(accepted) >= num_samples:
                break
        else:
            rejected.append(item)

    summary = {
        "status": "pass" if len(accepted) >= num_samples else "blocked",
        "claim_boundary": (
            "Deterministic procedural whole-aircraft voxel bundle for aircraft-validity "
            "gate execution. Not a public-source claim-bearing aircraft corpus."
        ),
        "grid_size": grid_size,
        "requested_samples": num_samples,
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "seed_start": seed_start,
        "max_attempts": max_attempts,
        "output_dir": str(output_dir.resolve()),
        "accepted": accepted,
        "rejected": rejected,
    }
    write_json(metadata_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build procedural aircraft-validity voxel inputs.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "build" / "protocol_final" / "generated_voxels"),
        help="Directory for generated .npy voxel artifacts.",
    )
    parser.add_argument(
        "--metadata",
        default=str(REPO_ROOT / "build" / "protocol_final" / "generated_voxels_metadata.json"),
        help="JSON metadata summary path.",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="Number of passing voxel artifacts to keep.")
    parser.add_argument("--grid-size", type=int, default=32, help="Voxel resolution.")
    parser.add_argument("--seed-start", type=int, default=0, help="First deterministic seed to try.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=128,
        help="Maximum number of design seeds to try before reporting blocked.",
    )
    args = parser.parse_args()

    summary = build_inputs(
        output_dir=Path(args.output_dir).resolve(),
        metadata_path=Path(args.metadata).resolve(),
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        seed_start=args.seed_start,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
