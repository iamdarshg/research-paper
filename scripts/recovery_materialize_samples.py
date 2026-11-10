#!/usr/bin/env python3
"""Materialize generated held-out voxels to .npy for independent screening.

Used so that the repository's own `CLI/aircraft_validity.py --input-dir` CLI can
be applied to the generated artifacts without going through this script's own
screening code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    OptimizedAircraftGenerator,
    _binarize_probability_grid_for_solver,
    load_grounded_manifest_records,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sampler", default="consistency")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = OptimizedAircraftGenerator(args.checkpoint, device=device)
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=int(generator.model_config.grid_resolution),
        latent_dim=int(generator.model_config.latent_dim),
        manifest_path=args.manifest,
    )
    records = load_grounded_manifest_records(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            condition = sample["condition_vector"].reshape(1, -1).to(
                device, dtype=torch.float32
            )
            record_id = records[index].get("sample_id", str(index))
            for seed in range(int(args.seeds)):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                probability = generator.generate(
                    condition_vector=condition,
                    sampler=args.sampler,
                    ddim_steps=50,
                )
                binary = _binarize_probability_grid_for_solver(
                    probability, threshold=args.threshold
                ).cpu().numpy()
                np.save(output_dir / f"{record_id}__{args.sampler}__seed{seed}.npy", binary)
    print(f"wrote {len(list(output_dir.glob('*.npy')))} artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
