#!/usr/bin/env python3
"""Evaluate a generator checkpoint against the aircraft-validity screen.

Pre-registered protocol (see evidence/research-paper-publish/
GENERATOR_RECOVERY_2026-11-05.md): the validity screen is
``CLI/aircraft_validity.evaluate_aircraft_validity`` with
``canonicalize=True`` (permutation-invariant orientation search) and a
*pre-declared* materialization threshold, applied unchanged to every sample.
No threshold is fitted to the evaluation set and no failing sample is
dropped.

Two arms are reported separately so that decoder capacity and sampler
quality are not conflated:

* ``reconstruction`` -- decode the dataset's structured latent for the record
  (the encoder side is given), materialize, screen.
* ``generation`` -- 4-step consistency sampling from noise conditioned on the
  record's condition vector, materialize, screen.

The ground-truth validity of every record in the split is also reported, so
the pass rate can be read against the achievable ceiling.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    OptimizedAircraftGenerator,
    _binarize_probability_grid_for_solver,
    load_grounded_manifest_records,
)
from aircraft_validity import evaluate_aircraft_validity


def _overlap(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = prediction.detach().cpu().bool()
    tgt = target.detach().cpu().bool()
    if pred.shape != tgt.shape:
        # Same nearest-neighbour resize the trainer applies when the model grid
        # differs from the corpus grid (never up-samples the corpus).
        tgt = (
            torch.nn.functional.interpolate(
                target.detach().cpu().float().unsqueeze(0).unsqueeze(0),
                size=tuple(pred.shape),
                mode="nearest",
            )
            .squeeze()
            .bool()
        )
    inter = int(torch.logical_and(pred, tgt).sum().item())
    union = int(torch.logical_or(pred, tgt).sum().item())
    return {
        "iou": inter / max(union, 1),
        "recall": inter / max(int(tgt.sum().item()), 1),
        "precision": inter / max(int(pred.sum().item()), 1),
    }


def _materialize(probability: torch.Tensor, threshold: float, target_occupancy):
    return _binarize_probability_grid_for_solver(
        probability,
        threshold=threshold,
        target_occupancy=target_occupancy,
    ).cpu()


def _screen(binary: torch.Tensor) -> Dict[str, Any]:
    report = evaluate_aircraft_validity(binary.numpy(), canonicalize=True)
    return {
        "status": report["status"],
        "failed_checks": report["failed_checks"],
        "metrics": report["metrics"],
        "permutation": report["canonicalization"]["permutation"],
    }


def _summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(samples)
    passed = sum(1 for sample in samples if sample["status"] == "pass")
    failures: Counter[str] = Counter()
    for sample in samples:
        failures.update(sample["failed_checks"])
    ious = [sample["overlap"]["iou"] for sample in samples if "overlap" in sample]
    occupancies = [sample["occupied_voxels"] / 32768.0 for sample in samples]
    return {
        "sample_count": total,
        "passed_sample_count": passed,
        "pass_rate": (passed / total) if total else 0.0,
        "failed_check_histogram": dict(sorted(failures.items(), key=lambda kv: -kv[1])),
        "mean_iou_vs_ground_truth": float(np.mean(ious)) if ious else 0.0,
        "mean_occupancy": float(np.mean(occupancies)) if occupancies else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, required=True,
                        help="Pre-declared, uncalibrated materialization threshold.")
    parser.add_argument("--target-occupancy", type=float, default=None,
                        help="Optional secondary arm: top-k materialization at a fixed "
                             "occupancy drawn from the TRAINING split only.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--samplers", default="consistency,ddim",
                        help="Comma-separated samplers to evaluate.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    generator = OptimizedAircraftGenerator(args.checkpoint, device=device)
    resolution = int(generator.model_config.grid_resolution)
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=resolution,
        latent_dim=int(generator.model_config.latent_dim),
        manifest_path=args.manifest,
    )
    records = load_grounded_manifest_records(args.manifest)
    samplers = [name.strip() for name in args.samplers.split(",") if name.strip()]

    report: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "grid_resolution": resolution,
        "latent_dim": int(generator.model_config.latent_dim),
        "materialization_threshold": float(args.threshold),
        "seeds": args.seeds,
        "ground_truth": [],
        "reconstruction": [],
        "generation": [],
    }

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            target = sample["geometry"].detach().cpu().float()
            condition = sample["condition_vector"].reshape(1, -1).to(
                device, dtype=torch.float32
            )
            record_id = records[index].get("sample_id", str(index))

            gt = _screen((target > 0.5).float())
            report["ground_truth"].append(
                {
                    "record": record_id,
                    "status": gt["status"],
                    "failed_checks": gt["failed_checks"],
                    "occupancy": float((target > 0.5).float().mean().item()),
                }
            )

            latent = sample["latent"].reshape(1, -1).to(device, dtype=torch.float32)
            probability = torch.sigmoid(generator.converter(latent))[0]
            binary = _materialize(probability, args.threshold, args.target_occupancy)
            recon = _screen(binary)
            recon.update(
                {
                    "record": record_id,
                    "occupied_voxels": int(binary.sum().item()),
                    "probability_std": float(probability.std().item()),
                    "probability_min": float(probability.min().item()),
                    "probability_max": float(probability.max().item()),
                    "overlap": _overlap(binary, target),
                }
            )
            report["reconstruction"].append(recon)

            for sampler_name in samplers:
                for seed in range(int(args.seeds)):
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    generated_probability = generator.generate(
                        num_steps=generator.diffusion_config.student_steps,
                        condition_vector=condition,
                        sampler=sampler_name,
                        ddim_steps=int(args.ddim_steps),
                    )
                    generated_binary = _materialize(
                        generated_probability, args.threshold, args.target_occupancy
                    )
                    gen = _screen(generated_binary)
                    gen.update(
                        {
                            "record": record_id,
                            "seed": seed,
                            "sampler": sampler_name,
                            "occupied_voxels": int(generated_binary.sum().item()),
                            "probability_std": float(generated_probability.std().item()),
                            "overlap": _overlap(generated_binary, target),
                        }
                    )
                    report["generation"].append(gen)

    report["summary"] = {
        "ground_truth": _summarize(
            [
                {**row, "overlap": {"iou": 1.0}, "occupied_voxels": 0}
                for row in report["ground_truth"]
            ]
        ),
        "reconstruction": _summarize(report["reconstruction"]),
        "generation": _summarize(report["generation"]),
        "generation_by_sampler": {
            sampler: _summarize(
                [row for row in report["generation"] if row.get("sampler") == sampler]
            )
            for sampler in samplers
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for arm in ("ground_truth", "reconstruction", "generation"):
        summary = report["summary"][arm]
        print(
            f"{arm}: {summary['passed_sample_count']}/{summary['sample_count']} pass "
            f"({summary['pass_rate']:.1%}) failed={summary['failed_check_histogram']}"
        )
    for sampler, summary in report["summary"]["generation_by_sampler"].items():
        print(
            f"generation[{sampler}]: {summary['passed_sample_count']}/{summary['sample_count']} pass "
            f"({summary['pass_rate']:.1%}) iou={summary['mean_iou_vs_ground_truth']:.3f} "
            f"occ={summary['mean_occupancy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
