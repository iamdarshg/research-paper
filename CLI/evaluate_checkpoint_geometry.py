#!/usr/bin/env python3
"""Evaluate grounded reconstruction and free generation from a training checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    OptimizedAircraftGenerator,
    _binarize_probability_grid_for_solver,
    load_grounded_manifest_records,
)
from aircraft_validity import evaluate_aircraft_validity


def _overlap_metrics(candidate: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    candidate_bool = candidate.detach().cpu().bool()
    target_bool = target.detach().cpu().bool()
    intersection = int(torch.logical_and(candidate_bool, target_bool).sum().item())
    union = int(torch.logical_or(candidate_bool, target_bool).sum().item())
    target_count = int(target_bool.sum().item())
    candidate_count = int(candidate_bool.sum().item())
    return {
        "recall": intersection / max(target_count, 1),
        "precision": intersection / max(candidate_count, 1),
        "iou": intersection / max(union, 1),
    }


def _projection(binary: np.ndarray, axis: int) -> np.ndarray:
    return np.max(binary, axis=axis)


def _probability_ranking_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, Any]:
    probabilities = probability.detach().cpu().float().reshape(-1)
    target_mask = target.detach().cpu().bool().reshape(-1)
    positive = probabilities[target_mask]
    negative = probabilities[~target_mask]
    occupied_count = max(1, int(target_mask.sum().item()))
    topk_cutoff = float(
        torch.topk(probabilities, k=min(occupied_count, probabilities.numel())).values[-1].item()
    )

    def quantiles(values: torch.Tensor, levels: List[float]) -> Dict[str, float]:
        if values.numel() == 0:
            return {f"q{int(level * 10000):04d}": 0.0 for level in levels}
        return {
            f"q{int(level * 10000):04d}": float(torch.quantile(values, level).item())
            for level in levels
        }

    return {
        "topk_cutoff": topk_cutoff,
        "positive_mean": float(positive.mean().item()) if positive.numel() else 0.0,
        "negative_mean": float(negative.mean().item()) if negative.numel() else 0.0,
        "mean_separation": (
            float(positive.mean().item() - negative.mean().item())
            if positive.numel() and negative.numel()
            else 0.0
        ),
        "positive_below_topk_cutoff_fraction": (
            float((positive < topk_cutoff).float().mean().item())
            if positive.numel()
            else 1.0
        ),
        "positive_quantiles": quantiles(positive, [0.10, 0.50, 0.90]),
        "negative_quantiles": quantiles(negative, [0.90, 0.99, 0.999, 0.9999]),
        "negative_max": float(negative.max().item()) if negative.numel() else 0.0,
    }


def _render_projection_grid(
    geometries: Iterable[tuple[str, np.ndarray]],
    output: Path,
) -> None:
    rows = list(geometries)
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(10, max(2.4 * len(rows), 4.8)),
        squeeze=False,
    )
    views = ((0, "top"), (1, "side"), (2, "front"))
    for row_index, (label, binary) in enumerate(rows):
        for column_index, (axis, view_name) in enumerate(views):
            panel = axes[row_index, column_index]
            panel.imshow(
                _projection(binary, axis),
                cmap="Greys",
                interpolation="nearest",
                vmin=0,
                vmax=1,
            )
            panel.set_title(f"{label}: {view_name}", fontsize=9)
            panel.set_xticks([])
            panel.set_yticks([])
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _evaluate_binary(
    binary: torch.Tensor,
    target: torch.Tensor,
    probability: torch.Tensor | None = None,
) -> Dict[str, Any]:
    binary_cpu = binary.detach().cpu().float()
    result = {
        "occupied_voxels": int(binary_cpu.sum().item()),
        "overlap": _overlap_metrics(binary_cpu, target),
        "aircraft_validity": evaluate_aircraft_validity(
            binary_cpu.numpy(),
            canonicalize=False,
        ),
    }
    if probability is not None:
        result["probability_ranking"] = _probability_ranking_metrics(
            probability,
            target,
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = OptimizedAircraftGenerator(args.checkpoint, device=device)
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=generator.model_config.grid_resolution,
        latent_dim=generator.model_config.latent_dim,
        manifest_path=args.manifest,
    )
    manifest_records = load_grounded_manifest_records(args.manifest)
    sample = dataset[args.sample_index]
    target = sample["geometry"].detach().cpu().float()
    target_occupancy = float(target.mean().item())
    geometry_threshold = float(generator.geometry_probability_threshold)
    geometries: List[tuple[str, np.ndarray]] = [("ground truth", target.numpy())]

    with torch.no_grad():
        latent = sample["latent"].unsqueeze(0).to(device)
        clean_probability = torch.sigmoid(generator.converter(latent))[0]
        clean_binary = _binarize_probability_grid_for_solver(
            clean_probability,
            threshold=geometry_threshold,
            target_occupancy=None,
        )

    report: Dict[str, Any] = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "sample_index": args.sample_index,
        "source_record": manifest_records[args.sample_index],
        "target_occupancy": target_occupancy,
        "geometry_probability_threshold": geometry_threshold,
        "materialization_mode": "fixed_global_threshold",
        "ground_truth_validity": evaluate_aircraft_validity(
            target.numpy(),
            canonicalize=False,
        ),
        "clean_reconstruction": _evaluate_binary(
            clean_binary,
            target,
            clean_probability,
        ),
        "generations": [],
    }
    geometries.append(("clean reconstruction", clean_binary.detach().cpu().numpy()))

    arrays: Dict[str, np.ndarray] = {
        "ground_truth": target.numpy(),
        "clean_reconstruction": clean_binary.detach().cpu().numpy(),
    }
    for seed in args.seeds:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            probability = generator.generate(
                None,
                num_steps=generator.diffusion_config.student_steps,
                condition_vector=sample["condition_vector"],
            )
            binary = _binarize_probability_grid_for_solver(
                probability,
                threshold=geometry_threshold,
                target_occupancy=None,
            )
        result = {
            "seed": seed,
            **_evaluate_binary(binary, target, probability),
        }
        report["generations"].append(result)
        label = f"generated seed {seed}"
        binary_np = binary.detach().cpu().numpy()
        geometries.append((label, binary_np))
        arrays[f"generated_seed_{seed}"] = binary_np

    np.savez_compressed(output_dir / "geometries.npz", **arrays)
    _render_projection_grid(geometries, output_dir / "projections.png")
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
