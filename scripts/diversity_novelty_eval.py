#!/usr/bin/env python3
"""Evaluate a generator checkpoint against the frozen diversity/novelty gate.

The gate (scripts/diversity_novelty_gate.py) is frozen before any model runs:
it combines the repository's old 12-check screen, the declared structural
articulation checks, voxel novelty against union(training), and diversity
metrics.  Negative controls (constant union, plus sign, three-box primitive)
are screened and reported in every run.  Nothing in this script is calibrated
against the checkpoint under test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CLI")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for directory in (CLI_DIR, SCRIPTS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from aircraft_diffusion_cfd import (  # noqa: E402
    AircraftDesignDataset,
    OptimizedAircraftGenerator,
    _binarize_probability_grid_for_solver,
    load_grounded_manifest_records,
)
from diversity_novelty_gate import (  # noqa: E402
    GATE_SPEC_HASH,
    corpus_report,
    novelty_fraction,
    sample_report,
)


def _load_train_union(train_manifest: str) -> np.ndarray:
    records = load_grounded_manifest_records(train_manifest)
    base = Path(train_manifest).parent
    union = np.zeros((32, 32, 32), dtype=bool)
    for record in records:
        path = base / str(record["geometry_path"])
        union |= np.load(path) > 0.5
    return union


def _materialize(probability: torch.Tensor, threshold: float) -> np.ndarray:
    return (
        _binarize_probability_grid_for_solver(probability, threshold=threshold)
        .detach()
        .cpu()
        .numpy()
    )


def _spearman(xs: list[float], ys: list[float]) -> float:
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = mean_rank
            i = j + 1
        return ranks

    n = len(xs)
    if n < 3:
        return 0.0
    rx, ry = rank(xs), rank(ys)
    dx = np.asarray(rx) - np.asarray(ry)
    denom = np.sqrt(((np.asarray(rx) - np.mean(rx)) ** 2).sum() * ((np.asarray(ry) - np.mean(ry)) ** 2).sum())
    return float(1.0 - 6.0 * (dx ** 2).sum() / max(n * (n * n - 1), 1.0)) if denom > 0 else 0.0


def condition_response(generated: list[dict]) -> dict:
    """Correlation between condition-vector distance and output distance."""
    n = len(generated)
    arrays = [g["binary"] for g in generated]
    iou_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(arrays[i], arrays[j]).sum()
            union = np.logical_or(arrays[i], arrays[j]).sum()
            iou_matrix[i, j] = iou_matrix[j, i] = float(inter / union) if union else 1.0
    cond = np.stack([g["condition"] for g in generated])
    cond_dist = np.linalg.norm(cond[:, None, :] - cond[None, :, :], axis=2)
    output_dist = 1.0 - iou_matrix
    upper = np.triu_indices(n, k=1)
    spearman = _spearman(cond_dist[upper].tolist(), output_dist[upper].tolist())
    return {"n": n, "condition_distance_output_distance_spearman": spearman,
            "mean_output_distance": float(output_dist[upper].mean()),
            "max_output_distance": float(output_dist[upper].max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--run-id", default="diversity-novelty-2026-11-06")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    generator = OptimizedAircraftGenerator(args.checkpoint, device=device)
    latent_dim = int(generator.model_config.latent_dim)
    records = load_grounded_manifest_records(args.holdout_manifest)
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=int(generator.model_config.grid_resolution),
        latent_dim=latent_dim,
        manifest_path=args.holdout_manifest,
    )
    train_union = _load_train_union(args.train_manifest)

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else None
    if artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict] = []
    reconstruction: list[dict] = []
    report = {
        "run_id": args.run_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "train_manifest": str(Path(args.train_manifest).resolve()),
        "holdout_manifest": str(Path(args.holdout_manifest).resolve()),
        "grid_resolution": int(generator.model_config.grid_resolution),
        "latent_dim": latent_dim,
        "threshold": args.threshold,
        "sampler": args.sampler,
        "seeds": args.seeds,
        "gate_spec_sha256": GATE_SPEC_HASH,
        "generated_samples": [],
        "negative_controls": [],
        "ground_truth": [],
    }

    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            record = records[index]
            record_id = record.get("sample_id", str(index))
            condition = np.asarray(sample["condition_vector"].tolist(), dtype=np.float32)
            target = sample["geometry"]
            gt_report = sample_report(target.numpy())
            report["ground_truth"].append({"record": record_id, **gt_report})

            # Reconstruction: the record's own structured latent.
            latent = sample["latent"].reshape(1, -1).to(device, dtype=torch.float32)
            probability = torch.sigmoid(generator.converter(latent))[0]
            binary = _materialize(probability, args.threshold)
            sample_report_row = sample_report(binary, train_union)
            reconstruction.append({
                "record": record_id,
                "occupied_voxels": int(binary.sum()),
                **sample_report_row,
            })

            # Generation: conditional seed sweep.
            condition_tensor = torch.from_numpy(condition).reshape(1, -1).to(device, dtype=torch.float32)
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                probability = generator.generate(
                    num_steps=generator.diffusion_config.student_steps,
                    condition_vector=condition_tensor,
                    sampler=args.sampler,
                    ddim_steps=args.ddim_steps,
                )
                binary = _materialize(probability, args.threshold)
                if artifact_dir:
                    np.save(artifact_dir / f"{record_id}__seed{seed}.npy", binary)
                sample_report_row = sample_report(binary, train_union)
                row = {
                    "record": record_id,
                    "seed": seed,
                    "condition": condition.tolist(),
                    "occupied_voxels": int(binary.sum()),
                    "binary": binary,
                    **sample_report_row,
                }
                generated.append(row)
                report["generated_samples"].append({k: v for k, v in row.items() if k != "binary"})

    # Negative controls: a model-free constant and two structured primitives.
    n = 32
    h = n // 2
    plus = np.zeros((n, n, n), dtype=np.float32)
    plus[h - 1 : h + 2, h - 10 : h + 11, h - 1 : h + 2] = 1.0
    plus[h - 1 : h + 2, h - 1 : h + 2, h - 10 : h + 11] = 1.0
    three_box = np.zeros((n, n, n), dtype=np.float32)
    three_box[15:17, 14:18, 4:28] = 1.0
    three_box[15:17, 3:29, 10:18] = 1.0
    three_box[16:23, 15:17, 23:28] = 1.0

    for name, array in [("constant_union(train)", train_union.astype(np.float32)),
                        ("plus_cross", plus), ("three_box", three_box)]:
        row = sample_report(array, train_union)
        row["name"] = name
        report["negative_controls"].append({k: v for k, v in row.items()})

    generated_report = corpus_report(
        [g["binary"] for g in generated],
        train_union,
        conditions=[g["record"] for g in generated],
    )
    recon_binaries = []
    for index in range(len(dataset)):
        sample = dataset[index]
        latent = sample["latent"].reshape(1, -1).to(device, dtype=torch.float32)
        probability = torch.sigmoid(generator.converter(latent))[0]
        recon_binaries.append(_materialize(probability, args.threshold))
    reconstruction_report = corpus_report(recon_binaries, train_union, conditions=[r["record"] for r in reconstruction])

    cr = condition_response(generated)
    report["generation_gate_report"] = generated_report
    report["reconstruction_gate_report"] = reconstruction_report
    report["condition_response"] = cr
    report["summary"] = {
        "new_gate_pass": bool(generated_report["new_gate_pass"]),
        "old_gate_pass_rate": generated_report["old_gate_pass_rate"],
        "novel_voxel_fraction_mean": generated_report["novel_voxel_fraction_mean"],
        "distinct_array_count": generated_report["diversity"]["distinct_array_count"],
        "nontrivial_pair_count": generated_report["diversity"]["nontrivial_pair_count"],
        "condition_distance_output_distance_spearman": cr["condition_distance_output_distance_spearman"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())