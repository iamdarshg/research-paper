#!/usr/bin/env python3
"""Diagnose where the generated-occupancy inflation enters the pipeline.

Read-only measurement on an existing checkpoint.  For a sample of held-out
records this reports, at the frozen materialization threshold 0.5 and without
any tuning:

  * the source sample's occupancy (target);
  * clean-latent decode: probability mass and materialized occupancy;
  * DDIM-50 generated latent: value summary (saturation at the [0,1] support
    boundary), decoded probability mass, materialized occupancy, IoU vs target;
  * distance of the generated latent to the nearest *training* latent, against
    the training latent cloud's own nearest-neighbour distance (is the
    generative latent on-manifold?);
  * decoder response on synthetic off-manifold probes (zero latent, mean
    training latent, uniform-random latent) to separate "decoder has an
    occupancy floor" from "generative latent is off-manifold".

Nothing here changes any model, loss, or gate.  It only measures.
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
    sample_latent_with_ddim,
)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _materialize(prob: torch.Tensor, threshold: float) -> np.ndarray:
    return (
        _binarize_probability_grid_for_solver(prob, threshold=threshold)
        .detach()
        .cpu()
        .numpy()
    )


def _decode(generator, latent: torch.Tensor, threshold: float) -> dict:
    prob = torch.sigmoid(generator.converter(latent.reshape(1, -1)))[0]
    binary = _materialize(prob, threshold)
    return {
        "probability_mass": float(prob.mean().item()),
        "occupied_voxels": int(binary.sum()),
        "occupancy_ratio": float(binary.mean()),
        "binary": binary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--records", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)
    generator = OptimizedAircraftGenerator(args.checkpoint, device=device)
    latent_dim = int(generator.model_config.latent_dim)

    train_dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=int(generator.model_config.grid_resolution),
        latent_dim=latent_dim,
        manifest_path=args.train_manifest,
    )
    holdout_dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=int(generator.model_config.grid_resolution),
        latent_dim=latent_dim,
        manifest_path=args.holdout_manifest,
    )
    holdout_records = load_grounded_manifest_records(args.holdout_manifest)

    train_latents = torch.stack(
        [train_dataset[i]["latent"] for i in range(len(train_dataset))]
    ).float()
    train_latent_mean = train_latents.mean(dim=0)
    train_latent_std = train_latents.std(dim=0)
    # Training-cloud scale: nearest-neighbour distance among a subsample.
    sample_size = min(200, len(train_latents))
    pick = torch.linspace(0, len(train_latents) - 1, sample_size).round().long()
    subsample = train_latents.index_select(0, pick)
    nn_dists = []
    for i in range(subsample.shape[0]):
        d = torch.norm(subsample - subsample[i], dim=1)
        d[i] = float("inf")
        nn_dists.append(float(d.min()))
    train_nn_mean = float(np.mean(nn_dists))
    train_nn_max = float(np.max(nn_dists))

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "threshold": args.threshold,
        "ddim_steps": args.ddim_steps,
        "seeds": args.seeds,
        "latent_dim": latent_dim,
        "train_latent_stats": {
            "mean_of_means": float(train_latents.mean()),
            "mean_of_stds": float(train_latents.std()),
            "min": float(train_latents.min()),
            "max": float(train_latents.max()),
            "fraction_at_support_boundary": float(
                ((train_latents <= 1e-6) | (train_latents >= 1.0 - 1e-6))
                .float()
                .mean()
            ),
            "nearest_neighbour_distance_mean": train_nn_mean,
            "nearest_neighbour_distance_max": train_nn_max,
        },
        "decoder_probes": {},
        "records": [],
    }

    with torch.no_grad():
        # --- decoder probes: what does the frozen decoder emit off-manifold? ---
        probes = {
            "zero_latent": torch.zeros(1, latent_dim, device=device),
            "train_mean_latent": train_latent_mean.reshape(1, -1).to(device),
            "uniform_random_latent": torch.rand(
                1, latent_dim, generator=torch.Generator().manual_seed(11)
            ).to(device),
            "binarized_train_mean_latent": (train_latent_mean > 0.5)
            .float()
            .reshape(1, -1)
            .to(device),
        }
        for name, latent in probes.items():
            probe = _decode(generator, latent, args.threshold)
            probe.pop("binary")
            report["decoder_probes"][name] = probe

        # --- per-record: clean latent vs free-running generated latent ---
        for index in range(min(args.records, len(holdout_dataset))):
            sample = holdout_dataset[index]
            record = holdout_records[index]
            record_id = record.get("sample_id", str(index))
            target = sample["geometry"].numpy() > 0.5
            condition = np.asarray(
                sample["condition_vector"].tolist(), dtype=np.float32
            )
            condition_tensor = (
                torch.from_numpy(condition).reshape(1, -1).to(device, dtype=torch.float32)
            )
            row = {
                "record": record_id,
                "target_occupied_voxels": int(target.sum()),
                "target_occupancy_ratio": float(target.mean()),
            }

            clean_latent = sample["latent"].reshape(1, -1).to(device, dtype=torch.float32)
            clean = _decode(generator, clean_latent, args.threshold)
            clean_iou = _iou(clean["binary"], target)
            clean.pop("binary")
            row["clean_latent_decode"] = {**clean, "iou_vs_target": clean_iou}

            generated = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                latent = sample_latent_with_ddim(
                    generator.diffusion_model,
                    generator.noise_schedule,
                    (1, latent_dim),
                    num_steps=args.ddim_steps,
                    condition=condition_tensor,
                    latent_value_min=0.0,
                    latent_value_max=1.0,
                )
                latent_cpu = latent.detach().float().cpu()
                decode = _decode(generator, latent, args.threshold)
                decode_iou = _iou(decode["binary"], target)
                decode.pop("binary")
                dists = torch.norm(train_latents - latent_cpu, dim=1)
                generated.append(
                    {
                        "seed": seed,
                        "latent_mean": float(latent_cpu.mean()),
                        "latent_std": float(latent_cpu.std()),
                        "latent_min": float(latent_cpu.min()),
                        "latent_max": float(latent_cpu.max()),
                        "latent_fraction_at_support_boundary": float(
                            ((latent_cpu <= 1e-6) | (latent_cpu >= 1.0 - 1e-6))
                            .float()
                            .mean()
                        ),
                        "latent_distance_to_nearest_train_latent": float(dists.min()),
                        "distance_to_train_latent_mean": float(
                            torch.norm(train_latent_mean - latent_cpu[0])
                        ),
                        **decode,
                        "iou_vs_target": decode_iou,
                    }
                )
            row["generated"] = generated
            report["records"].append(row)
            print(
                f"{record_id}: target {row['target_occupied_voxels']}, "
                f"clean {row['clean_latent_decode']['occupied_voxels']}, "
                f"gen {[g['occupied_voxels'] for g in generated]}",
                flush=True,
            )

    # --- aggregate ---
    target_occ = np.array([r["target_occupied_voxels"] for r in report["records"]], dtype=float)
    clean_occ = np.array(
        [r["clean_latent_decode"]["occupied_voxels"] for r in report["records"]], dtype=float
    )
    gen_occ = np.array(
        [[g["occupied_voxels"] for g in r["generated"]] for r in report["records"]], dtype=float
    )
    gen_mass = np.array(
        [[g["probability_mass"] for g in r["generated"]] for r in report["records"]], dtype=float
    )
    clean_mass = np.array(
        [r["clean_latent_decode"]["probability_mass"] for r in report["records"]], dtype=float
    )
    gen_nn = np.array(
        [[g["latent_distance_to_nearest_train_latent"] for g in r["generated"]] for r in report["records"]],
        dtype=float,
    )
    report["aggregate"] = {
        "target_occupied_mean": float(target_occ.mean()),
        "clean_occupied_mean": float(clean_occ.mean()),
        "generated_occupied_mean": float(gen_occ.mean()),
        "generated_occupied_min": float(gen_occ.min()),
        "generated_occupied_max": float(gen_occ.max()),
        "generated_over_target_ratio_mean": float((gen_occ.mean(axis=1) / target_occ).mean()),
        "clean_over_target_ratio_mean": float((clean_occ / target_occ).mean()),
        "clean_probability_mass_mean": float(clean_mass.mean()),
        "generated_probability_mass_mean": float(gen_mass.mean()),
        "target_probability_mass_mean": float((target_occ / 32768.0).mean()),
        "generated_latent_nn_distance_mean": float(gen_nn.mean()),
        "train_latent_nn_distance_mean": train_nn_mean,
        "generated_iou_vs_target_mean": float(
            np.mean([[g["iou_vs_target"] for g in r["generated"]] for r in report["records"]])
        ),
        "clean_iou_vs_target_mean": float(
            np.mean([r["clean_latent_decode"]["iou_vs_target"] for r in report["records"]])
        ),
        "generated_latent_saturation_mean": float(
            np.mean(
                [[g["latent_fraction_at_support_boundary"] for g in r["generated"]] for r in report["records"]]
            )
        ),
        "train_latent_saturation_mean": float(
            ((train_latents <= 1e-6) | (train_latents >= 1.0 - 1e-6)).float().mean()
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(report["decoder_probes"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
