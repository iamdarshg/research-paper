#!/usr/bin/env python3
"""Run a monitored training job with convergence and oscillation checks."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    CFDConfig,
    DiffusionConfig,
    ModelConfig,
    OptimizedDiffusionTrainer,
    TrainingConfig,
    aircraft_collate_fn,
    infer_conditioning_dim,
    resolve_grounded_grid_size,
)
from training_stability import compute_core_loss, summarize_stability


def _build_epoch_dataset(
    dataset: Dataset,
    *,
    max_samples_per_epoch: int,
    subset_seed: int,
) -> Dataset:
    if max_samples_per_epoch <= 0 or max_samples_per_epoch >= len(dataset):
        return dataset

    rng = random.Random(subset_seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples_per_epoch])


def _build_history_payload(
    *,
    args: argparse.Namespace,
    device: torch.device,
    history: List[Dict[str, Any]],
    stability: Dict[str, Any],
    checkpoint_path: str,
) -> Dict[str, Any]:
    return {
        "config": {
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "grid_size_requested": args.grid_size,
            "grid_size_resolved": args.resolved_grid_size,
            "learning_rate": args.learning_rate,
            "solver": args.solver,
            "cpu_threads": args.cpu_threads,
            "max_samples_per_epoch": args.max_samples_per_epoch,
            "subset_seed": args.subset_seed,
            "stability_metric": args.stability_metric,
            "convergence_window": args.convergence_window,
            "convergence_target": args.convergence_target,
            "convergence_cv_threshold": args.convergence_cv_threshold,
            "convergence_drift_threshold": args.convergence_drift_threshold,
            "oscillation_cv_threshold": args.oscillation_cv_threshold,
            "early_stop_on_convergence": args.early_stop_on_convergence,
            "save_every": args.save_every,
        },
        "device": str(device),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "history": history,
        "stability": stability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-monitored training sweep with stability checks.")
    parser.add_argument("--manifest", required=True, help="Grounded manifest used for training.")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--solver", default="D3Q27")
    parser.add_argument("--save-dir", default="./checkpoints_monitored")
    parser.add_argument("--history-output", default="./build/monitored_training/history.json")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-samples-per-epoch", type=int, default=6)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--stability-metric", default="core_loss")
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--convergence-target", type=float, default=20.0)
    parser.add_argument("--convergence-cv-threshold", type=float, default=0.08)
    parser.add_argument("--convergence-drift-threshold", type=float, default=0.35)
    parser.add_argument("--oscillation-cv-threshold", type=float, default=0.30)
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    except RuntimeError:
        pass

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=args.grid_size,
        latent_dim=args.latent_dim,
        manifest_path=args.manifest,
    )
    resolved_grid_size = resolve_grounded_grid_size(
        args.grid_size,
        detected_grid_size=dataset.grid_size,
        solver=args.solver,
        source_label=args.manifest,
    )
    args.resolved_grid_size = resolved_grid_size
    epoch_dataset = _build_epoch_dataset(
        dataset,
        max_samples_per_epoch=args.max_samples_per_epoch,
        subset_seed=args.subset_seed,
    )

    print(f"Using device: {device}")
    print(f"CPU threads capped at: {args.cpu_threads}")
    print(f"Using grounded lattice resolution: {resolved_grid_size}^3")
    print(f"Training samples per epoch: {len(epoch_dataset)}/{len(dataset)}")

    model_config = ModelConfig(
        latent_dim=args.latent_dim,
        base_grid_resolution=resolved_grid_size,
        grid_resolution=resolved_grid_size,
    )
    if model_config.conditioning_dim == 0:
        model_config.conditioning_dim = infer_conditioning_dim()

    diffusion_config = DiffusionConfig(teacher_steps=1000, student_steps=4)
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disconnection_penalty=30.0,
        precision="float32",
        enable_pipeline_parallelism=False,
    )
    cfd_config = CFDConfig(base_grid_resolution=resolved_grid_size, solver_type=args.solver)

    train_loader = DataLoader(
        epoch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=aircraft_collate_fn,
    )

    trainer = OptimizedDiffusionTrainer(model_config, diffusion_config, training_config, cfd_config, device=device)

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history_output = Path(args.history_output).resolve()
    history_output.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    final_checkpoint_path = str((save_dir / "final_monitored_model.pt").resolve())

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        metrics = trainer.train_epoch(train_loader, grid_size=resolved_grid_size)
        metrics = {
            "epoch": epoch + 1,
            **{key: float(value) for key, value in metrics.items()},
        }
        metrics["core_loss"] = compute_core_loss(metrics)
        metrics["total_minus_aero"] = float(metrics["loss"] - metrics["aerodynamic"])
        history.append(metrics)

        stability = summarize_stability(
            history,
            metric=args.stability_metric,
            window=args.convergence_window,
            convergence_target=args.convergence_target,
            convergence_cv_threshold=args.convergence_cv_threshold,
            convergence_drift_threshold=args.convergence_drift_threshold,
            oscillation_cv_threshold=args.oscillation_cv_threshold,
        )

        print(
            "Stability "
            f"status={stability['status']} "
            f"metric={args.stability_metric} "
            f"mean={stability.get('metric_stats', {}).get('mean', 0.0):.4f} "
            f"cv={stability.get('metric_stats', {}).get('cv', 0.0):.4f}"
        )
        if stability.get("suspected_root_cause"):
            print(f"Suspected instability root cause: {stability['suspected_root_cause']}")

        payload = _build_history_payload(
            args=args,
            device=device,
            history=history,
            stability=stability,
            checkpoint_path=final_checkpoint_path,
        )
        history_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_monitored_ep{epoch + 1}.pt"
            trainer.save_checkpoint(str(checkpoint_path))

        if args.early_stop_on_convergence and stability["converged"]:
            print(f"Early stopping at epoch {epoch + 1}: convergence criteria met.")
            break

    trainer.save_checkpoint(final_checkpoint_path)
    print(f"Final monitored checkpoint saved to {final_checkpoint_path}")
    print(f"History written to {history_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
