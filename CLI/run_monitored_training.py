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
from torch.optim.lr_scheduler import CosineAnnealingLR

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
from experiment_config import GLOBAL_CONFIG_PATH, config_value
from training_stability import compute_core_loss, summarize_stability
from sdf_utils import prepare_edt_workspace


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


def _geometry_promotion_metrics(
    promotion: Dict[str, Any],
) -> tuple[Dict[str, float], tuple[float, float, float, float]]:
    metrics = {
        "promotion_reconstruction_topk_recall": float(
            promotion.get("reconstruction_topk_recall", 0.0)
        ),
        "promotion_generated_topk_recall": float(
            promotion.get("generated_topk_recall", 0.0)
        ),
        "promotion_generated_worst_topk_recall": float(
            promotion.get("generated_worst_topk_recall", 0.0)
        ),
        "promotion_generated_aircraft_valid_fraction": float(
            promotion.get("generated_aircraft_valid_fraction", 0.0)
        ),
        "promotion_gate_passed": float(promotion.get("status") == "pass"),
    }
    rank = (
        metrics["promotion_generated_aircraft_valid_fraction"],
        metrics["promotion_generated_worst_topk_recall"],
        metrics["promotion_generated_topk_recall"],
        metrics["promotion_reconstruction_topk_recall"],
    )
    metrics["geometry_selection_metric"] = 1.0 - rank[2]
    return metrics, rank


def _build_history_payload(
    *,
    args: argparse.Namespace,
    device: torch.device,
    history: List[Dict[str, Any]],
    stability: Dict[str, Any],
    checkpoint_path: str,
    model_config: ModelConfig,
    best_checkpoint_path: str | None = None,
    best_geometry_metric: float | None = None,
    initial_geometry_promotion: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "config": {
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None,
            "warm_start_from": (
                str(Path(args.warm_start_from).resolve())
                if args.warm_start_from
                else None
            ),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "global_config": str(GLOBAL_CONFIG_PATH),
            "latent_dim": model_config.latent_dim,
            "grid_size_requested": args.grid_size,
            "grid_size_resolved": args.resolved_grid_size,
            "learning_rate": args.learning_rate,
            "converter_learning_rate": config_value("training", "converter_learning_rate", 1e-3),
            "consistency_student_learning_rate": config_value(
                "training", "consistency_student_learning_rate", 2e-4
            ),
            "solver": args.solver,
            "cpu_threads": args.cpu_threads,
            "max_samples_per_epoch": args.max_samples_per_epoch,
            "subset_seed": args.subset_seed,
            "stability_metric": args.stability_metric,
            "convergence_window": args.convergence_window,
            "convergence_target": args.convergence_target,
            "convergence_cv_threshold": args.convergence_cv_threshold,
            "convergence_drift_threshold": args.convergence_drift_threshold,
            "required_geometry_loss_max": args.required_geometry_loss_max,
            "oscillation_cv_threshold": args.oscillation_cv_threshold,
            "early_stop_on_convergence": args.early_stop_on_convergence,
            "save_every": args.save_every,
            "direct_solver_loss_weight": args.direct_solver_loss_weight,
            "direct_solver_steps": args.direct_solver_steps,
            "direct_solver_directions": args.direct_solver_directions,
            "direct_connectivity_weight": args.direct_connectivity_weight,
            "direct_aircraft_validity_weight": args.direct_aircraft_validity_weight,
            "direct_solver_perturbation": args.direct_solver_perturbation,
            "direct_solver_perturbation_grid_size": args.direct_solver_perturbation_grid_size,
            "direct_solver_gradient_clip": config_value(
                "training", "direct_solver_gradient_clip", 1.0
            ),
        },
        "device": str(device),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "best_checkpoint_path": (
            str(Path(best_checkpoint_path).resolve()) if best_checkpoint_path else None
        ),
        "best_geometry_metric": best_geometry_metric,
        "initial_geometry_promotion": initial_geometry_promotion,
        "history": history,
        "stability": stability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-monitored training sweep with stability checks.")
    parser.add_argument("--manifest", required=True, help="Grounded manifest used for training.")
    parser.add_argument("--num-epochs", type=int, default=int(config_value("training", "num_epochs", 200)))
    parser.add_argument("--batch-size", type=int, default=int(config_value("training", "batch_size", 1)))
    parser.add_argument("--latent-dim", type=int, default=int(config_value("model", "latent_dim", 192)))
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=float(config_value("training", "learning_rate", 2e-4)))
    parser.add_argument("--solver", default=str(config_value("cfd", "solver", "D3Q27")))
    parser.add_argument("--save-dir", default="./checkpoints_monitored")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--warm-start-from", default=None)
    parser.add_argument("--history-output", default="./build/monitored_training/history.json")
    parser.add_argument("--save-every", type=int, default=int(config_value("training", "save_interval", 25)))
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-samples-per-epoch", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--stability-metric", default="optimization_loss")
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--convergence-target", type=float, default=20.0)
    parser.add_argument("--convergence-cv-threshold", type=float, default=0.08)
    parser.add_argument("--convergence-drift-threshold", type=float, default=0.35)
    parser.add_argument(
        "--required-geometry-loss-max",
        type=float,
        default=float(config_value("training", "required_geometry_loss_max", 0.20)),
    )
    parser.add_argument("--oscillation-cv-threshold", type=float, default=0.30)
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument("--direct-solver-loss-weight", type=float, default=float(config_value("training", "direct_solver_loss_weight", 1.0)))
    parser.add_argument("--direct-solver-steps", type=int, default=int(config_value("training", "direct_solver_steps", 5)))
    parser.add_argument("--direct-solver-directions", type=int, default=int(config_value("training", "direct_solver_directions", 16)))
    parser.add_argument("--direct-connectivity-weight", type=float, default=float(config_value("training", "direct_connectivity_weight", 1.0)))
    parser.add_argument("--direct-aircraft-validity-weight", type=float, default=float(config_value("training", "direct_aircraft_validity_weight", 1.0)))
    parser.add_argument("--direct-solver-perturbation", type=float, default=float(config_value("training", "direct_solver_perturbation", 0.15)))
    parser.add_argument("--direct-solver-perturbation-grid-size", type=int, default=int(config_value("training", "direct_solver_perturbation_grid_size", 12)))
    args = parser.parse_args()
    if args.resume_from and args.warm_start_from:
        parser.error("--resume-from and --warm-start-from are mutually exclusive")

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
    prepare_edt_workspace((resolved_grid_size,) * 3)

    model_config = ModelConfig.scaled_for_corpus(
        int(dataset.metadata.get("unique_geometry_count", len(dataset))),
        resolved_grid_size,
        conditioning_dim=infer_conditioning_dim(),
        latent_dim=args.latent_dim,
    )
    if args.resume_from:
        checkpoint_metadata = torch.load(
            args.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_model_config = ModelConfig(**checkpoint_metadata["model_config"])
        if int(checkpoint_model_config.grid_resolution) != int(resolved_grid_size):
            raise ValueError(
                "Resume checkpoint grid resolution does not match the grounded dataset: "
                f"{checkpoint_model_config.grid_resolution} != {resolved_grid_size}"
            )
        if int(checkpoint_model_config.latent_dim) != int(args.latent_dim):
            raise ValueError(
                "Resume checkpoint latent width does not match --latent-dim: "
                f"{checkpoint_model_config.latent_dim} != {args.latent_dim}"
            )
        model_config = checkpoint_model_config
    if int(dataset.latent_dim) != int(model_config.latent_dim):
        dataset = AircraftDesignDataset(
            num_samples=0,
            grid_size=resolved_grid_size,
            latent_dim=model_config.latent_dim,
            manifest_path=args.manifest,
        )
    epoch_dataset = _build_epoch_dataset(
        dataset,
        max_samples_per_epoch=args.max_samples_per_epoch,
        subset_seed=args.subset_seed,
    )

    print(f"Using device: {device}")
    print(f"CPU threads capped at: {args.cpu_threads}")
    print(f"Using grounded lattice resolution: {resolved_grid_size}^3")
    print(f"Training samples per epoch: {len(epoch_dataset)}/{len(dataset)}")

    diffusion_config = DiffusionConfig(teacher_steps=1000, student_steps=4)
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disconnection_penalty=30.0,
        precision="float32",
        enable_pipeline_parallelism=False,
        direct_solver_loss_weight=args.direct_solver_loss_weight,
        direct_solver_interval=1,
        direct_solver_steps=args.direct_solver_steps,
        direct_solver_directions=args.direct_solver_directions,
        direct_solver_perturbation=args.direct_solver_perturbation,
        direct_solver_perturbation_grid_size=args.direct_solver_perturbation_grid_size,
        direct_connectivity_weight=args.direct_connectivity_weight,
        direct_aircraft_validity_weight=args.direct_aircraft_validity_weight,
        require_direct_solver_every_iteration=True,
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
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    elif args.warm_start_from:
        trainer.warm_start_checkpoint(args.warm_start_from)
    trainer.scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=max(1, int(args.num_epochs)),
    )

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history_output = Path(args.history_output).resolve()
    history_output.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    final_checkpoint_path = str((save_dir / "final_monitored_model.pt").resolve())
    best_checkpoint_path = str((save_dir / "best_geometry_model.pt").resolve())
    best_geometry_metric = float("inf")
    best_promotion_rank = (-1.0, -1.0, -1.0, -1.0)
    selection_interval = max(1, int(diffusion_config.student_steps))
    initial_geometry_promotion = None

    if args.resume_from or args.warm_start_from:
        python_rng_state = random.getstate()
        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        baseline_promotion = trainer.evaluate_geometry_promotion_gate(train_loader)
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        baseline_metrics, best_promotion_rank = _geometry_promotion_metrics(
            baseline_promotion
        )
        best_geometry_metric = baseline_metrics["geometry_selection_metric"]
        initial_geometry_promotion = {
            **baseline_metrics,
            "status": str(baseline_promotion.get("status", "fail")),
            "source_checkpoint": str(
                Path(args.resume_from or args.warm_start_from).resolve()
            ),
        }
        trainer.save_checkpoint(best_checkpoint_path)
        print(
            "Initial geometry promotion baseline: "
            f"valid_fraction={best_promotion_rank[0]:.6g}, "
            f"worst_recall={best_promotion_rank[1]:.6g}, "
            f"mean_recall={best_promotion_rank[2]:.6g}"
        )

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        metrics = trainer.train_epoch(train_loader, grid_size=resolved_grid_size)
        metrics = {
            "epoch": epoch + 1,
            **{key: float(value) for key, value in metrics.items()},
        }
        trainer.scheduler.step()
        for group in trainer.optimizer.param_groups:
            group_name = str(group.get("name", "unnamed"))
            metrics[f"learning_rate_{group_name}"] = float(group.get("lr", 0.0))
        metrics["core_loss"] = compute_core_loss(metrics)
        metrics["selected_as_best_geometry_checkpoint"] = 0.0
        metrics["geometry_selection_evaluated"] = 0.0
        if (epoch + 1) % selection_interval == 0:
            promotion = trainer.evaluate_geometry_promotion_gate(train_loader)
            metrics["geometry_selection_evaluated"] = 1.0
            promotion_metrics, promotion_rank = _geometry_promotion_metrics(promotion)
            metrics.update(promotion_metrics)
            if promotion_rank > best_promotion_rank:
                best_promotion_rank = promotion_rank
                best_geometry_metric = metrics["geometry_selection_metric"]
                trainer.save_checkpoint(best_checkpoint_path)
                metrics["selected_as_best_geometry_checkpoint"] = 1.0
        else:
            metrics["geometry_selection_metric"] = float("nan")
        history.append(metrics)

        stability = summarize_stability(
            history,
            metric=args.stability_metric,
            window=args.convergence_window,
            convergence_target=args.convergence_target,
            convergence_cv_threshold=args.convergence_cv_threshold,
            convergence_drift_threshold=args.convergence_drift_threshold,
            oscillation_cv_threshold=args.oscillation_cv_threshold,
            required_geometry_loss_max=args.required_geometry_loss_max,
        )

        metric_stats = stability.get("metric_stats", {})
        latest_metric = metrics.get(args.stability_metric, 0.0)
        print(
            "Stability "
            f"status={stability['status']} "
            f"metric={args.stability_metric} "
            f"mean={metric_stats.get('mean', latest_metric):.4f} "
            f"cv={metric_stats.get('cv', 0.0):.4f}"
        )
        if stability.get("suspected_root_cause"):
            print(f"Suspected instability root cause: {stability['suspected_root_cause']}")

        payload = _build_history_payload(
            args=args,
            device=device,
            history=history,
            stability=stability,
            checkpoint_path=final_checkpoint_path,
            model_config=model_config,
            best_checkpoint_path=best_checkpoint_path,
            best_geometry_metric=(
                best_geometry_metric if np.isfinite(best_geometry_metric) else None
            ),
            initial_geometry_promotion=initial_geometry_promotion,
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
