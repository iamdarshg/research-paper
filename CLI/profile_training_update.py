#!/usr/bin/env python
"""Reproducible single-update profiler for the training pipeline.

Per docs/to_be_read.md PHASE 1. Exercises ONE representative optimizer update
through the real trainer path (96^3, batch=1, D3Q27, real coordinate decoder,
real direct/SPSA solver, real backward pass, real optimizer step) using a fixed
sample and fixed RNG, then reports per-phase wall times and per-update
statistics (mean/median/p90/p95).

Modes:
  --full-update    complete optimizer updates through train_epoch (default)
  --direct-only    the SPSA direct-solver phase only (no data/model loss);
                   drives trainer.direct_solver_loss on a fixed-shape field
  --solver-only    isolated CFD solves on the base geometry (warm q-cache floor)

Each instrumented phase ends with torch.cuda.synchronize() so its wall time
includes the GPU work an async launch would otherwise defer to the next .item().

Usage:
  python CLI/profile_training_update.py --warmup 1 --iterations 3 --full-update
  python CLI/profile_training_update.py --direct-only --iterations 1
  python CLI/profile_training_update.py --solver-only --iterations 5
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPO_ROOT / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import aircraft_diffusion_cfd as adc  # noqa: E402
import advanced_lbm_solver as lbs  # noqa: E402
from experiment_config import config_value  # noqa: E402
from run_monitored_training import (  # noqa: E402
    _build_epoch_dataset,
    _prepare_geometry_threshold_for_run,
    aircraft_collate_fn,
    prepare_edt_workspace,
)

# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
TIMERS: dict[str, list[float]] = {}  # name -> [total_s, call_count]


def _instrument(cls_or_module, name):
    owner = getattr(cls_or_module, name, None)
    if owner is None:
        return

    def make_wrapper(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            entry = TIMERS.setdefault(name, [0.0, 0.0])
            entry[0] += time.perf_counter() - t0
            entry[1] += 1.0
            return result

        wrapper.__name__ = name
        return wrapper

    setattr(cls_or_module, name, make_wrapper(owner))


def install_instrumentation():
    _instrument(adc.AdvancedCFDSimulator, "simulate_aerodynamics")
    _instrument(adc.AdvancedCFDSimulator, "init_flow_field")
    _instrument(adc.AdvancedCFDSimulator, "compute_aerodynamic_coefficients")
    _instrument(lbs.D3Q27Solver, "collide_and_stream")
    _instrument(lbs.D3Q27Solver, "_get_q")
    _instrument(adc, "evaluate_aircraft_validity")
    _instrument(adc, "_direct_measured_objective_for_single")
    # ---- model-phase instrumentation (the un-instrumented ~72s/update) ----
    _instrument(adc.OptimizedDiffusionTrainer, "_compute_consistency_loss")
    _instrument(adc.ConsistencyModel, "fast_inference")
    _instrument(adc.LatentTo3DConverter, "forward")
    _instrument(adc.LatentTo3DConverter, "forward_flat_indices")
    _instrument(adc.LatentTo3DConverter, "_checkpointed_coordinate_chunk")
    _instrument(adc.LatentTo3DConverter, "_encode_coordinates")
    # diffusion_model, student_model, and teacher_model are all
    # LatentDiffusionUNet instances, so one class-level instrument covers the
    # total UNet forward cost (diffusion + consistency student + teacher).
    _instrument(adc.LatentDiffusionUNet, "forward")


def _report_phases():
    order = (
        "_direct_measured_objective_for_single",
        "simulate_aerodynamics",
        "collide_and_stream",
        "_get_q",
        "init_flow_field",
        "compute_aerodynamic_coefficients",
        "evaluate_aircraft_validity",
    )
    print(f"\n{'phase':<42} {'calls':>6} {'total_s':>9} {'per_call_ms':>11} {'share':>7}")
    grand = TIMERS.get("_direct_measured_objective_for_single", [0.0, 0.0])[0]
    for name in order:
        total_s, count = TIMERS.get(name, [0.0, 0.0])
        if count <= 0:
            continue
        per = total_s * 1000.0 / count
        share = (total_s / grand * 100.0) if grand > 0 else 0.0
        print(f"{name:<42} {int(count):>6} {total_s:>9.2f} {per:>11.1f} {share:>6.1f}%")
    return grand


def _stats(values):
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": sorted(values)[max(0, int(0.90 * len(values)) - 1)],
        "p95": sorted(values)[max(0, int(0.95 * len(values)) - 1)],
    }


# ---------------------------------------------------------------------------
# Builders (mirror select_recovery_checkpoint / run_monitored_training wiring)
# ---------------------------------------------------------------------------
def build_trainer_and_loader(checkpoint: Path, manifest: Path, device, samples: int, solver: str):
    # weights_only=False is required and intentional: the candidate may be an
    # interruption-safe run-state checkpoint that embeds torch rng state and
    # compatibility mappings. This is a trusted local artifact from our own
    # run, never untrusted input.
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_config = (
        adc.ModelConfig(**metadata["model_config"])
        if "model_config" in metadata
        else adc.ModelConfig(**metadata["compatibility"]["configuration"]["model_config"])
    )
    resolved_grid_size = int(model_config.grid_resolution)
    prepare_edt_workspace((resolved_grid_size,) * 3)

    dataset = adc.AircraftDesignDataset(
        num_samples=0,
        grid_size=resolved_grid_size,
        latent_dim=int(model_config.latent_dim),
        manifest_path=str(manifest),
    )
    epoch_dataset = _build_epoch_dataset(
        dataset, max_samples_per_epoch=samples, subset_seed=0, split="train"
    )
    loader = DataLoader(
        epoch_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=aircraft_collate_fn,
    )
    training_config = adc.TrainingConfig(
        num_epochs=1,
        batch_size=1,
        learning_rate=float(config_value("training", "learning_rate", 2e-5)),
        direct_solver_loss_weight=1.0,
        direct_solver_interval=1,
        direct_solver_steps=5,
        direct_solver_directions=16,
        direct_solver_perturbation=0.15,
        direct_solver_perturbation_grid_size=12,
        direct_connectivity_weight=1.0,
        direct_aircraft_validity_weight=1.0,
        overfit_geometry_gate_samples=samples,
        promotion_generation_seeds=6,
        require_direct_solver_every_iteration=True,
    )
    cfd_config = adc.CFDConfig(
        base_grid_resolution=resolved_grid_size,
        solver_type=solver,
        use_fused_stream_bfl=True,
    )
    diffusion_config = adc.DiffusionConfig(teacher_steps=1000, student_steps=4)
    trainer = adc.OptimizedDiffusionTrainer(
        model_config,
        diffusion_config,
        training_config,
        cfd_config,
        device=device,
    )
    trainer.load_checkpoint(str(checkpoint))
    _prepare_geometry_threshold_for_run(trainer, loader, resume_run_state=None)
    return trainer, loader


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_full_update(trainer, loader, warmup: int, iterations: int, profile_cuda: bool) -> dict:
    trainer.scheduler_step_per_update = True
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, warmup + iterations)
    )
    trainer.update_metrics_callback = None
    trainer.run_state_checkpoint_callback = None
    trainer.run_state_checkpoint_path = None
    trainer.stop_after_updates = int(trainer.global_step) + warmup + iterations

    update_times = []
    if profile_cuda:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            t0 = time.perf_counter()
            trainer.train_epoch(loader, grid_size=96, start_batch=0)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        trace_path = REPO_ROOT / "build" / "perf" / "baseline" / "update_trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print("chrome trace:", trace_path)
    else:
        t0 = time.perf_counter()
        trainer.train_epoch(loader, grid_size=96, start_batch=0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    total = t1 - t0
    # train_epoch does not report per-update times; report the aggregate and
    # the per-phase table which is summed over every update.
    update_times.append(total / max(1, warmup + iterations))
    return {
        "mode": "full_update",
        "updates": warmup + iterations,
        "total_wall_s": total,
        "update_stats": _stats([total / max(1, warmup + iterations)]),
        "phases": {k: {"total_s": v[0], "calls": int(v[1])} for k, v in TIMERS.items()},
    }


def run_direct_only(trainer, iterations: int, seed: int = 1234) -> dict:
    field = torch.randn(1, 96, 96, 96, device=trainer.device)
    spec = adc.DesignSpec(target_speed=90.0, wingspan_limit_m=1.2)
    times = []
    for i in range(iterations):
        torch.manual_seed(seed + i)
        t0 = time.perf_counter()
        _ = trainer.direct_solver_loss(
            field, spec, trainer.cfd_simulator, seed=seed + i
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    _report_phases()
    return {
        "mode": "direct_only",
        "iterations": iterations,
        "per_call_s": times,
        "stats": _stats(times),
        "phases": {k: {"total_s": v[0], "calls": int(v[1])} for k, v in TIMERS.items()},
    }


def run_solver_only(trainer, iterations: int) -> dict:
    field = torch.rand(1, 96, 96, 96, device=trainer.device)
    solver_geom = adc._canonical_training_geometry_to_solver_xyz(
        (field.squeeze(0) > 0.5).float()
    ).to(trainer.cfd_simulator.device)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = trainer.cfd_simulator.simulate_aerodynamics(solver_geom, steps=5)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    _report_phases()
    return {
        "mode": "solver_only",
        "iterations": iterations,
        "per_call_s": times,
        "stats": _stats(times),
        "note": "warm q-cache floor on the SAME geometry; real SPSA solves have a cold q-cache per perturbation",
        "phases": {k: {"total_s": v[0], "calls": int(v[1])} for k, v in TIMERS.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(REPO_ROOT / "build" / "recovery_ladder_20260814" / "step1305.pt"))
    ap.add_argument("--manifest", default=str(REPO_ROOT / "build" / "grounded_combined_1k_20260716" / "manifest.jsonl"))
    ap.add_argument("--warmup", type=int, default=1, help="updates for warmup (absorbs Triton JIT)")
    ap.add_argument("--iterations", type=int, default=3, help="measured updates / solver calls")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--profile-cuda", action="store_true", help="emit a chrome-trace of the full update")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full-update", action="store_true", help="complete optimizer updates (default)")
    mode.add_argument("--direct-only", action="store_true", help="SPSA direct-solver phase only")
    mode.add_argument("--solver-only", action="store_true", help="isolated CFD solves (warm q-cache)")
    ap.add_argument("--output", default=str(REPO_ROOT / "build" / "perf" / "baseline" / "profile_result.json"))
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device:", device)
    print("torch:", torch.__version__, "cuda:", torch.version.cuda if torch.cuda.is_available() else "n/a")

    if args.solver_only:
        trainer, _ = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=1, solver=str(config_value("cfd", "solver", "D3Q27")),
        )
        result = run_solver_only(trainer, args.iterations)
    elif args.direct_only:
        install_instrumentation()
        trainer, _ = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=1, solver=str(config_value("cfd", "solver", "D3Q27")),
        )
        result = run_direct_only(trainer, args.iterations)
    else:
        install_instrumentation()
        trainer, loader = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=args.warmup + args.iterations,
            solver=str(config_value("cfd", "solver", "D3Q27")),
        )
        result = run_full_update(trainer, loader, args.warmup, args.iterations, args.profile_cuda)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
