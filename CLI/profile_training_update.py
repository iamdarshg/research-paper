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

Per-update wall time (--full-update) is measured at the trainer's natural
per-update boundary (update_metrics_callback), warmup updates are unmeasured,
and the stats (mean/median/p90/p95) are computed over the M>=5 measured updates
only -- warmup is never in the denominator. A single synchronize at each update
boundary folds the just-launched GPU work into the wall delta; the per-call
synchronize inside the instrumented phase wrappers is optional
(--sync-per-call) and never feeds the wall numbers.

--fresh-init constructs the trainer WITHOUT loading a checkpoint and with a
procedural synthetic loader, so --full-update runs on any worktree with no
checkpoint present (the per-update cost is dominated by the SPSA solves +
decoder + solver, which are to first order weights-independent).

Usage:
  python CLI/profile_training_update.py --warmup 1 --iterations 5 --full-update
  python CLI/profile_training_update.py --warmup 1 --iterations 5 --full-update --fresh-init
  python CLI/profile_training_update.py --direct-only --iterations 1
  python CLI/profile_training_update.py --solver-only --iterations 5
"""
import argparse
import json
import logging
import pickle
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

# ---------------------------------------------------------------------------
# Trusted checkpoint loading (security: CWE-502 safe-deserialization gate)
# ---------------------------------------------------------------------------
# The exception set torch's weights_only=True loader raises when it rejects a
# checkpoint that embeds non-whitelisted globals (run-state RNG, custom
# compatibility objects). Depending on how the pickle was produced this is any
# of these, not just pickle.UnpicklingError.
_WEIGHTS_ONLY_FALLBACK_EXCEPTIONS = (
    pickle.UnpicklingError,
    AttributeError,
    TypeError,
    ModuleNotFoundError,
    ImportError,
    EOFError,
)

# Only checkpoints under this root are ever eligible for the weights_only=False
# fallback. These are trusted local artifacts from our own runs at explicit
# paths, never untrusted input.
_TRUSTED_CHECKPOINT_ROOT = REPO_ROOT / "build"


def _is_trusted_checkpoint_path(path) -> bool:
    """True when ``path`` resolves inside the trusted build/ checkpoint root."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    try:
        trusted_root = _TRUSTED_CHECKPOINT_ROOT.resolve()
    except OSError:
        return False
    return resolved == trusted_root or trusted_root in resolved.parents


def _load_checkpoint_metadata(checkpoint: Path):
    """Load checkpoint metadata preferring the safe weights_only=True loader.

    ``weights_only=True`` rejects any checkpoint that embeds non-whitelisted
    globals by raising one of ``_WEIGHTS_ONLY_FALLBACK_EXCEPTIONS``. We fall back
    to the unsafe ``weights_only=False`` loader ONLY for a trusted local
    artifact that resolves under the build/ root, and we log a warning when we
    do. Untrusted paths re-raise: we never deserialize untrusted input.
    """
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except _WEIGHTS_ONLY_FALLBACK_EXCEPTIONS as exc:
        if not _is_trusted_checkpoint_path(checkpoint):
            logging.getLogger(__name__).error(
                "weights_only=True rejected %s (%s); refusing weights_only=False "
                "fallback for an untrusted checkpoint path",
                checkpoint,
                exc,
            )
            raise
        logging.getLogger(__name__).warning(
            "weights_only=True rejected %s (%s); falling back to "
            "weights_only=False for trusted local checkpoint under %s",
            checkpoint,
            exc,
            _TRUSTED_CHECKPOINT_ROOT,
        )
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

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


def _instrument(cls_or_module, name, sync_per_call: bool = True):
    owner = getattr(cls_or_module, name, None)
    if owner is None:
        return

    def make_wrapper(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            if sync_per_call and torch.cuda.is_available():
                torch.cuda.synchronize()
            entry = TIMERS.setdefault(name, [0.0, 0.0])
            entry[0] += time.perf_counter() - t0
            entry[1] += 1.0
            return result

        wrapper.__name__ = name
        return wrapper

    setattr(cls_or_module, name, make_wrapper(owner))


def install_instrumentation(sync_per_call: bool = True):
    """Instrument the hot phases.

    ``sync_per_call`` makes the per-call ``torch.cuda.synchronize()`` optional:
    when False the phase table is approximate (deferred GPU work is excluded)
    but per-update wall numbers -- which come from the update-boundary callback
    in ``run_full_update``, never from per-call syncs -- stay exact. Defaults to
    True for the legacy direct/solver phase-attribution callers.
    """
    _instrument(adc.AdvancedCFDSimulator, "simulate_aerodynamics", sync_per_call)
    _instrument(adc.AdvancedCFDSimulator, "init_flow_field", sync_per_call)
    _instrument(adc.AdvancedCFDSimulator, "compute_aerodynamic_coefficients", sync_per_call)
    _instrument(lbs.D3Q27Solver, "collide_and_stream", sync_per_call)
    _instrument(lbs.D3Q27Solver, "_get_q", sync_per_call)
    _instrument(adc, "evaluate_aircraft_validity", sync_per_call)
    _instrument(adc, "_direct_measured_objective_for_single", sync_per_call)
    # ---- model-phase instrumentation (the un-instrumented ~72s/update) ----
    _instrument(adc.OptimizedDiffusionTrainer, "_compute_consistency_loss", sync_per_call)
    _instrument(adc.ConsistencyModel, "fast_inference", sync_per_call)
    _instrument(adc.LatentTo3DConverter, "forward", sync_per_call)
    _instrument(adc.LatentTo3DConverter, "forward_flat_indices", sync_per_call)
    _instrument(adc.LatentTo3DConverter, "_checkpointed_coordinate_chunk", sync_per_call)
    _instrument(adc.LatentTo3DConverter, "_encode_coordinates", sync_per_call)
    # diffusion_model, student_model, and teacher_model are all
    # LatentDiffusionUNet instances, so one class-level instrument covers the
    # total UNet forward cost (diffusion + consistency student + teacher).
    _instrument(adc.LatentDiffusionUNet, "forward", sync_per_call)


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


def _format_su_line(stats: dict, warmup: int, n: int) -> str:
    """Single machine-parseable stdout line with the per-update cost.

    Format (per the on-demand speed-tool requirement):
      s/u mean=12.34 median=12.30 p90=12.60 over=5 updates (warmup=1)
    ``n`` is the number of MEASURED updates (warmup excluded).
    """
    mean = stats.get("mean")
    median = stats.get("median")
    p90 = stats.get("p90")
    if mean is None or median is None or p90 is None:
        mean = median = p90 = float("nan")
    return (
        f"s/u mean={mean:.2f} median={median:.2f} p90={p90:.2f} "
        f"over={n} updates (warmup={warmup})"
    )


# ---------------------------------------------------------------------------
# Builders (mirror select_recovery_checkpoint / run_monitored_training wiring)
# ---------------------------------------------------------------------------
def build_trainer_and_loader(
    checkpoint: Path,
    manifest: Path,
    device,
    samples: int,
    solver: str,
    *,
    fresh_init: bool = False,
):
    if fresh_init:
        # Fresh-init mode: construct the trainer with DEFAULT (unloaded) weights
        # and a procedurally-generated synthetic loader so --full-update runs on
        # any worktree with no checkpoint present. The per-update cost is
        # dominated by the SPSA solves + decoder + solver, which are to
        # first-order weights-independent -- a fresh-init run is the right
        # on-demand A/B tool for code changes. The checkpoint is deliberately
        # never read here.
        model_config = adc.ModelConfig()
    else:
        # Prefer the safe weights_only=True loader; fall back to
        # weights_only=False ONLY for a trusted local artifact under build/ (see
        # _load_checkpoint_metadata). This is a trusted local artifact from our
        # own run at an explicit path, never untrusted input.
        metadata = _load_checkpoint_metadata(checkpoint)
        model_config = (
            adc.ModelConfig(**metadata["model_config"])
            if "model_config" in metadata
            else adc.ModelConfig(**metadata["compatibility"]["configuration"]["model_config"])
        )
    resolved_grid_size = int(model_config.grid_resolution)
    prepare_edt_workspace((resolved_grid_size,) * 3)

    if fresh_init:
        dataset = adc.AircraftDesignDataset(
            num_samples=samples,
            grid_size=resolved_grid_size,
            latent_dim=int(model_config.latent_dim),
            seed=0,
        )
        # Synthetic loader: every record is "train" so the loader yields exactly
        # `samples` batches. The procedural split_assignments would otherwise
        # strip ~30% of records and starve warmup+iterations, silently
        # under-measuring the per-update stats.
        dataset.metadata = dict(dataset.metadata)
        dataset.metadata["split_assignments"] = ["train"] * len(dataset)
    else:
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
        direct_solver_loss_weight=float(config_value("training", "direct_solver_loss_weight", 1.0)),
        direct_solver_interval=int(config_value("training", "direct_solver_interval", 1)),
        direct_solver_steps=int(config_value("training", "direct_solver_steps", 5)),
        direct_solver_directions=int(config_value("training", "direct_solver_directions", 16)),
        direct_solver_perturbation=float(config_value("training", "direct_solver_perturbation", 0.15)),
        direct_solver_perturbation_grid_size=int(config_value("training", "direct_solver_perturbation_grid_size", 12)),
        direct_connectivity_weight=float(config_value("training", "direct_connectivity_weight", 1.0)),
        direct_aircraft_validity_weight=float(config_value("training", "direct_aircraft_validity_weight", 1.0)),
        overfit_geometry_gate_samples=samples,
        promotion_generation_seeds=int(config_value("training", "promotion_generation_seeds", 6)),
        require_direct_solver_every_iteration=bool(config_value("training", "require_direct_solver_every_iteration", True)),
    )
    cfd_config = adc.CFDConfig(
        base_grid_resolution=resolved_grid_size,
        solver_type=solver,
        use_fused_stream_bfl=True,
    )
    diffusion_config = adc.DiffusionConfig(
        teacher_steps=int(config_value("diffusion", "timesteps", 1000)),
        student_steps=int(config_value("diffusion", "student_steps", 4)),
    )
    trainer = adc.OptimizedDiffusionTrainer(
        model_config,
        diffusion_config,
        training_config,
        cfd_config,
        device=device,
    )
    if fresh_init:
        # No checkpoint load: global_step stays 0 and weights are the model
        # defaults. The geometry threshold comes from the config fixed value
        # (calibrate_geometry_materialization_threshold is False in config.yaml,
        # so _prepare_geometry_threshold_for_run is a config-fixed no-op).
        _prepare_geometry_threshold_for_run(trainer, loader, resume_run_state=None)
    else:
        trainer.load_checkpoint(str(checkpoint))
        _prepare_geometry_threshold_for_run(trainer, loader, resume_run_state=None)
    return trainer, loader


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _profile_grid_size(trainer) -> int:
    """Return the common model/solver resolution used by this profiler."""
    model_size = getattr(getattr(trainer, "model_config", None), "grid_resolution", None)
    solver_size = getattr(getattr(trainer, "cfd_simulator", None), "resolution", None)
    sizes = [int(value) for value in (model_size, solver_size) if value is not None]
    if not sizes:
        raise RuntimeError("cannot determine profiler grid size from trainer")
    if len(set(sizes)) != 1:
        raise RuntimeError(
            "trainer model/solver grid mismatch: "
            f"model={model_size}, solver={solver_size}"
        )
    return sizes[0]


def run_full_update(trainer, loader, warmup: int, iterations: int, profile_cuda: bool) -> dict:
    grid_size = _profile_grid_size(trainer)
    trainer.scheduler_step_per_update = True
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=max(1, warmup + iterations)
    )
    trainer.run_state_checkpoint_callback = None
    trainer.run_state_checkpoint_path = None
    trainer.stop_after_updates = int(trainer.global_step) + warmup + iterations

    # Per-update wall via the trainer's NATURAL per-update boundary
    # (update_metrics_callback, invoked at the end of every optimizer update).
    # Warmup updates are unmeasured; only `iterations` measured updates feed the
    # stats, so the denominator is iterations only. A single synchronize at each
    # boundary folds the just-launched GPU work of the update into the wall
    # delta -- per-call instrument syncs never feed these numbers.
    update_times: list[float] = []
    boundary: dict = {"seen": 0, "t_prev": None}

    def _on_update(metrics: dict) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        boundary["seen"] += 1
        seen = boundary["seen"]
        if boundary["t_prev"] is not None and warmup < seen <= warmup + iterations:
            update_times.append(now - boundary["t_prev"])
        boundary["t_prev"] = now

    trainer.update_metrics_callback = _on_update

    def _train() -> None:
        # Initialise t_prev before the first update so warmup=0 measures update
        # 1 (otherwise the first boundary would only set t_prev).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        boundary["t_prev"] = time.perf_counter()
        trainer.train_epoch(loader, grid_size=grid_size, start_batch=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if profile_cuda:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            _train()
        trace_path = REPO_ROOT / "build" / "perf" / "baseline" / "update_trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        print("chrome trace:", trace_path)
    else:
        _train()

    stats = _stats(update_times)
    print(_format_su_line(stats, warmup, len(update_times)))
    return {
        "mode": "full_update",
        "updates": warmup + iterations,
        "warmup": warmup,
        "measured_updates": len(update_times),
        "total_wall_s": sum(update_times),
        "update_stats": stats,
        "phases": {k: {"total_s": v[0], "calls": int(v[1])} for k, v in TIMERS.items()},
    }


def run_direct_only(trainer, iterations: int, seed: int = 1234) -> dict:
    grid_size = _profile_grid_size(trainer)
    field = torch.randn(1, grid_size, grid_size, grid_size, device=trainer.device)
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
    grid_size = _profile_grid_size(trainer)
    field = torch.rand(1, grid_size, grid_size, grid_size, device=trainer.device)
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
    ap.add_argument("--checkpoint",
                    default=str(REPO_ROOT / "build" / "recovery_ladder_20260814" / "step1305.pt"),
                    help="checkpoint to load (ignored with --fresh-init)")
    ap.add_argument("--manifest",
                    default=str(REPO_ROOT / "build" / "grounded_combined_1k_20260716" / "manifest.jsonl"),
                    help="grounded manifest (ignored with --fresh-init; a procedural synthetic loader is used)")
    ap.add_argument("--fresh-init", action="store_true",
                    help="construct the trainer WITHOUT loading weights and with a procedural synthetic "
                         "loader, so --full-update runs on any worktree with no checkpoint present "
                         "(per-update cost is weights-independent to first order)")
    ap.add_argument("--warmup", type=int, default=1, help="updates for warmup (absorbs Triton JIT); unmeasured")
    ap.add_argument("--iterations", type=int, default=5,
                    help="measured updates / solver calls (--full-update requires M>=5 per spec-1)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-instrument", action="store_true",
                    help="skip per-call-sync instrumentation; measure the production update path")
    ap.add_argument("--sync-per-call", action="store_true",
                    help="synchronize inside each instrumented call (accurate phase table; optional -- "
                         "per-update wall numbers come from the update boundary, never from per-call syncs)")
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

    sync_per_call = args.sync_per_call and not args.no_instrument

    if args.solver_only:
        trainer, _ = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=1, solver=str(config_value("cfd", "solver", "D3Q27")),
            fresh_init=args.fresh_init,
        )
        result = run_solver_only(trainer, args.iterations)
    elif args.direct_only:
        install_instrumentation(sync_per_call=sync_per_call)
        trainer, _ = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=1, solver=str(config_value("cfd", "solver", "D3Q27")),
            fresh_init=args.fresh_init,
        )
        result = run_direct_only(trainer, args.iterations)
    else:
        if args.iterations < 5:
            ap.error(
                f"--full-update requires --iterations >= 5 (spec-1: per-update stats "
                f"over M>=5 measured iterations); got {args.iterations}"
            )
        if not args.no_instrument:
            install_instrumentation(sync_per_call=sync_per_call)
        trainer, loader = build_trainer_and_loader(
            Path(args.checkpoint), Path(args.manifest), device,
            samples=args.warmup + args.iterations,
            solver=str(config_value("cfd", "solver", "D3Q27")),
            fresh_init=args.fresh_init,
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
