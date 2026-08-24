"""Bounded before/after for Task 10 through the REAL trainer direct-solver path.

Builds the trainer exactly like profile_training_update.py --direct-only, then
times ONE direct_solver_loss call (base solve + 32 SPSA probes + post-processing)
at the model's grid resolution for _DIRECT_SOLVER_BATCH_CHUNK = 1 (sequential)
and = 4 (batched default). Reports per-call wall time and the instrumented
phase breakdown (simulate_aerodynamics / collide_and_stream / ...).

Usage:
    python CLI/measure_task10_direct.py
"""
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPO_ROOT / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import aircraft_diffusion_cfd as adc  # noqa: E402
from experiment_config import config_value  # noqa: E402
from profile_training_update import build_trainer_and_loader, install_instrumentation  # noqa: E402

CHECKPOINT = REPO_ROOT / "build" / "recovery_ladder_20260814" / "step1305.pt"
MANIFEST = REPO_ROOT / "build" / "grounded_combined_1k_20260716" / "manifest.jsonl"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    install_instrumentation()
    trainer, _ = build_trainer_and_loader(
        CHECKPOINT, MANIFEST, device, samples=1,
        solver=str(config_value("cfd", "solver", "D3Q27")),
    )
    field = torch.randn(1, trainer.model_config.grid_resolution,
                        trainer.model_config.grid_resolution,
                        trainer.model_config.grid_resolution, device=device)
    spec = adc.DesignSpec(target_speed=90.0, wingspan_limit_m=1.2)
    print("grid:", trainer.model_config.grid_resolution)

    results = []
    for chunk in (1, 4):
        old = adc._DIRECT_SOLVER_BATCH_CHUNK
        adc._DIRECT_SOLVER_BATCH_CHUNK = chunk
        try:
            # warmup call absorbs Triton JIT for the given path
            _ = trainer.direct_solver_loss(field, spec, trainer.cfd_simulator, seed=1234)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = trainer.direct_solver_loss(field, spec, trainer.cfd_simulator, seed=1234 + chunk)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        finally:
            adc._DIRECT_SOLVER_BATCH_CHUNK = old
        results.append({"chunk": chunk, "direct_loss_call_s": wall})
        print(f"chunk={chunk}: direct_solver_loss call = {wall:.3f} s")

    from profile_training_update import TIMERS
    phases = {k: {"total_s": v[0], "calls": int(v[1])} for k, v in TIMERS.items()}
    result = {"results": results, "phases": phases}
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))
    out = REPO_ROOT / "build" / "perf" / "task10" / "direct_before_after.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
