# 96³ Per-Update Baseline Profile

**Date:** 2026-08-15
**Protocol:** `docs/to_be_read.md` Phase 1 — measure before changing.
**Instrument:** `CLI/profile_training_update.py` (bounded, no checkpointing).

## Hardware under test

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GiB, Ada, compute cap 8.9) |
| CPU | AMD Ryzen 7 7735HS (8 cores / 16 threads) |
| CUDA / torch | cu130 / 2.9.1+cu130 |
| Backend | `fused_stream_bfl` (D3Q27, MRT) |

This is the "cheap box" class: a mid-tier 8 GB laptop GPU with 8 CPU cores.
Optimizations are judged by (speedup × instance cost), not by wall clock alone.

## Methodology

- One representative optimizer update on the step-1305 checkpoint
  (`build/recovery_ladder_20260814/step1305.pt`), batch=1, 96³, threshold 0.5,
  real coordinate decoder, real direct/SPSA solver (33 solves/update: 1 base +
  16 Rademacher directions × 2 antithetic), real backward, real optimizer step.
- Fixed sample and fixed RNG (seed per mode).
- Each instrumented phase ends with `torch.cuda.synchronize()` so wall time
  includes the GPU work an async launch would otherwise defer.
- Warmup update absorbs Triton JIT + model load; N measured updates follow.
- Statistics: mean / median / p90 / p95 per the protocol.

## Measured per-update cost stack

`CLI/profile_training_update.py --full-update --warmup 1 --iterations 3`
on the step-1305 checkpoint. 4 updates, 452.2 s total, **113.0 s/update**
steady-state (mean == median == p90 == p95: all 4 updates were ~113 s).
132 `_direct_measured_objective_for_single` calls = 33 solves × 4 updates.
660 `collide_and_stream` calls = 132 solves × 5 steps; 660 `_get_q` calls =
5/solve, of which only the first per geometry computes EDT (132 EDTs ≈ 87.7 s).

| subsystem | s/update | share | bottleneck type | 192³ scaling | candidate |
|---|---|---|---|---|---|
| **model fwd/bwd + data + optimizer** (un-instrumented) | **~72.5** | **64%** | opaque; needs own profile | ~×4 | torch.compile / bf16(opt-in) / stream-overlap |
| CPU EDT/SDF (`_get_q`) | 21.9 | 19% | serial scipy EDT (0.66 s/solve) | ×8 | thread-parallel EDT (measured 3.61× on 8 threads) |
| GPU collision + stream (excl. EDT) | 13.2 | 12% | launch-bound (Python loop in `compute_moment_equilibrium`) | ×8 | batch the 33 solves (doc 6A) |
| CPU validity (connected components) | 3.9 | 3% | serial scipy `label` | ×8 | thread-parallel |
| coefficients (`_shape_drag_correction`) + init_flow + binarize/transfer | ~4.0 | 3% | scalar + `.item()` syncs | ×8 | parallelize / batch |

## Bottleneck ranking

1. **Model fwd/bwd + data loss (~72.5 s/update, 64%)** — the largest and least
   characterized. Must be instrumented separately (forward vs backward, per
   module, per `.item()`/sync) before any solver-side change pays off the top
   line. Likely culprits: consistency-model + coordinate-decoder forward at
   96³/latent-192, loss aggregation with host syncs, the direct-solver backward
   replay site.
2. **CPU EDT (`_get_q`) 21.9 s/update** — 132× serial scipy EDT. Thread-parallel
   EDT measured at **3.61×** on 8 threads in isolation (thread-local EDT
   workspaces already landed in `sdf_utils.py`); the same geometry set is
   already CPU-resident in `_direct_measured_objective_for_single` (no extra
   transfer).
3. **GPU collision+stream 13.2 s/update** — 660 steps × ~20 ms GPU compute.
   Launch/overhead bound (the 27-iteration Python loop in
   `compute_moment_equilibrium` builds ~81 small kernels per step). Batched
   SPSA (doc 6A) collapses the 891 launches/update.
4. CPU validity 3.9 s/update — thread-parallel `label`.
5. coefficients + init_flow ~4 s/update.

**Top-line implication:** the direct-solver phase is only ~36% of the update.
Reaching the 8× target requires attacking the model phase (biggest), the EDT
(threads), and the solver (batching) — not any single one.

---

## Post-Optimization (2026-08-15) — Task 12 re-profile + fix round

Same command, same checkpoint, same hardware, after Tasks 1-10 all merged, with
the FINAL default `_DIRECT_SOLVER_BATCH_CHUNK = 1`. Full report:
`.superpowers/sdd/optimization-plan/task-12-report.md`.

| metric | baseline (112 s/u) | Task 9 milestone | Task 12 final (default C=1) |
|---|---|---|---|
| per-update wall | 113.0 s/u | 60.23 s/u | **62.66 s/u** (63/67/61/59 s) |

**Baseline reconciliation:** the plan anchor (and this table's cumulative
factors) use **112 s/u** as the mandated baseline anchor; the raw measured
baseline above was **113.04 s/u** (452.17 s / 4 updates). The 1.79× cumulative
factor in this document uses the 112 plan anchor.

**Fix round (Task 12 R=1):** the old merged default (C=4) regressed the real
full-update. Bounded re-profiles (`--warmup 1 --iterations 3`, 4 updates each)
via a module-attribute override harness (`build/perf/baseline/profile_chunk_override.py`,
no source edit for the measurement runs):

| chunk | full-update | CPU validity | diagnosis |
|---|---|---|---|
| C=1 (**final default**) | **62.66 s/u** | 7.83 s/u | recovers the Task 9 floor |
| C=2 | 117.22 s/u (124/117/119/109 s) | 14.63 s/u | mild VRAM paging |
| C=4 (old merged default) | 183.63 s/u (209/167/166/192 s) | 23.75 s/u | severe VRAM paging |

**Cumulative vs baseline with the FINAL default: 112 → 62.66 s/u = 1.79×**
(44.1% faster; Task 9 milestone measured 1.86× — the small gap is run-to-run
variance). The Task 10 batched path is a SCALE WIN (isolated 1.12×) that is
VRAM-bound on 8 GiB: the C>=2 batched workspaces (~2.7 GB at C=2, ~5.4-7 GB at
C=4) do not fit alongside the training model → GPU ~97% VRAM, OS paging, CPU
validity (scipy `label`) 2-6× slower. The default is therefore kept sequential
(C=1); the batched path remains available and parity-gated for ≥16 GB VRAM boxes.

Per-phase at the final default C=1 (s/update). Profiler phases are nested — the
outer `_direct_measured_objective_for_single` wraps the inner solver phases
(`_get_q`, `collide_and_stream`, `simulate_aerodynamics`), so do not sum rows:

| phase | baseline | Task 12 (default C=1) | note |
|---|---|---|---|
| CPU validity (scipy `label`) | 3.9 | 7.83 | 2× baseline (EDT-threading core contention; see Task 8/9 optimization-log rows) |
| `_direct_measured_objective_for_single` | 40.70 | 17.04 | Task-9 EDT threading win held |
| coordinate decoder (`_checkpointed_coordinate_chunk`) | 14.85 | 15.52 | ~flat |
| GPU collision+stream (all solves) | 13.2 | 13.83 | ~flat (sequential path) |
| EDT `_get_q` (all solves, threaded) | 21.9 | 8.00 | Task-9 threading held |
| decoder `forward_flat_indices` | 19.96 | 10.74 | Task-6 fusion held |
| `forward` (full-grid decode) | 5.94 | 5.49 | ~flat |
| `init_flow_field` | 0.64 | 0.44 | ~flat |

**Top-line implication (unchanged):** the model fwd/bwd + optimizer residual is
still the largest cost and the next 8×-target bottleneck. The fix round shows the
Task 10 batching win does not transfer to 8 GiB — it is a memory-footprint
problem, not a speed-of-light problem. Default C=1 restores the ~60 s/u floor.
