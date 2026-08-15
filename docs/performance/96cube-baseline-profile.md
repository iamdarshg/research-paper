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

## Post-Optimization (2026-08-15) — Task 12 re-profile

Same command, same checkpoint, same hardware, after Tasks 1-10 all merged:
`CLI/profile_training_update.py --full-update --warmup 1 --iterations 3`
(4 updates, 734.52 s total). Full report: `.superpowers/sdd/optimization-plan/task-12-report.md`.

| metric | baseline (112 s/u) | Task 9 milestone | Task 12 re-profile (merged C=4) |
|---|---|---|---|
| per-update wall | 113.0 s/u | 60.23 s/u | **183.63 s/u** (209/167/166/192 s) |

**Cumulative is a REGRESSION at the merged default**: 112 → 183.63 s/u =
**0.61×** (slower). This is caused by Task 10's C=4 batch workspace (~5.4-7 GB)
not fitting on 8 GiB alongside the training model → GPU 97%, system ~0 free RAM,
OS paging. Tasks 1-9 alone measured **60.23 s/u = 1.86×** at the Task 9
milestone; the C=4 default should be lowered to **C=2/C=1** for the 96³
continuation to recover that.

Per-phase before/after (s/update; Task 12 captures only the 1 instrumented base
solve — the 32 batched probes are in the un-instrumented residual):

| phase | baseline | Task 12 (merged C=4) | note |
|---|---|---|---|
| CPU validity (scipy `label`) | 3.9 | **23.75** | 6× regression — OS paging (memory pressure) |
| coordinate decoder (`_checkpointed_coordinate_chunk`) | 14.85 | 17.29 | mildly elevated |
| decoder `forward_flat_indices` | 19.96 | 12.47 | Task-6 fusion held |
| EDT `_get_q` (base solve) | 21.9 (all solves) | 2.37 (base only) | Task-9 threading + batched probes un-instrumented |
| GPU collision+stream (base solve) | 13.2 | 5.41 (base only) | Task-8 fusion held; batched probes un-instrumented |
| un-instrumented residual | ~72.5 (model fwd/bwd+opt) | ~105.2 (incl. 32 batched probes) | model block still dominant |

**Top-line implication (unchanged, sharper):** the model fwd/bwd + optimizer
residual is still the largest cost. The 3× update-level regression is a memory
footprint problem (C=4 batching on 8 GiB), not a speed-of-light problem — the
isolated direct-solver call is genuinely 1.12× faster. Fix the memory (C=2/C=1),
then the model block is the next 8×-target bottleneck.
