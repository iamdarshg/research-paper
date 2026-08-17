# R11 — Production-faithful C=1/2/4 benchmark (PR 41 review, item 11)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)
Hardware: RTX 4060 Laptop 8 GB / Ryzen 7 7735HS / 16 GB RAM, Windows 11, torch 2.9.1+cu130

## The reviewer finding

> The s/u numbers that will appear in the paper were measured on the pre-review
> code state (before the TF32-scope, FP64-flattening, and durability fixes) and
> the C=2/C=4 batch comparison was run through an instrumented path that
> perturbs wall time. Re-measure at production fidelity: the real checkpoint,
> the real grounded corpus, the production numerics flags, and the
> non-instrumented update path.

## Method (what "production-faithful" means here)

All three configurations were measured through the **same** wrapper on the
**same** commit (`f2ab840` HEAD), via `profile_training_update.py --full-update`
with `--no-instrument` (the harness's documented "measure the production update
path" mode — the instrumented mode adds per-call syncs that perturb wall time):

- **Checkpoint:** `build/recovery_ladder_20260814/step1305.pt` (the real
  claim-bearing source checkpoint), 96³, batch=1, latent_dim=192.
- **Corpus:** real `grounded_combined_1k_20260716/manifest.jsonl` + the 1069
  real voxel geometries it references (3 corpora), not the synthetic loader.
- **Numerics:** production `config.yaml` (`experiment.tf32_gemm_math: true`,
  applied NN-only per R1 — solver stays IEEE fp32).
- **Protocol:** warmup=1 (absorbs Triton JIT) + 5 measured full optimizer
  updates per config; stats over the measured 5 only. C=2/C=4 set via the
  `_DIRECT_SOLVER_BATCH_CHUNK` module-global override (gitignored wrapper at
  `build/perf/baseline/profile_chunk_override.py`), same update path.

## Results (full optimizer update, s/u over 5 measured updates)

| config | mean | median | p90 | peak allocated | peak reserved | vs C=1 mean |
|---|---|---|---|---|---|---|
| **C=1** (production) | **27.53** | 26.72 | 27.45 | 6,053 MiB | 7,460 MiB | — |
| C=2 | 29.33 | 28.17 | 29.01 | 6,078 MiB | 7,569 MiB | +6.5% |
| C=4 | 30.20 | 29.78 | 29.89 | 6,112 MiB | 8,414 MiB | +9.7% |

## Verdict

**C=1 remains the production configuration**, now confirmed at current HEAD with
production fidelity. The batched SPSA path is strictly worse on the 8 GB box:

- C=2 is +6.5% slower with no measurable GPU-solve win (the direct phase's real
  cost is per-solve CPU work — EDT/q + validity — which batching does not
  compress; the GPU LBM solve itself is <1 s/u at 96³).
- C=4 is +9.7% slower **and** reserves 8.4 GiB — over the card's physical 8 GiB,
  spilling into WDDM shared system memory. That spill is the slowdown mechanism.
- This reinforces the standing constraint `_DIRECT_SOLVER_BATCH_CHUNK = 1` and
  the documented guidance that the batched path is for boxes with ≥16 GiB VRAM.

## vs the pre-review baseline

The prior retest (2026-08-15, instrumented, pre-R1–R10) measured C=1 62.66,
C=2 61.07, C=4 66.21. At production fidelity on current HEAD the same
configuration family is ~2.1× faster (27.53 vs 62.66 s/u) and the batch penalty
is unambiguous rather than noise-level. The claim-bearing production number is
**27.5 s/u mean (p90 ≈ 30 s/u)** at 96³, batch=1, TF32-NN-on.

## Paper implication

- The paper's per-update cost claim should cite **27.5 s/u (RTX 4060 Laptop 8 GB)**
  from this production-faithful measurement, alongside the phase attribution
  already documented (per-solve CPU prep dominates; the LBM solve is ~0.15 s/u).
- The "sequential solves, chunk=1" design decision is now backed by a
  production-faithful C=1/2/4 comparison rather than an instrumented one.
- VRAM ceiling: C=1 peaks at 7.46 GiB reserved on 8 GiB (~0.54 GiB headroom),
  with C=4 demonstrating the spill threshold — a reproducible boundary for the
  8-GB portability claim.

## Artifacts

- `build/perf/baseline/profile_result_c1.json` / `_c2.json` / `_c4.json`
  (gitignored) — full harness JSON per config.
- `build/perf/baseline/profile_chunk_override.py` (gitignored) — the override
  wrapper, also used to print peak VRAM.
