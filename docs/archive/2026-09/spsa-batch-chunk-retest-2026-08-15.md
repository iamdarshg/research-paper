# Batched SPSA chunk re-test post-Task-34 — C=2 / C=4 (2026-08-15)

**Question:** Task 34 (commit 384854e) compressed the LBM batch workspace (sparse
q + 2-buffer ping-pong; C=4 peak 4.853→4.142 GB isolated, paging gone). The prior
finding ("validity (scipy label) slows 2-6x, batched path parity-gated for
≥16 GiB") was measured **with** paging. Does batching now pay on the 8 GB box?

**Method:** full optimizer update via the same `profile_chunk_override.py` →
`profile_training_update.py --full-update` harness used for the 62.66 s/u C=1
baseline. step1305.pt, 96³, batch=1, warmup=1 + 2 measured updates per chunk.

| config | full-update s/u | vs C=1 | peak allocated | peak reserved |
|---|---|---|---|---|
| C=1 (doc baseline) | 62.66 | — | — | — |
| C=2 | 61.07 | **+2.5%** | 5,790 MiB | 7,260 MiB |
| C=4 | 66.21 | **−5.7%** | 5,818 MiB | 7,704 MiB |

C=1 reference is well-established: documented 62.66 s/u + the de-black-box Run A
(65.35 s under profiler ≈ 62.6 unprofiled).

## Verdict: KEEP `_DIRECT_SOLVER_BATCH_CHUNK = 1`

- **C=2 is noise-level** (+2.5%), **C=4 is slower** (−5.7%) and dangerously close
  to the VRAM ceiling (7.70 GB reserved on 8 GB → ~490 MiB headroom).
- The phase table shows why batching cannot win here: the direct phase's real cost
  is **per-solve CPU work**, which batching does not compress —
  `evaluate_aircraft_validity` = 12.9 s/u (C=2) / 16.7 s/u (C=4), and
  `_checkpointed_coordinate_chunk` ≈ 14 s/u. The GPU LBM solve itself is 0.15 s/u
  (fused `_stream_bfl_kernel`).
- Batching only compresses the GPU solve — already negligible. Post-Task-34 the
  paging regression is gone (hence C=4 is 66 s/u, not the pre-Task-34 183 s/u),
  but the batch path still does not beat sequential on 8 GB.
- Consistent with the documented guidance (batched path for boxes with
  ≥16 GiB VRAM).

## Implication for P6

The CPU-side validity metric extraction (~12.9 s/u, scipy label + `.item()`/any
drains + nonzero scans) and the decoder checkpoint recompute (~14 s/u) are the
real direct-phase costs, not the LBM launches. These are the targets of the
CPU-offload + fusion analysis (Task #40) — not the batch chunk.

## Artifacts
- `build/perf/baseline/profile_result_c2.json` / `_c4.json` — phase tables.
- `build/perf/baseline/profile_c2.log` / `_c4.log` — peak-VRAM lines.
