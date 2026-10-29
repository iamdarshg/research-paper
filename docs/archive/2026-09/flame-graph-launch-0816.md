# Flame graph accounting for kernel-launch CPU time (2026-08-16)

**What changed vs the old flame graph:** the previous flame graphs
(`flame-graph-c1.md`, `residual-blackbox-report.md`) showed GPU busy per phase
and a ~23.5 s "residual" black box. The user's ask: *"not launch time, i mean
time used to launch kernels and shit"* — i.e. put the **actual CPU time spent
in kernel-launch machinery** (`cudaLaunchKernel`, Command Buffer Full,
`cudaStreamSynchronize`) into the flame graph as explicit bars. This document
does that.

Device: RTX 4060 Laptop 8 GiB, Ryzen 7 7735HS, 96³, batch=1,
`_DIRECT_SOLVER_BATCH_CHUNK=1`, fp32, on branch `experiment/kernel-fusion-launch`
(HEAD `18de04a`). Source trace: `build/perf/baseline/update_trace.json`
(fresh `--profile-cuda` run of ONE full update, 2026-08-16 22:56, the
profiled wall is 49.8 s incl. ~2.7 s profiler overhead). Honest no-instrument
speed measured same day: **44.80 s/u mean** (`profile_speedcheck_0816.json`).

Parser: `build/perf/baseline/launch_phase_attribution.py` — classifies every
GPU kernel **and** every CPU-side launch/sync event into phase windows that are
**auto-calibrated from the trace's own markers** (first/last `_stream_bfl` for
the solver window, final dense `direct_copy` burst for the optimizer write-back).

---

## Honest speed check (double-check, as requested)

| run | date | s/u | notes |
|---|---|---|---|
| pre-lever honest (no-instrument) | 2026-08-15 20:41 | 52.82 | pre-P6g levers, pre-PR-41 merge |
| **current HEAD honest (no-instrument)** | **2026-08-16 22:48** | **44.80** (mean of 3; one thermal hiccup 59.7 s) | 4 updates, warmup 1 |
| current HEAD profiled (trace) | 2026-08-16 22:56 | 49.8 (incl. ~2.7 s profiler overhead) | 1 update, `--profile-cuda` |

So the honest current speed is **~45 s/u**, still ~5 s above the 40 s/u
launch gate. The profiler run adds ~2.7 s of overhead, so the trace's 48.6 s
kernel span corresponds to the ~46 s honest wall.

## The one-update flame (48.6 s kernel span, GPU busy 32.4 s = 66.7% util)

Columns: `launch` = CPU time inside `cudaLaunchKernel`(+`cuLaunchKernel`,
`cudaLaunchKernelExC`); `cmdbf` = CPU time blocked in `Command Buffer Full`
(launch queue full — the GPU cannot absorb launches fast enough);
`sync` = CPU time in `cudaStreamSynchronize`. All are the **actual CPU time
spent in the launch layer**, per phase.

| phase | kernels | GPU busy (s) | launch (s) | cmdbf (s) | sync (s) | GPU idle gaps (s) |
|---|---:|---:|---:|---:|---:|---:|
| decoder_forward (0–15.6s) | 16,861 | 11.65 | 5.25 | 5.22 | 0.00 | 3.67 |
| diffusion_consistency | 1,703 | 0.01 | 0.00 | 0.00 | 0.00 | 0.26 |
| solver / SPSA (15.6–20.9s) | 28,933 | 2.44 | 0.83 | 0.00 | 0.03 | 2.89 |
| backward (20.9–46s) | 26,327 | 17.87 | 9.75 | 11.33 | 0.33 | 7.44 |
| optimizer_save (46s+) | 17,871 | 0.47 | 0.49 | 0.00 | 0.44 | 1.93 |
| **TOTAL** | **91,695** | **32.44** | **16.32** | **16.55** | **0.80** | **16.18** |

ASCII (each cell = 2 s): `#` = GPU busy, `*` = CPU launch+cmdbf.

```
phase                     GPU busy                         CPU launch+cmdbf
decoder_forward           #####                              *****
solver                    #                                 *
backward                  ########                           **********
optimizer_save
```

## The launch-machinery readout (this is the ask)

**CPU launch + Command Buffer Full + sync = 16.32 + 16.55 + 0.80 = 33.7 s of
CPU time across a 48.6 s wall — the launch layer is now the dominant cost,
and it is concentrated in the two GEMM phases.**

| phase | CPU launch+cmdbf+sync (s) | % of phase CPU+GPU | read |
|---|---|---|---|
| **backward** | **21.4** | **54.5%** | 26.3k launches on 17.9 s GPU; the 6.5 s gap (81.8% wall) is the backward→optimizer handoff |
| **decoder_forward** | **10.5** | **47.4%** | 16.9k launches on 11.7 s GPU; biggest single launch source after backward |
| optimizer_save | 0.9 | 66.6% | 17.9k launches but tiny GPU work (~0.5 s) |
| solver / SPSA | 0.9 | 25.9% | 28.9k launches but tiny per-kernel GPU (fused `_stream_bfl` = 0.15 s total) |
| diffusion_consistency | 0.0 | 0% | negligible |

### What shrank since the residual report (2026-08-15 → now)

The 65 s residual trace had **launch 15.1 + cmdbf 12.8 + sync 9.1 = 37 s of CPU
machinery**. The fresh trace shows:

- **`cudaStreamSynchronize` collapsed: 9.1 s → 0.8 s (−91%)** — the Task 1 / Task 7
  metric-sync batching and the PR-41 `batch_guard_dot_reads`/`deferred_solver_reads`
  design work have already landed the sync win. Explicit syncs are no longer a
  cost class.
- **`cudaLaunchKernel` is flat (15.1 → 16.3 s)** and **Command Buffer Full grew
  (12.8 → 16.6 s)** — the launch-count reduction did not land yet; the CPU is
  still issuing ~91k launches per update, and the GPU queue now fills *more*
  because the solver/decoder are faster. This is the remaining launch tax.

### What this means for getting below 40 s/u

The two big bars are `backward` (17.9 s GPU) and `decoder_forward` (11.7 s
GPU) — both GEMM-bound (the decoder MLP). The CPU launch tax on them is 21.4 +
10.5 = **~32 s of CPU machinery feeding ~29.5 s of GPU GEMM work**. Cutting
launches is the single highest-value lever left, and the two OFF experiment
levers attack exactly this:

| lever | what it does | est. s/u | status |
|---|---|---|---|
| `deferred_solver_reads` | enqueue all 33 SPSA solves with NO host scalar reads; read back in ONE batched `.tolist()` + one sync (bit-identical occupancy/coefficients) | ~5–9 | built, OFF |
| `batch_guard_dot_reads` | collapse per-active-guard `.item()` guard-dot reads into ONE deferred stacked read (bit-identical fp64 dots) | ~1–3 | built, OFF |
| `graph_decode_mlp` | CUDA-graph the decoder chunk MLP forward (13 launches/chunk → 3) | 0.99× measured, no speedup | OFF, stays OFF |

## Idle gaps (the GPU is idle 16.2 s of 48.6 s)

Top stalls by wall position (prev-kernel phase, seconds):

```
#1  6464.9 ms @ 81.8% wall  [backward      ] reduce_kernel ... -> unrolled_elementwise ...
#2  1367.7 ms @ 29.2% wall  [decoder_forward] unrolled_elementwise -> _stream_bfl_kernel   (solver entry drain)
#3   156.6 ms @  0.6% wall  [decoder_forward] vectorized_elementwise -> nchwToNhwc ...
```

The single 6.5 s stall at 81.8% wall is the backward→optimizer_save handoff
(the `.item()`/sync drain before the copy burst). The 1.4 s at 29.2% is the
decoder→solver boundary (solver entry scalar reads).

## Raw artifacts

- `build/perf/baseline/update_trace.json` — fresh chrome trace, current HEAD.
- `build/perf/baseline/profile_speedcheck_0816.json` — honest no-instrument s/u.
- `build/perf/baseline/launch_phase_attribution.py` — the parser (auto-calibrates windows).
- `build/perf/baseline/residual_trace.json` / `residual-blackbox-report.md` — the Aug-15 65 s baseline this is compared against.
