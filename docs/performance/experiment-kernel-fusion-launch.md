# Experiment: kernel-fusion launch reduction (branch `experiment/kernel-fusion-launch`)

Status: **experimental, off by default**. Built under the standing GPU constraint
(approved training job PID 24436 running; no GPU work may launch while it runs).
The harness and its CPU-only fallback path are verified; the GPU microbenchmark
is written but **must not be run until the training job ends or the user opens a
GPU window**. Training was stopped safely 2026-08-16 (validated checkpoint at
step 2338); the GPU is now free for the microbenchmark and parity validation.

## What the branch tries

The profiled training update launches **~114,264 CUDA kernels** (cudaLaunchKernel
self-CPU 15.06 s) and the GPU is idle **39.4 % of wall time** (24.86 s of 63.17 s).
Directive (user, 2026-08-16): "fuse all used kernels into one" + "run a compile
sorta thing before every launch to take the numbers from the config and set them
into the kernel" + "cache these launches in such a way they remain in memory so
that it needs like 10 us instead of 1140".

The mechanism that does all three is **CUDA-graph capture/replay**:
* the "compile before launch" = capture bakes the exact kernel sequence with
  shape/dtype/config constants fixed (kernel specialization);
* the "cache in memory" = the captured graph object + its private memory pool
  persist; each `graph.replay()` re-enqueues the whole sequence in ~2-10 us of
  CPU instead of per-launch overhead (~1.1 ms/op in the trace).

## Target 1 (built): coordinate-decoder chunk-forward CUDA graph

File: `CLI/kernel_fusion_graph.py` (`DecodeMLPGraph`).

* 96^3 = 884,736 voxels; `coordinate_chunk_size` = 32,768 -> **exactly 27 full
  chunks per decode**, every chunk `[32768, 231]` -> one fixed-shape graph serves
  every chunk (no variable tail).
* Each chunk runs the decoder MLP (coordinate_input Linear+SiLU, residual blocks,
  coordinate_output Linear) ~ 13 un-fused kernels -> ~7,848+ launches/update
  forward, doubled by checkpoint recompute.
* `torch.utils.checkpoint(use_reentrant=False)` forward runs under
  `torch.no_grad()` (confirmed in torch/utils/checkpoint.py), so a **graph replay
  is safe in the forward path** (no autograd edges needed).
* **Autograd safety guard** (the load-bearing decision): the checkpoint BACKWARD
  recompute runs with grad enabled and must rebuild the autograd graph to flow
  gradients to `latent`. A graph replay returns a detached tensor -> silent zero
  gradients. `DecodeMLPGraph.__call__` therefore falls back to eager whenever
  `torch.is_grad_enabled()`.

CPU-only verification done (no GPU touched):
- capture refuses on CPU and records the reason;
- eager fallback is bitwise-identical to the unwrapped MLP;
- grad-enabled calls retain autograd (requires_grad preserved).

## Target 2 (assessed, NOT built here): solver step

`advanced_lbm_solver.py:_apply_bfl_boundary` has a Python `for i in range(1,27)`
loop with data-dependent host branches (`if not torch.any(active)`,
`if torch.any(q_low)`). A CUDA graph baked at capture would replay the branch
taken for the *capture-time geometry*, which changes per training sample ->
**correctness hazard. Not graph-capturable as-is.** The solver's fusion path is
Triton kernel fusion (already partially landed via `fused_stream_bfl`), not CUDA
graphs.

## Honest recovery estimate (workflow wf_da6a83b6-ae6)

* **Decoder: GPU-compute-bound, not launch-bound.** ~598 GFLOP/chunk x 654
  chunks/update at ~5-6 TFLOPs effective fp32 ~ the measured 30.8 s. The graph
  recovers only CPU launch/Python overhead (~1-1.5 s) + Command-Buffer-Full
  backpressure (12.8 s / 1059 events) — valuable CPU headroom, not the 25 s of
  GPU idle.
* **The 25 s GPU idle lives in the solver phase**: 33 sequential SPSA solves,
  each with ~4-5 device->host barriers (stacked `.tolist()`,
  occupancy `.item()`, `_effective_drag_link_metric_exponent .item()`, full-96^3
  D2H `compute_tensor_content_hash`). None feed control flow inside the probe
  loop, so they can be deferred to **one batched D2H read + one sync**. That is
  the single largest recoverable idle.
* `_DIRECT_SOLVER_BATCH_CHUNK = 1` is **deliberate** (aircraft_diffusion_cfd.py:
  4089-4098): batched C=2/C=4 measured 117/183 s/u vs 62.66 sequential because
  `[C,27,96^3]` workspaces page on the 8 GB 4060. The deferred-read design must
  work on the sequential path.
* Graph-pool footprint for a [32768, 896] capture is estimated ~350 MiB
  persistent — **"do not land on 8GB"** per workflow. Mitigation: capture at a
  smaller `rows` (e.g. 8192) and loop 4 replays per chunk; the pool shrinks ~4x
  at the cost of more replays (still 2 launches each vs 24 eager). The
  microbenchmark measures the real pool delta.

## Full phase attribution of the 24.86 s GPU idle (workflow wf_da6a83b6-ae6)

Per-phase windows of the 63.17 s traced update (phase_attribution.py, 0/114,264
kernels unclassified):

| phase | window (into wall) | GPU busy | GPU idle | util | kernels |
|---|---|---|---|---|---|
| decoder_forward | 0-15.55 s (0-24.6 %) | 12.46 s | 2.73 s | ~80 % | 36,870 |
| diffusion_consistency | interleaved w/ decoder | 0.05 s | 0.31 s | - | 9,481 |
| solver (SPSA direct-solve) | 15.55-32.71 s (24.6-51.8 %) | 5.36 s | **11.84 s** | 31 % | 30,184 |
| backward | 32.71-60.5 s (51.8-95.8 %) | 20.14 s | **7.68 s** | 72 % | 26,091 |
| optimizer_save (copy burst) | 60.5-63.2 s (95.8-100 %) | 0.30 s | 2.30 s | - | 11,638 |

Top-25 gaps: 20 solver (4.4 s of 50-200 ms per-solve drains), 4 backward (6.0 s
incl. the single 5852 ms end drain), 1 decoder startup.

### The three recoverable levers (ranked)

1. **Solver scalar-read drain (11.84 s idle, ~9 s recoverable).** The sequential
   33-probe SPSA loop gates every solve on ~6 host scalar reads (validity
   `.tolist()` aircraft_validity.py:456, occupancy `.item()` :4307, 15-scalar
   coeff `.tolist()` :1640, nonempty bool :3417, drag-exponent `.item()`
   advanced_lbm_solver.py:343, full-96^3 D2H `compute_tensor_content_hash`
   utils.py:42) -> ~600 serialized barriers/update. None of the probe values
   feed control flow inside the loop (only post-loop accumulation + records),
   so all reads can defer to ONE batched pinned D2H + one sync after the 33rd
   solve. `_assemble_direct_solver_components` (aircraft_diffusion_cfd.py:4433)
   already exists as a byte-faithful assembly tail -> reuse verbatim. Zero
   memory risk (pinned [33,~20] floats); CPU-parity-gateable; keeps solves
   sequential (batching C>=2 is OFF on 8 GB - measured 117/183 s/u vs 62.66).
   Also folds in the 2888 ms SPSA-batch-prep entry drain by pre-rolling the
   per-probe binarize/canonical/hash into the Task-9 pre-warm.

2. **End-of-update optimizer boundary (5.85 s GPU-silent + 2.30 s copy burst).**
   Between the last loss scalar read (:54.7 s) and the save burst (:60.5 s)
   the trainer serializes the host against the GPU: 3x full-model fp64 gradient
   reconstruction + `.item()` guard-dot reads (aircraft_diffusion_cfd.py:
   6932-6947, one per active guard), `clip_grad_norm_` x3 (:6905-6907),
   `capture_gradients` x2 (:6908/:6919), `project_improvement_gradients_
   against_guards`, and `_update_ema` (:5872). `offload_optimizer_state_
   between_steps` is FALSE in config.yaml (confirmed), so the copy burst is NOT
   the offload - it is the guard/gradient-capture/EMA churn. Fix direction:
   compute the guard dots GPU-side (fp64 cat + dot + a single deferred .item()),
   or accept them but overlap with the next forward. NOT the SPSA loop (corrects
   the pre-workflow belief).

3. **Decoder launch overhead (2.73 s idle, healthy ~80 % util).** This branch's
   CUDA graph (Target 1) applies here - recovers the CPU launch/Python overhead,
   not this idle (launch-gap scale). Conditional on 8 GB memory headroom.

## MSI Center utilization analysis (profile.CSV, 2026-08-16) — what's still underutilised

Workflow wf_12a39185-15e (6 lenses) over `build/perf/profile_training_window.csv`
(27,872 samples, 2.008 s cadence, 15.55 h of the continuation run). The user
asked to "find exactly what's still underutilised"; the five lenses resolve the
question that the profiler snapshot (39.4 % GPU idle per update) could not.

**Headline: the GPU is power-capped, not idle.** `lim_power_yn = Yes` on 73.6 %
of samples (~11.4 h). At load > 90 % the GPU clamps to an ~85 W TGP (mean 82.7 W,
never touches the ~95 W laptop TGP) and the clock sags from a 2535-2550 MHz boost
peak to mean 2068-2071 MHz. Temperatures never fire a limiter (gpu_temp max
69.7 °C vs 87 °C limit; hotspot max 87.9 vs 100; ~12-18 °C headroom) and thermal
limiter = No on ALL samples. So the envelope is **not** heat: the power budget is
the lever. Raising the TGP (MSI Center / driver) directly buys compute on the
compute-bound decoder (~12.5 s busy/update, ~80 % util) and backward (~20.1 s,
72 %) phases.

- **GPU idle is real but mostly short-burst, sync-bound — not CPU-bound.** 14.3 %
  of samples sit < 30 % load (~2.2 h/day), structured as 2265 low-load runs with
  median length 2.0 s (1328 single-sample blips); only 298 runs exceed 5 s
  (longest 124.5 s; 31 runs in the 16-20 s band ~ the per-update host drain).
  The profiler's per-update 39.4 % idle is authoritative; the 2 s sampler smears
  the short solver/backward drains up into mid-util samples.
- **CPU is heavily underused (mean 22.4 %, ~3.6/16 threads; max ~6.25 threads).**
  During GPU-idle samples CPU mean is 27.3 % — the CPU is ALSO idle, so the GPU
  idle is a serialization/sync/latency drain (the 33 sequential SPSA solves +
  end-of-update host drain), **not** a CPU-compute-bound one. Adding CPU threads
  will not help; de-serializing the reads is the lever.
- **VRAM is thin-but-stable: min 566 MiB, p5 573 MiB free (pinned at ~7,615/8,188
  allocated), zero samples < 256 MiB.** RAM paging is active (pagefile 32-38 %)
  but no OOM on either. Implication: a 350 MiB persistent CUDA-graph pool (lever
  3) would consume 61 % of the standing buffer, leaving a worst-case 216-223 MiB
  margin — **below the 256 MiB comfort line for ~95 % of samples. Lever 3 stays
  OFF (decoder graph)**; the memory cost is not justified by the ~1-1.5 s of CPU
  launch overhead it recovers.

**Net priority update (supersedes the pure code-fix ordering):**
1. **Raise the GPU power budget (envelope, user action)** — largest and most
   certain lever for the compute-bound phases; thermal headroom makes it safe.
2. **Lever 1 (deferred batched solver reads, ~9 s)** and **lever 2 (GPU-side
   guard-dot reads, ~1-3 s of the 5.85 s end drain)** — the code-side
   de-serialization the CPU lens confirms is the real idle driver.
3. **Lever 3 (decoder CUDA graph) — NOT WORTH IT on 8 GB** (memory margin below
   comfort line; recovers only ~1-1.5 s of launch overhead on a phase that is
   already power-capped-compute-bound).

## How to run the GPU microbenchmark (deferred)

When no training job is on the GPU:

```
python CLI/kernel_fusion_graph.py --chunks 27 --iters 20 [--eager]
```

Reports: eager vs graph ms/update, per-op vs per-replay CPU enqueue latency
(the "10 us vs 1140 us" claim), parity max abs diff vs COMPONENT_ATOL, and the
graph-pool reserved-memory delta.
