# Experiment: kernel-fusion launch reduction (branch `experiment/kernel-fusion-launch`)

Status: **experimental, off by default**. Built under the standing GPU constraint
(approved training job PID 24436 running; no GPU work may launch while it runs).
The harness and its CPU-only fallback path are verified; the GPU microbenchmark
is written but **must not be run until the training job ends or the user opens a
GPU window**.

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

## How to run the GPU microbenchmark (deferred)

When no training job is on the GPU:

```
python CLI/kernel_fusion_graph.py --chunks 27 --iters 20 [--eager]
```

Reports: eager vs graph ms/update, per-op vs per-replay CPU enqueue latency
(the "10 us vs 1140 us" claim), parity max abs diff vs COMPONENT_ATOL, and the
graph-pool reserved-memory delta.
