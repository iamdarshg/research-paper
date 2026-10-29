# The ~23.5 s/u Residual — De-Black-Boxed (2026-08-15)

**Question:** the flame-graph (`flame-graph-c1.md`) showed a 62.66 s/update where the
instrumented phases sum to ~39 s and **~23.5 s is unattributed**. Every optimization
claim stopped at that wall. This report opens the black box.

**Method:** two passes over the same full optimizer update on the step-1305
checkpoint (`build/recovery_ladder_20260814/step1305.pt`), 96³, batch=1,
`_DIRECT_SOLVER_BATCH_CHUNK = 1` (the frozen default), real direct/SPSA solver
(33×5-step solves), real backward, real optimizer step:

| pass | instrument | purpose | update wall |
|---|---|---|---|
| A | `torch.profiler` CPU+CUDA, `with_stack=True` | attribute autograd/optimizer/GPU-op time | 65.35 s (profiler overhead ≈ 2.7 s) |
| B | `cProfile` | pure-Python time: telemetry, `.item()`, loops, thread waits | 73.78 s (cProfile overhead) |

No phase wrappers were installed: the profiler IS the instrumentation, so
CPU/GPU overlap is un-distorted. No checkpoint writes, telemetry-only, single
process. Raw artifacts: `build/perf/baseline/residual_trace.json` (chrome trace),
`build/perf/baseline/residual_cprofile.pstats`.

---

## The answer, in one sentence

**The residual is not one thing — it is four, in roughly this order of
fixability: (1) 60 separate backward passes (~16.6 s engine time), (2) 11,987
`.item()` sync drains feeding tensorboard (~7.8 s pure + ~19 s cumtime),
(3) launch/command-buffer/sync CPU overhead (~37 s of CPU across a 65 s wall —
the CPU cannot feed the GPU), and (4) thread-lock contention (~6.9 s) plus
activation-checkpoint unpack hooks (~4.9 s).**

The GPU is doing only ~40 s of real work per update; the other ~25 s of wall
time is the CPU serializing on launches, syncs, and `.item()` calls. The LBM
solver is **no longer a GPU bottleneck** (fused `_stream_bfl_kernel`: 0.152 s
CUDA across all 165 steps) — the solver fusion and batching work is done.

---

## Pass A — torch.profiler (CPU self time, top 14 of 45)

| name | self CPU | % of 72.1 s | calls | read |
|---|---|---|---|---|
| `cudaLaunchKernel` | 15.06 s | 20.9 | **111,215** | pure kernel-launch overhead |
| `Command Buffer Full` | 12.82 s | 17.8 | 1,059 | CUDA launch queue fills → CPU blocks feeding GPU |
| `cudaStreamSynchronize` | 9.13 s | 12.7 | 2,845 | explicit syncs |
| `DirectSolverSPSAFunction` (py) | 8.26 s | 11.5 | 1 | the whole SPSA block's Python |
| `aten::nonzero` | 4.56 s | 6.3 | 967 (7.4 ms ea) | full-grid CPU scans (validity metrics, SDF occupancy, loss index extraction) |
| `cudaMemsetAsync` | 2.54 s | 3.5 | 9,442 | tiny per-launch buffers |
| `cudaMemcpyAsync` | 1.98 s | 2.7 | 6,517 | H2D/D2D churn |
| `aten::addmm` | 1.55 s | 2.2 | 5,508 | forward GEMM dispatch |
| `AddmmBackward0` (autograd) | 1.21 s | 1.7 | 4,111 | backward GEMM dispatch; **total CPU 12.22 s** |
| `aten::_local_scalar_dense` | 0.27 s | 0.4 | 2,196 | the `.item()` primitive; **total CPU 5.97 s** |
| `aten::to` / `_to_copy` | 0.14+0.44 s | — | 28,900/19,167 | dtype/device churn; total CPU ~5.3 s each |

**Launch + command-buffer + sync = ~37 s of CPU** (15.1 + 12.8 + 9.1) across a
65 s wall. This is the signature of a CPU-bound trainer that cannot keep the GPU
fed — 111k kernel launches per update.

## Pass A — GPU (self CUDA time, 39.93 s total)

| kernel | self CUDA | % | read |
|---|---|---|---|
| `ampere_sgemm_128x64_tn` | 13.67 s | 34.2 | addmm (Linear) GEMM |
| `ampere_sgemm_64x64_nn` | 6.33 s | 15.9 | mm GEMM |
| `ampere_sgemm_64x64_nt` | 6.20 s | 15.5 | mm GEMM |
| `vectorized_elementwise` (×2) | 2.92 + 2.17 s | 12.7 | silu/where/etc. |
| `aten::silu` | 2.17 s | 5.4 | |
| `reduce_kernel` | 1.13 s | 2.8 | sums |
| `Memcpy HtoD` | 1.09 s | 2.7 | 304 transfers |
| `_stream_bfl_kernel` | **0.152 s** | 0.4 | **the entire fused LBM solve block (165 steps)** |

**GEMM family (addmm + mm) = 26.2 s = 65.6% of GPU time** — the model + decoder
MLP, forward and backward. GPU time is not the solver; it is the network.

## Pass B — cProfile (top lines)

| line | calls | tottime | cumtime | read |
|---|---|---|---|---|
| tensorboard `record_writer.write` | 8 | 5.71 s | **73.7 s** | file writes spanning the whole update |
| tensorboard `add_scalar` | 12/11 | 0.01 s | **63.7 s** | ~12 scalars/update, each forcing `.item()` syncs |
| `TensorBase.item` | **11,987** | **7.79 s** | 19.2 s | the metric-drain serialization point |
| `_EngineBase.run_backward` | **60** | **16.62 s** | 18.7 s | 60 backward calls (0.28 s engine overhead each) |
| `_direct_measured_objective_for_single` | 33 | 0.08 s | 67.5 s | the SPSA block (child work) |
| `_thread.lock.acquire` | 166 | 6.88 s | 7.67 s | EDT/validity pool lock contention |
| checkpoint `unpack_hook` | 7,576 | 0.14 s | **4.85 s** | decoder activation-checkpoint bookkeeping |
| `_backward_full_grounded_coordinate_loss` | 1 | 0.72 s | **11.2 s** | one heavy backward site |
| `_decode_coordinate_features` / `_decode_latent_coordinate_chunk` | 436 | 0.82+0.04 s | 7.9 + 8.1 s | decoder chunks (incl. recompute) |

---

## Where the ~23.5 s went (reconciliation)

| residual component | measured cost | evidence | fixes |
|---|---|---|---|
| **Backward passes** | ~16.6 s engine (60 calls) | cProfile `run_backward` 16.6 s; profiler `AddmmBackward0` 20.25 s CUDA | merge branches → fewer `retain_graph` backwards (6K); audit `_backward_full_grounded_coordinate_loss` (11.2 s single) |
| **`.item()` drains + tensorboard** | ~7.8 s pure, ~19 s cumtime | 11,987 `.item()`; 12 `add_scalar`/update | GPU-resident metric reduction, sync once, batch writes (6R/6E; scan ranks 8/10/18) |
| **Launch + command-buffer + sync** | ~37 s CPU (overlaps 40 s GPU) | `cudaLaunchKernel` 15.1 s / 111k, `Command Buffer Full` 12.8 s, `cudaStreamSynchronize` 9.1 s | CUDA graphs for fixed-shape fwd/bwd (6O), `torch.compile` (6I), kernel fusion (6C) |
| **Thread-lock contention** | 6.9 s | `_thread.lock.acquire` 6.88 s | pool rebalance EDT 8→4 / validity 2→4-6 (rank 1) |
| **Activation-checkpoint unpack** | 4.9 s | `unpack_hook` 7,576 calls | measure checkpointing value at 96³ (6N); likely disable for small chunks |
| **`aten::nonzero` CPU scans** | 4.6 s | 967 × 7.4 ms | vectorize validity metric index extraction (rank 8) |
| **Copy/memset churn** | ~4.5 s | 9,442 memset + 6,517 memcpy + 304 HtoD | pin host buffers, reduce H2D, batch |

Not summed — these overlap (e.g. `.item()` syncs ARE the tensorboard cumtime,
launch overhead is CPU-side over the GPU wall). The point is attribution, and
each row is independently fixable.

---

## What this changes

1. **The next three optimizations are NOT solver work.** The LBM is 0.15 s GPU
   per update; the solver batching/fusion is done. The measured-order targets are
   now: (i) batch `.item()`/telemetry, (ii) merge the 60 backwards, (iii) cut
   launches (CUDA graphs / `torch.compile`).
2. **The 65.35 s update runs ~40 s of GPU and ~37 s of CPU feed/sync overhead.**
   The GPU is not saturated; the CPU is the limiting resource. This is the exact
   "B" stop-condition evidence the brief asks for — the bottleneck is now the
   Python/CUDA-launch layer, quantifiable per row above.
3. **Projected ceiling with the listed fixes:** roughly 65 → 40-45 s/u (recover
   most of the ~25 s CPU-overhead gap that does not overlap GPU) before hitting
   the ~40 s GPU-compute floor. Reaching below that requires shrinking the
   GEMM floor (model/decoder) itself — 6I/6J territory with parity gates.

## Raw artifacts
- `build/perf/baseline/residual_trace.json` — chrome trace of Run A (526 MB).
- `build/perf/baseline/residual_cprofile.pstats` — cProfile of Run B.
- `build/perf/baseline/residual_deblackbox.py` — the probe (gitignored scratch).
- `build/perf/baseline/residual_deblackbox.log` — full tables.
