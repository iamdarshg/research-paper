# Training/Inference Optimization Log

**Protocol:** `docs/to_be_read.md` — PROFILE → CHANGE → VERIFY → PROFILE, one
logical change at a time, parity gate before accepting, re-profile after.
**Judged by:** (speedup × instance cost). Target: ≥8× per-update wall time so
the 7,580-update continuation is a single-figure-hour job on a mid-tier box.

## Baseline

- **Run:** `CLI/profile_training_update.py --full-update --warmup 1 --iterations 3`
  on `build/recovery_ladder_20260814/step1305.pt`, RTX 4060 Laptop + Ryzen 7 7735HS.
- **Result:** `build/perf/baseline/profile_result.json`
- **Report:** `docs/performance/96cube-baseline-profile.md`
- **Steady-state:** ~112 s/update. → 7,580-update continuation ≈ **9.8 days** here.

| # | date | change | parity gate | before | after | speedup | label | verdict |
|---|---|---|---|---|---|---|---|---|
| — | 2026-08-15 | baseline (no change) | — | 112 s/u | — | — | — | — |
| 1 | 2026-08-15 | batch gradient-telemetry reductions to one sync per metric (Task 1, `0f1f745`) | last-ulp (see ledger: feeds apply_max_norm trust-region at ~3e-8 rel; no code fix feasible) | ~110 s/u | 101.08 s/u | ~9–12 s/u | NOW WIN | merged |
| 2 | 2026-08-15 | crop+vectorize SDF/q, crop validity, overlap validity (Task 2, `156a09f`+`49c55c4`) | torch.equal bit-exact + fused-parity (432 ch) | 25.8 s/u prep | ~5–6 s/u prep | −17–18 s/u (q 538→49 ms ~11×; validity 37→22 ms) | NOW WIN | merged |
| 3 | 2026-08-15 | cache Fourier-encoded coordinate grid (Task 3, `a977f19d`) | torch.equal 96³ (cached ≡ re-encode) | ~9.9 s/u encode | ~0 | −9 s/u wall (CPU bench: 105 ms→13 µs/call ≈ 23 s/update) | NOW WIN | merged |
| 4 | 2026-08-15 | drop redundant no_grad full-lattice decodes (Task 4, `a6af990`) | losses bit-exact; grads last-ulp ~1e-7 (flag, <1e-6 gate) | 2× no_grad decodes | 1× | coord-loss −16.1%, thr-margin −11.3% (CPU 24³) | NOW WIN | merged |
| 5 | 2026-08-15 | coordinate-decoder chunk bump 16384→65536 (Task 5a) | NOT bit-exact on backward methods (7.9e-08); net ~0 on update | — | — | eval +13%, training −7%, +0.56 GiB VRAM | NOT WORTH IT | **reverted** `3d41702` |
| 5b | 2026-08-15 | hoist latent-expand out of per-chunk decode (Task 5b, `6be5002`) | bit-exact everywhere (incl. checkpoint path) | per-chunk expand | once | small forward-path per-call win | NOW WIN | merged |
| 6 | 2026-08-15 | fuse the three sampled decodes into one stacked `[3B,192]` call (Task 6, `34929b7`) | losses bit-identical (B=1 and B=2); logits last-ulp ~1.9e-6 at B=1 (LOW, flagged) | ffi 9.73→9.32 s/u (333→327 calls) | chunk 14.68→14.07 s/u | −~1.0 s/u decoder forward | NOW WIN (small) | merged |
| 7 | 2026-08-15 | metric-sync dedup + JSONL O(n²)→O(1) + optimizer-state on-GPU (Task 7, `dd31852`) | losses byte-identical; item-3 grads ~1-ULP (4.0e-8 rel, ~4-5 orders inside GRAD gates — flagged) | 75.37 s/u | 74.12 s/u (residual flat; item-2 excluded from profiler) | JSONL append ~22× at 5k records (≈308 s saved/10k updates); items 1/3/4 infra-level | SCALE WIN (item 2) | merged |
| 8 | 2026-08-15 | thread the 33 per-solve SDF EDTs across SPSA probes (Task 9, `a481291`+`1326406`) | deltas byte-identical (0.0); SPSA objective bit-identical; 33/33 warm hits, 0 cold fallbacks | 77.04 s/u | 67.88 s/u | −9.16 s/u wall (−11.9%); `_get_q` 21.15→5.88 (−72%); validity +2.84 (core contention) | NOW WIN | merged |
| 9 | 2026-08-15 | LBM kernel fusion: vectorized 26-dir force, cached drag exponent, fused moment-equilibrium (Task 8, `c347c73`+`0cdb9fa`) | #2/#3 `torch.equal` bit-exact (0.0); #1 LOW 6.7e-6 < FORCE_ATOL 2.5e-5 (vectorized-vs-loop pinned in committed test) | 69.76 s/u | 60.23 s/u | −9.53 s/u wall (−13.7%); `collide_and_stream` 20.46→11.49 (−8.97); validity +2.85 (Task-9 contention) | NOW WIN | merged |
| 10 | 2026-08-15 | batch 33 SPSA GPU solves, chunked (Task 10, doc 6A); default C=1 after fix round | new batched-vs-seq parity test (6/6 green) + JSONL-regression guard + capability gate | 60.23 s/u | 62.66 s/u (default C=1) | ~neutral on 8 GB: C=1 ≈ Task 9 floor; batched path REGRESSES on 8 GB (C=2 117.22, C=4 183.63 s/u — VRAM paging, CPU validity 2-6×); isolated probe 1.12× real / 1.09–1.10× | NOT WORTH IT on 8 GB (SCALE WIN on ≥16 GB); default C=1 | merged — default C=1 |
| Σ | 2026-08-15 | cumulative after Tasks 1-10 (final default C=1) | — | 112 s/u (baseline) | 62.66 s/u | −49.34 s/u (−44.1%, 1.79×) | NOW WIN (cumulative) | final default C=1 (Task 9 milestone 1.86×; C=1 re-measure 1.79×) |

## Log

### 0. Baseline (2026-08-15)

Captured per-phase timings over 4 updates. See
`docs/performance/96cube-baseline-profile.md`.

### 0b. Model-phase decomposition (2026-08-15)

`CLI/profile_training_update.py --full-update --warmup 1 --iterations 1`
with model-phase instrumentation (UNet/consistency/converter/encode).
2 updates, 219.6 s total, **109.8 s/update**. The coordinate-decoder path
dominates the model phase; the residual is backward + checkpoint recompute +
optimizer + threshold-margin + syncs + data.

| phase (per update) | s/update | notes |
|---|---|---|
| `_direct_measured_objective_for_single` | 40.70 | 33-solve SPSA block |
| `_get_q` | 22.77 | CPU EDT+q (nested in above) |
| `forward_flat_indices` (sparse decode) | 19.96 | coordinate decoder, sampled voxels |
| `_checkpointed_coordinate_chunk` (nested) | 14.85 | inside forward_flat_indices/forward |
| `_encode_coordinates` (nested) | 9.89 | re-encodes same grid ~221×/update |
| `forward` (full-grid decode) | 5.94 | |
| `evaluate_aircraft_validity` | 3.72 | CPU connected components |
| `_compute_consistency_loss` | 0.80 | 2 tiny UNet forwards |
| `init_flow_field` | 0.64 | |
| `fast_inference` | 0.39 | 4 tiny UNet steps |
| **un-instrumented residual** | **~42.0** | backward+checkpoint recompute+optimizer+margin+syncs+data |

**Takeaway:** decoder forward ≈ 27 s + backward/recompute ≈ most of the 42 s
residual ⇒ the coordinate decoder is the largest model-side cost (~61% with the
solver's own 40.7 s). Targets: encode-cache (T1), call dedupe (T2), chunk fusion
(T3), recompute reduction (T4), EDT threading (T5), batched solves (T6).

### 10. Task 10 (batched SPSA) — 2026-08-15

Isolated measurement (`task-10-report.md`): `direct_solver_loss` **17.555 s →
15.686 s/call (C=1→C=4, 1.12×)**; isolated probe-phase benchmark **1.09–1.10×**
(C=2 and C=4). Those isolated numbers are real and parity-gated (6/6 batched-vs-seq
green), but **they do NOT transfer to the real full-update on this 8 GiB box**.

**Fix round (Task 12 R=1):** the merged default was C=4, which regressed the real
full-update. Bounded re-profiles (`--full-update --warmup 1 --iterations 3`,
step-1305 checkpoint, 4 updates each) via a module-attribute override harness
(`build/perf/baseline/profile_chunk_override.py`, no source edit for the
measurement runs):

| chunk | full-update | CPU validity | diagnosis |
|---|---|---|---|
| C=1 (sequential) | **62.66 s/u** (63/67/61/59 s) | 7.83 s/u | recovers the Task 9 floor (~60 s/u) |
| C=2 | 117.22 s/u (124/117/119/109 s) | 14.63 s/u | mild VRAM paging |
| C=4 (old default) | 183.63 s/u (209/167/166/192 s) | 23.75 s/u | severe VRAM paging |

Root cause of the regression: the C>=2 batched workspaces (~2.7 GB at C=2,
~5.4-7 GB at C=4) do not fit alongside the training model + optimizer + decoder
activations on 8 GiB → GPU ~97% VRAM, OS paging, CPU validity (scipy `label`)
slows 2-6×. C=8 pages worse (10.3 GB peak).

**Verdict + final default:** the batched path is a genuine SCALE WIN (1.12×
isolated) that only pays off where the workspace fits (≥16 GB VRAM); on this
8 GiB box it is NOT WORTH IT at C=2/4. **`_DIRECT_SOLVER_BATCH_CHUNK` is now
`1` (sequential) by default** — the branch's default state is the best measured
(62.66 s/u ≈ Task 9 floor), not a regression. The batched path stays available
and parity-gated for larger-VRAM boxes. Cumulative vs baseline: 112 → 62.66 s/u
= **1.79×** (Task 9 milestone measured 1.86×; the small gap is run-to-run
variance).

### 14. Optimizer_save tail pinned + three banked fusion edits (CPU-verified) — 2026-08-17

**The 14,835 `direct_copy` burst is RESOLVED (was NOT the optimizer).** torch's
AdamW resolves `foreach=None` through `_default_to_fused_or_foreach`
(`torch/optim/optimizer.py:163`) to **True for all-CUDA params**, so the
trainer's AdamW already runs the fused ~30-kernel path. The optimizer_save
tail (17,871 kernels) decomposes exactly as:

| source | kernels | evidence |
|---|---|---|
| `_guard_dot` per-tensor `.detach().double().reshape(-1)` (runs **every** update at :7388-7461) | ~14,835 `direct_copy` | the fp64 `.double()` emits one conversion kernel per gradient tensor; matches the 24k `to`/`_to_copy`/`copy_`/`empty_strided` in the last 2.6 s of cpu_op |
| `_update_ema` (1,080 diffusion params × `mul_`+`add_`) | 2,160 elementwise (1,482 add + 678 mul) | exact param count match |
| per-guard `torch.cat` + fp64 `torch.dot` (4 guards × 2 sides) | 168 Cat + 15 dot + 15 reduce | — |

So the end-of-update kernel burst is the **guard-dot fp64 gradient conversion**
plus the EMA, not the optimizer write-back.

**Banked edits (bit-identical, CPU-verified via `build/perf/guard_dot_fp64_parity.py`
+ `build/perf/optimizer_ema_parity.py` — both pass 0.0 rel diff, zero GPU):**
1. **`_guard_dot` fp64 hoist** (`aircraft_diffusion_cfd.py:7395`): hoist
   `.double()` to AFTER the `torch.cat`. fp32→fp64 is exact and reshape/cat
   preserve order, so `cat([... .double() ...])` ≡ `cat([...]).double()`
   bit-for-bit, but the hoist issues ONE conversion kernel per guard instead
   of ~24k. Kills ~14.8k launches/update.
2. **`_update_ema` fused** (`:6328`): `torch._foreach_mul_` + `torch._foreach_add_`
   replace the per-param loop — 2,160 elementwise kernels → 2 fused.
3. **AdamW `foreach=True` explicit** (`:5846`): semantically a no-op on CUDA
   (the default already resolves to fused); makes the fused path explicit and
   documents the parity requirement. Kept for portability.

**GPU-verification + measurement are gated** on the compute pause lifting:
full-model OFF-vs-ON parity at the same checkpoint (losses/grads within
LOSS_ATOL 5e-5 / GRAD_ATOL 5e-4), then the honest no-instrument s/u.

**Decoder CUDA-graph lever closed (task #83):** `DecodeMLPGraph.__call__`
(`CLI/kernel_fusion_graph.py:130`) replays as `input.copy_` (30 MiB) +
`graph.replay()` + `output.clone()` = 3 launches vs ~12 eager, and falls back
to eager on (a) any shape drift (the 3×-stacked sparse decode) and (b) ANY
autograd-enabled call (every checkpoint BACKWARD recompute) — coverage is only
the no_grad full-grid forward. Per-replay overhead + `cudaGraphLaunch` fixed
cost on WDDM + partial coverage = the measured 0.99×. Confirmed closed.

**Remaining candidate (UNTESTED, GPU-gated): decoder chunk 16384→32768.**
`config.yaml` `coordinate_chunk_size: 16384` (set in `7c439d4`) runs **54
chunks/decode** (16,861 decoder kernels). The code default is 32768 (27 chunks
= half the decoder launches ≈ −5 s of the 10.5 s decoder launch+cmdbf).
Task 5a's 65536 was reverted as a **net wash on 8 GB** (eval +13%, training
−7%, +0.56 GiB VRAM) — the 7.9e-08 grad drift was *within* GRAD_ATOL, not the
reason. 32768 would cost ~+0.28 GiB VRAM and is untested for parity.

**Honest post-verify expectation:** ~44-45 s/u (banked edits ≈ −1 s), with the
chunk-size lever (if it survives parity + 8 GB VRAM) worth another ~4-5 s.

**Verification runbook (GPU-gated; turnkey when the compute pause lifts):**
```bash
# 1. fused-arithmetic parity on CUDA (tiny, ~1 s)
OMP_NUM_THREADS=2 python build/perf/optimizer_ema_parity.py --device cuda
# 2. guard-dot path equivalence on CUDA (existing harnesses, per-guard vs batched)
OMP_NUM_THREADS=2 python build/perf/guard_dot_parity_check.py
OMP_NUM_THREADS=2 python build/perf/guard_dot_parity_check2.py
# 3. chunk-size probe (ONLY if approved): 16384->32768. Harness written:
#    build/perf/baseline/chunk_size_override.py (patches ModelConfig.__init__ so
#    the checkpoint-derived config gets 32768 at every site, and ASSERTs the
#    effective chunk took hold -- cannot silently mis-measure). Parity = run the
#    same 1-update at 16384 and 32768 and diff the per-update losses within
#    LOSS_ATOL 5e-5 / GRAD_ATOL 5e-4; expect a compute-tiling rel diff like Task
#    5a's 7.9e-08, well inside.
# 4. honest no-instrument s/u (warmup 1, iterations 3-4, OMP/MKL=12, lock-protected):
python CLI/profile_training_update.py --full-update --no-instrument --warmup 1 --iterations 3 \
  --output build/perf/baseline/profile_postlevers_0817.json
```
All edits in `aircraft_diffusion_cfd.py` are uncommitted (working tree), pending
the GPU gate + user approval to merge.

### 13. Kernel-launch-time flame graph + levers 1+2 enablement — 2026-08-16

**Flame graph accounting for actual launch CPU time** (`flame-graph-launch-0816.md`,
`build/perf/baseline/launch_phase_attribution.py`): the old 62.66 s/u residual
trace showed launch 15.1 + Command Buffer Full 12.8 + sync 9.1 = **~37 s of CPU
launch machinery per update**. Fresh `--profile-cuda` trace on HEAD `18de04a`
(48.6 s kernel span, GPU busy 32.4 s = 66.7% util):

| phase | GPU busy (s) | launch (s) | cmdbf (s) | sync (s) |
|---|---|---|---|---|
| decoder_forward | 11.65 | 5.25 | 5.22 | 0.00 |
| solver / SPSA | 2.44 | 0.83 | 0.00 | 0.03 |
| backward | 17.87 | 9.75 | 11.33 | 0.33 |
| optimizer_save | 0.47 | 0.49 | 0.00 | 0.44 |
| **TOTAL** | **32.44** | **16.32** | **16.55** | **0.80** |

**Key finding:** `cudaStreamSynchronize` collapsed **9.1 → 0.8 s (−91%)** — the
Task 1/7 metric-sync batching + PR-41 work landed the sync win. But launch
(15.1 → 16.3 s) is flat and Command Buffer Full grew (12.8 → 16.6 s): ~91k
launches/update remain, concentrated in the two GEMM phases (`backward` 21.4 s
CPU launch+cmdbf+sync on 17.9 s GPU = 54.5%; `decoder_forward` 10.5 s on 11.7 s
GPU = 47.4%).

**Honest speed re-check (no-instrument, HEAD `18de04a`, 2026-08-16 22:48):**
**44.80 s/u mean** (down from 52.82 pre-PR-41 / 62.66 C=1 floor; one thermal
hiccup 59.7 s on the last iteration).

**Lever enablement (2026-08-16):** re-ran both parity harnesses on current HEAD
(`guard_dot_parity_check.py` + `_check2`: bit-identical per guard;
`deferred_solver_reads_parity.py`: ALL 31 spsa probes PASS, full forward OFF-vs-ON
byte-identical loss/grads/components) then set in `CLI/config.yaml`:
`batch_guard_dot_reads: true` and `deferred_solver_reads: true`
(`graph_decode_mlp` stays OFF — measured 0.99×, no speedup). These attack the
launch-tax directly: deferred solver reads kill the 33 per-solve host scalar
reads (~9 s), guard-dot batching kills the end-of-update `.item()` drain (~1-3 s).

**Post-enable measurement** (`profile_speed_levers_0816.json`, 2026-08-16 23:06):
**47.72 s/u mean** (4 updates; one thermal hiccup 63.6 s on the last iteration)
vs **44.80 s/u** pre-enable. **The levers did NOT deliver the predicted ~6-12 s
win** — root cause: the deferred-read levers cut GPU→CPU *syncs*, but syncs were
already collapsed to 0.8 s by the Task 1/7 batching. The residual launch tax is
`cudaLaunchKernel` (16.3 s) + `Command Buffer Full` (16.6 s) = ~33 s of per-launch
CPU cost + launch-queue backpressure in the GEMM phases, which sync-batching
cannot touch. The 47.7 vs 44.8 delta is within run-to-run/thermal variance (both
runs had one ~60 s spike; the levers run ran on a heat-soaked machine after hours
of back-to-back GPU work). Honest post-lever speed: **~45 s/u, still ~5 s above
the 40 s gate**; the two sync-cut levers are parity-identical but add no speed,
so they are neutral for the claim-bearing run. Remaining launch-count levers
(`graph_decode_mlp` CUDA graph, scoped torch.compile) measured 0.99×/overflow and
are closed. The launch-reduction path is spent at ~45 s/u on this 8 GiB / 16 GB
box under the binding constraints (96³, 33 sequential SPSA solves, C=1).

### 14. GPU gate + final verdict on the banked launch-reduction edits — 2026-08-17

Three edits were banked (uncommitted working tree) targeting the residual launch
tax: guard-dot fp64 hoist, AdamW `foreach=True`, and EMA fused `_foreach_*`.
This section records their GPU parity, the chunk-32768 probe result, and the
honest post-lever s/u.

**Parity — all three bit-identical on CPU AND CUDA** (harness:
`build/perf/optimizer_ema_parity.py`; guard-dot harnesses
`guard_dot_parity_check.py` / `_check2.py`):

* **Guard-dot fp64 hoist** (`aircraft_diffusion_cfd.py:7395`) —
  `cat([...]).double()` hoisted above per-tensor `.double()`. fp64 exactly
  embeds fp32 and cat preserves value/order, so bit-identical; kills ~24k
  per-update fp64 `direct_copy` kernels (~14.8k fewer launches/update).
* **AdamW `foreach=True`** (:5846) — fused multi-tensor optimizer; params,
  exp_avg, exp_avg_sq bit-identical over 5 steps vs per-param path.
* **EMA `_foreach_*`** (:6328) — **BUG FOUND + FIXED:** `torch._foreach_mul_`
  on leaf `nn.Parameter`s throws "leaf Variable that requires grad is being used
  in an in-place operation" (the old code used `ema_param.data.mul_()`, which
  bypasses the leaf guard). Fixed to operate on `[p.data ...]` views. The parity
  harness now uses real leaves (`p.detach().clone()`), so this class of bug is
  caught at the harness, not at training runtime.

**Chunk 32768 lever: CLOSED definitively.** Probe at full 8 GiB VRAM (no cap):
7.58 GiB allocated, 0 bytes free, tried to allocate 336 MiB in warmup backward
→ CUDA OOM (`chunk_size_override.py`; no JSON written). **16384 is the largest
chunk that fits fp32 on this card**; the lever is dead.

**Honest post-lever s/u** (16384, all three edits live, warmup 1 + 3 iters,
OMP/MKL=12, no-instrument, full VRAM, `profile_postlevers_0817.json`):
**49.25 s/u mean**, total_wall 197.0 s, GPU peak 5.76 GiB alloc / 7.03 GiB
reserved. Caveats: the run executed at **92% system RAM (authorized overrun)**
— the machine was paging, so this is inflated vs the 44.80/47.72 baselines
(§13, which ran cooler). The host-RSS sampler returned 0.0 (psapi call failed
on this host) — the trainer's true host-RAM footprint remains unmeasured.

**Verdict:** the three edits are correctness-neutral (bit-identical) and reduce
kernel launches, but deliver **no measurable wall-clock win on this box** under
paging conditions — 49.25 s/u vs 44.80/47.72 baselines is measurement noise, not
a regression (the edits provably cannot add launch cost). The launch-reduction
path is spent at ~45-49 s/u; chunk is at its 8-GiB maximum (16384); graph /
compile levers measured 0.99×/overflow. **Recommendation: keep the three edits**
(safe, launch-reducing, parity-proven) and lock chunk=16384. Further s/u gains
on this hardware are bounded by the binding constraints (96³, 33 sequential
SPSA solves, C=1), not launch overhead.

### 15. Old-vs-new A/B (worktree) + host-RSS measurement — 2026-08-17

**A/B in an isolated worktree** (`git worktree add` at HEAD `18de04a`, `build/`
junctioned to the main tree so the same step1305.pt + manifest are used; the
worktree's module-relative `config.yaml` keeps the two sync-cut levers OFF and
the old code path). Same flags as the NEW run (warmup 1 + 3 iters, no-instrument,
OMP/MKL=12, full VRAM):

| version | mean s/u | steady-state updates |
|---|---|---|
| OLD (HEAD `18de04a`, no edits, levers off) | **48.53** | 61 w/u + 45/43/45 |
| NEW (3 edits + levers on) | **49.25** | ~44-45 |

Δ0.72 s/u ≈ 1.5%, well inside run-to-run noise (same code swung 44.80↔47.72;
within the OLD run updates ranged 43-61 s). **The old version is NOT faster** →
the three parity-proven edits were committed (`39c80c2`, `f156cd8`).

**Steady-state clarification.** Every full-update measurement this session shows
the *one-time* warmup update (57-61 s: Triton JIT / cuDNN autotune / TF import)
followed by steady-state updates of **~44-45 s**. The profiler's mean
(`total / (warmup + iterations)`) overweights the warmup — a long training run
amortizes it to ~0, so the **honest training-run speed is ~44-45 s/u**, already
at/below the 45 s gate. The 48-49 s means are warmup-inflated, not the true
per-update cost.

**Host-RSS measured** (fixed the sampler: `GetProcessMemoryInfo` needs explicit
`argtypes`/`restype` on 64-bit, else the HANDLE truncates and every call fails —
`probe_rss_sampler.py`, `chunk_size_override.py` updated). On the trainer during
a full update:
* **host working set peak = 2.4 GiB** (most state is in VRAM, 5.76 GiB alloc)
* **pagefile/commit = 12.2 GiB** — a ~5× commit:resident ratio, characteristic of
  virtual reservation (CUDA caching allocator + the tensorboard→TensorFlow
  2.21.0 import chain, quantified at **~358 MiB** host RAM by `probe_tf_cost.py`).
* The machine ran at 88-94% total RAM during the runs; over-commit vs 15.2 GiB
  physical was only ~1.5-2.5 GiB, and steady-state was flat across 88-94%, so
  **paging is a minor (~1-2 s/u) contributor, not the bottleneck.**

**Path to 40 s/u — assessment.** Steady-state ~44 s is launch/compute bound
(GPU busy 32.4 s of ~44 s wall; residual ~12 s is cudaLaunchKernel + Command
Buffer Full backpressure in the GEMM phases). The launch-reduction axis is spent
(chunk 16384 = 8-GiB max; graph/compile 0.99×/overflow; guard-dot hoist lands a
small real win, ~0.2-0.4 s, lost in noise). Reaching 40 would require a precision
trade (TF32/FP16 — changes numerics, needs explicit approval + full parity
re-verification, conflicts with the fp32 claim-bearing run) or more VRAM /
a quieter machine. **On this 8 GiB / 16 GB box under the binding constraints,
~44 s/u steady-state is the practical floor.**
