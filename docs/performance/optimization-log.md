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
| 10 | 2026-08-15 | batch 33 SPSA GPU solves, chunked C=4 (Task 10, doc 6A) | new batched-vs-seq parity test (6/6 green) + JSONL-regression guard + capability gate | 60.23 s/u | 183.63 s/u | +123.40 s/u (+204.9%) at C=4 on 8 GB — full-path REGRESSION (memory pressure); isolated direct-solver call 17.555→15.686 s (1.12×); isolated probe 1.09–1.10× | NOT WORTH IT at C=4 on 8 GB (SCALE WIN on ≥16 GB) | merged — run continuation at C=2/C=1 |
| Σ | 2026-08-15 | cumulative after Tasks 1-10 (merged C=4 default) | — | 112 s/u (baseline) | 183.63 s/u | +71.63 s/u (+63.95%, 0.61×) — REGRESSION as merged | REGRESSION as merged | run continuation at C=2/C=1 (Tasks 1-9 alone = 60.23 s/u = 1.86×) |

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

Re-profile (Task 12, `--full-update --warmup 1 --iterations 3`, 4 updates,
734.52 s): **183.63 s/u** vs the Task 9 row's **60.23 s/u** — a **3.05×
regression**, not the plan's estimated −11–12 s/u win. Root cause: the C=4 batch
workspace (~5.4 GB isolated, ~7 GB in the real path) does not fit alongside the
training model + optimizer + decoder activations on 8 GiB → GPU 97%
(7.9/8.2 GB), system ~0 free RAM (484 MB), profile process committing 22.8 GB
private / 6.9 GB WS → OS paging. CPU validity (scipy `label`) regressed 6×
(3.9 → 23.75 s/u); per-update 209/167/166/192 s. C=8 pages worse (10.3 GB peak).

**Verdict: NOT WORTH IT at C=4 on 8 GB** (a genuine batching win, 1.12×, that
only pays off where the workspace fits — ≥16 GB VRAM). **Recommendation: run the
96³ continuation with `_DIRECT_SOLVER_BATCH_CHUNK = 2` (or 1)** — Task 10's own
report called C=2 the "safer choice on memory-constrained runs" — and/or
`empty_cache()` before the probe phase. No CLI/ code changed here (Task 12
constraint). Cumulative at the merged default is 112 → 183.63 s/u = **0.61×**
(regression); Tasks 1-9 alone = 60.23 s/u = **1.86×** at C=1/C=2.
