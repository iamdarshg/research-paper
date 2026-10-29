# Flame graph of the training update at C=1 (2026-08-15)

Device: NVIDIA RTX 4060 Laptop 8 GiB (this laptop), Ryzen 7 7735HS (8 cores),
torch 2.9.1+cu130. Measured with `CLI/profile_training_update.py --full-update`
on `build/recovery_ladder_20260814/step1305.pt`, `_DIRECT_SOLVER_BATCH_CHUNK = 1`
(the frozen default). Source: `build/perf/baseline/profile_result_c1.json`.

This document is the "flame graph" of one 62.66 s training update: where the time
goes, what has already been fused/optimized in each bar, and what the honest
floors are on this box. A companion section (ranked remaining fusion
opportunities) is appended below from the 2026-08-15 fusion-scan.

---

## The one-update budget (62.66 s/u)

The phases below are **partially nested and partially overlapped — do not sum the
table naively.** The nesting rules are stated under "structure".

| bar | s/u | what it is | already optimized by | floor vs fixable |
|---|---|---|---|---|
| **Un-instrumented residual** | **~23.5** | SPSA Rademacher direction generation, replay backward (28.1M), optimizer+EMA step, telemetry/JSONL fsync, per-metric `.item()`/`.cpu()` syncs, Python loop overhead | Task 1 (sync batching), Task 7 (JSONL O(n²)→O(1), optimizer on-GPU) | **unattributed — the largest single bar, and a black box** |
| **SPSA solve block** (`_direct_measured_objective_for_single`) | **17.04** | the 33 per-solve objectives (1 base + 16 dirs × 2 antithetic) | Task 8 (LBM fusion), Task 9 (threaded EDT), Task 10 (batched path, SCALE WIN ≥16 GB) | memory-bandwidth floor |
| └ `simulate_aerodynamics` | 15.45 | 33 solver simulations | — | — |
| └ `collide_and_stream` | 13.83 | 165 fused LBM steps (33 × 5) | Task 8: vectorized 26-dir force, cached drag exponent, fused moment-equilibrium | bandwidth floor (~72 ms/step) |
| **Decoder decodes** (`_checkpointed_coordinate_chunk`) | **15.52** | 872 chunk calls; real decode 10.74 + **4.78 chunk-loop/encode/cat overhead** | Task 3 (cached Fourier grid), Task 5b (latent-expand hoist), Task 6 (3-way stacked decode) | partial — the 4.78 is attackable |
| └ `forward_flat_indices` | 10.74 | sparse coordinate decode (436 calls) | Task 6 (cat→one call→chunk) | FLOP floor |
| **CPU EDT q-field** (`_get_q`) | **8.00** | threaded Euclidean distance transform, 165 calls | Task 2 (crop+vectorize), Task 9 (thread across cores) | CPU core; **overlaps the GPU** |
| **CPU validity** (`evaluate_aircraft_validity`) | **7.83** | connected components, scipy `label`, 33 calls | Task 2 (crop+overlap) | CPU core; **overlaps the GPU** |
| `forward` (full-grid decode) | 5.49 | diffusion model forward on the latent | Task 4 (dropped redundant full-lattice decodes) | FLOP floor |
| `fast_inference` | 0.35 | 4 tiny UNet steps | — | small |
| `_compute_consistency_loss` | 0.29 | — | — | small |
| `init_flow_field` | 0.44 | solver init | — | small |
| `_encode_coordinates` | 0.015 | coordinate encode | Task 3 (cached) | negligible |

## Structure: how the bars nest and overlap

1. **Nesting.** `_direct_measured_objective_for_single` (17.04) wraps the solver
   region: it contains `simulate_aerodynamics` (15.45), which contains
   `collide_and_stream` (13.83). So the solver block is ~17 s/u of wall time, of
   which ~14 s is the fused LBM kernel.
2. **CPU/GPU overlap.** The objective block (17.04) is *less* than
   collide(13.83) + EDT(8.00) + validity(7.83) = 29.66 — so the threaded EDT and
   the connected-components validity already run while the GPU executes the LBM
   solves. Those ~16 s of CPU work only cost wall-clock where the CPU is the
   slower side. This is the Task 9 design (thread the 33 per-solve EDTs across
   the 8 cores).
3. **The residual.** The top-level non-nested instrumented sum is ~39 s/u; the
   remaining ~23.5 s is everything the profiler does not wrap. It is larger than
   any single instrumented bar. The immediate next measurement is a
   function-level flame graph (py-spy / `torch.profiler`) over one full update to
   attribute it.

## Baseline → now (112 → 62.66 s/u = 1.79×)

| bar | baseline s/u | now (C=1) s/u | moved by |
|---|---|---|---|
| `collide_and_stream` | 35.13 | 13.83 | Task 8 kernel fusion; Task 10 solver-block rework |
| `_get_q` (EDT) | 21.92 | 8.00 | Task 2 crop+vectorize, Task 9 threading |
| `evaluate_aircraft_validity` | 3.86 | 7.83 | **regressed** — core contention with the threaded EDT |
| `simulate_aerodynamics` | 36.52 | 15.45 | above |
| `_direct_measured_objective_for_single` | 40.48 | 17.04 | above |
| decoder + forward (was inside the residual) | ~residual | 15.52 + 5.49 | Tasks 3/4/5b/6 |
| un-instrumented residual | ~72.5 | ~23.5 | Tasks 1/7 (syncs, fsync, optimizer offload) + decoder work |

The residual shrank ~72.5 → ~23.5 s/u, and that is where the largest remaining
cost still hides. The full optimization log is in
`docs/performance/optimization-log.md`.

## The floors on this box (what cannot shrink)

- **The LBM is memory-bandwidth-bound**, not launch-bound: ~72 ms/step × 165
  steps ≈ 12-14 s/u of population traffic is the irreducible kernel cost at
  D3Q27 / fp32 / 96³. Fewer launches or a fused mega-solve recovers overhead, not
  traffic.
- **8 GiB VRAM ceiling**: the Task 10 batched path's workspaces (~2.7 GB at C=2,
  ~5.4-7 GB at C=4) do not fit beside the training model, so the default is
  C=1 sequential. The batched path is a SCALE WIN that pays off on ≥16 GB boxes.
- **No precision switches, no step/direction cuts** (frozen): 5 steps, 16
  directions, D3Q27, fp32.

## Open: ranked remaining fusion opportunities

Appended from the 2026-08-15 fusion-scan (5 deep-read subagents + synthesis).
Estimates are s/u on the C=1 profile and are **not additive** — LBM gains
overlap (phase ceiling ~11-12 s/u of the 13.83 s/u collide bar) and CPU gains
overlap the GPU solves. The realistic ceiling with all listed fixes is
**~3.2× (≈34-38 s/u)**.

| # | fix | component | est. s/u | effort | risk | 8 GB |
|---|---|---|---|---|---|---|
| 1 | Rebalance EDT/validity thread pools (EDT 8→4, validity 2→4-6) | aircraft_diffusion_cfd.py:4017/4024 | ~3 (2-4) | S | LOW-MED | ✓ |
| 2 | Fuse rho+ux/uy/uz velocity reductions into one `[4,27]@[27,N]` tensordot | advanced_lbm_solver.py:464-475, 583-592 | ~2.9 | S-M | LOW | ✓ |
| 3 | Merge duplicate direct-solver full-lattice decodes (no_grad 6416 + grad 6583 → one grad decode + `.detach()`) | aircraft_diffusion_cfd.py:6416/6583 | ~2.6 | M | MED (verify 8 GB peak) | ✓ |
| 4 | Fuse force accumulation + upwind drag proxy into one Triton boundary kernel | advanced_lbm_solver.py:572-577, 289-294 | ~2.7 | M | MED | ✓ |
| 5 | Single final backward for the margin path | aircraft_diffusion_cfd.py:6036-6142 | ~2.2 | M | MED (re-run recovery gate) | ✓ |
| 6 | Fuse MRT collision into one Triton kernel | advanced_lbm_solver.py:491-525 | ~2.7 | M-L | HIGH | ✓ |
| 7 | Hoist per-probe geometry prep into a one-time pre-pass | aircraft_diffusion_cfd.py:4211-4217 | ~2.0 | S-M | LOW-MED | ✓ |
| 8 | Vectorize validity heuristic metrics (~192 `.item()`/`.any()` syncs) | aircraft_validity.py:171-174 | ~1.5 | S | LOW | ✓ |
| 9 | Vectorize 32-probe construction (batched trilinear) | aircraft_diffusion_cfd.py:4731-4762 | ~1.25 | S | MODERATE | ✓ |
| 10 | Batch ~20-30 gradient-telemetry reductions into one flat buffer | multiobjective_gradients.py:115-136 | ~1.1 | S | LOW | ✓ |
| 11 | Drop redundant `nan_to_num` full-array passes | advanced_lbm_solver.py:461, 523, 580 | ~0.9 | S | MED | ✓ |
| 12 | Fuse optimizer step + EMA (fused AdamW) | aircraft_diffusion_cfd.py:6879-6888 | ~1.0 | S-M | MED (parity) | ✓ |
| 13 | Persistent 5-step LBM kernel | advanced_lbm_solver.py:458-611 | ~2.75 marginal after 2/4/6/11 (or ~11-12 total if first) | L (multi-week) | VERY HIGH | ✓ |
| 14 | Cache `solid_u8` + inlet `feq_in` per geometry hash | advanced_lbm_solver.py:543, 346-363 | ~0.03 | S | LOW | ✓ |
| 15 | Keep AdamW state on GPU (offload=false) | aircraft_diffusion_cfd.py:506-507 | ~0.35 | S | LOW-MED | ✓ |
| 16 | Fuse the two grounded full-lattice grad decodes (latent + generation_latent) | aircraft_diffusion_cfd.py:5914/6083 | ~0.4 | M | MED | ✓ |
| 17 | Split decoder input Linear (no per-chunk cat temp) | aircraft_diffusion_cfd.py:3094-3097 | ~0.3 + VRAM relief | M | LOW | ✓ |
| 18 | Collapse per-probe/per-update `.item()` drains | aircraft_diffusion_cfd.py:4242-4254, 4864-4910 | ~0.3 | S | LOW | ✓ |
| 19 | Warm base-solve q synchronously + hash CPU geometry directly | aircraft_diffusion_cfd.py:4149-4159 | ~0.3 | S | LOW-MED | ✓ |
| 20 | Gate debug SPSA parity work + enable `with_stack` profiler export | aircraft_diffusion_cfd.py:4916-4928 | ~0.1 + de-black-boxes the ~23.5 residual | S | LOW | ✓ |
| 21 | Batched SPSA probes with persistent buffers (C=2-4) | aircraft_diffusion_cfd.py:4047, 4805-4863 | ~0 on 8 GB (measured regression C=2 117 / C=4 183); -6 to -10 on ≥16 GB | L | HIGH | ✗ |
| 22 | fp16 q + packed 1-bit boundary links (batch workspace) | advanced_lbm_solver.py:1501-1508 | ~0 on 8 GB (saves ~0.9 GiB at C=4; still overflows) | M | MED | ✓* |

**Hard floor (~34-38 s/u ≈ 2.9-3.3×; realistic ~3.2× with all fixes).**
Four irreducible terms on this 8 GB box: (1) the two full-lattice
coordinate-decoder passes + backward + activation-checkpoint recompute
(~15-20 s/u, chunk size VRAM-capped at 16384); (2) latent UNet forwards +
direct replay backward + data/consistency losses (~8-12 s/u); (3)
optimizer/EMA + gradient-capture/guard/telemetry (~3-5 s/u); (4) the fused
LBM solve floor (~1-2 s/u; ~0.76 s/u pure memory bound for 165 steps).
**8× (14 s/u) is out of reach on this 8 GB laptop** — it needs ≥16-24 GB
VRAM (batched solves + bigger decoder chunks + fused batch kernels), a
smaller model/decode, or multi-GPU. The gap is dominated by the decoder
floor and the model fwd/bwd+optimizer floor, both 8 GB-VRAM-capped, not code
inefficiency.

**Key caveats (from the scan):**
- LBM-phase gains overlap: the velocity-reduce + force + MRT + `nan_to_num` +
  persistent-kernel estimates sum ~18 s/u but draw from a 13.83 s/u bar; the
  honest phase ceiling is ~11-12 s/u, and the persistent kernel's marginal
  gain AFTER the staged fusions is ~2.75 s/u (the naive 4.4× cumulative is an
  artifact).
- The ~23.5 s/u residual-attack estimates sum against genuinely irreducible
  terms (direct replay backward, data-loss retain_graph backward, decoder
  backward dispatch); realistic residual recovery is ~5-7 s/u.
- Batched SPSA (rank 21) measurably regresses on this 8 GB box; it is only a
  ≥16 GB play. **The LBM state-compression work (sparse q + buffer collapse +
  gated bf16, Task 34) is the 8 GB lever that changes this.**
- fused=True AdamW, the margin single-backward, and the MRT collision fusion
  all change last-ulp op order — each must re-run the checkpoint-frozen
  recovery parity gate before adoption.
- `run_monitored_training.py:716` defaults to `pytorch_reference`, which
  silently runs the slower 27-roll path; continuation must pass
  `fused_stream_bfl` or any comparison is invalid.
- **Recommended next step (scan):** thread-pool rebalance (rank 1) first,
  then the velocity-reduce tensordot (rank 2), and flip on `with_stack` so
  the ~23.5 s/u residual stops being a black box.
