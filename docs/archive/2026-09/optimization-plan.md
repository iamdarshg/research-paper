# Optimization Plan — Training-Update Speedup (96³, D3Q27, SPSA)

**Goal:** ≥8× per-update wall-time reduction on the cheap box (RTX 4060 Laptop +
Ryzen 7 7735HS), driving the 7,580-update continuation from ~9.8 days to hours.
**Measured baseline:** 109.8–113.0 s/update (mean == median == p90 == p95).
**Protocol:** `docs/to_be_read.md` — PROFILE → CHANGE → VERIFY → PROFILE, one
logical change at a time, exact-parity gate before accepting, re-profile after.
**Judge each change:** (speedup × instance cost); label NOW WIN / SCALE WIN /
BOTH / NOT WORTH IT; record mean/median/p90/p95.

## Global constraints (MANDATORY — every implementer and reviewer enforces these)

1. **Exact-parity gates.** `tests/test_direct_solver_fused_parity.py` (fused vs
   pytorch_reference at 32³; COMPONENT_RTOL=1e-3, GRAD_ATOL=5e-4, LOSS_ATOL=5e-5)
   must pass after any solver change. New batched-vs-sequential parity test for
   batched SPSA. Converter changes must preserve identical output (or fp32
   rounding absorbed by existing loss tolerances — only if flagged).
2. **No lowering solver steps (5) or SPSA directions (16).** Never claim a
   speedup that comes from fewer/different solves.
3. **No silent precision switches** (fp32→fp16/tf32). Flag any approximate
   proposal as parity-risk, never present it as exact.
4. **No deleting, overwriting, renaming, or modifying original failed-run
   artifacts.** Do not discard grounded corpus evidence.
5. **Do not change** 96³ resolution, latent_dim=192, conditioning_dim=22, or the
   [Z,Y,X] geometry convention.
6. **No training runs.** Probes only (bounded profiler, bounded parity tests).
   Long training requires explicit per-run user approval.
7. **Benchmark discipline.** Report mean/median/p90/p95; bound profile runs
   (≤5 updates). No silent precision switches in reporting either.
8. **No start of self-training or a multi-hour run.**

## Measured per-update cost stack (baseline, step-1305)

| subsystem | s/update | share | source |
|---|---|---|---|
| direct-solver phase (`_direct_measured_objective_for_single`, 33 solves) | 40.7 | 37% | profile |
| └ GPU collision+stream (`collide_and_stream`) | 35.6 | | profile |
| └ CPU EDT/q (`_get_q`) | 22.8 | | profile |
| └ CPU validity | 3.7 | | profile |
| coordinate-decoder `forward_flat_indices` (sparse) | 19.96 | 18% | model profile |
| └ `_checkpointed_coordinate_chunk` (nested) | 14.85 | | model profile |
| └ `_encode_coordinates` (nested, re-encodes every call) | 9.89 | | model profile |
| coordinate-decoder `forward` (full grid) | 5.94 | 5% | model profile |
| consistency loss + fast_inference (tiny UNets) | ~1.2 | 1% | model profile |
| **un-instrumented residual** (gradient-telemetry syncs + redundant decodes + optimizer + margin + syncs) | **~42.0** | 38% | residual |

**Top-line:** the residual is now explained: gradient-telemetry sync storm
(~18–20k GPU→CPU syncs/update) + redundant no_grad full-lattice decodes +
metric syncs + optimizer offload + JSONL fsync. Coordinate-decoder forward
decodes ≈ 27 s. Direct-solver phase = 40.7 s.

## Task 1 — Batch gradient-telemetry reductions (was T9-item1)

**Why:** ~18–20k GPU→CPU syncs/update drain the stream. Biggest single win.
**Where:** `CLI/multiobjective_gradients.py` — `gradient_l2_norm` L85
(`isfinite().all().item()`), L110 (`max().item()`), L115-120
(`vector_norm(...,dtype=float64).item()`), L168-172 & L206-210
(`sum(double).item()`); `gradient_cosine_similarity`/dot loops; called from
`CLI/aircraft_diffusion_cfd.py` L5940-5972, L6016, L6025, L6066, L6072,
L6089-6108, L6115, L6126-6146. Each loop walks all ~190 optimizer tensors with
full-param reductions + `.item()` sync; `_validate_gradient_tensor` (L74-89)
does `isfinite().all()` full-param reduction per tensor per norm (~6,600/update).

**Change:** replace per-tensor `.item()` reductions with ONE batched reduction +
one sync: `g = torch.cat([p.grad.flatten() for p in params if p.grad is not
None]); total = g.norm()`. Cosine/dot via `torch.dot` on concatenated flats.
Keep the `guard_dot` correctness gate (L6126-6146) but evaluate as ONE
`torch.dot(...).item()` over concatenated tensors.

**Parity:** EXACT-SAFE for optimization behavior — telemetry feeds `last_*`
dicts/callbacks only, never the optimizer. Last-ulp changes in logged norms OK.

**Verify:** full-update profile (2 updates) — expect gradient-telemetry share to
collapse; assert optimizer step count + losses unchanged. Est. −20–40 s/u.
Complexity: S.

## Task 2 — Crop + vectorize SDF/q, crop validity, overlap validity (SDF sweep P1+P2+P3)

**Why:** the aircraft is ~1–4% of the 96³ box — EDT over the full volume is
almost entirely wasted work. Measured (RTX 4060 + 8-core, real 96³ aircraft
voxel, 0.93% occupancy): full-domain `compute_all_link_distances` 854 ms/solve
→ cropped+vectorized 173 ms/solve (4.9×, bit-exact); cropped connected-
components label 7.2× (bit-exact). Cropping is a bigger win than threading,
is local, and stays EXACT-SAFE.
**Where:** `CLI/sdf_utils.py` `compute_all_link_distances` (L77-120),
`compute_sdf` (L42); `CLI/aircraft_validity.py` `_heuristic_metrics`
connected-components block (L81-88); `CLI/aircraft_diffusion_cfd.py`
`_direct_measured_objective_for_single` (L3934-3940, `geometry_cpu` at L3884).

**Change (P1, primary):** crop SDF+q to the solid bounding box. `solid =
voxel_grid > 0.5`, bbox via `torch.nonzero`, crop = bbox expanded by margin 2
(covers every crossing fluid cell: those are ≤1 cell outside the bbox; their q
neighbor is the solid inside). Run `compute_sdf` on the crop only, pad, stack
all 26 shifted neighbor slices into one `[26,cD,cH,cW]` tensor, compute q with
`clamp(sdf/(sdf-neighbor+1e-12))` then `where(crossing, q, 1.0)`, scatter into
full-size `q_all` (init 1.0); return all-ones early for empty geometry. Read
the 27 shifts once via `.item()` (not 78 per-direction reads; or pass
`_stream_shifts` from advanced_lbm_solver.py L74). P1 alone: −16–18 s/u.
**(P2):** crop the connected-components label to the solid bbox in validity
(largest_component_fraction invariant under the bbox crop; other metrics stay
on full grid); −0.7–1.5 s/u. **(P3):** overlap `evaluate_aircraft_validity`
with the LBM solve — module-level `_VALIDITY_POOL = ThreadPoolExecutor(max_workers=2)`,
submit validity (pure function of `geometry_cpu`, no dep on `cfd_results`)
before `simulate_aerodynamics`, collect after; −1.5–2 s/u.
Combined: geometry-prep ~25.8 s → ~5–6 s/u (~20 s, ≈18% of the 113 s update).

**Parity:** EXACT-SAFE — verified `torch.equal` bit-exact at 32³/96³ on real
aircraft voxels and empty/full/single-cell/corner/boundary-touching/dense
masks. Existing fused-parity test must pass.
**Verify:** solver-only profile + full-update profile; assert identical
objective values. Complexity: M. Diff sketch in the ledger (SDF sweep
aba967ef4e3da0665).

## Task 3 — Cache the Fourier-encoded coordinate grid (was T1)

**Why:** `_encode_coordinates` runs ~221×/update on (subsets of) the SAME grid.
**Where:** `CLI/aircraft_diffusion_cfd.py` LatentTo3DConverter `_coordinates`
(L2918), `_encode_coordinates` (L2929), `forward` (L3010), `forward_flat_indices`
(L2992). Current: each call re-encodes its selected coords.

**Change:** precompute the encoded grid for the full 96³ coordinate lattice once
(cache buffer, like `_coordinate_grid`), and in `forward_flat_indices` do
`encoded_full.index_select(0, flat_indices)` instead of re-encoding the subset.
`forward` uses the cached full encoding directly. Invalidate on device/dtype/
fourier_bands change.

**Parity:** EXACT-SAFE — `sin`/`cos` of the same inputs, cached. Identical
output.
**Verify:** run the converter twice (cached vs uncached) assert identical; then
full-update profile. Est. −9 s/u (WALL — the profile measures `_encode_coordinates`
at 9.89 s/u; the compute-bound estimate of ~15 ms is wrong, the cost is overhead).
Complexity: S.

## Task 4 — Drop redundant no_grad full-lattice decodes (was T4)

**Why:** `latent` is fully decoded 5×/update; two no_grad passes exist only to
compute detached scalar metrics — pure recomputation.
**Where:** `CLI/aircraft_diffusion_cfd.py` `_backward_full_grounded_threshold_margin`
(L5313; no_grad pass L5360-5376, grad pass L5394-5424) and
`_backward_full_grounded_coordinate_loss` (L5157; no_grad L5177-5209, grad
L5261+), plus sampled clean decode (L5667).

**Change:** accumulate `chunk_loss` (positive/negative sums) inside the grad
pass and return the detached scalar from the same running sums; delete the
first `with torch.no_grad():` loop. Do NOT merge the two `latent` backprops in
this task (separate MEDIUM-risk change — defer).

**Parity:** EXACT-SAFE — same arithmetic, gradients unchanged.
**Verify:** run a full update twice (before/after) assert identical loss scalars
and gradient L2 norms; full-update profile. Est. −8–18 s/u. Complexity: M.

## Task 5 — Coordinate-decoder chunk-loop fusion (was T3)

**Why:** chunked decode loop (27 chunks @ 32,768 for 96³) with per-chunk
expand/cat temporaries + gradient-checkpoint recompute in backward.
**Where:** `CLI/aircraft_diffusion_cfd.py` LatentTo3DConverter
`_decode_latent_coordinate_chunk` (L2946), `_checkpointed_coordinate_chunk`
(L2964), `forward` (L3010), `forward_flat_indices` (L2992), `_effective_coordinate_chunk_size`
(L2982). Config `coordinate_chunk_size=32768`.

**Change (each independently gated):** (a) raise `coordinate_chunk_size` to
131,072 (fewer, larger GEMMs) IF peak VRAM allows; (b) precompute the coord
batch tensor once (avoid re-expanding `latent[:, None, :]` per chunk);
(c) evaluate disabling gradient checkpointing per-chunk where the 8 GiB GPU
allows (checkpoint recompute ≈ doubles backward cost) — memory-gated;
(d) split the input `Linear(231→256)` (converter sweep Opt 1): the concat of
`[latent, fourier-encoded coords]` → build the linear's two inputs separately
and do `W1@x1 + W2@x2 + b` instead of one GEMM over the cat — cuts the cat
temporary and the checkpoint-recompute graph.

**Parity:** EXACT-SAFE for (a)/(b) (identical GEMMs, fp32); (c) changes recompute
vs saved-activations but same arithmetic — verify identical gradients; (d) is
LOW-parity (fp32 summation order differs) — gate it on the existing loss
tolerances, flag it, keep it a separate commit.
**Verify:** full-update profile; assert identical losses/gradients. Est. −5–8 s/u.
Complexity: M.
**Framing (converter sweep ada1b428a4746b6e3):** the decoder is OVERHEAD-bound,
not compute-bound — the profile measures ~91 ms wall per `forward_flat_indices`
call vs ~2 ms of actual GEMM. The wins are per-call overhead (fewer/bigger
chunks, one encode, one input GEMM), not FLOP reduction.

## Task 6 — Deduplicate redundant decode calls (was T2)

**Why:** ~219 `forward_flat_indices` calls/update (438/2 updates). Many decode
the SAME sampled coordinates for different latents; some are repeated.
**Where:** train_epoch sparse path (L5667, L5671, L5689), `_backward_full_grounded_threshold_margin`
(L5365, L5398), `_backward_full_grounded_coordinate_loss` (L5265), plus
L5182/L5328/L5702/L5711/L5868.
**Change:** (1) find decodes of identical `flat_indices` across latents and reuse
the decoded logits where the loss permits (same coords, different latent ⇒
different logits — dedupe only where the INPUT is identical); merge repeated
full-grid decodes of the same latent. Prefer eliminating calls over caching.
(2) Fuse the THREE sampled decodes at L5667/L5671/L5689 (converter sweep Opt 3):
they decode the SAME `flat_indices` for three different latents → one stacked
`[3*B,192]` call, one coord gather/encode, one chunk loop → 3 calls become 1,
removing 2/3 of the per-call overhead. Gate on output equality vs the three
separate calls.
**Parity:** (1) EXACT-SAFE only where inputs are truly identical; flag any caching
of approximate reuse. (2) LOW parity — stacking changes fp32 GEMM batch ordering;
gate on loss tolerances and flag it.
**Verify:** full-update profile; assert identical losses. Est. −10–15 s/u.
Complexity: M.

## Task 7 — Metric syncs + JSONL fsync + optimizer offload (was T9 items 2-4)

**Why:** ~27-40 hot-path `.item()` syncs after each backward; `_append_jsonl`
fsync + full-file recount O(n²); optimizer-state offload ~430 MiB/u.
**Where:** `CLI/aircraft_diffusion_cfd.py` L5561, L6162-6167, L6194-6196, pbar
L6245-6258, callback dict L6261-6363, `_analytic_occupancy_logit_gradient`
L4895; `CLI/run_monitored_training.py` `_append_jsonl` L43-63; optimizer offload
`move_optimizer_state` L6150/L6156 (flag `offload_optimizer_state_between_steps`
L402-404).

**Change:** (1) read each loss `.item()` ONCE into a CPU float, reuse in
totals/postfix/callback; refresh pbar every N updates; (2) `_append_jsonl` →
plain `open(...,"ab")` + `flush`, drop the re-read/recount/sha256 (use an
in-memory counter); (3) set `offload_optimizer_state_between_steps=false` if
VRAM allows (assert no OOM at 96³); (4) delete `torch.cuda.empty_cache()` at
L6375-6376.

**Parity:** EXACT-SAFE (logging/offload only).
**Verify:** full-update profile. Est. −3–8 s/u. Complexity: S.

## Task 8 — LBM collision/stream kernel fusion (LBM sweep #1+#2+#3)

**Why:** `collide_and_stream` measured 118.8 ms/step at 96³; the dominant costs
are NOT the collision math — force accumulation (54.6 ms), per-step drag-exponent
host sync (13.3 ms), moment equilibrium (6.7 ms), boundary copies (6.7 ms).
**Where:** `CLI/advanced_lbm_solver.py` `_accumulate_momentum_exchange_force_nosync`
(L230-252), `_effective_drag_link_metric_exponent` (L254-259),
`_accumulate_projected_pressure_drag_proxy` (L261-269), `compute_moment_equilibrium`
(L182-192), `collide_stream` (L686-748).

**Change (all pure PyTorch, single-file, parity-verified by the sweep):**
(#1, primary) vectorize the 26-dir force loop into 3 broadcasts —
`f_in_all = self.f_pre_stream[1:27]*boundary_links`,
`f_out_all = self.f_temp[self._opposite_links]*boundary_links`,
`sum_all = f_in_all + f_out_all`, return `(self._force_ex_links*sum_all).sum(),
(self._force_ez_links*sum_all).sum()`; precompute `self._opposite_links` and
`self._force_ex_links = self.ex[1:27].float().view(26,1,1,1)` in `__init__` (L80).
54.6→6.2 ms/step (~6 s/u). LOW parity (1e-13 ≪ FORCE_ATOL 2.5e-5; passes fused gate).
(#2) cache `_effective_drag_link_metric_exponent` once per solve in `collide_stream`
after geom_hash: `self._solver.drag_link_metric_exponent =
self._solver._effective_drag_link_metric_exponent(geometry_mask)`; 13.3→4.2 ms/step
(~1 s/u). EXACT-SAFE (same float, cached).
(#3) fuse `compute_moment_equilibrium` 27-loop into a tensor-product broadcast
(same operand order a*9+b*3+c → bitwise-identical):
`mx/my/mz` stacked from `[ones, u, u*u+cs2]`, then
`(rho * mx[:,None,None] * my[None,:,None] * mz[None,None,:]).reshape(27,*rho.shape)`;
6.7→1.7 ms/step (~0.55 s/u). EXACT-SAFE (`torch.equal` verified on live state).
Secondary (defer): #4 `_apply_domain_boundaries` Triton/index fusion (EXACT, M,
~0.75 s/u), #5 upwind-only projected drag (LOW, S, ~0.1 s/u), #6 `solid_u8`
per-geom_hash cache (EXACT, S, minor). Mega-kernel (#9, MEDIUM parity) explicitly
NOT first. Driver-level #7 (SDF prefetch) / #8 (validity overlap) are already
covered by Tasks 2/9 — do NOT duplicate.

**Parity:** #2/#3 EXACT-SAFE; #1 LOW (reduction order, 1e-13) — fused-vs-reference
test at 32³ must pass (measured force 7.6e-6 < 2.5e-5, field 2.1e-7 < 4e-6).
**Verify:** solver-only profile + fused-parity test. Est. −7.5 s/u. Complexity: M.
Spec from LBM sweep (agent af19073f157b656f5).

## Task 9 — Thread the 33 per-solve SDF computations (SDF sweep P4; replaces old q-cache premise)

**Why (sweep correction):** the `_boundary_links` single-entry cache genuinely
misses on every probe, but it is CHEAP (~1–2 ms of GPU mask ops/solve, not in
the top-4 profile) and the 33 probe geometries are all distinct — there is NO
cross-solve q reuse to capture. The original "q/boundary-links cache reuse"
premise is disproven. The remaining lever is P4: thread the 33 SDF computations
across solves.
**Where:** `DirectSolverSPSAFunction.forward` (`CLI/aircraft_diffusion_cfd.py`
L4132-4236); `_get_q` (`CLI/advanced_lbm_solver.py` L161-175);
`_warm_sdf_cache` on the solver; `_clear_direct_solver_geometry_caches`
(L4026-4044).
**Change:** in forward, draw all `direction_count` deltas first (same seeded
`generator`, same order → identical deltas), threshold all 33 probe geometries
exactly as `_direct_measured_objective_for_single` does (L3881), submit the 33
SDF computations to `ThreadPoolExecutor(max_workers=8)` (thread-local EDT
workspaces). Store results in `_warm_sdf_cache[hash]` on the solver; `_get_q`
pops it before `compute_all_link_distances`. Validity rides the same pool.
Memory: keep warm entries on CPU, transfer to GPU lazily inside `_get_q` /
pop after each solve — do NOT hold all 33 q tensors on GPU (3.1 GB).
**Parity:** EXACT-SAFE if deltas are drawn in the original order and each EDT
is a single deterministic scipy call on the same binary geometry.
**Verify:** solver-only + full-update profile; assert identical objective
values. Est. −3–4 s/u (after Task 2's P1 lands). Complexity: L — do ONLY after
Task 2 lands.

## Task 10 — Batch the SPSA perturbation solves (was T6, OPT-2, doc 6A)

Full design captured in the plan's T6 note from sweep agent a162ccbd804e8196b
(GO; chunked C=4-8; flat-matmul collision; new Triton `stream_bfl_d3q27_batch`;
mask-multiply momentum exchange; batched-vs-sequential parity test at 32³ and
96³ reusing COMPONENT_RTOL=1e-3, COMPONENT_ATOL=5e-5, GRAD_ATOL=5e-4,
LOSS_ATOL=5e-5; assert identical 16 deltas; per-item f ≤ 4e-6 at 5 steps).
**Est. −11–12 s/u. Highest complexity. Gated by its new parity test.**
Details: `docs/superpowers/plans/optimization-plan.md` Task 10 spec + plan
notes. Generate the full brief from the plan text before dispatch.

## Task 11 — Verify `mission_profile_stress_probe.py` on GPU (was T10)

Rewritten probe compiles (syntax OK); run a bounded GPU verification (no
training), fix runtime issues, commit. Covers legacy task #12.

## Task 12 — Re-profile + final optimization report (was T11)

Full-update profile after all merges; update `docs/performance/optimization-log.md`
+ `docs/performance/96cube-baseline-profile.md` with measured before/after +
labels; scaling note for 192³; final report to user. Covers legacy #11/#15.

## Ordering / dependencies

Tasks 1-7 are independent and EXACT-SAFE → dispatch in order 1,2,3,4,5,6,7.
Task 8's spec is resolved (LBM sweep #1+#2+#3). Task 9's spec is resolved
(SDF sweep P4) — dispatch after Task 2 (P4 builds on P1's crop). Task 10
(batched SPSA) is the riskiest — dispatch after 1-9, gated by its parity test.
Task 11 independent. Task 12 last. Re-profile after each merge; update the optimization log with
measured before/after + label (NOW WIN / SCALE WIN / BOTH / NOT WORTH IT).
Optimistic sum ≈ 100 s → ~10 s/u (11×); realistic ≈ 3-5×.
