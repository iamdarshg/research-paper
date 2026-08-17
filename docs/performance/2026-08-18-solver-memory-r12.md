# R12 — Cheap solver-memory optimizations (PR 41 review, item 12)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)
Hardware: RTX 4060 Laptop 8 GB / Ryzen 7 7735HS / 16 GB RAM, Windows 11, torch 2.9.1+cu130

## The two changes

Both are numerics-neutral (bit-identical output, verified) — they shrink the
per-solve GPU working set, not the math.

### (a) `compute_link_q`: per-direction evaluation (sdf_utils.py)

The C=1 sequential q-algebra (production default backend is
`pytorch_reference`, which calls `_get_q` → `compute_link_q` once per solve,
33×/update) built the full `[26, 96, 96, 96]` q field by materializing ~5
**stacked** `[26, D, H, W]` fp32 temporaries together: `sdf_neighbors` (via
`torch.stack` of 26 shifted slices), `crossing`, `denom`, `q`, and the
`where`-result — ~5 × 88 MB at 96³ ≈ 440 MB peak just to produce an 88 MB
result.

`compute_link_q` now evaluates the identical per-element formula one direction
at a time, writing directly into the pre-filled 1.0 `q_all`; only `[D, H, W]`
working-set temporaries (~3.5 MB each) are ever live. Every element's fp32
arithmetic is unchanged, so the result is bit-identical (the batch width of the
vectorization does not affect per-element IEEE determinism). The cold
`compute_all_link_distances` path delegates to `compute_link_q`, so it inherits
the fix.

### (b) Cache `max_count` in the BFL sparse table (advanced_lbm_solver.py + d3q27_kernels.py)

`stream_bfl_d3q27_batch_compressed` did a host sync `max_count =
int(pair_count.max().item())` on every solve to size its correction grid. The
sparse table is cached keyed by `(geom_hashes, C, res)`, so `max_count` is
geometry-static. `_build_bfl_sparse_tables` now stores it once at build time
(`"max_count": int(pair_count.max().item())`), and the kernel reads the cached
int (with a `.get` fallback recompute for legacy hand-built dicts).

## Evidence

### Bit-identity

The parity gate (gitignored probe `build/perf/baseline/r12_q_peak_probe.py`)
runs the frozen pre-R12 `compute_link_q` body against the new one on a real 96³
geometry on the GPU:

```
[parity] torch.equal(old, new) = True
```

The five solver/BFL parity test files all pass unchanged (37 passed / 2
skipped), including the Task 10 sequential-vs-batched SPSA parity and Task 34
sparse-q/two-buffer parity. Full suite: **512 passed / 2 skipped** (baseline
preserved).

### Peak VRAM, isolated (fresh process per impl, 8 iterations)

| implementation | max allocated | max reserved |
|---|---|---|
| old stacked form | 591.8 MiB | 627.0 MiB |
| new per-direction form | 122.4 MiB | 142.6 MiB |

The q-algebra's peak reserved drops **~470 MiB** (~627 → ~143 MiB).

### Peak VRAM + wall time, integrated (production-faithful C=1 harness)

Same harness as R11 (real step1305 checkpoint, real 1069-geometry corpus,
`--no-instrument`, warmup=1 + 5 measured updates):

| metric | R11 (pre-R12) | R12 | Δ |
|---|---|---|---|
| s/u mean | 27.53 | **27.17** | −0.36 (noise) |
| peak reserved | 7,460 MiB | **7,334 MiB** | **−126 MiB** |
| peak allocated | 6,053 MiB | 6,053 MiB | 0 |

The global peak reserved drops ~126 MiB — the q-algebra transient was partially
setting the global envelope; it now sits below the solver's persistent
footprint. Headroom on the 8 GiB card improves from ~0.54 to ~0.67 GiB.
Wall time is flat despite 26× more kernel launches per q-algebra, confirming
the direct phase is CPU-prep-bound (EDT/q + validity), not GPU-launch-bound.

## Paper implication

- The 8-GB portability claim gets a slightly larger measured headroom bound
  (~0.67 GiB at the production configuration).
- Because both changes are bit-identical, **no reproducibility/numerics claim
  in the paper changes** — this is purely a memory-surface improvement.

## Artifacts

- `build/perf/baseline/r12_q_peak_probe.py` (gitignored) — parity + peak probe.
- `build/perf/baseline/profile_result_c1_r12.json` (gitignored) — C=1 harness JSON.
