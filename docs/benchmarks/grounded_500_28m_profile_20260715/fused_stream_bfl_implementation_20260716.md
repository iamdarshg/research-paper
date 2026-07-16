# Fused Stream/BFL Kernel: Implementation And Parity Evidence

Generated: `2026-07-16`

Status: implemented and parity-gated; **disabled by default**. Training may
select it per run with `--lbm-stream-bfl-backend fused_stream_bfl` on
`CLI/run_monitored_training.py`. The scientific-regression gate (plan gate 8,
full low-Mach/OpenFOAM fixture rerun with the fused backend) has not yet been
executed, so default-on promotion remains blocked.

## What Was Implemented

Following [mrt_fusion_parallelization_plan.md](mrt_fusion_parallelization_plan.md):

### Phase 1: host-side waste removal (`CLI/advanced_lbm_solver.py`)

1. `ext_force=None` fast path: the direct-training path no longer allocates a
   `3 x N^3` zero field per step nor reduces it with `torch.any`.
2. The 27-element MRT relaxation vector `S` is cached per `(omega, s_e, s_h)`
   instead of being constructed as a new CUDA tensor every step.
3. `v_prev` (a `3 x N^3` stack) is materialized only on steps that actually
   run the convergence check; five-step training solves with
   `check_every = 10` never build it.

These are arithmetic identities; they change no loss value.

### Phase 3: fused pull-stream + BFL kernel (`CLI/d3q27_kernels.py`)

`_stream_bfl_kernel` / `stream_bfl_d3q27` fuse, per LBM step, what was
previously 27 full-volume `torch.roll` calls plus a 26-direction Python BFL
loop with up to three `torch.any` host synchronizations per direction:

- periodic pull streaming with exact `torch.roll` semantics;
- boundary-link detection (fluid cell, in-domain solid neighbor) on device;
- the exact q-low and q-high BFL interpolation formulas, including the
  reference's zero-padded out-of-domain neighbor semantics;
- one write per `(direction, cell)` with no cross-lane dependencies, so no
  host synchronization at all.

Momentum-exchange force accumulation gained a `_nosync` variant used with the
fused backend: the reference loop verbatim minus the per-direction
`torch.any` early-outs (an empty-selection sum contributes exactly 0.0, so
results are bitwise identical to the reference loop on identical state —
verified directly).

Domain boundaries remain the separate reference implementation, per the plan's
first-correct-version guidance. Inlet/outlet/slip, momentum-exchange reads,
and projected-drag accounting are unchanged.

### Backend selection and provenance

- `LBMPhysicsConfig.use_fused_stream_bfl` (default `False`) and a
  `CFDConfig.use_fused_stream_bfl` override (default `None` = defer).
- `run_monitored_training.py --lbm-stream-bfl-backend` records the choice in
  the run-history `config` block.
- Unsupported environments (no CUDA/Triton) fall back explicitly to
  `pytorch_reference`; the simplified periodic bounce kernel
  (`stream_bounce_d3q27`) is never substituted.

## Lattice Opposite-Table Correction

The audit found that the D3Q27 `opposite` table in `CLI/lbm_utils.py` paired
the twelve edge directions with vectors that were not their geometric
negations. That is a physics defect rather than an acceptable calibration
choice, so it was corrected before resumed training. A lattice invariant test
now requires both `e[opposite[i]] == -e[i]` and
`opposite[opposite[i]] == i` for all 27 directions.

The fused and reference backends were rerun after the correction. All 15 CUDA
kernel/direct-objective parity tests pass, as do the six core solver tests.
The fused kernel therefore matches the corrected reference rather than merely
reproducing the old table defect.

## Parity Evidence (gates 1-7)

`tests/test_d3q27_kernel_parity.py` (13 tests) and
`tests/test_direct_solver_fused_parity.py` (2 tests), all passing on the
RTX 4060 Laptop GPU, torch 2.9.1+cu130, triton 3.5.1, 2026-07-16:

- one-step field parity on sphere/cube/face-touching/disconnected fixtures:
  max population diff 1.9e-9 (gate 1e-8);
- five-step field, macroscopic, and force parity on the same fixtures plus a
  real 96^3 corpus aircraft (gate 4e-6 fields; forces gated at
  rel 5e-4 / abs 2.5e-5 — see the measured-envelope note below);
- empty and full masks; q distributions verified to exercise both BFL
  branches (q < 0.45 and q > 0.55 links present);
- fused-path bitwise repeatability across runs;
- full direct-objective component parity (aero, drag, lift, occupancy,
  connectivity, aircraft-validity, total) reference vs fused;
- SPSA loss and gradient parity with identical seeds and deltas
  (gradient max-diff and norm gates).

Post-correction verification commands:

```text
python -m pytest tests/test_solver.py -q
6 passed

python -m pytest tests/test_d3q27_kernel_parity.py tests/test_direct_solver_fused_parity.py -q
15 passed
```

Residual differences are FMA contraction inside the Triton kernel: computed on
identical state, the force summation is bitwise equal to the reference; the
only divergence is <= 1.8e-7 field drift projected through a momentum sum
whose gross term magnitude (~380) cancels to a net of ~1e-2. The envelope is
therefore absolute, scaled to gross magnitude, not relative to the small net.

## Measured Performance

`CLI/profile_d3q27_kernels.py` (CUDA events, median of 5, warm caches;
`build/d3q27_kernel_profile_20260716/`):

| Geometry | Grid | Ref step (ms) | Fused step (ms) | Step speedup | Ref stream+BFL (ms) | Fused stream+BFL (ms) | Segment speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sphere | 32 | 186.4 | 35.9 | 5.19x | 116.5 | 0.14 | 812x |
| cube | 32 | 119.2 | 33.4 | 3.57x | 84.0 | 0.11 | 746x |
| sphere | 96 | 185.1 | 76.3 | 2.43x | 124.2 | 1.38 | 90x |
| cube | 96 | 177.9 | 69.9 | 2.54x | 107.8 | 1.07 | 101x |
| real 96^3 aircraft | 96 | 292.4 | 115.1 | 2.54x | 193.4 | 2.50 | 77x |

The remaining fused-step cost is dominated by the dense MRT collision
transforms (plan Phase 2) and the still-CPU SciPy EDT (plan Phase 5), which
this change intentionally does not touch.

A short post-correction `96^3` rerun retained the performance result. On the
real AircraftVerse fixture, the complete step improved from `243.02 ms` to
`84.10 ms` (`2.89x`), the stream/BFL segment improved from `149.23 ms` to
`2.03 ms` (`73.42x`), and maximum population difference was `1.86e-9`.
Machine-readable output is in
`build/d3q27_kernel_profile_opposite_fix_20260716/kernel_profile.json`.

### One-update A/B (plan delivery step 8)

Full 33-evaluation direct SPSA objective (16 antithetic pairs, 5 D3Q27 steps,
96^3, real corpus aircraft `AircraftVerse_1.zip.design_10007`, identical
seeds/deltas, backward included):

| Backend | Seconds per direct objective | Loss | Gradient norm |
| --- | ---: | ---: | ---: |
| pytorch_reference | 82.1 | 0.654597 | 1.624641 |
| fused_stream_bfl | 27.5 | 0.654598 | 1.625097 |

Speedup 2.99x; loss diff 4.2e-7; gradient max diff 6.0e-6; gradient norm
relative diff 2.8e-4 — all inside the measured kernel-parity envelope. With
the direct objective owning 51.29% of profiled epoch wall time, this projects
to roughly a 1.5x end-to-end epoch speedup before any Phase 2/5 work.

## Remaining Before Default-On

1. Plan gate 8: rerun the low-Mach and OpenFOAM comparison fixtures with the
   fused backend and confirm the claim boundary is unchanged.
2. Optional phases 2/4/5/6 (fused MRT collision, batched antithetic pairs,
   EDT pipeline, CUDA graphs) remain unimplemented.
