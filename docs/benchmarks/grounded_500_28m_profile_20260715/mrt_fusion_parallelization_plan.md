# D3Q27 MRT Fusion And Parallelization Plan

Generated: `2026-07-15`

Status: proposed; do not modify the active six-hour training process.

## Objective

Reduce the wall time of the direct D3Q27 objective while retaining every one of
the current loss evaluations:

- one base solver evaluation;
- sixteen SPSA plus/minus pairs;
- five D3Q27 steps per evaluation;
- aerodynamic, connectivity, and aircraft-validity contributions;
- the same `96^3` thresholded geometry and BFL link distances.

No surrogate, skipped solver call, reduced grid, reduced SPSA direction count,
or diagnostic-only replacement is in scope. Parallel execution is acceptable
only when all 33 objective values are still computed and consolidated into the
same loss and gradient estimate.

## Measured Starting Point

The live profile attributes 51.29% of sampled training wall time to the direct
solver. Within the full training loop:

| Region | Inclusive sample share |
| --- | ---: |
| D3Q27 `collide_and_stream` | 47.64% |
| BFL boundary application | 36.89% |
| SDF and BFL link preparation | 11.02% |
| SciPy EDT | 10.76% |
| Momentum-exchange force accumulation | 5.42% |

The hottest individual solver leaves are the masked BFL assignments at
`advanced_lbm_solver.py:340` and `advanced_lbm_solver.py:336`, with 9.33% and
5.16% of all raw samples. The BFL loop also executes host-visible
`torch.any(...)` decisions at lines 312, 335, and 339.

At 26 non-rest directions, five steps, and 33 evaluations, the current code
performs exactly 4,290 `active` host checks per optimizer update and as many as
8,580 additional q-branch checks. These checks fragment CUDA work with up to
12,870 CPU/GPU synchronization points.

The MRT collision is the main arithmetic block. Each step applies two dense
`27 x 27` transforms over 884,736 lattice cells. This is approximately 1.29
billion multiply-accumulates per step and 212.8 billion across the 165 D3Q27
steps in one optimizer update.

The observed GPU is an RTX 4060 Laptop GPU, compute capability 8.9, with 8,188
MiB VRAM. Peak usage was 3,421 MiB, leaving enough measured headroom to start
with a two-geometry solver batch. Nsight Compute 2025.4 is installed locally.

## Existing Kernel Boundary

`CLI/d3q27_kernels.py` already contains a Triton streaming/bounce kernel. It is
not a valid replacement for the active solver because it:

- uses periodic modulo addressing;
- implements simple bounce-back instead of q-dependent BFL interpolation;
- does not apply the current inlet, outlet, and slip boundaries;
- does not preserve momentum-exchange and projected-drag accounting;
- does not fuse or replace the MRT collision transforms.

The current `use_triton_streaming = false` default must remain until a new path
passes the parity gates below. The existing kernel can supply scaffolding for
dispatch and build detection, but not the physics implementation.

## Target Execution Shape

The first fused design should use two GPU kernels per LBM step rather than one
monolithic kernel:

1. **Fused MRT collision kernel:** compute macroscopic quantities, moment
   transform, equilibrium moments, relaxation, conservation overwrite, inverse
   transform, and `f_post` without materializing global `K`, `Keq`, or `K_post`.
2. **Fused pull-stream and BFL kernel:** pull the appropriate neighboring
   `f_post`, apply q-dependent BFL interpolation and nonperiodic boundary rules,
   write `f_next`, and emit block-local force partials.

A small deterministic reduction kernel may consolidate force partials. Domain
boundaries should remain a separate kernel in the first correct version if
fusing them obscures parity. This still reduces hundreds of Python-launched
operations to a bounded handful of launches per step.

Trying to fuse collision and streaming immediately is high risk. Streaming
needs post-collision values from neighboring cells; doing both in one global
kernel either requires a large halo/shared-memory tile or recomputes neighboring
collisions. The two-kernel design preserves a clear synchronization boundary
and avoids that duplication.

## Phase 0: Reproducible Kernel Benchmark

Create `CLI/profile_d3q27_kernels.py` before changing solver behavior.

The benchmark should:

- use fixed cube, sphere, NASA CRM, public CAD, and generated SPSA-perturbation
  geometries at `32^3`, `64^3`, and `96^3`;
- initialize reference and candidate solvers from identical populations;
- time SDF, q construction, collision, streaming, BFL, force reduction,
  macroscopic recomputation, and coefficient extraction separately;
- use CUDA events for device time and wall timers for end-to-end time;
- add NVTX ranges and collect Nsight Compute launch, occupancy, register-spill,
  DRAM, L2, and achieved-throughput counters;
- save machine-readable JSON and a short Markdown summary;
- record PyTorch CUDA/TF32 settings because reassociation and TF32 can alter the
  numerical baseline.

The benchmark must run after the active trainer reaches a checkpoint or stops.
Profiling another solver concurrently would contaminate timings and slow the
training run.

## Phase 1: Remove Proven Host-Side Waste

Make low-risk changes against the existing PyTorch solver first. Each change
gets an independent benchmark and parity check.

1. Replace `ext_force = torch.zeros(...)` plus `torch.any(ext_force != 0)` with
   an explicit host boolean. The direct-training path passes no external force,
   so it currently allocates a `3 x 96^3` zero field and reduces it every step
   only to discover that it is zero.
2. Cache the 27-element relaxation vector `S` for a fixed `omega` instead of
   constructing a new CUDA tensor every step.
3. Allocate `v_prev` only on an actual convergence-check step. Five-step
   training solves use `check_every = 10`, so the current `3 x 96^3` stack is
   produced every step but never read by the convergence branch.
4. Pass the content hash derived from the already-available CPU binary mask
   into the solver. Do not copy the same mask GPU-to-CPU again for hashing.
5. Compute post-step macroscopic fields only on a convergence-check step and the
   final step, unless a parity test proves an intermediate consumer needs them.
6. Rate-limit the 33 repeated solver announcements per optimizer update.

This phase does not alter the MRT or BFL equations. It establishes a cleaner
reference and removes synchronization points that would otherwise obscure the
fused-kernel profile.

## Phase 2: Fused MRT Collision Kernel

Add a new collision entry point in `CLI/d3q27_kernels.py`; retain the PyTorch
implementation as the reference backend.

### Data layout

- Preserve the contiguous structure-of-arrays layout `[batch, 27, cells]`.
- Use a compile-time `27 x 27` moment basis and inverse basis.
- Process a `BLOCK_CELLS` tile for one or more independent geometries.
- Read each population once, retain working values in registers, and write only
  final `f_post` to global memory.
- Do not materialize full-grid `K`, `Keq`, or `K_post` tensors.

### Numerical order

The first kernel should implement the same operation order as the PyTorch
reference as closely as practical: macroscopic reduction, equilibrium moment
construction, MRT relaxation, conserved-moment overwrite, then inverse
transform. Do not enable fast math or introduce an algebraically collapsed
collision matrix until the direct implementation passes all parity gates.

An algebraic `f_post = A(omega) f + B(omega) Keq` form may be benchmarked later,
but it changes floating-point association and therefore needs its own evidence.

### Register-pressure search

D3Q27 can require at least 27 population values plus moment accumulators per
cell. Benchmark `BLOCK_CELLS` and warp counts rather than assuming one launch
shape. Reject candidates that spill materially into local memory. If a single
kernel cannot retain the required state without spilling, use two fused kernels
for forward moments/relaxation and inverse moments rather than silently losing
performance.

Expose the kernel through a `torch.library` custom operation or an equivalently
explicit dispatcher. Autograd is not required inside this black-box solver;
SPSA supplies the measured derivative outside it.

## Phase 3: Fused Streaming And BFL Kernel

Replace the 27 `torch.roll` calls and the Python direction loop with a pull-style
kernel indexed by direction, geometry batch, and destination cell.

For each destination population, the kernel must:

1. derive the nonperiodic source coordinate;
2. load the streamed post-collision population;
3. read the exact cached boundary-link and q value;
4. apply the current q-low or q-high BFL equation entirely on device;
5. apply the reference inlet/outlet/slip semantics;
6. write `f_next` once;
7. produce momentum-exchange and projected-drag partials where applicable.

The current BFL implementation reads an already-streamed opposite population.
The fused kernel must derive that same source population explicitly from
`f_post`; it cannot reuse the simple reflected value from the existing Triton
kernel. Add direction-by-direction unit fixtures before full solver tests.

Force accumulation should use deterministic block reductions followed by a
fixed-order final reduction. Avoid unordered global floating-point atomics if
they make repeated solver results vary.

This kernel removes the Python loop, all BFL `torch.any` host checks, repeated
boolean-index temporaries, and full-volume `torch.roll` allocations.

## Phase 4: Batch Antithetic SPSA Pairs

After the fused kernels support a leading solver-batch dimension, evaluate each
SPSA plus/minus pair as a batch of two:

```text
base geometry:              batch 1
direction 0 (+eps, -eps):   batch 2
direction 1 (+eps, -eps):   batch 2
...
direction 15 (+eps, -eps):  batch 2
```

This changes 33 serial solver launches into one base launch plus 16 batched
pair launches while still computing all 33 objective values. It also matches
the structure of the finite-difference gradient: each plus/minus result is
consumed together.

Start with batch two. Increase to batch four only if Nsight shows unused SM
capacity and measured peak VRAM remains below the configured safety ceiling.
Do not batch all 33 states; that wastes memory and weakens failure isolation.

The batched solver must return one drag, lift, and force record per geometry.
Connectivity and aircraft-validity values remain per geometry and are combined
with the matching CFD value before the SPSA difference is calculated.

## Phase 5: CPU/GPU Pipeline

Parallelize preparation around the batched GPU solver without changing solver
result order:

```text
CPU workers:  mask/hash + EDT/SDF + validity for pair k+1
copy stream:  pinned SDF/mask transfer for pair k+1
GPU stream:   fused D3Q27 solve for pair k
host:         consolidate all loss components for pair k-1
```

Required changes:

- replace the single global EDT lock/workspace with a bounded two-workspace
  pool so plus and minus preparation can proceed concurrently;
- use two pinned host staging buffers and two GPU q/boundary buffers;
- synchronize buffers with CUDA events rather than global
  `torch.cuda.synchronize()` calls;
- run geometry-only connectivity/validity work while the same geometry's CFD is
  on the GPU, then join before forming total loss;
- keep at most one direction pair ahead to bound memory and preserve deterministic
  failure reporting.

SciPy EDT releases native work but its behavior must be confirmed under two
workers. If parallel EDT increases memory pressure or is slower, retain one EDT
worker and overlap it only with GPU execution.

## Phase 6: CUDA Graph Capture

Once tensor addresses and launch shapes are stable, capture flow initialization
and the fixed five-step kernel sequence in a CUDA graph. Copy each new geometry,
q field, and boundary-link field into static input buffers before replay.

Graph capture is the final optimization, not the first. It reduces residual
Python launch overhead but cannot repair an inefficient BFL or spilling MRT
kernel. Keep a normal launch path for debugging and unsupported environments.

## Correctness Gates

The reference PyTorch solver remains authoritative until every gate passes.

1. **Preprocessing parity:** binary mask and content hash identical; q and
   boundary-link tensors identical before the fused kernels execute.
2. **One-step field parity:** compare `f_post`, `f_next`, rho, velocity, pressure,
   and force partials after each stage.
3. **Five-step parity:** compare populations, macroscopic fields, accumulated
   force, drag, lift, and projected-drag values.
4. **Conservation:** mass and conserved moments must be no worse than the
   reference solver on empty, solid, symmetric, and real-aircraft geometries.
5. **Direct-loss parity:** compare every aero, connectivity, validity, occupancy,
   and total-loss component for the base and all 32 perturbations.
6. **Gradient parity:** with identical SPSA deltas, compare the unclipped and
   clipped gradient tensors, norms, and resulting parameter update.
7. **Repeatability:** repeated fused runs must remain within the reference
   backend's measured repeatability envelope. Establish that envelope before
   setting numeric tolerances; do not invent a convenient tolerance afterward.
8. **Scientific regression:** rerun the existing low-Mach and OpenFOAM comparison
   fixtures. The fused path may be faster but may not change the solver's claim
   boundary or improve a comparison by changing the equations.

Test fixtures should include axial and diagonal boundaries, q values below,
equal to, and above 0.5, all 26 moving directions, domain faces/corners, empty
and full masks, disconnected geometries, and real `96^3` aircraft.

## Performance Gates

Report both kernel and end-to-end measurements. A kernel-only speedup is not
enough if EDT or synchronization still dominates.

- BFL host synchronizations per update: target zero.
- Full-volume temporary moment grids: target zero outside `f`/`f_post`.
- Solver-segment GPU duty cycle: target sustained operation rather than the
  current alternating idle/saturated pattern.
- Peak VRAM: keep a measured safety margin below the 8,188 MiB device limit;
  begin with a 7,000 MiB stop threshold for experiments.
- Numerical gates: all must pass before performance is considered.
- End-to-end output: report seconds per direct objective, seconds per optimizer
  update, solver calls per second, and projected epoch time.

No fixed speedup is a correctness gate, but the fused path should not be enabled
by default unless it materially improves end-to-end update time. If MRT fusion
spills registers or BFL fusion regresses physics, keep the faster validated
subset of changes and retain the reference implementation for the rest.

## Implementation Surface

Expected files:

- `CLI/d3q27_kernels.py`: fused MRT, stream/BFL, and deterministic reduction
  kernels plus runtime dispatch.
- `CLI/advanced_lbm_solver.py`: reference/fused backend selection, batched solver
  state, cached constants, and removal of redundant host synchronization.
- `CLI/sdf_utils.py`: bounded EDT workspace pool and pinned staging support.
- `CLI/aircraft_diffusion_cfd.py`: batched plus/minus direct-objective evaluation
  and ordered component consolidation.
- `CLI/profile_d3q27_kernels.py`: CUDA-event/NVTX/Nsight benchmark harness.
- `tests/test_d3q27_kernel_parity.py`: direction, field, force, and batch parity.
- `tests/test_direct_solver_fused_parity.py`: 33-value objective and SPSA-gradient
  parity.

Add global configuration keys for backend, solver batch size, pipeline depth,
and CUDA graph use. Record those values in checkpoint/run metadata. Unsupported
CUDA/Triton environments must fall back explicitly to `pytorch_reference` and
must never silently use the old simplified bounce kernel.

## Delivery Order

1. Benchmark and golden-state fixtures.
2. Host synchronization and redundant-allocation cleanup.
3. Fused MRT collision kernel.
4. Fused pull-stream/BFL and force reduction.
5. Batch-two antithetic solver API.
6. Two-buffer CPU/GPU preprocessing pipeline.
7. CUDA graph capture.
8. Full one-update A/B test, then one complete corpus epoch.
9. Enable only after parity, memory, and scientific regression evidence passes.

This order isolates numerical failures and provides useful speedups even if the
most aggressive fusion stage proves unsuitable for the RTX 4060 register or
memory hierarchy.
