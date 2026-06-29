# Expanded GPU Training Design

## Objective

Train a fresh `96^3` aircraft generator with 7-8 million unique trainable
parameters, at least 600 unique provenance-backed aircraft geometries, and the
measured CFD/connectivity/aircraft-validity objective on every optimizer
iteration. Remove the measured CPU synchronization bottlenecks without changing
the solver physics or weakening the geometry promotion gate.

## Evidence Behind The Design

The active run uses 752 manifest records but only 409 unique geometry files.
Its trainable model has 956,777 parameters, including only 71,169 parameters in
the raw-coordinate geometry decoder. Epoch-15 raw generations remained
block-like and failed aircraft validity.

A 60-second `py-spy` profile is stored at:

`build/profiles/g96_epoch20_20260629/python_flamegraph.svg`

During that window:

- mean GPU utilization was 33.12%;
- median GPU utilization was 23%;
- GPU memory averaged 7,925.93 MiB of 8,188 MiB;
- the Python process performed negligible steady-state disk I/O;
- most sampled stacks were in the three sequential SPSA solver probes;
- the dominant chain was D3Q27 collision/streaming, BFL boundary work, and
  SciPy Euclidean distance transforms.

The bottleneck is therefore CPU computation and CPU/GPU synchronization around
new solver geometries, not SSD throughput or insufficient DataLoader workers.

## Model Architecture

The fresh model will target 7.0-8.0 million unique trainable parameters.
Checkpoint-only copies, EMA state, and the frozen consistency teacher do not
count toward this budget.

The intended configuration is:

- latent width: 96;
- diffusion encoder channels: `[64, 96, 128]`;
- diffusion decoder channels: `[128, 96, 64]`;
- coordinate decoder width: 512;
- coordinate decoder depth: four residual hidden layers;
- coordinate encoding: raw coordinates plus six Fourier frequency bands;
- model precision: BF16 where supported;
- CFD and measured objective accumulation: FP32.

The coordinate decoder must support activation checkpointing and fixed-size
chunked evaluation. Its configuration belongs in `ModelConfig` and must be
serialized in checkpoints. Loading an older checkpoint must retain the old
decoder defaults rather than silently changing its architecture.

The parameter-count test must instantiate the final configuration and fail
unless unique trainable parameters are between 7.0 and 8.0 million.

## Grounded Corpus Expansion

The training manifest must contain at least 600 unique geometry identities.
Record count alone is not sufficient.

Ingestion order:

1. Remaining distinct HiLiftAeroML geometry variants, using one canonical
   surface per configuration rather than treating angle-of-attack runs as new
   geometry.
2. Official NASA UAM OpenVSP reference vehicles.
3. NASA CRM candidate groups that resolve to whole-aircraft geometry.
4. Additional license-qualified Airshow records only when their geometry
   identity is not already present.

Each accepted geometry must have:

- a stable geometry identity and content hash;
- source URL or local source-catalog identity;
- license/provenance metadata;
- a local voxel artifact at `96^3`;
- aircraft-validity results;
- family/configuration identity for split control.

Component-only CAD, failed conversions, zero-volume geometry, duplicate content,
and records without an acceptable provenance lane are excluded. Train and
holdout splits must be grouped by geometry family so repeated flight conditions
or related variants cannot cross the split boundary.

The corpus builder must report:

- manifest record count;
- unique geometry count;
- content-duplicate count;
- validity pass/fail counts;
- source-collection counts;
- split counts by unique geometry and family.

The 600-geometry requirement is fail-closed.

## Windows-Aware Host Cache

The host corpus cache will keep one canonical tensor per unique geometry and
reference it from all condition records. Geometry occupancy is stored as
`uint8` or boolean data in system memory and converted to model dtype only on
the target device.

The loader will use:

- `num_workers=0` on Windows to avoid spawn-time duplication of the corpus;
- a two-slot pinned-memory staging ring;
- `non_blocking=True` device transfers;
- no per-iteration file reads, directory scans, hashing subprocesses, or
  checkpoint serialization;
- bounded prefetching of only the next small batch.

Cold EMA/checkpoint state may be kept in CPU memory or copied in bounded chunks
when benchmarks show a net VRAM benefit. Active solver populations, boundary
data, and the current SPSA probe remain GPU-resident. Offloading data needed in
the next solver step is prohibited because it would add synchronization and
reduce throughput.

Persistent DataLoader workers are not used unless a benchmark proves that a
worker does not duplicate the canonical cache on Windows.

## Exact GPU Boundary Preparation

The current path copies each new occupancy mask to CPU for a content hash,
copies it again for two SciPy distance transforms, and then returns boundary
data to the GPU. The replacement path will create one ephemeral
`GeometryBoundaryContext` for each immutable base/plus/minus SPSA geometry.
That context is reused across all LBM steps in that solver call and released
afterward.

The preferred SDF backend is CuPy's exact GPU
`cupyx.scipy.ndimage.distance_transform_edt`, connected to PyTorch with DLPack
to avoid host copies. SciPy remains an explicit reference/fallback backend.

Backend selection is fail-closed:

- `gpu_exact` requires CUDA, CuPy, and a passing startup probe;
- `scipy_reference` remains available for tests and unsupported machines;
- an unavailable or failed GPU backend must not silently select an approximate
  distance transform.

Numerical parity tests compare SDF values, BFL `q` values, boundary links,
forces, drag/lift terms, and final direct objective values on fixed geometries.

## BFL And Solver Kernels

The existing Triton kernel implements simple link bounce-back and is not
equivalent to the active BFL boundary scheme. It must not be enabled for this
run.

A separate BFL-equivalent Triton kernel may replace the Python 26-direction
loop only after it matches the reference implementation for:

- low- and high-`q` interpolation branches;
- non-periodic neighbor indexing;
- opposite-direction writes;
- active boundary masks;
- momentum-exchange force accounting.

The solver will expose a training-coefficient mode that computes every quantity
used by the optimization objective while deferring report-only vorticity,
Q-criterion, and visualization reductions. This does not remove CFD,
connectivity, validity, force, drag, lift, occupancy, or convergence terms from
the optimizer.

Base, positive, and negative SPSA probes remain sequential because they share
mutable solver state and the GPU lacks memory for three independent `96^3`
D3Q27 solvers.

## Transfer And Compute Scheduling

The next pinned batch may transfer on a dedicated CUDA stream while the current
model/solver work executes. The consumer stream must wait on an event before
using the batch.

Within one optimizer iteration:

1. Consume the staged latent, condition, and compact geometry.
2. Run diffusion and sampled reconstruction work.
3. Materialize the full coordinate-decoder grid.
4. Build and solve base, positive, and negative SPSA geometries sequentially.
5. Consolidate CFD, connectivity, validity, and occupancy into the measured
   direct loss.
6. Backpropagate the SPSA gradient through the geometry decoder and model.
7. Stage the next compact host batch without retaining stale solver geometry.

Generic diffusion kernels will not be imported merely because they are
available. A custom or external kernel must correspond to a measured hot
operation, support the installed CUDA/PyTorch versions, preserve numerical
semantics, and pass the reference tests.

## Safeguards And Failure Handling

The existing requirements remain:

- direct solver interval equals one;
- direct solver, connectivity, and aircraft-validity weights are nonzero;
- direct solver coverage equals optimizer iteration count;
- no final checkpoint promotion without reconstruction top-k recall and
  generated aircraft-validity gates;
- no wall-clock training timeout.

Additional failures:

- abort before training if unique geometry count is below 600;
- abort before training if parameter count is outside 7.0-8.0 million;
- reject a GPU SDF or BFL backend that fails parity;
- reject a configuration whose one-iteration memory probe exceeds the safe
  VRAM ceiling;
- preserve the current run and checkpoint until the replacement probe passes.

## Verification And Benchmarks

Verification proceeds in this order:

1. Unit tests for corpus deduplication, compact storage, split grouping, model
   configuration, parameter count, Fourier encoding, and checkpoint round-trip.
2. SDF/BFL parity tests against SciPy/PyTorch reference implementations.
3. Direct-objective forward and SPSA-gradient parity tests.
4. Full repository test suite.
5. A fixed-seed one-iteration `96^3` memory and correctness probe.
6. A fixed-seed 20-iteration reference-versus-optimized benchmark.
7. A fresh full-corpus run only after all prior gates pass.

The benchmark records wall time, CPU utilization, GPU utilization, VRAM, system
RAM, disk I/O, transfer time, SDF time, BFL time, solver time, decoder time, and
optimizer time.

Acceptance targets:

- peak VRAM no greater than 8,050 MiB;
- no objective or force drift outside documented numerical tolerance;
- no per-iteration SSD reads in steady-state training;
- materially higher sustained GPU utilization than the 33.12% baseline;
- lower median iteration time than the current 5-7 second range;
- at least 600 unique accepted geometries;
- 7.0-8.0 million unique trainable parameters.

These performance targets do not relax scientific or geometry gates.

## Claim Boundary

More parameters, higher utilization, and a larger grounded corpus do not prove
aircraft validity, aerodynamic superiority, or publication-grade solver
validation. Only generated artifacts that pass the existing geometry,
condition-response, solver-validation, baseline, and final-evidence gates may
support those claims.
