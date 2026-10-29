# Model-collapse root cause and fail-closed safeguards

## Finding

The downloaded L4 checkpoint was not merely rendered with the wrong threshold.
The F-16-like and Cessna-like projections both produced an almost constant
probability field:

- F-16-like: min `0.4989901`, max `0.4990200`, mean `0.4990049`, standard
  deviation `3.5e-6`, zero occupied voxels at threshold `0.5`.
- Cessna-like: min `0.4989879`, max `0.4990174`, mean `0.4990023`, also zero
  occupied voxels at threshold `0.5`.

Moving the threshold only changed the number of noisy specks. It did not recover
an aircraft-shaped field, so this was a model/optimization failure rather than
an image-viewer failure or a single bad threshold.

The decisive checkpoint evidence was the optimizer state. The diffusion and
consistency groups had Adam step `6008`, while the coordinate-converter group
had step `94`. Since `6008 // 64 + 1 = 94`, the converter was updated only on
the sparse full-lattice cadence. Its moment norms were initialization-scale.

The deployed archive made that cadence possible by combining two assumptions
that are incompatible:

1. `freeze_decoder_for_generated_paths` removed converter gradients from the
   sampled data, generated, consistency, direct, and topology-guard branches.
2. The archive then ran the exact grounded full-lattice decoder backward only
   every 64 updates.

Therefore 63 of every 64 updates had no coordinate-converter gradient. The
denoiser and consistency optimizers advanced, but the geometry decoder stayed
near its initial bias and emitted a field around 0.5 everywhere. The current
canonical trainer already uses the exact grounded decoder backward on every
update; the new cadence validator prevents a future sparse configuration from
reintroducing the contradiction.

There is also deployment drift: the archived worker was labelled with source
commit `6e57aac` while extracting `repo-d657364.tar.gz`, and its recorded
command contains options that are not present in the current monitored-run
surface. The worker now refuses to launch an archive unless it contains the
collapse watchdog implementation. The archive must be rebuilt and its content
hash/commit metadata updated before another cloud run is authorized.

## Runtime safeguards

`evaluate_model_collapse` is called after every optimizer update. After a
bounded 128-update warm-up, it observes the actual generated/free-running
probabilities and checks:

- nonfinite forward, solver, loss, or converter-gradient values;
- constant probability fields with tiny standard deviation and span;
- blank or solid materialization;
- a missing/tiny coordinate-converter gradient;
- a coordinate-converter Adam step that falls behind `global_step`.

Numerical failures and optimizer-step stalls are hard failures. Geometric
symptoms require two consecutive observations to avoid stopping on one noisy
update. A triggered decision is written to the append-only update record,
stored in run-state metadata, and persisted through a forced atomic
`latest_run_state.pt` save before the trainer exits the epoch. An exact resume
from a collapse-paused run-state is refused so it cannot silently repeat the
same failure; a reviewed warm-start can be used after the cause is corrected.

Promotion evaluation has a second guard for empty, blank/solid, or materially
non-unique multi-seed outputs. This catches failures that might evade a single
per-update sampled field.

## Scope boundary

These safeguards make collapse fail closed and preserve the evidence needed for
recovery. They do not prove that a future checkpoint is scientifically valid.
The next run still needs the independent F-16/Cessna projection checks, a
multi-seed promotion pass, finite-gradient telemetry, and a source/archive
identity check before any long horizon is accepted.

## 2026-09-01 guarded L4 canary provenance

The repair canary always used the SHA-256-verified monitored run state from
global step `6008`; warm-starts intentionally reset the new optimizer lineage
to step zero. No failed boot committed an optimizer update.

- v6 verified the checkpoint but exposed the shared latent-matrix schema.
- v7 fixed latent indexing but the eager 8,027 x 128-cubed geometry materialization
  approached the 12 GiB cgroup limit and was stopped before OOM.
- v8 used lazy NPY geometry and failed closed on the nested monitored-run-state
  model schema.
- v9 unwrapped that schema and then rejected a student-only consistency state.
- v10 accepted only omitted `teacher_model.*` consistency keys, loaded all 126
  student keys, and resynchronized the teacher from the student. Its first
  forward/backward attempt was rejected before optimizer commit because BF16
  projection quantization left a `-5.76486e-06` residual cosine.
- v11 narrows the numerical-residue allowance to `-1e-5`; larger post-projection
  conflicts still fail closed. The focused multiobjective suite passed 17 tests.

The VM service has `MemoryMax=12 GiB`, no swap, `OOMPolicy=stop`, an automatic
shutdown fuse, and durable exit/status/telemetry upload. The v11 fuse is 25
minutes so the cumulative L4 work remains below the user-authorized compute
budget.

## v11 throughput regression and correction

The v11 runtime log disproved the hypothesis that the first update alone was
slow. The first three committed updates took `69.81`, `71.08`, and `70.87`
seconds respectively. The run then spent about another `94.3` seconds entering
the next update before failing in `project_conflicting_gradient`; it did not
hit an OOM (`memory.peak=6.68 GiB`, `oom=0`, `oom_kill=0`).

The dominant cost is the direct objective: each update performs one base plus
32 antithetic SPSA probe solves at `128^3`, with five D3Q27 steps per configured
solver call and the associated CPU validity/SDF work. The L4 worker inherited
the safe local setting `_DIRECT_SOLVER_BATCH_CHUNK=1`, so all 33 solves were
serialized even though the L4 may have sufficient VRAM headroom. The local and
L4 worker defaults remain sequential for the first recovery canary. Chunk size
`2` is still available through the explicit `RESEARCH_DIRECT_SOLVER_BATCH_CHUNK`
override, but is not enabled until the sequential-versus-batched objective
parity check and a live VRAM measurement pass independently.

The separate v11 failure was numerical rather than performance-related. BF16
gradient buffers were projected in BF16 and then measured as having a residual
cosine of `-5.83282e-05` against the data anchor. The correction performs the
projection and fail-closed residual check in FP64 and converts the accepted
gradient back to its original dtype only afterward. A focused regression suite
now covers both the strict FP32 path and the bounded BF16 compatibility path.

## Precision policy and recovery gate

The corrected trainer keeps trainable parameters and optimizer state in FP32.
CUDA BF16 is used only inside autocast model computation; the noise schedule and
solver-facing logits remain FP32. Projection dot products and norms widen one
parameter tensor at a time for FP64 reduction, while the projected gradient is
applied in tensorwise FP32 scratch, so the fix does not create a full-model FP64
copy. Startup and update metadata record `parameter_dtype`, `compute_dtype`,
`optimizer_master_dtype`, and `projection_reduction_dtype` so deployment drift
is visible in the durable run state.

The next cloud attempt is a staged canary, not a continuation of the failed
long run: first verify that an optimizer step is nonzero and that the collapse,
gradient, and post-writeback guards remain healthy; only then measure the L4
probe-chunk setting. Recovery gates are 4, 8, 16, and 32 committed updates.
Chunk-2 is not accepted by memory estimates alone: it must also pass the
sequential-versus-batched objective parity check and a live VRAM measurement.
The source step-6008 checkpoint remains immutable and the VM stays stopped
until those local gates are complete.

## Checkpoint streaming and local retention

Long runs publish an immutable checkpoint plus a SHA-256/byte-count manifest at
every 100-update boundary. `CLI/watch_cloud_checkpoints.py` polls the latest
manifest, downloads to a quarantined `.partial` path, verifies both declared
properties, and only then atomically accepts it. Before downloading a newer
checkpoint, it removes the prior payload previously accepted by this watcher,
so two full checkpoint payloads never coexist on the local SSD. Durable JSONL
evidence and generation IDs remain, but mutable state retains only the newest
checkpoint path.

Each accepted checkpoint runs both the F-16-like and Cessna-like mission probes.
The probe records probability min/max/mean/standard deviation, occupancy across
thresholds 0.30 through 0.70, aircraft-validity diagnostics, compressed arrays,
and isometric/top/side/front projections. A read-only Luna agent independently
observes cloud health, update telemetry, local acceptance, and inference results.
