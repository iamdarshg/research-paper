# Consistency And Multiobjective Training Stabilization Plan

Generated: `2026-07-17`

Status: implementation proposed; no new training run may begin until the
implementation and preflight gates below pass.

## Goal

Resume from the preserved pre-1k checkpoint and produce a verifiably improved
`96^3` four-step aircraft generator without weakening, skipping, or replacing
the direct D3Q27 objective. Stabilize consistency distillation, prevent any one
loss branch from monopolizing the student update, retain all measured
aerodynamic/connectivity/validity contributions, and promote only checkpoints
that improve or preserve geometry validity, diversity, and overlap.

## Evidence That Motivates This Work

The completed 758-update run over the 1,069-geometry combined corpus used:

- one base and sixteen antithetic SPSA pairs per optimizer update;
- 33 direct D3Q27 solver calls per update and 25,014 calls in total;
- the fused stream/BFL backend at `96^3`;
- nonzero aerodynamic, connectivity, and aircraft-validity terms on every
  optimizer update.

The run completed successfully at the process level, but the final checkpoint
failed checkpoint promotion:

| Metric | Preserved starting checkpoint | Final checkpoint |
| --- | ---: | ---: |
| Generated aircraft valid fraction | 0.375000 | 0.104167 |
| Generated mean top-k recall | 0.234770 | 0.223368 |
| Generated worst top-k recall | 0.159605 | 0.151361 |
| Reconstruction top-k recall | 0.234801 | 0.231885 |

The final epoch's mean consistency term was `945,849,900.86`, while the direct
solver term was `0.70`. Console evidence shows repeated consistency spikes up
to approximately `3.0e11`, always on the every-20-batch consistency path. The
final generator learned a stronger top-view aircraft planform but produced
boundary-saturated side views and only three distinct fields from six fixed
seeds.

The code-level mismatch is:

- diffusion training cycles over the actual four-step inference schedule
  `[999, 666, 333, 0]`;
- consistency distillation samples arbitrary timesteps from `[0, 999]`;
- consistency uses an unbounded raw MSE;
- the teacher is copied from the current diffusion model immediately before
  each sparse consistency update;
- the raw consistency gradient is mixed with generated-geometry and measured
  solver gradients before one final student gradient clip.

This plan treats the mismatch as the first failure to fix. It does not assume
that corpus scale or model capacity is the limiting factor.

## Non-Negotiable Invariants

1. The recovery source is
   `build/grounded_500_28m_full_20260715/checkpoints/best_geometry_model.pt`.
   It must remain byte-for-byte untouched.
2. Every optimizer update at `96^3` must execute one base D3Q27 solve and all
   sixteen plus/minus SPSA pairs: exactly 33 measured solver evaluations.
3. Aerodynamic, connectivity, and aircraft-validity values must be computed
   from the thresholded geometry used by the solver and must all contribute to
   the applied gradient.
4. No surrogate may replace a solver result or derivative estimate.
5. No objective may be relabeled as a report-only diagnostic to remove it from
   optimization.
6. Gradient balancing may limit a branch's influence, but a finite nonzero raw
   branch gradient must retain a finite nonzero applied contribution.
7. Critical nonfinite model outputs, losses, solver values, or gradients must
   fail closed. They must not be silently converted to zero.
8. Model width, latent dimension, grid size, corpus membership, and solver
   fidelity remain fixed during stabilization. Scaling changes are considered
   only after a stable full-corpus result.
9. The final checkpoint is never promoted merely because a process finishes.
   It must beat or preserve the fixed-seed source-checkpoint baseline.

## Task 0: Preserve And Fingerprint Recovery Inputs

**Files:**

- Create: `build/training_stabilization_20260717/baseline_fingerprints.json`
- Reuse:
  `build/grounded_500_28m_full_20260715/checkpoints/best_geometry_model.pt`
- Reuse: `build/grounded_combined_1k_20260716/manifest.jsonl`

- [ ] Hash the source checkpoint and manifest with SHA-256.
- [ ] Record checkpoint tensor finiteness, parameter counts, `global_step`,
  optimizer group names, and configured learning rates.
- [ ] Re-run the fixed promotion set and the six fighter-style seeds and store
  all scalar metrics, binary hashes, and projection images.
- [ ] Verify that the copied best checkpoint in the failed run has identical
  model tensors to the source checkpoint.
- [ ] Mark
  `build/grounded_1k_28m_20260716/checkpoints/final_monitored_model.pt` as a
  rejected comparison artifact, never as a resume source.

**Gate:** all hashes and baseline metrics are recorded and reproducible before
any optimizer code is exercised.

## Task 1: Add A Read-Only Branch-Stability Preflight

**Files:**

- Create: `CLI/diagnose_training_branches.py`
- Test: `tests/test_training_branch_diagnostics.py`

The preflight must load the preserved checkpoint without modifying it and
evaluate 16 fixed validation records at all four inference timesteps. For each
record and timestep, record:

- teacher and student prediction RMS, absolute maximum, and finite status;
- raw consistency MSE and robust consistency candidate loss;
- student parameter-gradient norms for generated reconstruction,
  consistency, and direct measured solver branches;
- direct aero, connectivity, and validity SPSA component-gradient norms;
- pairwise cosine similarity among the three student gradient branches.

The direct objective preflight uses the same 33 solver evaluations as training.
No low-fidelity substitute is allowed.

The report establishes observed finite p50, p95, p99, and maximum values.
Branch trust-region limits must be derived from this report and written to the
run configuration. They must not be guessed after looking at a favorable
training result.

**Gate:** all four inference timesteps are finite on the source checkpoint and
the report contains independently measured nonzero gradients for aero,
connectivity, and validity.

## Task 2: Make Consistency Distillation Match Inference

**Files:**

- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/config.yaml`
- Test: `tests/test_consistency_model.py`
- Test: `tests/test_mission_conditioning.py`

- [ ] Add explicit consistency configuration:
  `consistency_interval`, `consistency_loss_type`,
  `consistency_huber_delta`, `consistency_timestep_sampling`, and
  `consistency_gradient_max_norm`.
- [ ] Retain the current sparse consistency frequency initially, but add a
  dedicated persisted `consistency_update_step`. Cycle that counter over
  `[999, 666, 333, 0]`; do not index the schedule with `global_step % 4`
  because an interval of 20 would repeatedly select the same timestep.
- [ ] Feed teacher and student the same latent, noise realization, timestep,
  and mission condition.
- [ ] Optimize a per-element Smooth L1/Huber consistency residual with a delta
  fixed before training from the source-checkpoint preflight.
- [ ] Record raw MSE from the same residual for comparability, but never use it
  as the unbounded optimizer scalar.
- [ ] Synchronize the consistency teacher from the diffusion EMA model instead
  of the just-updated online diffusion model.
- [ ] Reorder checkpoint loading so the EMA weights load before the teacher is
  synchronized.
- [ ] Save and restore `consistency_update_step` while accepting old
  checkpoints that do not contain it.
- [ ] Raise a descriptive error on nonfinite teacher output, student output,
  residual, or consistency gradient.

**Tests:**

- the dedicated consistency counter cycles all four exact inference levels;
- checkpoint resume continues the cycle;
- the teacher is copied from EMA, not the online model;
- Huber and raw MSE are equal near zero but Huber has bounded influence for a
  synthetic extreme residual;
- mission conditions reach both teacher and student;
- nonfinite inputs fail closed.

## Task 3: Preserve Solver Components Through SPSA Backward

**Files:**

- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/config.yaml`
- Test: `tests/test_aerodynamic_loss.py`
- Test: `tests/test_direct_solver_fused_parity.py`

The existing plus/minus solver evaluations already compute enough information
to expose component derivatives. Reuse each solve; do not add another CFD run.

- [ ] Return aero, connectivity, and validity components from every plus and
  minus objective evaluation.
- [ ] Accumulate three SPSA gradient estimates with the same Rademacher
  directions:
  `g_aero`, `g_connectivity`, and `g_validity`.
- [ ] Verify before clipping that their weighted sum equals the existing total
  objective gradient to numerical tolerance.
- [ ] Apply independently configured max-norm trust regions to each component,
  derived from Task 1. Trust regions may reduce an extreme norm but may not
  amplify a zero/tiny component or replace it.
- [ ] Sum the three applied component gradients and return that sum through the
  custom autograd function.
- [ ] Record raw norm, applied norm, scale factor, and pairwise cosine
  similarity for all three components.
- [ ] Keep the scalar forward value exactly equal to measured
  `aero + weighted connectivity + weighted validity`.
- [ ] Replace `nan_to_num(..., nan=0)` on the direct objective with explicit
  finite validation and failure. A failed solver must never appear as a
  zero-loss success.

**Tests:**

- call count remains `1 + 2 * directions`;
- component gradients sum to the legacy total when trust regions do not bind;
- forcing only one mock component to vary changes only that component's raw
  gradient;
- all three finite nonzero component gradients remain nonzero after balancing;
- fused and reference BFL backends retain scalar-loss and applied-gradient
  parity.

## Task 4: Separate And Recombine Student Gradient Branches

**Files:**

- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `CLI/multiobjective_gradients.py`
- Test: `tests/test_multiobjective_gradients.py`
- Test: `tests/test_aerodynamic_loss.py`

The student receives three semantically different gradient branches:

1. grounded/generated data losses;
2. consistency distillation;
3. direct measured CFD/connectivity/validity.

Compute them sequentially so the direct solver still runs after the first
neural graph is released:

1. Backpropagate diffusion, clean reconstruction, generated reconstruction,
   and latent reconstruction.
2. Capture the student data-gradient buffer and clear only student gradients.
3. Backpropagate the robust consistency loss when scheduled, capture its
   student gradient, and clear only student gradients.
4. Run all 33 direct solver evaluations, recompute the generated path, inject
   the component-balanced SPSA voxel gradient, capture the resulting student
   gradient, and clear only student gradients.
5. Apply a preflight-derived max-norm trust region to each student branch.
6. Sum all present branch gradients into the student parameters.
7. Apply the existing final whole-model gradient clip and execute one optimizer
   step.

The combiner must never normalize a zero or tiny gradient upward. Each applied
branch is `raw_gradient * min(1, configured_limit / raw_norm)`. This makes the
operation a transparent trust region rather than a hidden objective rewrite.

Record raw norm, applied norm, scale factor, and cosine similarity for each
student branch on every update. Keep diffusion and converter gradients in
their current ownership boundaries.

**Tests:**

- each synthetic branch contributes to the final gradient;
- an extreme consistency branch is limited without changing the data or
  direct branches;
- tiny gradients are not amplified;
- nonfinite gradients fail before `optimizer.step()`;
- sequential recomputation matches a small all-graphs-retained reference;
- direct-solver coverage remains exactly 100 percent.

## Task 5: Replace The One-Epoch Learning-Rate Reset

**Files:**

- Modify: `CLI/run_monitored_training.py`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/config.yaml`
- Test: `tests/test_consistency_model.py`
- Test: `tests/test_training_stability.py`

The monitored runner currently constructs an epoch-based cosine schedule with
`T_max=num_epochs`. A one-epoch continuation therefore records zero learning
rates at the end of every invocation.

- [ ] Add a run-local, optimizer-update-based schedule with a configurable
  nonzero floor.
- [ ] Calculate its horizon from the actual number of planned optimizer
  updates, including `--max-samples-per-epoch`.
- [ ] Step it after each successful optimizer update, not once per epoch.
- [ ] Version scheduler metadata in checkpoints.
- [ ] Resume the scheduler state only when continuing the same run/horizon.
  Starting a new probe from a promoted checkpoint creates a fresh schedule
  from configured optimizer-group learning rates.
- [ ] Preserve separate diffusion, converter, and consistency-student rates.
- [ ] Record current rates every update in the live JSON stream and every
  checkpoint.

**Gate:** no configured optimizer group reaches zero during a 40-update or
120-update probe unless zero was explicitly configured.

## Task 6: Strengthen Promotion Against Collapse And Boundary Artifacts

**Files:**

- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/run_monitored_training.py`
- Modify: `CLI/config.yaml`
- Test: `tests/test_overfit_stop.py`
- Test: `tests/test_training_stability.py`
- Test: `tests/test_aircraft_validity.py`

Extend the fixed-seed promotion result with:

- generated binary geometry hash and unique fraction;
- mean and worst largest-component fraction;
- mean and worst normalization-boundary fraction;
- counts for `span_sanity`, `normalization_margin`, and connectivity failures;
- generated mean and worst top-k recall;
- absolute pass/fail thresholds from the existing validity evaluator;
- non-regression checks against the source checkpoint baseline.

Use at least six fixed seeds. A candidate may be promoted only when:

- the existing reconstruction and generated-validity gates pass;
- solver coverage for the training stage is 100 percent;
- no critical finite check failed;
- generated unique fraction does not regress beyond a fixed predeclared
  tolerance;
- normalization-margin and connectedness metrics do not regress;
- generated worst-seed recall does not regress beyond a fixed predeclared
  tolerance;
- the lexicographic quality rank improves over the source or currently
  promoted checkpoint.

The source checkpoint remains the output `best_geometry_model.pt` when a
candidate fails. The rejected candidate is retained under an explicit
`rejected_*` name with its report for forensic comparison.

## Task 7: Make Update-Level Health Visible And Actionable

**Files:**

- Modify: `CLI/run_monitored_training.py`
- Modify: `CLI/training_tui.py`
- Modify: `CLI/watch_training_progress.py`
- Test: `tests/test_watch_training_progress.py`
- Test: `tests/test_resource_monitor.py`

- [ ] Write one atomic JSONL update record after every optimizer step.
- [ ] Include all neural loss values, raw/applied branch gradient norms,
  consistency timestep and teacher/student RMS, three SPSA component values and
  gradient norms, solver call count, learning rates, elapsed time, GPU
  utilization, VRAM, and CPU memory.
- [ ] Make the TUI distinguish sampled GPU utilization from unknown values;
  never display a hard-coded 100 percent.
- [ ] Show within-epoch loss and branch norms from JSONL rather than parsing
  wrapped `tqdm` console lines.
- [ ] Add fail-fast health rules for nonfinite values, missing solver calls,
  and consistency values above a preflight-derived ceiling.
- [ ] On a health failure, save a rejected emergency checkpoint and report,
  leave the last promoted checkpoint untouched, and exit nonzero.

The telemetry is evidence for whether optimizer inputs were applied; it is not
a substitute for any loss computation.

## Task 8: Verification Before GPU Training

Run in this order:

1. Focused consistency and gradient tests.
2. Direct objective and fused-kernel parity tests.
3. Promotion and monitoring tests.
4. Existing complete test suite.
5. A tiny CPU/mock-solver trainer integration test that verifies backward
   ordering and checkpoint compatibility.
6. A single `96^3` GPU update using the real fused D3Q27 solver.

Required commands will include:

```powershell
python -m pytest tests/test_consistency_model.py tests/test_mission_conditioning.py tests/test_multiobjective_gradients.py -q
python -m pytest tests/test_aerodynamic_loss.py tests/test_direct_solver_fused_parity.py tests/test_d3q27_kernel_parity.py -q
python -m pytest tests/test_overfit_stop.py tests/test_training_stability.py tests/test_watch_training_progress.py -q
python -m pytest -q
```

**Gate:** all tests pass, the one-update real solver smoke test records exactly
33 calls, and all three direct component gradients are finite and nonzero.

## Task 9: Staged Full-Physics Recovery Runs

All stages resume from a promoted checkpoint and retain `96^3`, D3Q27,
`fused_stream_bfl`, batch size 1, five LBM steps, sixteen SPSA directions, and
all direct objective components.

### Stage A: 40 Updates

- Resume from the preserved source checkpoint.
- Use a deterministic 40-record training subset and the fixed validation set.
- Expected solver calls: `40 * 33 = 1,320`.
- Expected wall time from the completed run: approximately 45 minutes plus
  promotion evaluation.

Pass only if:

- all 40 updates have 33 solver calls;
- no finite/health rule fails;
- all four consistency timesteps are exercised;
- consistency raw and robust metrics remain inside the predeclared preflight
  envelope;
- the candidate passes absolute and baseline-relative promotion gates;
- fixed-seed diversity does not collapse.

### Stage B: 120 Updates

- Resume only from the promoted Stage A checkpoint.
- Use a different deterministic subset seed.
- Expected solver calls: `120 * 33 = 3,960`.
- Expected wall time: approximately 2 to 2.5 hours plus evaluation.

Apply the same gates, and require no negative trend in generated validity,
diversity, boundary fraction, or worst-seed recall relative to Stage A.

### Stage C: Full 758-Record Epoch

- Resume only from the promoted Stage B checkpoint.
- Train over the complete `train` split of the 1,069-record corpus.
- Expected solver calls: `758 * 33 = 25,014`.
- Expected wall time: approximately 13.5 hours.

The completed candidate is promoted only if it passes every absolute and
baseline-relative gate. A process return code of zero is necessary but not
sufficient.

## Task 10: Final Evidence And Decision

**Files:**

- Create:
  `docs/benchmarks/grounded_1k_28m_stabilized_20260717/README.md`
- Create:
  `docs/benchmarks/grounded_1k_28m_stabilized_20260717/loss_and_gradient_report.md`
- Create:
  `docs/benchmarks/grounded_1k_28m_stabilized_20260717/geometry_validation_report.md`

Record:

- source, code, config, manifest, and checkpoint hashes;
- exact update and solver-call counts;
- raw and applied gradient distributions for every branch/component;
- learning-rate trace;
- promotion metrics before and after every stage;
- fixed-seed projection grids and binary hashes;
- generated validity failure distributions;
- runtime, GPU utilization, VRAM, CPU memory, and disk throughput;
- all test commands and results;
- rejected checkpoints and exact rejection reasons;
- claim boundaries: internal LBM optimization evidence is not independent PDE
  validation or proof of flightworthiness.

Only after Stage C passes should the paper's methods, results, figures, and
sentence-level analysis be updated. Failed or rejected runs may be discussed as
ablation/debugging evidence but must not be presented as successful model
results.

## Commit Structure

Use small logical commits:

1. `test: capture consistency and branch instability`
2. `fix: align and robustify consistency distillation`
3. `feat: preserve measured solver component gradients`
4. `fix: balance sequential student gradient branches`
5. `fix: make resumed learning-rate schedules update based`
6. `feat: reject collapsed geometry checkpoints`
7. `feat: stream grounded optimizer health metrics`
8. `docs: report stabilized 1k recovery evidence`

Do not commit generated checkpoints, downloaded corpora, raw voxel arrays, or
large runtime logs. Commit compact JSON summaries, plots, projection sheets,
and documentation only when repository policy permits their size.

## Stop Conditions

Stop implementation and investigate before any long run when:

- a test shows that a solver component is dropped or receives a zero applied
  gradient despite a finite nonzero raw gradient;
- fused/reference parity changes outside existing tolerances;
- old checkpoint loading changes source model tensors;
- a critical NaN or infinity is silently replaced;
- the real one-update smoke test does not execute exactly 33 solver calls;
- a short probe cannot preserve the source checkpoint's fixed-seed promotion
  metrics.

Do not respond to a failed short probe by immediately increasing parameters,
adding more data, reducing solver fidelity, or relaxing promotion thresholds.
Use the recorded branch/component evidence to identify the failed objective
path first.
