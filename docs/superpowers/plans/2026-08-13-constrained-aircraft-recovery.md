# Constrained Aircraft Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long 96^3 CFD-integrated training recoverable and prevent aggregate-loss improvements from destroying reconstruction, connectivity, or aircraft validity.

**Architecture:** Add atomic mid-epoch run-state checkpoints that restore the exact optimizer, scheduler, RNG, sample-order, and update position. Replace unconstrained summation of measured objective gradients with a deterministic priority-aware projection that retains every measured component but prevents occupancy and aerodynamic gradients from taking first-order steps uphill on reconstruction, connectivity, or validity. Add a calibrated-threshold margin term to grounded reconstruction, then gate the implementation with unit tests and a short real-CFD 96^3 non-regression smoke.

**Tech Stack:** Python 3.12, PyTorch, SciPy EDT, D3Q27 MRT LBM, Triton fused stream/BFL, pytest, JSON/JSONL run artifacts.

## Global Constraints

- Do not use aerodynamic, geometry, connectivity, or validity surrogates in place of the real measured objective.
- All occupancy, aerodynamic, connectivity, and aircraft-validity values must continue to be computed and incorporated into the final optimization update.
- Do not remove, disable, zero-weight, skip, or relabel a difficult loss component as a diagnostic.
- Do not use target occupancy to carve generated geometry with top-k selection; materialization remains the fixed calibrated global threshold.
- The calibrated geometry threshold remains checkpoint metadata and is the same threshold used by training, promotion, evaluation, inference, and export.
- Direct gradients continue to come from finite differences of the real solver objective. No learned proxy, straight-through fake measurement, or cached label may replace solver evaluations.
- Preserve corpus separation: `train` is used for gradient updates, `val` for promotion, and `holdout`/`test` remain untouched by training and checkpoint selection.
- Do not launch a full 758-update or 1,069-record epoch. The implementation phase may run unit tests and one bounded 32-update 96^3 smoke only.
- Do not claim aerodynamic ground truth when the LBM reports non-convergence. Preserve convergence and force-stability metadata in every update record.
- Use atomic checkpoint writes and never overwrite the source checkpoint.
- Follow TDD: write a failing test, observe the intended failure, implement the minimum behavior, and rerun the focused test before broader verification.

## Failure Evidence To Preserve

The completed recovery epoch is the regression fixture and must not be rewritten:

- Run: `build/grounded_1k_free_running_recovery_20260718_restart1`
- Updates: 758/758; exit code 0; 25,014 measured solver calls.
- First-to-last 100-update optimization loss: 10.301322 to 8.239930.
- Occupancy loss: 0.187139 to 0.003789.
- Aerodynamic loss: 2.056208 to 0.356581.
- Connectivity loss: 0.013808 to 0.630541.
- Validation generated-valid fraction: 0.0 to 0.1041666667.
- Validation largest-component fraction: 0.9872801371 to 0.5276170658.
- Validation reconstruction recall: 0.2431656392 to 0.0.
- Validation uniqueness: 0.9479166667 to 0.5416666667.
- Promotion correctly rejected the final state and retained the source checkpoint.

---

### Task 1: Atomic Mid-Epoch Checkpoint And Exact Resume

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/run_monitored_training.py`
- Modify: `CLI/training_tui.py`
- Test: `tests/test_consistency_model.py`
- Test: `tests/test_overfit_stop.py`
- Test: `tests/test_watch_training_progress.py`

**Interfaces:**
- Add CLI option `--checkpoint-every-updates N`, with `0` disabling mid-epoch checkpoints.
- Add CLI option `--resume-run-state PATH`, distinct from `--resume-from`, because the former resumes an interrupted run while the latter starts a new continuation run from model weights.
- Persist `epoch_index`, `completed_in_epoch`, deterministic sample order or sampler state, total optimizer update, model state, all optimizer states, scheduler state, mixed-precision state when present, Python RNG, NumPy RNG, CPU Torch RNG, all CUDA RNG states, geometry threshold metadata, and configuration needed to reject incompatible resumes.
- Preserve `updates.jsonl` on exact resume and append from the next update. Starting a new run may truncate it.
- Emit `run_state_checkpoint_path`, `resumed_from_update`, and `remaining_in_epoch` in history/update metadata for the TUI.

- [ ] **Step 1: Write failing serialization and compatibility tests**

Create tests that save a run state after a deterministic toy update, perturb every model/optimizer/RNG value, restore it, and assert exact equality. Add incompatibility cases for manifest identity, grid size, latent dimension, split, and sample count.

- [ ] **Step 2: Write a failing interrupted-versus-uninterrupted equivalence test**

Use a tiny deterministic dataset and inexpensive mocked measured objective. Compare an uninterrupted four-update trajectory with a two-update save plus two-update resume. Assert identical sample order, global steps, learning rates, model parameters, optimizer state, and nonduplicated JSONL update indices.

- [ ] **Step 3: Implement one atomic run-state writer**

Write to a sibling temporary file, flush and close it, then replace the target with `os.replace`. Keep only `latest_run_state.pt` plus an optional previous-known-good file during replacement; do not accumulate hundreds of 464 MB checkpoints.

- [ ] **Step 4: Resume at the next unprocessed sample**

Do not replay already-completed updates. Configure the scheduler for the original run horizon before loading its state. Reject a resume when immutable configuration differs, with an error naming every mismatched field.

- [ ] **Step 5: Surface resume state in the TUI**

Show `completed/total`, resumed update, last run-state checkpoint age/path, and whether the current process is a fresh run or exact resume. Keep the monitor read-only.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
pytest tests/test_consistency_model.py tests/test_overfit_stop.py tests/test_watch_training_progress.py -q
```

Expected: all selected tests pass, with the interrupted-versus-uninterrupted equality test included.

- [ ] **Step 7: Commit**

```powershell
git add CLI/aircraft_diffusion_cfd.py CLI/run_monitored_training.py CLI/training_tui.py tests/test_consistency_model.py tests/test_overfit_stop.py tests/test_watch_training_progress.py
git commit -m "feat: resume CFD training within an epoch"
```

### Task 2: Fixed-Threshold Grounded Reconstruction Margin

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/config.yaml`
- Modify: `CLI/config.py`
- Test: `tests/test_consistency_model.py`
- Test: `tests/test_geometry_promotion_integrity.py`

**Interfaces:**
- Add a grounded reconstruction margin loss that consumes predicted probabilities or logits, binary target voxels, and the fixed calibrated probability threshold.
- Positive target voxels are penalized when their probability falls below `threshold + positive_margin` (clamped below 1); negative target voxels are penalized when their probability rises above `threshold - negative_margin`.
- Normalize positive and negative regions independently before combining them so sparse aircraft voxels cannot be drowned by empty space.
- Record `threshold_positive_margin_loss`, `threshold_negative_margin_loss`, positive/negative voxel counts, and configured weights per optimizer update.

- [ ] **Step 1: Write failing margin-loss tests**

Cover: zero loss for correctly separated probabilities; positive loss for disappearing aircraft voxels; positive loss for false solid voxels; finite gradients; independent class normalization; and use of the checkpoint-calibrated threshold rather than an inferred batch top-k threshold.

- [ ] **Step 2: Implement the loss with no geometry carving**

The target mask may supervise reconstruction. It must never replace, intersect, dilate, select, or post-process generated geometry passed to CFD or promotion.

- [ ] **Step 3: Integrate the margin into the ordinary grounded data branch**

Keep existing MSE and geometry losses. Add the margin as a separately logged component. Ensure its gradient participates in the reconstruction/data anchor used by conflict projection.

- [ ] **Step 4: Add configuration validation**

Require nonnegative margins and weights, and require `threshold + positive_margin < 1`. Persist effective values in run history.

- [ ] **Step 5: Run focused tests**

```powershell
pytest tests/test_consistency_model.py tests/test_geometry_promotion_integrity.py -q
```

- [ ] **Step 6: Commit**

```powershell
git add CLI/aircraft_diffusion_cfd.py CLI/config.yaml CLI/config.py tests/test_consistency_model.py tests/test_geometry_promotion_integrity.py
git commit -m "feat: protect grounded geometry at materialization threshold"
```

### Task 3: Priority-Constrained Measured Multi-Objective Gradients

**Files:**
- Modify: `CLI/multiobjective_gradients.py`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/config.yaml`
- Modify: `CLI/config.py`
- Test: `tests/test_multiobjective_gradients.py`
- Test: `tests/test_aerodynamic_loss.py`

**Interfaces:**
- Treat grounded reconstruction, connectivity, and aircraft validity as guard objectives.
- Treat occupancy and aerodynamics as improvement objectives.
- Before combining measured component gradients, project each improvement gradient against every active guard gradient whenever their dot product is negative. For gradient descent, the accepted combined gradient must have nonnegative dot product with each active guard gradient within numerical tolerance.
- Connectivity guard activation uses the real measured dominant-largest-component shortfall `max(0, 0.70 - largest_component_fraction)` and retains existing measured connectivity terms.
- Validity remains the full measured aircraft-validity objective; do not replace it with only the 0.70 connectivity gate.
- Apply deterministic ordering and log pre/post cosine, projection norm, active guard set, and accepted norm for every component.
- If projection consumes an improvement gradient completely, log a zero accepted norm but retain the measured loss value and solver calls.

- [ ] **Step 1: Write failing synthetic-vector tests**

Test aligned gradients, one conflict, multiple guards, zero guard, exactly opposing gradients, deterministic ordering, finite outputs, and the invariant that the final accepted direction has nonnegative dot product with each active guard.

- [ ] **Step 2: Write a failing direct-objective integration test**

Use controlled SPSA component gradients where occupancy/aero lower their own losses while increasing connectivity. Assert that all measured values are logged, all finite-difference calls still execute, and the accepted update does not point uphill on connectivity or validity.

- [ ] **Step 3: Implement constrained projection in `multiobjective_gradients.py`**

Keep the math independent of the trainer and usable with named gradient dictionaries. Do not silently mutate caller tensors. Use float64 for dot products/norm decisions, then return gradients in their original dtype/device.

- [ ] **Step 4: Integrate it after component SPSA estimation and before model backpropagation**

Preserve the existing outer projection against grounded data, but make the direct branch internally constraint-aware first. The final direct loss scalar remains the complete measured scalar for reporting; only the update direction is constrained.

- [ ] **Step 5: Add explicit guard telemetry**

Every update record must show whether reconstruction, connectivity, and validity guards were active and whether occupancy or aero was projected. This is optimization evidence, not a diagnostic-only computation.

- [ ] **Step 6: Run focused tests**

```powershell
pytest tests/test_multiobjective_gradients.py tests/test_aerodynamic_loss.py -q
```

- [ ] **Step 7: Commit**

```powershell
git add CLI/multiobjective_gradients.py CLI/aircraft_diffusion_cfd.py CLI/config.yaml CLI/config.py tests/test_multiobjective_gradients.py tests/test_aerodynamic_loss.py
git commit -m "feat: constrain measured aircraft objectives"
```

### Task 4: Promotion Non-Regression Gate For Short Recovery Runs

**Files:**
- Modify: `CLI/run_monitored_training.py`
- Modify: `CLI/training_stability.py`
- Modify: `CLI/training_tui.py`
- Test: `tests/test_training_stability.py`
- Test: `tests/test_geometry_promotion_integrity.py`
- Test: `tests/test_watch_training_progress.py`

**Interfaces:**
- Compare baseline and candidate promotion reports using fixed seeds and the same validation records.
- A short recovery candidate is acceptable only when all configured conditions pass:
  - generated occupancy error decreases;
  - mean largest-component fraction is at least `0.70` and does not regress;
  - reconstruction recall is no worse than baseline minus `0.02`;
  - generated aircraft-valid fraction strictly improves when the baseline is below `0.50`, or remains at least `0.50` otherwise;
  - uniqueness is no worse than baseline minus `0.05`;
  - fixed-global-threshold and calibrated-threshold checks remain true.
- Persist every comparison, delta, threshold, and pass/fail result. Never select a checkpoint from aggregate geometry rank when a hard condition fails.

- [ ] **Step 1: Write table-driven failing gate tests**

Include the exact completed-epoch regression: occupancy and aero improve, validity reaches 0.1041666667, but largest component falls to 0.5276170658 and reconstruction recall to zero. Assert rejection.

- [ ] **Step 2: Implement explicit directional gate evaluation**

Return a structured report with named conditions and observed/baseline/delta/threshold fields. Avoid a single opaque score.

- [ ] **Step 3: Integrate checkpoint selection and TUI rendering**

Selection requires all hard conditions. The TUI must display the failing condition names and candidate deltas.

- [ ] **Step 4: Run focused tests**

```powershell
pytest tests/test_training_stability.py tests/test_geometry_promotion_integrity.py tests/test_watch_training_progress.py -q
```

- [ ] **Step 5: Commit**

```powershell
git add CLI/run_monitored_training.py CLI/training_stability.py CLI/training_tui.py tests/test_training_stability.py tests/test_geometry_promotion_integrity.py tests/test_watch_training_progress.py
git commit -m "feat: gate aircraft recovery on topology non-regression"
```

### Task 5: Verification And Bounded Real-CFD Smoke

**Files:**
- Modify: `docs/benchmarks/training_objective_integrity_20260718.md`
- Create: `docs/benchmarks/constrained_recovery_smoke_20260813.md`
- Generated artifacts only: `build/constrained_recovery_smoke_20260813/`

**Interfaces:**
- The smoke uses the real 96^3 D3Q27 solver with `fused_stream_bfl`, 33 measured calls per optimizer update, 32 training updates, the `train` split, and fixed-seed `val` promotion.
- Enable mid-epoch checkpointing at 8-update intervals and prove one exact stop/resume before completing all 32 updates.
- The smoke is evidence for directional correctness, not a trained-model or publication-quality aerodynamic claim.

- [ ] **Step 1: Run the complete automated test suite**

```powershell
pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run an 8-update interrupted segment**

Use a new output directory, `--max-samples-per-epoch 32`, and `--checkpoint-every-updates 8`. Stop only after the atomic run-state artifact exists and update 8 is fully recorded.

- [ ] **Step 3: Resume the same run state and complete 32 updates**

Use `--resume-run-state` and verify the final JSONL contains exactly updates 1 through 32 once each, with no duplicate solver work for updates 1 through 8.

- [ ] **Step 4: Evaluate smoke acceptance conditions**

The smoke passes only if occupancy error improves while connectivity loss, reconstruction recall, and largest-component fraction satisfy Task 4. A failed smoke is a valid implementation result and must not be hidden by aggregate loss.

- [ ] **Step 5: Write the evidence report**

Record commands, commit SHA, source checkpoint SHA-256, manifest SHA-256, solver call count, LBM convergence counts, before/after promotion metrics, guard activations/projections, exact-resume proof, resource summary, and every failed or passed gate.

- [ ] **Step 6: Commit documentation only after verification**

```powershell
git add docs/benchmarks/training_objective_integrity_20260718.md docs/benchmarks/constrained_recovery_smoke_20260813.md
git commit -m "docs: record constrained recovery evidence"
```

## Final Acceptance

- The full automated suite passes from a clean process.
- Exact resume produces the same continuation as uninterrupted training in deterministic tests.
- A real interrupted 96^3 smoke resumes without duplicate or skipped updates.
- All 33 real solver evaluations still run per update.
- Occupancy, aerodynamics, connectivity, validity, and reconstruction all affect the accepted gradient or explicitly constrain it.
- The candidate checkpoint is selected only if every directional promotion condition passes.
- No full epoch starts automatically.
- The implementation report distinguishes optimization evidence from claim-bearing aerodynamic validation.
