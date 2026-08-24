# Luna Implementation Report: Constrained Aircraft Recovery

Status: `SUPERSEDED_PENDING_REREVIEW`

## Summary

Tasks 1 through 5 were implemented in order on branch
`codex/constrained-aircraft-recovery`, based at `feef092`. This report records
the implementation through review round 3 and is not a closure statement.
Round 4 found additional mixed-batch guard transport and generated-path
converter-freeze defects, and round 5 found a separate per-sample
`DesignSpec` transport defect. Their changes and evidence are recorded in the
round-specific reports. No real `96^3` smoke was run in rounds 2 through 5, so
live interruption/resume and promotion remain unverified. No full epoch was
launched.

## Commits

- `1032014` `feat: resume CFD training within an epoch`
- `4a92e4a` `feat: protect grounded geometry at materialization threshold`
- `9f0edb8` `feat: constrain measured aircraft objectives`
- `798829b` `feat: gate aircraft recovery on topology non-regression`
- `9da675c` `fix: honor bounded interrupted training segments`
- `1368133` `fix: stabilize measured guard projection`
- `f4e29c0` `fix: close reviewed constrained recovery invariants`
- `432c8a9` `fix: close round-two recovery integration defects`
- `36d73f8` `fix: repair production gradient lifecycle`
- documentation evidence commit containing this report

Changed implementation and test paths across these commits:

- `CLI/aircraft_diffusion_cfd.py`
- `CLI/config.py`
- `CLI/config.yaml`
- `CLI/multiobjective_gradients.py`
- `CLI/run_monitored_training.py`
- `CLI/training_stability.py`
- `CLI/training_tui.py`
- `tests/test_consistency_model.py`
- `tests/test_constrained_recovery_review.py`
- `tests/test_multiobjective_gradients.py`
- `tests/test_training_stability.py`

Documentation paths added or updated for this work:

- `docs/benchmarks/training_objective_integrity_20260718.md`
- `docs/benchmarks/constrained_recovery_smoke_20260813.md`
- `.superpowers/sdd/constrained-aircraft-recovery/luna-implementation-report.md`

## Test Evidence

Each task followed red/green TDD evidence:

1. Task 1 red: imports failed because the run-state API did not exist. Green:
   `pytest tests/test_consistency_model.py tests/test_overfit_stop.py tests/test_watch_training_progress.py -q` -> `42 passed`.
   The interruption-contract regression tests added later passed as `2 passed`.
2. Task 2 red: `grounded_threshold_margin_loss` was missing. Green:
   `pytest tests/test_consistency_model.py tests/test_geometry_promotion_integrity.py -q` -> `35 passed`.
3. Task 3 red: the measured guard projection API was missing. Green:
   `pytest tests/test_multiobjective_gradients.py tests/test_aerodynamic_loss.py -q` -> `43 passed`.
   After smoke-exposed numerical hardening, the focused projection suite passed
   as `17 passed`.
4. Task 4 red: the directional promotion gate was missing. Green:
   `pytest tests/test_training_stability.py tests/test_geometry_promotion_integrity.py tests/test_watch_training_progress.py -q` -> `21 passed`.
5. Task 5 pre-smoke gate: `pytest -q` -> `409 passed, 3 warnings in 255.50s (0:04:15)`.
   Review-round focused suites -> `120 passed, 3 warnings`; dedicated review
   regressions -> `13 passed, 3 warnings`. Fresh post-round-1 full suite:
   `pytest -q` -> `423 passed, 3 warnings in 281.04s (0:04:41)`.

## Review-Round Red/Green

The first review regression module was initially red at collection because the
required RNG-neutral loader helper did not exist. That round's full set passed
`13 passed, 3 warnings`. For round 2, the new production-path tests were
initially red at collection because the final guard, runner fingerprint, saved
threshold, cadence-reset, and data-anchor APIs did not exist. After
`432c8a9`, the dedicated review module passed `18 passed, 3 warnings`.
Affected solver/trainer/promotion suites passed `84 passed, 3 warnings`.
The fresh full suite passed `428 passed, 3 warnings in 284.9s`.

Round 2 specifically verifies that the final parameter update remains
non-uphill against separate data, connectivity, and validity parameter-space
guards; exact resume restores the saved threshold before compatibility
comparison; the nested fingerprint covers model, objective, optimizer,
consistency, clipping, EMA, branch limits, reconstruction, and margin
settings; resumed next-epoch cadence starts at zero; and the exact
full-lattice calibrated margin is present in the captured student data anchor.

Round 3 initially failed with the required production tests: the shape-drag
fields were outside `LBMPhysicsConfig`, and the one-update trainer path had no
all-group gradient lifecycle or active-guard replay invariant. After
`36d73f8`, dedicated production regressions passed `6 passed, 3 warnings` and
the affected suites passed `101 passed, 3 warnings`. The train_epoch fixture
observes diffusion, converter, and consistency-student parameters at the
actual optimizer step, confirms inactive validity is not replayed, checks
replay isolation, and verifies post-clip active-guard dot products.

## Smoke, Resume, and Promotion

The exact smoke command and exact intended resume command are recorded in
`docs/benchmarks/constrained_recovery_smoke_20260813.md`.

The first real run completed one optimizer update with 33 measured solver
calls, then reached an end-of-epoch coverage assertion that incorrectly used
the full 32-update loader length after an intentional stop. `9da675c` fixes
coverage to the processed prefix and makes checkpoint cadence relative to the
current segment.

The retry completed one real optimizer update with 33 measured solver calls,
then failed before update 2 because a float32 projection residual left an
improvement direction numerically uphill against the validity guard.
`1368133` performs guard projection in float64 and fail-closes a residual
conflicting improvement direction to zero. The retry therefore provides real
solver-call and objective telemetry, but not a valid 8-update checkpoint.

Resume evidence is a deterministic toy 2-update interruption plus 2-update
continuation test. It compares sample/update sequence, model parameters,
optimizer state, scheduler state, JSONL steps, and RNG continuation against
an uninterrupted four-update trajectory. The run-state path stores model,
optimizer, scheduler, scaler, EMA, RNG, threshold, sample order, objective-
defining configuration, outer epoch position, and durable log metadata.
Atomic replacement, log-ahead truncation, checkpoint-ahead fail-closed
handling, `.previous` recovery, and exact-vs-model-only resume distinctions
are tested. No real `latest_run_state.pt` was created because neither smoke
reached update 8.

Round 3 smoke command: `NOT RUN BY INSTRUCTION`. The exact bounded command
and exact `--resume-run-state` continuation command remain recorded in the
smoke evidence document. The earlier one-update real attempt provides only
historical evidence of 33 measured solver calls; it predates `432c8a9` and is
not a validation of the current final-guard integration. Promotion result:
`NOT RUN`. No candidate was promoted, and no promotion claim is made.

## Scientific and Engineering Concerns

- No real smoke was run in rounds 2 or 3, so live exact-resume behavior and
  condition-based promotion remain unverified on the 96-cubed run.
  Deterministic exact-resume behavior, crash-safe reconciliation, split
  integrity, saved-threshold compatibility, complete objective fingerprint,
  next-epoch cadence reset, and promotion-baseline identity are verified in
  focused tests only.
- The internal five-step LBM smoke reported `lbm_converged=0`; its measured
  forces are optimization inputs, not external-PDE ground truth or
  claim-bearing aerodynamic coefficients.
- No real smoke was run in rounds 2 or 3 by instruction. The round-2 fresh
  full pytest result was `428 passed`; the round-3 fresh full pytest result was
  `431 passed, 3 warnings in 298.29s (0:04:58)`.
- The fixed global materialization threshold, held-out promotion split, and
  full validity objective remain intact. The `0.70` largest-component value is
  a hard guard and was not substituted for the validity objective.

## Integrity Confirmation

No surrogate was added or used. No solver component was removed, disabled, or
replaced by an evaluator shortcut. All measured scalar objective values and
the 33 solver calls per real optimizer update were preserved in smoke
telemetry. The implementation retains the full reconstruction, occupancy,
aerodynamic, connectivity, and aircraft-validity paths.
