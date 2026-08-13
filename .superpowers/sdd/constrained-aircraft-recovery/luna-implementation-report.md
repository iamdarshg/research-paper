# Luna Implementation Report: Constrained Aircraft Recovery

Status: `DONE_WITH_CONCERNS`

## Summary

Tasks 1 through 5 were implemented in order on branch
`codex/constrained-aircraft-recovery`, based at `feef092`. Tasks 1 through 4
are complete and committed. Task 5's focused and full-suite gates were run;
the permitted real `96^3` smoke was started twice with the exact measured
solver configuration, but execution time did not allow the requested
8-update interruption/resume proof or promotion evaluation. No full epoch was
launched.

## Commits

- `1032014` `feat: resume CFD training within an epoch`
- `4a92e4a` `feat: protect grounded geometry at materialization threshold`
- `9f0edb8` `feat: constrain measured aircraft objectives`
- `798829b` `feat: gate aircraft recovery on topology non-regression`
- `9da675c` `fix: honor bounded interrupted training segments`
- `1368133` `fix: stabilize measured guard projection`
- documentation commit containing this report: `docs: record constrained recovery implementation evidence`

Changed implementation and test paths across these commits:

- `CLI/aircraft_diffusion_cfd.py`
- `CLI/config.py`
- `CLI/config.yaml`
- `CLI/multiobjective_gradients.py`
- `CLI/run_monitored_training.py`
- `CLI/training_stability.py`
- `CLI/training_tui.py`
- `tests/test_consistency_model.py`
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
   A later full-suite run was intentionally stopped at the user's request
   after the final projection hardening; the post-hardening focused suite
   passed, but no post-hardening full-suite result is claimed.

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

Resume evidence is bounded to focused unit tests and runner/CLI inspection:
the run-state path stores model, optimizer, scheduler, scaler, EMA, RNG,
threshold, sample order, and compatibility state; atomic replacement and
exact-vs-model-only resume distinctions are tested. No real `latest_run_state.pt`
was created because neither smoke reached update 8.

Promotion result: `NOT RUN`. No candidate was promoted, and no promotion
claim is made.

## Scientific and Engineering Concerns

- The requested real smoke did not complete the 8-update interruption point,
  so live exact-resume behavior and condition-based promotion remain
  unverified on the 96-cubed run. The exact command and state mechanism are
  present and focused-tested.
- The internal five-step LBM smoke reported `lbm_converged=0`; its measured
  forces are optimization inputs, not external-PDE ground truth or
  claim-bearing aerodynamic coefficients.
- The post-hardening full pytest pass was stopped by the user's instruction;
  only the post-hardening focused projection tests were run. The last complete
  full-suite result is the `409 passed` result listed above.
- The fixed global materialization threshold, held-out promotion split, and
  full validity objective remain intact. The `0.70` largest-component value is
  a hard guard and was not substituted for the validity objective.

## Integrity Confirmation

No surrogate was added or used. No solver component was removed, disabled, or
replaced by an evaluator shortcut. All measured scalar objective values and
the 33 solver calls per real optimizer update were preserved in smoke
telemetry. The implementation retains the full reconstruction, occupancy,
aerodynamic, connectivity, and aircraft-validity paths.
