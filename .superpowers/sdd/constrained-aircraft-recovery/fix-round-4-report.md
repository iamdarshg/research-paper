# Fix Round 4 Implementation Report

Status: `IMPLEMENTED_PENDING_REREVIEW`

## Scope

Implemented both round-4 findings on
`codex/constrained-aircraft-recovery`, starting from `d340871`.
Implementation commit: `e6f9016` (`fix: align recovery guard and decoder gradients`).
No push, merge, real long smoke, or full training epoch was performed.

## Mixed-Batch Guard Transport

`DirectSolverSPSAFunction` now preallocates one full batch-shaped gradient
buffer for each topology guard. Active samples write to their original batch
index and inactive positions remain exact zeros. The ordered union of active
guard names is computed across the batch and is protected from the generic
first-record telemetry aggregation.

Production regressions cover all required batch-size-2 patterns:

- first sample inactive, later sample connectivity-active;
- first sample connectivity-active, later sample inactive;
- connectivity active on the first sample and validity active on the second.

For every pattern, the test exercises both direct production SPSA aggregation
and a real one-update `train_epoch` replay. It checks `[B, ...]` guard shapes,
zero inactive slices, nonzero active slices, the ordered guard union, the mean
of two controlled base values, one evaluation record, and six controlled
objective calls for two samples with one SPSA direction. Round 4 did not vary
or prove per-sample `DesignSpec` transport; that separate integrity defect and
its weighted-scalar evidence are covered by round 5.

## Generated-Path Converter Freeze

`freeze_decoder_for_generated_paths=True` now filters converter parameter
entries from every captured generated branch before branch recombination:
generated data/reconstruction and exact margin, consistency when applicable,
direct solver replay, and active topology guard replay. Upstream diffusion and
consistency-student gradients are preserved.

The separate full-lattice clean grounded converter gradient is captured after
the filter and added back to the data anchor. Lifecycle telemetry records clean
grounded converter norm and generated converter norms before and after the
switch. The production test inspects gradients at the actual optimizer-step
boundary: enabled mode removes all generated converter entries while retaining
a nonzero clean grounded converter update; disabled mode retains generated
converter entries and produces a distinct converter update.

The switch was already serialized through the complete `TrainingConfig`
fingerprint. Round 4 adds an explicit exact-resume fingerprint assertion for
the disabled value.

## TDD Evidence

Initial focused run on unmodified production code:

```text
4 failed, 1 passed, 20 deselected, 3 warnings in 53.85s
```

The failures matched the findings: union overwrite for first-inactive and
split-guard batches, `[1, ...]` instead of `[2, ...]` for first-active, and no
captured-buffer freeze telemetry or behavior.

Focused green run:

```text
5 passed, 20 deselected, 3 warnings in 63.91s
```

## Verification

```text
pytest tests/test_constrained_recovery_review.py -q
25 passed, 3 warnings in 91.49s

pytest tests/test_aerodynamic_loss.py tests/test_direct_solver_fused_parity.py tests/test_multiobjective_gradients.py tests/test_consistency_model.py tests/test_training_branch_diagnostics.py tests/test_overfit_stop.py -q
97 passed, 3 warnings in 81.09s

pytest -q
435 passed, 3 warnings in 311.63s
```

The three warnings are existing `pkg_resources` namespace/deprecation warnings.

## Integrity And Concerns

No surrogate was introduced. No solver call or measured objective component
was removed, disabled, relabeled, or replaced. Fixed-threshold materialization,
held-out split integrity, scalar aggregation, and call accounting remain
intact.

This round provides automated production-path evidence only. No current-code
real `96^3` smoke was run by instruction, so live exact resume, a completed
eight-update checkpoint boundary, LBM convergence behavior, and promotion
remain unverified. The implementation must remain pending re-review and must
not be described as closed or production-ready.
