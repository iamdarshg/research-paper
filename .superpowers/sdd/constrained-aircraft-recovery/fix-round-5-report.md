# Fix Round 5 Implementation Report

Status: `IMPLEMENTED_PENDING_REREVIEW`

## Scope

Implemented the final-breaker per-sample `DesignSpec` finding on
`codex/constrained-aircraft-recovery`, starting from `f1dbc59`.
Implementation commit: `aa701c0` (`fix: preserve per-sample solver design specs`).
No push, merge, real smoke, or full training epoch was performed.

## Root Cause

`aircraft_collate_fn` correctly retained one `DesignSpec` object per sample,
but `train_epoch` replaced the collated list with its first element. The direct
SPSA autograd function also accepted only one spec and forwarded that same
object to every sample's base, plus, and minus measured evaluations. Mixed
batches therefore used sample zero's mission weights and related fields for
all solver objectives.

## Implementation

`train_epoch` now passes the collated spec value through unchanged.
`DirectSolverSPSALoss` and `DirectSolverSPSAFunction` accept either one
`DesignSpec` or a sequence. The direct batch boundary normalizes the input,
requires a sequence length of one or exactly the direct-objective batch size,
validates every sequence entry, and selects the corresponding spec once per
sample before all base and antithetic evaluations.

A lone `DesignSpec` and a one-element sequence broadcast across the batch for
compatibility. A mismatched sequence fails before any measured objective call.
Solver calls, component aggregation, batch-aligned guards, and backward
gradient construction are otherwise unchanged.

## Production Tests

The call-recording objective uses two geometrically controlled samples with
different weights:

- sample 0: `space_weight=0.80`, `drag_weight=0.15`, `lift_weight=0.05`;
- sample 1: `space_weight=0.05`, `drag_weight=0.25`, `lift_weight=0.70`.

Both the direct `DirectSolverSPSALoss` test and a one-update production
`train_epoch` test assert that sample zero's base/plus/minus calls receive its
spec and sample one's base/plus/minus calls receive its distinct spec. They
also assert six calls, the mean of the correctly weighted base objectives,
inequality from the first-spec-for-both result, unchanged evaluation/call
counts, and finite direct gradients.

Separate tests cover direct `DesignSpec` broadcast, one-element-sequence
broadcast, and fail-fast rejection of three specs for a batch of two.

## TDD Evidence

Initial dedicated run on the round-4 implementation:

```text
3 failed, 1 passed, 25 deselected, 3 warnings in 64.08s
```

The direct sequence test passed the tuple unchanged to a single-sample
objective, the mismatch was not rejected, and production `train_epoch` reused
sample zero's spec for all six calls. The legacy single-spec case passed.

Final dedicated run after adding one-element-sequence coverage:

```text
5 passed, 25 deselected, 3 warnings in 52.01s
```

## Verification

```text
pytest tests/test_constrained_recovery_review.py -q
30 passed, 3 warnings in 101.70s

pytest tests/test_aerodynamic_loss.py tests/test_direct_solver_fused_parity.py tests/test_multiobjective_gradients.py tests/test_consistency_model.py tests/test_training_branch_diagnostics.py tests/test_overfit_stop.py -q
97 passed, 3 warnings in 84.32s

pytest -q
440 passed, 3 warnings in 341.45s
```

The warnings are the existing `pkg_resources` and namespace-package
deprecation warnings.

## Documentation Correction And Concerns

The two round-4 mixed-batch statements now describe their evidence as
controlled scalar aggregation rather than proof of per-sample mission-spec
integrity. This round supplies the fresh weighted and call-recording evidence
for that stronger claim.

No surrogate was introduced. No solver call, measured component, guard,
fixed-threshold path, or held-out integrity check was removed or weakened.

No current-code real `96^3` smoke was run by instruction. Live exact resume at
the eight-update checkpoint boundary, real LBM convergence behavior, and
promotion remain unverified. The implementation remains pending re-review and
must not be described as closed or production-ready.
