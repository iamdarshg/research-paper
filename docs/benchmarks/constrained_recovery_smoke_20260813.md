# Constrained Recovery Smoke Evidence

Status: `IMPLEMENTED_PENDING_REREVIEW`

Round-5 status: implementation commit `aa701c0` passed dedicated, affected,
and full automated suites. This document's real-smoke artifacts predate all
five review-fix rounds; no new real smoke was run by instruction. This is not
closure or promotion evidence.

## Scope

This record covers the single permitted bounded real smoke for Tasks 1 through
5. It used the real `96^3` D3Q27 solver with the `fused_stream_bfl` backend,
the actual free-running four-step generator, 16 antithetic SPSA directions,
and the measured direct objective. It did not launch a full epoch.

## Exact Command

The bounded interruption command was:

```text
python -u CLI/run_monitored_training.py --manifest build/grounded_combined_1k_20260716/manifest.jsonl --num-epochs 1 --batch-size 1 --latent-dim 192 --grid-size 96 --learning-rate 2e-5 --lr-min-ratio 0.10 --solver D3Q27 --lbm-stream-bfl-backend fused_stream_bfl --save-dir build/constrained_recovery_smoke_20260813_retry/checkpoints --resume-from build/training_stabilization_20260718/intrinsic_threshold_smoke/checkpoints/best_geometry_model.pt --history-output build/constrained_recovery_smoke_20260813_retry/history_segment.json --updates-output build/constrained_recovery_smoke_20260813_retry/updates.jsonl --save-every 0 --no-save-final-checkpoint --cpu-threads 2 --max-samples-per-epoch 32 --subset-seed 0 --training-split train --promotion-split val --promotion-evaluation-samples 16 --promotion-generation-seeds 6 --no-stop-on-promotion-pass --direct-solver-loss-weight 1.0 --direct-solver-steps 5 --direct-solver-directions 16 --direct-connectivity-weight 1.0 --direct-aircraft-validity-weight 1.0 --direct-solver-perturbation 0.15 --direct-solver-perturbation-grid-size 12 --checkpoint-every-updates 8 --stop-after-updates 8
```

The exact-resume continuation command, verified against the runner's
`--resume-run-state` contract but not executed because no checkpoint reached
the eighth-update boundary, is:

```text
python -u CLI/run_monitored_training.py --manifest build/grounded_combined_1k_20260716/manifest.jsonl --num-epochs 1 --batch-size 1 --latent-dim 192 --grid-size 96 --learning-rate 2e-5 --lr-min-ratio 0.10 --solver D3Q27 --lbm-stream-bfl-backend fused_stream_bfl --save-dir build/constrained_recovery_smoke_20260813_retry/checkpoints --resume-run-state build/constrained_recovery_smoke_20260813_retry/checkpoints/latest_run_state.pt --history-output build/constrained_recovery_smoke_20260813_retry/history_segment.json --updates-output build/constrained_recovery_smoke_20260813_retry/updates.jsonl --save-every 0 --no-save-final-checkpoint --cpu-threads 2 --max-samples-per-epoch 32 --subset-seed 0 --training-split train --promotion-split val --promotion-evaluation-samples 16 --promotion-generation-seeds 6 --no-stop-on-promotion-pass --direct-solver-loss-weight 1.0 --direct-solver-steps 5 --direct-solver-directions 16 --direct-connectivity-weight 1.0 --direct-aircraft-validity-weight 1.0 --direct-solver-perturbation 0.15 --direct-solver-perturbation-grid-size 12 --checkpoint-every-updates 8
```

## Evidence

- The first corrected-run attempt completed one optimizer update and logged
  `direct_solver.call_count=33`, `global_step=423`, and
  `remaining_in_epoch=31`. It then stopped at the old end-of-epoch coverage
  assertion; this was fixed in `9da675c`.
- The retry completed one optimizer update with the same 33 measured solver
  calls and the same `global_step=423` start checkpoint lineage. It then
  failed before update 2 with a float32 residual in the measured guard
  projection. The float64 projection and fail-closed fallback were added in
  `1368133`.
- Both smoke `updates.jsonl` files contain exactly one optimizer-update record;
  neither contains the requested 8 updates. Neither checkpoint directory has
  `latest_run_state.pt`, because the configured cadence is eight completed
  updates and the smoke did not reach that boundary.
- The first update preserved the measured scalar objective and called the
  solver 33 times. Its telemetry included the active validity guard,
  connectivity fraction, occupancy/aerodynamic components, SPSA norms, and
  the fixed threshold metadata.
- Promotion result: `NOT RUN`. The process did not complete an epoch and no
  candidate was promoted.

## Review-Round Verification Boundary

The round-1 focused review suite passed `13 tests`, the round-1 combined
focused suites passed `120 tests`, and its fresh full suite passed `423 tests`
with three existing warnings. Round 2's dedicated review suite passed `18
tests`, the affected solver/trainer/promotion suites passed `84 tests`, and a
fresh full suite passed `428 tests` with the same three warnings. Round 2 tests
verify separate data/connectivity/validity parameter-space guards, saved
threshold restoration before compatibility comparison, complete nested
objective fingerprinting, next-epoch cadence reset, and exact full-lattice
margin participation in the student data anchor. These tests do not
substitute for a real 96-cubed interruption/resume.

## Integrity Boundary

`--resume-run-state` remains distinct from model-only `--resume-from`.
Run-state compatibility, RNG restoration, atomic replacement, bounded cadence,
processed-prefix solver coverage, durable log reconciliation, and promotion
baseline identity have focused tests. The real smoke did not produce a state
file from which to demonstrate a live continuation.

The run used measured LBM evaluations and did not introduce a surrogate. No
solver component or evaluator was removed. The transient smoke reported
`lbm_converged=0`; its internal force values are optimization inputs only, not
external-PDE ground truth or publication claims.

Round-3 production-boundary regressions cover the complete optimizer parameter
groups, active measured guard transport, replay isolation, saved threshold and
cadence orchestration, and restoration of the LBM shape-drag configuration
fields. They remain test evidence only and do not establish live 96^3 resume
or promotion behavior.

The round-3 dedicated review suite passed `21 tests`; affected focused suites
passed `101 tests`; and the fresh full suite passed `431 tests` with three
existing environment warnings. No real smoke was run.

## Round-4 Automated Boundary

Round 4 corrected batch-position transport for active topology guards and made
`freeze_decoder_for_generated_paths` operate on the captured gradient buffers
that are restored before the optimizer step. Batch-size-2 tests cover first
inactive/later active, first active/later inactive, and distinct guards on the
two samples. Each case preserves two controlled base values, performs six
controlled objective calls for one SPSA direction, produces full `[B, ...]`
guard tensors with exact zeros at inactive positions, and replays the union of
active guards through `train_epoch`. Those round-4 fixtures did not vary the
samples' `DesignSpec` values, so they were not evidence of per-sample mission
weight integrity.

The converter-freeze production test observes gradients immediately before
the optimizer step. With the switch enabled, generated data, direct, and
topology converter entries are removed while the clean grounded converter
gradient remains nonzero. With the switch disabled, generated converter
entries remain. The switch is present in the exact-resume training-config
fingerprint.

Verification for `e6f9016`: dedicated round-4 selection `5 passed`; complete
review module `25 passed`; affected suites `97 passed`; full `pytest -q` `435
passed, 3 warnings in 311.63s`. The warnings are existing `pkg_resources`
deprecations. No real smoke was run, so current-code live 96-cubed resume and
promotion remain unverified.

## Round-5 Design-Spec Boundary

Round 5 carries the batch-aligned `DesignSpec` sequence preserved by collation
through `train_epoch` and `DirectSolverSPSALoss`. For sample index `i`, its
base, plus, and minus measured calls receive `design_specs[i]`. A lone
`DesignSpec` and a one-element sequence retain broadcast compatibility; any
other sequence length must equal the direct-objective batch size and is
rejected before a measured call.

The batch-size-2 controlled objective uses deliberately different
`space_weight`, `drag_weight`, and `lift_weight` values and records every
received spec. Direct-loss and production `train_epoch` tests both verify all
six calls, the mean of the two correctly weighted base objectives, inequality
from the known-wrong first-spec-for-both result, unchanged call accounting,
and finite gradients.

Verification for `aa701c0`: dedicated design-spec selection `5 passed`;
complete review module `30 passed`; affected suites `97 passed`; full
`pytest -q` `440 passed, 3 warnings in 341.45s`. No real smoke was run, so
current-code live 96-cubed resume and promotion remain unverified.
