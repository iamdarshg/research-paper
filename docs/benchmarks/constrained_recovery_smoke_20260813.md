# Constrained Recovery Smoke Evidence

Status: `DONE_WITH_CONCERNS`

Round-2 status: the implementation fixes from `432c8a9` passed the fresh full
suite. This document's real-smoke artifacts predate both review-fix rounds;
no new real smoke was run by instruction.

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
