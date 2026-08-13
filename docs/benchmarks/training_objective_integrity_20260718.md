# Training Objective Integrity Audit (2026-07-18)

## Question

Why did scalar loss decrease while generated geometries became less
aircraft-like?

## Answer

The old training and promotion paths did not evaluate the same object.
Training evaluated a target-conditioned one-step denoising result, while
promotion generated a four-step sample from random noise. Both paths then used
the paired target's occupied-voxel count to select the prediction's top-ranked
voxels. That top-k operation hid the generator's actual occupancy distribution.

The preserved checkpoint is not a known-good aircraft generator. Under a single
fixed threshold, its free-running outputs are mostly or entirely solid.
Therefore, previous decreasing losses must not be presented as evidence that
free-running aircraft quality improved.

## Measured Evidence

The rejected 120-update Stage B candidate had a lower training objective but
worse geometry promotion:

- generated aircraft valid fraction: `0.375 -> 0.3125`
- generated mean recall: `0.23370 -> 0.22715`
- generated worst recall: `0.15873 -> 0.14132`
- generated largest-component fraction: `0.99945 -> 0.99951`

The target-oracle audit found that the source checkpoint's probability field
could be converted into a target-sized mask even when its intrinsic occupancy
was wrong. A fixed global threshold was calibrated once from 16 clean grounded
reconstructions:

- threshold: `0.9752044081687927`
- grounded occupied fraction: `0.005089865997433662`
- materialized clean-reconstruction fraction: `0.005089724436402321`
- calibration voxels: `14,155,776`

With that threshold frozen, a four-sample, three-seed baseline measured:

- generated occupied fraction: mean `0.777802`, min `0.303192`, max `1.0`
- generated aircraft valid fraction: `0.0`
- normalization-boundary fraction: `0.535258`
- target occupied fraction: `0.005243`
- failed validity checks included occupancy envelope, span, planform sparsity,
  normalization margin, and tail/body plausibility

The standalone fixed-threshold evaluator confirmed the failure visually and
numerically. Seed 0 occupied all `884,736` voxels; seed 1 occupied `884,191`.
The evidence is in:

- `build/training_stabilization_20260718/intrinsic_geometry_visualization/evaluation.json`
- `build/training_stabilization_20260718/intrinsic_geometry_visualization/projections.png`

## Root Causes

1. **Target-oracle materialization.** Each output was top-k carved to the
   paired target's occupancy. Occupancy error was therefore approximately zero
   by construction, and identical probability fields could produce different
   hashes against different targets.
2. **Training/inference mismatch.** Direct CFD optimized a one-step
   target-conditioned denoising state. Promotion used the actual four-step
   free-running consistency path.
3. **Solver-axis mismatch.** Canonical model geometry is `[Z,Y,X]`; the D3Q27
   solver consumes `[X,Y,Z]`. The old direct path sent the tensor unchanged.
4. **Calibrated drag fallback.** Five-step LBM evaluations are unconverged.
   The old selector preferred a calibrated shape correction when the raw force
   was not converged, despite the direct objective being described as raw CFD.
5. **Validity dilution.** Averaging 12 violation scores reduced one complete
   gate failure to roughly `0.0833`.
6. **Gradient cancellation.** In Stage B, data and direct gradients had
   negative cosine on 60 of 120 updates, including 39 below `-0.5`.
7. **Incorrect zero-incidence lift objective.** The old `1 - |CL|` term
   rewarded large transient or asymmetric lift even though no angle-of-attack
   or load target was supplied.
8. **Promotion diversity inflation.** Seeds restarted for every target, while
   target-specific top-k counts changed the resulting hashes.

## Implemented Corrections

- One corpus-calibrated threshold is persisted in checkpoints and frozen for
  direct CFD, promotion, inference, export, and standalone evaluation.
- Target occupancy remains a scalar loss reference only. It never chooses
  which voxels exist.
- Direct CFD now evaluates a seeded four-step free-running sample. The same
  initial noise is replayed through all four differentiable steps after the
  sequential solver calls.
- Model `[Z,Y,X]` geometry is explicitly permuted to solver `[X,Y,Z]`.
- Direct optimization requires finite raw momentum-exchange coefficients and
  never consumes calibrated or surrogate drag. Signed drag is retained in
  telemetry; physical drag magnitude enters the loss.
- Zero-incidence absolute lift is penalized as a residual.
- Occupancy, raw aerodynamics, connectivity, and aircraft validity receive
  independent SPSA gradients from the same solver evaluations before their
  bounded gradients are combined.
- Aircraft validity uses mean plus worst violation, preserving hard failures.
- Conflicting direct model gradients are projected against the grounded-data
  gradient instead of being allowed to move uphill on the data objective.
- Promotion uses globally distinct seeds, zero condition vectors for the
  explicitly unconditioned corpus, and `canonicalize=False`.
- Promotion records intrinsic occupancy and rejects generated mean occupancy
  outside `[0.0005, 0.25]`.
- Incremental checkpoint ranking includes absolute occupancy error, so a model
  can be retained for meaningful progress before it reaches the final 50%
  aircraft-valid gate.
- Resumed runs retain the source checkpoint in place instead of writing an
  unnecessary duplicate. Checkpoint writes are atomic.

## Real 96-Cubed Smoke

The final one-update smoke used:

- `96^3` geometry
- fused stream/BFL backend
- 16 antithetic SPSA directions
- 33 sequential raw LBM calls
- all four measured components
- the actual four-step free-running generator

For the evaluated free-running sample:

- occupied fraction: `0.885310`
- occupancy loss: `0.289185`
- raw drag magnitude: `2.434944`
- signed raw drag: `-2.434944`
- zero-incidence lift residual: `3.754263`
- raw aerodynamic loss: `2.079981`
- aircraft-validity loss: `1.398357`
- connectivity loss: `0.00002681`
- total direct loss: `3.767550`
- occupancy SPSA norm before/after component bound: `1.223650 / 1.0`
- model-level direct gradient norm after bound: `0.25`
- data/direct cosine: `0.043117`
- wall time for the optimizer update: `65.9 s`

Evidence:

- `build/training_stabilization_20260718/free_running_cfd_smoke_v5/updates.jsonl`
- `build/training_stabilization_20260718/free_running_cfd_smoke_v5/history.json`

## Claim Boundary

The objective and gates now measure the free-running geometry honestly, but the
model has not yet produced a passing aircraft. The current five-step raw LBM
force is also transient: the smoke reported `lbm_converged=0`,
`force_stability=1.0`, and reversed signed drag. These values are valid
optimization inputs from the internal solver, not external-PDE ground truth and
not claim-bearing aerodynamic coefficients.

The next continuation should be described as recovery training, not final model
training. It should retain only candidates that improve the fixed validation
rank, and publication figures must come from the fixed-threshold evaluator.

## Constrained Aircraft Recovery Implementation (2026-08-13)

Tasks 1 through 4 are implemented on `codex/constrained-aircraft-recovery`.
The focused tests and the full suite passed before the final numerical guard
projection hardening; the post-hardening focused projection tests also passed.
The permitted real `96^3` smoke was bounded with the exact `D3Q27` /
`fused_stream_bfl` configuration and completed one optimizer update with 33
measured solver calls. It then stopped before the eighth-update checkpoint:
the first run exposed the intentional-interruption coverage contract, and the
retry exposed a float32 residual in measured guard projection. Those fixes are
committed, but the final 8-update smoke and exact-resume continuation were not
completed within the execution window. No promotion result is claimed.

This remains an internal recovery-training result. The smoke's five-step raw
LBM force was transient (`lbm_converged=0`) and is not external-PDE ground
truth or a claim-bearing aerodynamic coefficient.

The first review round (`f4e29c0`) added deterministic trajectory,
crash-reconciliation, promotion-baseline, compatibility, final-guard, and
calibrated-margin tests. Round 2 (`432c8a9`) addressed its then-scoped integration
findings with production-path tests for separate parameter-space topology
guards, saved-threshold compatibility, complete objective fingerprinting,
next-epoch cadence reset, and exact full-lattice margin capture in the student
data anchor. Fresh round-2 verification passed `428` tests with three
warnings. No additional real smoke was run in round 2, and no live 96^3
resume or promotion result is claimed.

Round 3 (`36d73f8`) repaired the production gradient lifecycle across the
diffusion, converter, and consistency-student optimizer groups. Production
`train_epoch` tests now verify that all required gradients survive branch
replay, only measured active topology guards are transported, each replay is
isolated, and the gradients at the actual optimizer step satisfy the active
guard invariant after clipping. The three shape-drag LBM configuration fields
were also restored as dataclass fields and covered by payload construction
tests. These are bounded CPU fixture results; no live 96^3 resume or promotion
claim is made. The round-3 dedicated production suite passed `21 tests`, the
affected focused suites passed `101 tests`, and the fresh full suite passed
`431 tests` with three existing warnings. These bounded tests do not replace
the unrun live smoke.

Round 4 (`e6f9016`) corrected two additional production lifecycle defects.
Measured topology guard gradients now remain aligned to original batch
positions, with exact zero tensors for inactive samples, and the ordered union
of active guard names cannot be overwritten by first-sample telemetry. The
generated-path decoder freeze now filters converter entries from captured
data, consistency, direct, and topology buffers before adding the separate
clean grounded converter gradient. Enabled and disabled behavior is tested at
the optimizer boundary, and the switch remains part of exact-resume
compatibility through the full training-config fingerprint.

Round-4 verification passed the dedicated selection (`5 tests`), complete
review module (`25 tests`), affected suites (`97 tests`), and full suite (`435
tests`, three existing warnings). This is automated implementation evidence,
not closure: no current-code live 96-cubed exact-resume run or promotion was
performed.
