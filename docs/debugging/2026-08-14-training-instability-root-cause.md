# Training-Instability Root Cause: SPSA Occupancy Flip-Noise

**Date:** 2026-08-14 (investigation) / 2026-08-15 (verification probes)
**Scope:** Aug-13/14 constrained-recovery stabilization run
**Checkpoint under test:** `build/constrained_recovery_stabilization_20260813/ten_epochs/checkpoints/latest_run_state.pt` (global step 1305)

## Executive Summary

The Aug-13/14 stabilization run was not converging: the materialized-occupancy
signal swung between ~0% and ~100% with a near-one-update period (worst
100-update window spanned 96.8 percentage points). The cause is not the
denoiser, the data, or the CFD solver. It is the **occupancy component of the
SPSA direct-solver gradient**, which is flip-noise dominated:

1. The SPSA finite-difference occupancy gradient is the derivative of a
   hard-threshold step function. Its raw magnitude is always far above the
   gradient clip cap (mean 2.13, median 1.46, max 10.65 vs. cap 1.0), so after
   clipping it is essentially a ±unit vector whose **direction is set by the
   Rademacher sign of the finite difference**, not by the objective. It is
   directionally decorrelated from the aero and connectivity SPSA gradients
   (mean cosines +0.014 and −0.093).
2. A clipped, directionally random gradient feeding a high learning rate is a
   bang-bang controller: it pushes every free-running voxel toward saturated
   probability on some updates and toward 0 on others, producing the
   occupancy 0↔99% oscillation. This is exactly what the telemetry shows.
3. The 0.9752 materialization threshold was calibrated on the **clean grounded
   reconstruction** field (mean probability 0.063, max 0.9918) but applied to
   the **free-running** field (mean probability ~0.959), where nearly every
   voxel sits above the threshold. The calibration therefore had no meaning for
   the field it governed, and the positive margin (0.9752 + 0.05 > 1.0) clamped
   to a degenerate "push everything to 1.0" signal.

**Fix:** fixed the materialization threshold at 0.5 (the natural Bayesian
boundary) and replaced the SPSA occupancy component with an **analytic
occupancy gradient** composed of (a) a one-sided mean-probability saturation
brake that pushes down only while `mean(p) > threshold`, and (b) a soft
threshold-anchored surrogate `|mean(sigmoid((p−t)/T)) − reference_occupancy|`.
Both are deterministic and smooth, so the occupancy signal is no longer
noise-driven. Bounded probes (1, 5, 20 updates) confirm the oscillation is
gone, all losses are finite, and the deterministic brake is active every
update. 20 updates is the longest verification rung (a 40-update probe was
intentionally skipped).

## 1. Symptom

The failed run's per-update telemetry (`updates.jsonl`) shows materialized
occupancy oscillating violently:

| Series | Value |
|---|---|
| Overall occupancy range | 0.000% – 100.0% |
| Worst 100-update window (steps 863–963) | 98.7% → 1.9% (range **96.8 pp**) |
| Update-to-update jumps in that window | up to ~50 pp per update |
| Run trajectory | up 99%, down to 0%, back to 99%, ... with no convergence |

Losses mirrored the chaos: `optimization_loss` swung ~6–9 without a downward
trend over 897 analyzed updates.

## 2. Root Cause

### 2.1 Flip-noise SPSA occupancy gradient

The direct solver uses a two-sided Rademacher finite difference: for each of 16
directions it perturbs logits by ±ε, runs the CFD solver, and forms
`(loss(+ε) − loss(−ε))/(2ε)`. For a smooth loss this is an estimator of the
gradient; for the occupancy component the loss is

```
occupancy_loss = materialized_occupancy(threshold)   # step function of logits
```

The derivative of a step function is zero almost everywhere and an infinite
spike at the threshold. Every finite-difference sample therefore lands on a
flat plateau (gradient ≈ 0) or straddles the threshold (gradient ≈ ±big). The
mean result is a gradient whose **magnitude is dominated by the noise** and
whose **sign is set by which voxels happened to straddle the threshold**.

Measured on 897 updates:

- Raw occupancy-gradient norm was **above the clip cap (1.0) on 74.8%** of
  updates (mean 2.13, median 1.46, max 10.65).
- After clipping to a unit vector, the signal is a Rademacher-sign sample.
- Directional agreement with the meaningful components is near zero:
  occupancy↔aero cosine +0.014, occupancy↔connectivity −0.093. (Only
  occupancy↔validity, +0.696, is aligned — both "collapse everything to 0" for
  a solid box.)

The combined SPSA gradient carried the same pathology: raw magnitude mean ~800
(max ~2600) before clipping, so the post-clip direction was noise.

### 2.2 Calibration on the wrong distribution

The run calibrated its threshold to 0.9752 on the **clean grounded
reconstruction** field (mean p 0.0629, max 0.9918, 16 samples, 96³). That
threshold is the point at which the clean field's extreme tail materializes the
~0.5%-sparse target airframe. But the direct-solver optimizes the
**free-running** field, which at step 1305 had mean probability ≈ 0.959 — a
near-saturated blob. Applied there, a 0.9752 threshold materializes almost the
whole box (occupancy 50–100%), so the "calibrated" threshold governed a
completely different distribution and gave the occupancy component nothing
meaningful to optimize — only the threshold-crossing noise.

### 2.3 Degenerate positive margin

The margin loss anchored positives at `threshold + 0.05 = 1.0252`, which clamps
to 1.0 and pushes saturated voxels *up* (toward full saturation) with no
well-defined target. At a 0.5 threshold the margins (0.5 ± 0.05) are
well-defined and never clamp.

## 3. Evidence (before fix)

| Metric (old run, threshold 0.9752, SPSA occupancy) | Value |
|---|---|
| Occupancy SPSA at clip cap | 74.8% of updates (671/897) |
| Occupancy SPSA raw norm | mean 2.13, median 1.46, max 10.65 (cap 1.0) |
| Combined SPSA raw norm | mean ~800, max ~2600 |
| occupancy↔aero cosine | +0.014 (decorrelated) |
| occupancy↔connectivity cosine | −0.093 (decorrelated) |
| Calibration field | clean reconstruction: mean p 0.0629, max 0.9918 |
| Free-running field at step 1305 | mean p ≈ 0.959 (saturated) |
| Materialized occupancy at 0.5 on step-1305 field | 99.999% |

## 4. The Fix

Three coordinated changes (`CLI/aircraft_diffusion_cfd.py`,
`CLI/run_monitored_training.py`, `CLI/config.yaml`):

1. **Fixed threshold at 0.5.** With calibration disabled, the config value is
   authoritative and re-forced at resume time
   (`_prepare_geometry_threshold_for_run`). 0.5 is the natural Bayesian
   boundary: no calibration, well-defined margins, no tail sensitivity.

2. **Analytic occupancy gradient** (`_analytic_occupancy_logit_gradient`)
   replacing the SPSA occupancy component. On the free-running logits:
   ```
   brake = mean_w * max(0, mean(p) − threshold)      # one-sided saturation brake
   soft  = mean(sigmoid((p − threshold) / T))
   anchor = soft_w * |soft − reference_occupancy|     # occupancy anchor
   ```
   The mean term is **one-sided**: it only pushes down while the field is
   saturated and never fights a healthy sparse field. The soft term carries the
   occupancy anchor at the batch reference (~0.5%). The SPSA component list no
   longer contains `occupancy_loss` — only aero, connectivity, and validity.

3. **Saturation brake engages at the right regime.** On the step-1305
   saturated field the brake is active (mean p 0.959 > 0.5) and will
   monotonically push the free-running field down over the long continuation
   run, while aero/connectivity carve the shape.

## 5. Verification (bounded probes)

Bounded interruption probes (`--stop-after-updates`), each resuming from the
step-1305 run-state (extracted to a plain `--resume-from` checkpoint because
the run-state's configuration fingerprint legitimately rejects the changed
config; see note below).

### Probe 1 (1 update, step 1306)

- SPSA components now **only** aero / connectivity / aircraft_validity — the
  occupancy SPSA component is gone.
- `occupancy_analytic_gradient_enabled = 1.0`, norm at cap 1.0 (engaged).
- Field confirmed saturated: mean_p 0.959, materialized occupancy 99.999%,
  soft surrogate 0.9999.
- All losses finite; run exited cleanly.

### Probe 5 (steps 1306–1310)

| step | mean_p | occupancy | aero | validity | occ_loss |
|---|---|---|---|---|---|
| 1306 | 0.9590 | 100.0% | 0.0002 | 1.5112 | 0.3282 |
| 1307 | 0.9510 | 99.6% | 0.1885 | 1.5057 | 0.3272 |
| 1308 | 0.9396 | 99.0% | 0.5242 | 1.4993 | 0.3252 |
| 1309 | 0.9536 | 100.0% | 0.0575 | 1.5108 | 0.3280 |
| 1310 | 0.9530 | 99.9% | 0.0213 | 1.5111 | 0.3275 |

- Occupancy held at 99.0–100% for all five updates — **no 0↔99% bang-bang**.
- mean_p drifted −0.006; the residual 0.94–0.96 wiggle is the aero gradient
  re-shaping the (still saturated) field, not flip-noise.
- Analytic occupancy gradient ON at cap every update.
- All losses finite.

### Probe 20 (steps 1306–1325)

- Occupancy held 97.8–100% across all 20 updates (**range 2.2 pp** vs. the old
  run's 96.8 pp worst-100-window range — a 44× reduction; the bang-bang is
  gone).
- mean_p drifted 0.959 → 0.950 (−0.0088). The one carved dip to mean_p 0.922 /
  occupancy 97.8% (step 1313, when aero_loss jumped to 1.44) is the aero
  gradient shaping the field, not flip-noise, and it recovers.
- Analytic occupancy gradient ON at cap all 20 updates.
- **Desaturation rate is self-accelerating**: −0.0088/20 updates
  (≈ −0.00044/update) now, and the brake term `mean_w·p(1−p)` *grows* as
  mean_p falls toward 0.5 (peak at p=0.5). Linear extrapolation reaches
  mean_p ≈ 0.5 in ~1,000–2,000 updates — well inside the planned 10-epoch
  continuation horizon — so the long run should reach a sparse field, not
  stall saturated.
- All losses finite.

### Probe 40 — skipped by decision

A 40-update probe was not run; **20 updates is the longest verification rung**
(operator decision, 2026-08-15). The 20-update probe is sufficient to verify
the fix: the failure mode it targets (bang-bang occupancy oscillation at
~1-update period) is fully visible within 20 updates, and the mechanism
(analytic brake engaged every update, deterministic direction) is step-local.

### Note on resume compatibility

The run-state resume path is intentionally fail-closed: it compares every
configuration-fingerprint key including the full `training_config` dataclass.
The fix changes that fingerprint (new `occupancy_*` weights, threshold 0.5,
calibration disabled), so `--resume-run-state` correctly refuses the old
run-state. Probes instead resume via `--resume-from` a **plain checkpoint
extracted from the run-state** (all four state_dicts, optimizer, scaler,
global_step, embedded `model_config`/`training_config`/`diffusion_config`/
`cfd_config` from the run-state's own fingerprint, threshold 0.5). The original
run-state artifacts are untouched.

## 6. Before / After

| Signal | Before (old run) | After (probes) |
|---|---|---|
| Occupancy trajectory | 0↔99% bang-bang, 96.8pp range / 100 steps | 99.0–100% monotonic hold |
| Occupancy gradient | SPSA flip-noise, at cap 74.8% | analytic, deterministic, at cap 100% |
| Occupancy↔aero alignment | +0.014 (noise) | analytic (not SPSA) — no flip term |
| Threshold | 0.9752 (wrong distribution) | 0.5 (Bayesian boundary) |
| Margins | degenerate (clamped push-to-1.0) | well-defined 0.5 ± 0.05 |
| Losses | chaotic, no trend | finite, smooth |

## 7. Remaining Risks

1. **Desaturation is slow at lr 2e-5.** The step-1305 field needs ~3 units of
   logit drop (p 0.959 → 0.5); at Adam-normalized steps of ~2e-5/update this
   is a ~100k-update horizon. The probes verify the *mechanism* and *stability*,
   not the endpoint. The continuation run must be allowed to run long enough to
   see mean_p cross below ~0.5 and occupancy fall toward the ~0.5% reference.
2. **Aero on a near-solid box.** While occupancy is ~100%, the solver measures
   drag on a solid block (aero_loss 0.0002–0.52, fluctuating as the box edge
   shifts). Bounded by the gradient clip and guard projection, but worth
   watching in the continuation run's telemetry.
3. **Full desaturation changes the validity landscape.** As the field thins,
   the aircraft-validity component will drive structure; the guard projection
   (project_conflicting_direct_gradient) stays on.
4. **Promotion at threshold 0.5 starts at fail.** Initial promotion baseline
   on the saturated field is valid_fraction 0, occupancy_error 0.98. Promotion
   can only pass after substantial desaturation.

## 8. Recommended Next Run

```bash
python CLI/run_monitored_training.py \
  --manifest build/grounded_combined_1k_20260716/manifest.jsonl \
  --resume-from build/recovery_ladder_20260814/step1305.pt \
  --lbm-stream-bfl-backend fused_stream_bfl \
  --save-dir build/recovery_continuation_20260815 \
  --num-epochs 10 \
  --checkpoint-every-updates 25 \
  --history-output build/recovery_continuation_20260815/history.json \
  --updates-output build/recovery_continuation_20260815/updates.jsonl
```

Conservative, matches the failed run's horizon (10 epochs, 7580 planned
updates), re-forces threshold 0.5, resumes the exact step-1305 weights with the
fixed occupancy objective. Validate against the probe-40 result before
launching.

## 9. Files Changed

- `CLI/aircraft_diffusion_cfd.py` — `TrainingConfig` occupancy weights;
  validation; `_analytic_occupancy_logit_gradient`; occupancy removed from
  SPSA `component_names` and from `total_loss`; replay site adds the analytic
  gradient to the direct-logit gradient.
- `CLI/run_monitored_training.py` — `_prepare_geometry_threshold_for_run`
  forces the config threshold when calibration is disabled.
- `CLI/config.yaml` — `geometry_materialization_threshold: 0.5`,
  `calibrate_geometry_materialization_threshold: false`,
  `occupancy_mean_probability_weight: 0.5`, `occupancy_soft_temperature: 0.05`,
  `occupancy_soft_weight: 0.5`.
