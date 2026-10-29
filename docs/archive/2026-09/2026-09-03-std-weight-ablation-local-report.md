# Local standard-deviation-weight ablation — 2026-09-03

## Decision

The appropriate local screen used a 16^3 model/grid, not the production 128^3
model. All 45 cells passed the finite-gradient and non-collapse gates under a
strict 1,200,000,000-byte per-process RSS guard. The mechanical minimum of mean
held-out balanced perplexity was `std_weight=0.5`, but its paired improvement
over `std_weight=0` was only `-0.000073336` PPL with an approximate five-pair
95% interval of `[-0.000567554, 0.000420881]`. That interval crosses zero.

Therefore the study does **not** justify changing the production coefficient
to 0.5. The conservative production recommendation is to keep the current
coefficient unchanged until a larger, production-path confirmation is run.
If a local-only default is required, `0.0` is preferable because it is tied on
PPL, has the highest mean Dice, and avoids paying for a term whose spread target
was not reached.

## Protocol

- Grid/model: dense 16^3, 32-dimensional latent, 96-wide two-layer decoder,
  409,792 parameters.
- Fixtures: 16 fixed synthetic aircraft-like training solids and 8 disjoint
  validation solids; data seed `20260910`.
- Paired repeats: five model-initialization/training-order seeds
  `20260911` through `20260915`. Every weight saw the same fixtures and the
  same initialization within a repeat.
- Weights: `0, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 4`.
- Optimization: 160 updates/cell, AdamW, learning rate `0.002`.
- Loss: the production `balanced_voxel_bce_with_logits`,
  `soft_dice_loss_with_logits`, and one-sided
  `geometry_probability_standard_deviation_loss(target_std=0.20)`.
- Selection: lowest mean final balanced perplexity among weights with 100%
  pass rate; Dice and probability span were secondary diagnostics.
- Resource safety: CPU execution and hard fail-closed RSS checks at cell start,
  model load, every update, and completion; maximum measured RSS was
  `668,400,000` bytes (~0.6684 GB).

## Aggregate results

Values are mean +/- sample standard deviation over five repeats.

| std weight | balanced PPL | soft Dice | probability std | probability span |
|---:|---:|---:|---:|---:|
| 0 | 1.046440 +/- 0.012077 | 0.802364 +/- 0.036897 | 0.156836 | 0.963679 |
| 0.01 | 1.046438 +/- 0.012075 | 0.802351 +/- 0.036898 | 0.156837 | 0.963667 |
| 0.05 | 1.046448 +/- 0.012066 | 0.802283 +/- 0.036859 | 0.156830 | 0.963633 |
| 0.1 | 1.046486 +/- 0.012023 | 0.802058 +/- 0.036689 | 0.156812 | 0.963559 |
| 0.25 | 1.046444 +/- 0.012007 | 0.801882 +/- 0.036717 | 0.156829 | 0.963436 |
| 0.5 | **1.046367 +/- 0.011831** | 0.801378 +/- 0.036607 | 0.156874 | 0.963280 |
| 1 | 1.046426 +/- 0.011542 | 0.799592 +/- 0.035857 | 0.156858 | 0.962968 |
| 2 | 1.046441 +/- 0.010969 | 0.792397 +/- 0.033471 | 0.157053 | 0.963892 |
| 4 | 1.046455 +/- 0.011808 | 0.780414 +/- 0.032984 | 0.157302 | 0.965182 |

## Paired comparison against weight 0

Mean PPL deltas are candidate minus the same-repeat baseline; negative is
better. Approximate 95% intervals use `t(4)=2.776` and are descriptive, not a
claim of a broad population estimate.

| std weight | mean delta PPL | approximate 95% interval | repeats better than baseline |
|---:|---:|---:|---:|
| 0.01 | -0.000002351 | [-0.000009607, 0.000004905] | 4/5 |
| 0.05 | +0.000008179 | [-0.000045421, 0.000061779] | 3/5 |
| 0.1 | +0.000045715 | [-0.000155794, 0.000247224] | 3/5 |
| 0.25 | +0.000004043 | [-0.000190868, 0.000198954] | 3/5 |
| 0.5 | -0.000073336 | [-0.000567554, 0.000420881] | 3/5 |
| 1 | -0.000014452 | [-0.001072777, 0.001043873] | 3/5 |
| 2 | +0.000000544 | [-0.003211898, 0.003212986] | 1/5 |
| 4 | +0.000014964 | [-0.006394508, 0.006424436] | 2/5 |

## Interpretation and boundary

The observed validation probability standard deviation remained near 0.157,
below the target 0.20, across all weights. Increasing the coefficient did not
meaningfully produce a more deviated field. At weights 1–4 it reduced Dice,
with weight 4 about 0.022 below baseline Dice on average. The tiny PPL
differences are therefore not evidence that the coefficient solves the
collapse problem.

This is still a local 16^3 diagnostic with synthetic fixtures. It does not
validate the 128^3 production model, the cloud gradient path, or the real
aircraft corpus. The production checkpoint is 1.81 GB on disk and its CPU
deserialization peaked around 3.35 GB RSS in the attempted full-resolution
study, so that study was correctly excluded from the 1.2 GB-capped local
screen rather than misrepresented as capped evidence.

## Artifacts

- Harness: `CLI/ablate_std_weight_tiny_repeated.py`
- Environment/provenance: `build/ablation_std_weight_tiny_repeated_20260903/environment.json`
- Full machine-readable results: `build/ablation_std_weight_tiny_repeated_20260903/results.json`
- Summary CSV: `build/ablation_std_weight_tiny_repeated_20260903/summary.csv`
