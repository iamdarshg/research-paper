# Airshow Direct Solver-In-Loop SPSA Report, 2026-06-21

This report records the direct solver-in-loop training update requested after
the Airshow loss debugging pass. It is a local smoke-training report, not an
external aerodynamic validation result.

## Executive Summary

The old detached-monitor path was not enough: exact connectivity and raw
aerodynamic scores could be logged, but they did not change model weights. The
trainer now has a direct measured solver loss:

```text
optimization_loss =
  model_losses
  + direct_solver_loss_weight * direct_solver_loss
```

`direct_solver_loss` materializes generated voxel probabilities, converts them
to binary geometry, runs the internal D3Q27 lattice-Boltzmann evaluator, adds an
optional exact connected-component penalty, and uses a two-sided SPSA
finite-difference estimate for the backward signal. No learned stand-in model is
used in this path.

The strongest local result is the continued `96^3` run. Its scheduled measured
solver objective dropped from `180.29967` after the matched two-epoch run to
`3.87060` after one continuation epoch and `7.48613` after the second
continuation epoch. The rise from `3.87060` to `7.48613` is a warning sign that
this short protocol can start to drift even while optimizer loss decreases.

## Derivative Options Reviewed

The solver value can influence training only if the optimizer receives some
usable derivative signal. These are the options reviewed for the current code:

| Option | Status | Reason |
| --- | --- | --- |
| Native PyTorch autograd through the exact score | Not available | Binary thresholding, SciPy connected components, and Python solver scalars break the analytic graph. |
| Per-voxel central finite differences | Rejected for this sweep | A full grid would require too many solver calls per scheduled update. |
| SPSA finite differences | Implemented | Two measured perturbation evaluations give a black-box gradient estimate independent of voxel count. |
| Low-frequency SPSA | Implemented and used | Perturbing an `8^3` field and upsampling it reduces high-frequency noise on sparse voxel geometries. |
| Evolutionary or genetic updates | Implemented for candidates | Useful after generation, but it does not directly update model weights unless wrapped into training. |
| Adjoint CFD or differentiable LBM | Future work | This would be cleaner mathematically, but the repository does not currently have an adjoint solver path. |
| OpenFOAM PDE labels | Validation route | OpenFOAM is the external comparison foundation, not the label source for this direct training sweep. |

The implemented path is therefore a measured black-box optimizer: it runs the
solver and estimates how to push probabilities using two-sided perturbations.
It is not analytic CFD backpropagation and it is not external PDE ground truth.

## Implementation Notes

Main code path:

- `CLI/aircraft_diffusion_cfd.py`
- `DirectSolverSPSALoss`
- `DirectSolverSPSAFunction`
- `_direct_measured_objective_for_single`
- `_binarize_probability_grid_for_solver`

Important configuration fields:

- `direct_solver_loss_weight`
- `direct_solver_interval`
- `direct_solver_steps`
- `direct_solver_perturbation`
- `direct_solver_perturbation_grid_size`
- `direct_solver_gradient_clip`
- `direct_connectivity_weight`
- `direct_solver_target_occupancy`

Metric semantics:

- `optimization_loss`: scalar used for backpropagation.
- `direct_solver_loss`: direct solver term averaged over every batch, including
  zero values on unscheduled batches.
- `direct_solver_eval_loss`: direct solver term averaged only over scheduled
  solver evaluations.
- `connectivity` and `aerodynamic`: detached monitors. They can be set to zero
  interval so compute is spent on the optimizer-facing direct solver term.

## Calibration Attempts

Several attempts were run before the final setting. Keeping them in the report
matters because the failures explain why the current hyperparameters are not
arbitrary.

| Attempt | Setting | Result | Interpretation |
| --- | --- | --- | --- |
| Raw threshold | 0.5 threshold, no target occupancy | Solver objective stayed nearly constant. | Scheduled geometries were empty or effectively empty. |
| Top-k 5% | `direct_solver_target_occupancy=0.05` | `32^3` scheduled eval worsened from `53.88` to `208.80`. | 5% materialization was too dense for the sparse Airshow corpus. |
| Top-k 1%, weak direct term | target 1%, lower weight/clip variants | Direct eval stayed high or worsened. | The direct term was present but not steering enough. |
| Strong low-frequency SPSA | target 1%, weight 0.2, `8^3` perturbations | 32/64 improved; 96 needed continuation. | This became the reported direct solver-in-loop setting. |

Corpus occupancy measurements used for calibration:

| Grid | Mean occupancy | Median occupancy |
| ---: | ---: | ---: |
| `32^3` | `0.01495` | `0.01202` |
| `64^3` | `0.00802` | `0.00655` |
| `96^3` | `0.00565` | `0.00458` |

## Final Run Configuration

The reported direct solver sweep used:

```text
--batch-size 1
--coordinate-decoder-threshold 1
--coordinate-training-samples 4096
--coordinate-positive-fraction 0.5
--full-diagnostic-interval 0
--direct-solver-loss-weight 0.2
--direct-solver-interval 32
--direct-solver-steps 3
--direct-solver-perturbation 0.25
--direct-solver-perturbation-grid-size 8
--direct-solver-gradient-clip 10.0
--direct-connectivity-weight 5.0
--direct-solver-target-occupancy 0.01
--connectivity-monitor-interval 0
--aerodynamic-monitor-interval 0
--solver D3Q27
--disable-pipeline
--enable-checkpointing
```

Metrics paths:

- `build/airshow_direct_solver_strong_grid_sweep_20260621/g32/checkpoints/training_metrics.json`
- `build/airshow_direct_solver_strong_grid_sweep_20260621/g64/checkpoints/training_metrics.json`
- `build/airshow_direct_solver_strong_grid_sweep_20260621/g96/checkpoints/training_metrics.json`
- `build/airshow_direct_solver_strong_grid_sweep_20260621/g96_more/checkpoints/training_metrics.json`

## Results

| Run | Optimizer loss | Geometry BCE | Direct solver loss | Scheduled solver eval | Scheduled eval count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `32^3`, epoch 2 | `2.101655` | `0.082671` | `5.830280` | `172.479119` | 12 |
| `64^3`, epoch 2 | `1.113592` | `0.050385` | `1.302243` | `38.524679` | 12 |
| `96^3`, epoch 2 | `2.059842` | `0.040661` | `6.094637` | `180.299673` | 12 |
| `96^3`, continuation epoch 1 | `0.683055` | `0.038023` | `0.130837` | `3.870600` | 12 |
| `96^3`, continuation epoch 2 | `0.606047` | `0.037385` | `0.253052` | `7.486129` | 12 |

The matched two-epoch comparison was not monotone for the direct solver
objective: `96^3` was worse than `64^3`. The continuation run was therefore
necessary. After continuation, the final `96^3` scheduled solver evaluation was
lower than both `32^3` and `64^3`, and geometry BCE also decreased with grid
size.

The best direct measured value occurred at the first continuation epoch
(`3.870600`), while the final continuation checkpoint had lower optimizer loss
but higher scheduled solver eval (`7.486129`). That means the run reached the
early edge of overfitting or estimator drift under the short local protocol.

## Ground-Truth Boundary

Grounded facts in this report:

- The corpus records are public Airshow geometry records admitted by license and
  geometry filters.
- The listed metrics come from local `training_metrics.json` files produced by
  the commands above.
- The direct solver loss calls the repository's internal D3Q27 evaluator during
  training.

Limits:

- Internal D3Q27 LBM values are internal-label evidence.
- SPSA estimates a useful black-box backward signal; it is not an analytic CFD
  adjoint.
- The sweep does not use OpenFOAM labels.
- The sweep does not prove generated aircraft validity, structural viability, or
  externally validated aerodynamic optimality.

## Paper Changes Made From This Report

The methodology now describes `L_solver-SPSA` as a scheduled direct measured
solver objective inside `optimization_loss`.

The results now include a direct solver grid-sweep table and a regenerated
grid-loss figure. The caption states that the plotted `96^3` point uses a
continuation run and that the right panel is an internal D3Q27 solver objective,
not external PDE validation.

The conclusion now says the repository has a direct internal-solver training
term with black-box SPSA gradients, while OpenFOAM remains the external PDE
validation foundation.

## Next Work

1. Save per-epoch continuation checkpoints when exploring overfit onset so the
   best direct-solver epoch can be selected without rerunning.
2. Run repeated seeds for `32^3`, `64^3`, and `96^3` because one local sweep is
   not enough for a claim about resolution scaling.
3. Add OpenFOAM comparison labels for a small fixed validation set and keep them
   separate from internal D3Q27 labels until the external-PDE gates pass.
4. Measure whether lower direct solver objective improves downstream
   aircraft-validity gates; current solver-loss improvement alone does not prove
   valid aircraft generation.
