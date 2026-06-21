# Airshow Training Loss Debug Report, 2026-06-20

This report records the debugging pass requested after the Airshow reruns showed
zero or unstable aerodynamic and connectivity losses. It is a training-semantics
report, not a new claim-bearing validation result.

## Root Cause Summary

Two separate issues were found.

1. Resumed training from the completed `32^3` checkpoint could silently inherit
   a zero optimizer learning rate from the old cosine scheduler state. That made
   the first source-valid fine-tunes run through all batches while leaving model
   weights byte-identical to the starting checkpoint.
2. The connectivity and aerodynamic terms in `CLI/aircraft_diffusion_cfd.py`
   are detached diagnostics in the current implementation. They can be logged
   and used for post-hoc ranking, but they do not provide gradients to the
   diffusion model.

The second point explains why treating the large aerodynamic scalar as part of
the main training loss was misleading. It changed the reported scalar, not the
actual direction of the optimizer update.

## Gradient Check

A local gradient probe was run against `ConnectivityLoss` and `AerodynamicLoss`
using a voxel tensor with `requires_grad=True`.

| Term | Observed value source | `requires_grad` | Backward behavior |
| --- | --- | --- | --- |
| Connectivity | thresholded tensor, SciPy connected-components label | false | backward raises: no grad function |
| Aerodynamic | thresholded tensor, internal solver output, wrapped scalar coefficients | false | backward raises: no grad function |

The implementation cause is direct:

- `ConnectivityLoss.forward()` thresholds the voxel grid, moves it through
  NumPy, runs `scipy.ndimage.label`, and returns a fresh scalar tensor.
- `AerodynamicLoss.forward()` thresholds the voxel grid, invokes the solver,
  consumes Python/float coefficient outputs, and wraps those values back into a
  scalar tensor.
- The training loop only computes the aerodynamic diagnostic every ten batches,
  so per-batch progress output showing `aero=0` on most batches is expected.

## Code Change

The trainer now separates the scalar used for backpropagation from the detached
diagnostic total:

\[
\mathcal{L}_{opt} =
\mathcal{L}_{mse}
+ \lambda_{geom}\mathcal{L}_{geom}
+ \lambda_{gen}\mathcal{L}_{gen}
+ \mathcal{L}_{cons}
\]

\[
\mathcal{D}_{total} =
\mathcal{L}_{opt}
+ \mathcal{D}_{conn}
+ \mathcal{D}_{aero}
\]

`loss` and `optimization_loss` now refer to the backpropagated value.
`diagnostic_total`, `connectivity`, and `aerodynamic` remain available for
monitoring and reports. This keeps the training log from implying that raw CFD
or connected-component labeling is a differentiable teaching signal.

The checkpoint resume path was also adjusted: when a loaded optimizer has zero
learning rate and the new run config provides a positive learning rate, the
optimizer LR is restored and the old completed scheduler state is not reloaded.

## Source-Valid Corpus Filter

The `32^3` Airshow corpus was filtered using the same first-pass aircraft
validity checks used for generated outputs.

| Item | Count |
| --- | ---: |
| Source records evaluated | 355 |
| Records kept | 176 |
| Records rejected | 179 |
| Filtered split counts | train 120, val 19, test 16, holdout 21 |
| Filtered manifest validation | pass |
| Filtered manifest SHA-256 | `0d149d981730871fbab792bd28a42248e529545d03f531fc8b829f0d21d25ccc` |

Failed source-check counts were:

| Check | Failed records |
| --- | ---: |
| symmetry | 117 |
| span_sanity | 70 |
| nonempty_occupancy | 22 |
| body_centerline_dominance | 13 |
| wing_body_balance | 12 |
| longitudinal_profile_variation | 8 |
| tail_body_plausibility | 6 |

This supports filtering the corpus for a stricter training probe, but it does
not convert the dataset into aerodynamic or structural ground truth.

## Source-Valid Fine-Tune Results

The first source-valid fine-tune before the LR fix completed but produced
byte-identical model states. After the LR restoration fix, a three-epoch
source-valid fine-tune from the `32^3` checkpoint produced this final
checkpoint:

- checkpoint:
  `build\airshow_training_20260620_g32_source_valid_lrfix\checkpoints\final_optimized_model.pt`
- checkpoint SHA-256:
  `657243eaeac0dfda9e7ca250770e860e73ad82834f591061d3609fb62145574c`

Observed epoch metrics:

| Epoch | Loss | MSE | Geometry | Generation reconstruction | Consistency | Connectivity diagnostic | Aerodynamic diagnostic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.41197 | 0.81644 | 0.04150 | 0.04655 | 0.01352 | 0.34075 | 40.15320 |
| 2 | 50.89676 | 0.76414 | 0.04170 | 0.04305 | 0.01020 | 0.05855 | 49.97912 |
| 3 | 62.10584 | 0.69973 | 0.03861 | 0.04022 | 0.01034 | 0.03569 | 61.28125 |

Those historical epoch losses include the old diagnostic total. Under the
patched trainer, the optimizer loss for epoch 3 would be the differentiable
components only: approximately `0.78890`.

The generated flight-path smoke tests from this checkpoint all remained
non-claim-bearing, but they improved relative to the one-epoch `32^3` result:

| Case | Occupancy | Failed checks | Length fraction x | Symmetry score | Raw D3Q27 Cd |
| --- | ---: | --- | ---: | ---: | ---: |
| short_takeoff_payload | 0.006226 | span_sanity | 0.25000 | 0.80392 | 0.58408 |
| high_speed_sprint | 0.005554 | span_sanity | 0.25000 | 0.69231 | 0.57352 |
| endurance_turning | 0.006104 | span_sanity | 0.28125 | 0.68000 | 0.44086 |

All three samples passed occupancy and symmetry checks, but all three remained
too short under the current `span_sanity` gate.

## Continuation Probe

An additional three-epoch continuation from the LR-fixed source-valid checkpoint
regressed generated validity:

- checkpoint SHA-256:
  `3ebac4d91d3222f2a8ccc4106db9564998a7dc2d465806245f0f675b67d834ac`

| Case | Occupancy | Failed checks | Length fraction x | Symmetry score | Raw D3Q27 Cd |
| --- | ---: | --- | ---: | ---: | ---: |
| short_takeoff_payload | 0.005798 | span_sanity | 0.15625 | 0.55789 | 0.55761 |
| high_speed_sprint | 0.005432 | symmetry, span_sanity | 0.21875 | 0.50562 | 0.56896 |
| endurance_turning | 0.005402 | symmetry, span_sanity | 0.15625 | 0.49153 | 0.71079 |

This argues against simply increasing epoch count on the current objective.
The generated samples are close enough to exercise the pipeline, but the
validity failures are systematic rather than random noise.

## Sequential Objective Optimizer Follow-Up

The repository now includes a sequential measured-objective optimizer in
`CLI/sequential_diagnostic_optimizer.py`, and
`CLI/run_airshow_flight_path_tests.py` can apply it before exporting each
generated flight-path artifact. The optimizer turns connectivity, validity, and
internal CFD scores into real candidate-selection losses:

```text
L_seq =
  w_conn * D_conn
  + w_aero * D_aero
  + w_valid * D_valid
  + w_occ * D_occ
```

Two methods are available:

- `genetic`: sequentially evaluates a small population, keeps elites, and
  mutates the next generation.
- `spsa`: uses two measured perturbation evaluations to estimate a gradient-like
  update direction in voxel-probability space.

Both modes evaluate one candidate at a time. This makes the diagnostic scores
real black-box optimization objectives, but it does not make the internal solver
or connected-component labeling differentiable PyTorch operations. Training
still backpropagates only through `L_opt`; the sequential objective optimizer
acts on generated candidates during evaluation/export.

Example command:

```powershell
python CLI\run_airshow_flight_path_tests.py `
  --checkpoint build\airshow_training_20260620_g32_source_valid_lrfix\checkpoints\final_optimized_model.pt `
  --manifest build\airshow_grounded_corpus_20260620_g32_source_valid\manifest.jsonl `
  --output-dir build\airshow_training_20260620_g32_source_valid_lrfix\flight_path_tests_seqopt `
  --grid-size 32 `
  --num-steps 4 `
  --cfd-steps 100 `
  --objective-optimizer genetic `
  --objective-population-size 4 `
  --objective-generations 2 `
  --cpu
```

The runner records the pre-optimization voxels, final selected voxels, objective
history, candidate counts, and measured best/initial losses in
`flight_path_results.json`.

## Recommended Next Approach

The next credible route is now to run and report the sequential optimizer rather
than pretending that raw solver scores are differentiable:

1. Generate multiple candidate voxel grids per design condition.
2. Run the aircraft-validity screen and internal solver on each candidate.
3. Rank or filter candidates using explicit logged criteria.
4. Train either a differentiable surrogate on the solver-labeled candidates or
   use a black-box optimization loop that records candidate scores and accepted
   updates.

Until the sequential loop is run at benchmark scale and shows reliable
candidate improvement, the paper should say that the repository has a
CFD-oriented scoring path plus an experimental black-box candidate optimizer,
not demonstrated CFD-guided gradient training.

## 2026-06-21 `96^3` Coordinate-Decoder Addendum

The `96^3` training follow-up uses a coordinate decoder rather than the dense
latent-to-voxel layer. To keep the sparse public Airshow objective faithful
while avoiding a full `96^3` decode on every batch, the high-resolution branch
now samples voxel coordinates and applies importance-weighted BCE. This means
occupied coordinates can be oversampled for signal without changing the
effective full-grid reconstruction objective.

Two failed probes are recorded because they explain the final setting:

- uniform coordinate sampling completed but drove generated fixed-threshold
  grids toward near-empty occupancies;
- unweighted 50% occupied-coordinate sampling completed but biased the decoder
  toward overly dense grids.

The final three-epoch `96^3` run used importance weighting and produced
checkpoint hash
`1bfdcfcf844010a0a5af463662bed94c7462748add21f0d337234b41d59774d3`.
Its optimizer loss decreased across epochs, but full-grid aero/connectivity
diagnostics were intentionally disabled during training to keep the run
sequential and memory-bounded. That zero diagnostic value should be read as
`not evaluated on this batch`, not as evidence of perfect aerodynamics or
connectivity.
