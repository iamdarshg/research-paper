# A/B: boolean-mask gather → index_select (lever 2, commit f654b28)

Date: 2026-08-17
Branch: `experiment/kernel-fusion-launch`
Context: task #88 "land last ~6 s/u" off TF32 steady-state.

## Hypothesis

The ~3.5 s/update GPU-idle stall between backward and the optimizer burst was
caused by the backward of `flat_probabilities[row][flat_target[row]]` in
`sparse_voxel_reconstruction_loss` (`aircraft_diffusion_cfd.py:1237`). A boolean-
mask gather's backward is `IndexBackward0` → `torch.nonzero` (device→host sync) +
`_index_put_impl_`, run ~83×/update. Replacing it with `index_select` on
precomputed `nonzero` long indices removes the sync; values and gradients are
bit-identical (`IndexSelectBackward0` scatters on-device).

## Parity (already proven, commit f654b28)

- 28/28 pattern-level checks bit-identical (means + grads, fractions 0.001–0.999,
  empty/all-True masks).
- 8/8 full-function `sparse_voxel_reconstruction_loss` checks bit-identical
  (loss + grad, batch 1/4/8 + empty-row guard).
- 6/6 unit tests pass.

## Measurement protocol

Same harness on both sides (`measure_steady_su.py`, hooks `tqdm.update` to read
per-update wall time; JIT-warmup update excluded). `ALLOW_RAM_OVERRUN=1`,
OMP/MKL threads 12, TF32 matmul ON (both commits post-TF32).

| side | commit | steady-state per-update |
|---|---|---|
| pre-fix | 6d86818 (`f654b28^`, worktree `../maskab-prefix`) | 33.17 s/u (update 2) |
| post-fix | f654b28 (HEAD) | 31.37 / 30.12 s/u (avg ~30.7) |

## Result

Measured win ≈ **2.5 s/u** (30.7 vs 33.17). The 3.5s stall does not fully
disappear from the per-update number — some residual bubbles remain in the
telemetry/optimizer tail — but the dominant stall is gone.

## Decision

Keep the fix (already committed f654b28). Bit-identical, net wall-clock win,
no precision or numerics change.
