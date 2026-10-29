# 16^3 mHC collapse comparison — 2026-09-03

## Outcome

The requested 16^3 stress comparison completed for six decoder variants over
three seeds. At the selected stress learning rate (`0.002`), every arm
collapsed under the joint production-style gate:

```
probability_std <= 1e-4 AND probability_span <= 1e-3
```

The plain mHC-style probe therefore prevented **0/3** collapses (0 percentage
point reduction versus baseline). The RMS-normalized mHC-style probe also
prevented **0/3**; it collapsed at update 8, 8, and 1. Pre-normalization and
RMS-plus-gate collapsed on update 1 for all three seeds. This experiment does
not support promoting any of these probes to the 128^3 training run.

## Exact scope

- Lattice: `16 x 16 x 16`.
- Fixtures: one deterministic aircraft-like fixture per case. This was
  required after the full-width mHC probe exceeded the local 8 GiB GPU when
  four fixtures were processed together.
- Baseline architecture: the selected 128^3 architecture shape, including
  latent width 512, coordinate decoder width 2048, depth 5, Fourier bands 6,
  and coordinate chunk 4096.
- Baseline parameter count: `43,096,065`.
- mHC-style parameter count: `43,096,390`.
- Optimizer: AdamW, learning rate `0.002`, weight decay `0.01`, gradient norm
  clipping at `1.0`.
- Loss: repository balanced voxel BCE + soft Dice + standard-deviation term
  (`target=0.20`, weight `0.25`).
- Seeds: `20260911`, `20260912`, `20260913`.
- Ceiling: 128 updates per case, stopping at the first observed collapse.
- Checkpoints: none written. The isolated result directory contains only 18
  small JSON result files.

## Results

| Variant | Collapse steps by seed | Collapse rate | Mean final probability std | Mean final probability span |
|---|---:|---:|---:|---:|
| baseline | 8, 8, 8 | 3/3 (100%) | 0 | 0 |
| mhc | 8, 8, 8 | 3/3 (100%) | 0 | 0 |
| mhc_rms | 8, 8, 1 | 3/3 (100%) | 1.63e-5 | 9.76e-5 |
| mhc_prenorm | 1, 1, 1 | 3/3 (100%) | 4.41e-6 | 9.56e-5 |
| mhc_gate | 8, 8, 8 | 3/3 (100%) | 0 | 0 |
| mhc_rms_gate | 1, 1, 1 | 3/3 (100%) | 2.43e-8 | 1.19e-7 |

All 18 cases reported finite gradients. The huge final perplexity proxies in
the saturated cases are not useful quality scores; they are retained as an
indicator that the output logits became numerically/extremely wrong after
collapse.

## Calibration and interpretation

With four fixtures, the baseline at learning rate `0.002` collapsed during
calibration at update 8 in the final run. An earlier repeat reached the same
gate at update 16, while the one-fixture repeat also reached it at update 8.
This narrow-boundary variability is why the final comparison uses the exact
one-fixture workload and reports the observed update, rather than claiming a
stable collapse time of 8 or 16.

The result is a negative stabilization result, not evidence that mHC is
generally ineffective. The implementation here is explicitly an mHC-style
static stream-routing probe: it adds identity-initialized Sinkhorn-constrained
stream mixing to the production coordinate residual blocks. It is not a
drop-in reproduction of a complete multi-stream mHC training stack, and the
single-fixture 16^3 workload is not a proxy for final 128^3 model quality.

## Harness incidents and fixes

Two earlier parent-process attempts were invalidated by CUDA allocator
pressure. The parent process retained its last autograd tensors across cases,
and aborted PTY sessions left two stale Python processes holding roughly 7.9
GiB of the local RTX 4060. The run was repaired by:

1. keeping canonical initial states on CPU;
2. releasing autograd locals before CUDA cache cleanup;
3. isolating every arm in `CLI/run_mhc_case_16.py`, giving each case a fresh
   CUDA context; and
4. verifying the final run had 18 result files, no OOM, and no checkpoint
   output.

The two failed parent runs are retained as provenance in their separate
`build/ablation_mhc_collapse_16_20260903*` directories and are not included in
the table above.

## Reproducibility artifacts

- Harness: `CLI/ablate_mhc_collapse_16.py`.
- Isolated runner: `CLI/run_mhc_case_16.py`.
- Final raw cases:
  `build/ablation_mhc_collapse_16_20260903_isolated/`.
- The final raw files are per-case JSON and do not contain model checkpoints.

## Decision

Do not restart the paid 128^3 L4 training from any of these arms. The next
useful experiment is to fix the mHC probe's high-learning-rate behavior and
repeat a multi-fixture 16^3 study at a stress schedule where baseline collapse
is late enough to compare sustained survival, with a separately reported
perplexity/geometry-quality gate.
