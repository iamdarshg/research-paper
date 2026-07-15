# Grounded 500-Geometry 28M Training Bottleneck Report

Generated: `2026-07-15`

## Scope

This report profiles the live `96^3`, 28.1-million-parameter training run without
changing or stopping it. It records what was measured, maps the measurements to
the training-loop implementation, and ranks exactness-preserving optimization
experiments. It does not claim that an optimization works until a controlled
before/after run has reproduced the loss, solver outputs, and gradients.

Profiled commit: `f88bf45 feat: train 28M model on multiscale aircraft latents`

Run command:

```powershell
python CLI/run_monitored_training.py `
  --manifest build/grounded_combined_500_20260715/manifest.jsonl `
  --num-epochs 10 `
  --batch-size 1 `
  --grid-size 96 `
  --cpu-threads 4 `
  --save-dir build/grounded_500_28m_full_20260715/checkpoints `
  --history-output build/grounded_500_28m_full_20260715/history.json `
  --save-every 1 `
  --stop-on-promotion-pass
```

The manifest has `549` unique records: `382` train, `56` validation, `52` test,
and `59` holdout. The model has `28,126,841` trainable parameters. The run uses
the checked-in defaults of `16` SPSA directions, `5` D3Q27 steps per direct
solver evaluation, and one direct solver loss on every optimizer update.

## Evidence And Method

The resource monitor sampled the process and NVIDIA telemetry every 10 seconds.
At the analysis cutoff it contained `740` valid samples over `7,563.3` seconds
(`2.101` hours). The live process remained responsive after profiling.

Two `py-spy 0.4.2` captures used nonblocking process reads:

- 90-second flamegraph at 25 Hz: `1,874` usable samples and `25` read errors.
- 60-second raw-stack capture at 25 Hz: `1,125` usable weighted samples and
  `30` read errors.

The checked-in flamegraph is [training_flamegraph.svg](training_flamegraph.svg).
The raw capture remains with the run at
`build/grounded_500_28m_full_20260715/resources/training_profile.raw`.

These are wall-time sampling profiles. An inclusive percentage assigns a sample
to every active caller in its stack, so nested rows are not additive. The phase
ownership rows are exclusive and do add to 100%. CUDA calls can also appear as
their Python launch or synchronization point rather than the exact GPU kernel.

## Resource Measurements

| Metric | Mean | P50 | P90 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| GPU compute utilization | 53.68% | 46% | 100% | 100% |
| Dedicated GPU memory | 3,336 MB | 3,371 MB | 3,403 MB | 3,421 MB |
| GPU memory-controller utilization | 25.66% | 34% | 48% | 60% |
| GPU power | 42.39 W | 15.25 W | 84.95 W | 86.09 W |
| Trainer CPU | 130.31% | 158.9% | 189.5% | 330.2% |
| System CPU | 22.96% | 22.5% | 34.4% | 50.1% |
| Trainer RSS | 980 MB | 865 MB | 1,595 MB | 2,606 MB |
| Whole-system memory | 86.05% | 85.1% | 90.7% | 94.3% |

On Windows, process CPU can exceed 100%; `130%` is approximately 1.3 logical
cores continuously occupied. GPU utilization was below 10% for `13.8%` of
samples and at least 80% for `39.5%`. The low median power and alternating
0-100% utilization are the signature of serialized launch/synchronization
phases, not steady GPU saturation.

Only about `3.42/8.19 GB` of VRAM was used at peak. The run is therefore not
VRAM-capacity-bound. Whole-system memory pressure is high, but the trainer's own
RSS peaked at 2.61 GB. No file-read or data-loader frame appeared as a material
hotspot, so the observed loop is not SSD-bound either.

## Sampled Phase Ownership

The second 60-second capture covered portions of multiple updates. Its exclusive
phase assignment was:

| Phase | Samples | Share |
| --- | ---: | ---: |
| Direct solver and SPSA | 577 | 51.29% |
| Exact full-grid grounded decoder loss | 349 | 31.02% |
| Sampled reconstruction / coordinate decode | 59 | 5.24% |
| Optimizer or optimizer-state transfer | 14 | 1.24% |
| Other autograd | 7 | 0.62% |
| Other model/control work | 119 | 10.58% |

Important inclusive regions in that same capture were:

| Inclusive stack region | Share of all samples |
| --- | ---: |
| Direct measured objective | 51.29% |
| D3Q27 `collide_and_stream` | 47.64% |
| BFL boundary application | 36.89% |
| Exact full-grid grounded loss | 31.02% |
| Coordinate Fourier encoding | 17.96% |
| `compute_all_link_distances` | 12.71% |
| Signed-distance calculation | 11.02% |
| SciPy Euclidean distance transforms | 10.76% |
| Momentum-exchange force accumulation | 5.42% |
| Aircraft validity evaluation | 1.24% |
| Optimizer-state movement | 1.16% |

The earlier 90-second capture assigned 37-39% to the direct objective and about
8% to the exact coordinate-loss caller. The difference is expected because the
captures intersected different parts of the sequential update. Both captures
agree on the hotspot ordering: D3Q27/BFL, coordinate decoding, and SDF work.

## Per-Update Critical Path

One optimizer update currently performs the following work in sequence:

1. Run the denoiser, consistency, latent reconstruction, and sampled geometry
   objectives.
2. Decode a full generated `96^3` logit field for the direct objective.
3. Detach that field and run the direct objective once at the base point plus
   `16` antithetic plus/minus pairs: `33` sequential solver evaluations.
4. For every solver evaluation, copy logits to CPU, select the target-occupancy
   top-k mask, copy the binary field to GPU, initialize the flow field, compute
   SDF/BFL link distances through two SciPy EDT calls, run five D3Q27 steps, and
   calculate connectivity and aircraft validity.
5. Recompute the generated student and decoder path once and inject the measured
   SPSA logit gradient into the neural network.
6. Compute the exact grounded reconstruction over all `884,736` voxels. With a
   chunk size of `16,384`, this is `54` chunks in a no-gradient statistics pass
   and another `54` decoder chunks in the backward pass.
7. Move AdamW state to GPU, step the optimizer, and move the state back to CPU.

This is `33 * 5 = 165` D3Q27 time steps per optimizer update. A complete
382-record training epoch therefore requests `12,606` direct solver evaluations
and `63,030` D3Q27 steps before promotion testing. This multiplier, not corpus
loading or attention, explains the multi-hour epoch.

## Bottleneck Analysis

### 1. BFL Boundary Handling And LBM Launch Granularity

`_apply_bfl_boundary` is the largest sampled implementation hotspot. It loops
over 26 lattice directions in Python and performs many small masked operations
and `torch.any` decisions. Those decisions can synchronize the CPU with CUDA.
The loop repeats for each of five LBM steps and each of 33 solver evaluations.

The existing fused Triton streaming path is disabled because it has not yet
matched the full physics path. Turning it on without parity evidence would be a
scientific shortcut, not a valid optimization. The credible experiment is to
fuse BFL operations while retaining the same q values, boundary populations,
momentum exchange, drag, and lift, then require numerical parity on a fixed set
of real and perturbed geometries before training uses it.

### 2. Repeated Deterministic Coordinate Encoding

`_encode_coordinates` was the largest avoidable leaf at `17.96%`. The converter
recomputes six bands of sine and cosine features for coordinates that never
change at a fixed grid size. The exact grounded objective invokes this path for
108 chunks per update, and full generated decodes encode the entire lattice
again.

Caching the FP32 encoded `96^3 x 39` coordinate table would use approximately
`131.6 MiB`, well inside the measured VRAM headroom. It changes no model value
or loss term. A parity test should require identical encoded coordinates and
decoder logits, and matching losses and gradients within the existing floating
point tolerance.

### 3. CPU SciPy SDF And Device Synchronization

Every new binary geometry crosses GPU to CPU in `compute_sdf`, runs two SciPy
`distance_transform_edt` calls, and copies an FP32 SDF back to GPU. This owns
about 11% of the raw profile and creates a hard synchronization boundary before
the LBM can continue.

The SDF cannot simply be replaced with a constant half-link distance. On one
real `96^3` corpus geometry, 57,942 crossing links had q values from
`0.3660254` to `0.6339746`; they were not all `0.5`. Any GPU EDT or direct q
kernel must reproduce those distances and solver coefficients. A content-hash
cache of scalar objective results is safe only when two SPSA probes produce the
same thresholded geometry; the duplicate rate must be measured first.

### 4. Exact Full-Lattice Decoder Work

The grounded decoder objective owns 31% of the raw window. Its first pass
computes exact global balanced-BCE and Dice statistics without gradients. Its
second pass recomputes every chunk and backpropagates the exact gradients. This
two-pass design bounds activation memory and preserves the full-lattice loss,
but doubles decoder forward work.

The first experiment should be coordinate-feature caching because it attacks a
measured subcost without changing this algorithm. A later custom autograd or
saved-logit implementation may remove recomputation, but it must demonstrate
the same scalar loss and per-parameter gradients. Replacing the full objective
with a sampled approximation would violate the training contract.

### 5. Optimizer-State Offload

The checked-in AdamW setup has about `214.6 MiB` of FP32 first/second moments
for 28.1 million parameters. Moving them to GPU and back transfers roughly
`429 MiB` per update. It was only `1.16%` of the sampled window, so this is not
the first bottleneck. Still, the measured VRAM headroom supports an A/B test
with optimizer state resident on GPU. That test should be performed after the
larger hotspots, and only after confirming peak memory during the optimizer
step rather than relying on average VRAM.

### 6. Work That Is Not Currently Dominant

- Aircraft validity and connectivity are real optimizer terms, but their CPU
  evaluation occupied only about 1.2% of samples.
- The 28M neural network and its attention blocks did not dominate either
  profile. Increasing parameter count would increase cost without addressing
  the current throughput limit.
- Data loading and SSD reads were not visible hotspots.
- Console output from 33 solver announcements per update is unnecessary noise,
  but it is secondary to solver and decoder work.

## Exactness-Preserving Experiment Order

Do not alter the live run. After its first checkpoint, use one fixed batch,
fixed SPSA directions, and fixed solver state for each comparison.

1. Cache Fourier-encoded coordinates. Compare logits, all loss components,
   gradients, wall time, GPU duty cycle, and peak VRAM.
2. Instrument hashes for all 33 thresholded SPSA masks. Add per-update scalar
   objective memoization only if duplicate masks occur; identical masks must
   return previously measured identical solver values.
3. Prototype a fused BFL boundary path. Gate it on per-step population fields,
   q values, momentum-exchange forces, and final aerodynamic coefficients.
4. Evaluate a GPU-resident exact EDT/q implementation. Reject it if crossing
   distances or final objective values exceed tolerance.
5. Test resident AdamW state, measuring the true optimizer-step peak.
6. Investigate an exact custom-autograd full-grid loss only after the simpler
   coordinate cache is measured.
7. Rate-limit repetitive solver logging.

For every experiment, all aero, connectivity, validity, reconstruction,
consistency, and latent losses remain in the final optimizer update. SPSA stays
at 16 antithetic directions unless a separate estimator-variance study justifies
a change. No surrogate objective, skipped CFD call, lower grid, sampled
full-lattice replacement, or unvalidated physics kernel qualifies as a speedup.

## Conclusion

The live run is healthy but alternates between GPU-saturated kernels and
serialized CPU/CUDA boundaries. Its principal cost is the required 33-evaluation
direct solver objective, especially Python-granular BFL work; its largest clearly
redundant computation is Fourier encoding of the same lattice coordinates. The
best first change is therefore the exact coordinate cache, followed by measured
mask reuse and a parity-gated BFL fusion. More parameters, more CPU workers, or
faster storage would not address the observed bottlenecks.
