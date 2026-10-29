# 128 cubed model-parameter ablation and selection

Date: 2026-09-01

## Decision

Use the checked-in corpus-scaling architecture for the next 128 cubed L4
repair canary:

- latent dimension: 512
- encoder channels: 128, 176, 224
- decoder channels: 224, 176, 128
- conditioning dimension: 22 (fixed by corpus schema)
- grouped attention: 8 query groups, 4 KV groups, 4 attention layers
- coordinate decoder: width 2048, depth 5, six Fourier bands
- coordinate chunk: 4096
- gradient checkpointing: enabled
- dense decoder: rejected at 128 cubed
- whole-model and converter compilation: disabled for this diagnostic

This preserves the downloaded checkpoint's latent, denoiser, conditioning, and
attention shapes while replacing its collapsed, heavily over-parameterized
coordinate head through an explicit weights-only warm start. It is a new
optimizer lineage; it is not an exact resume.

A direct CPU compatibility probe loaded the step-6008 converter state into the
selected head through overlap migration. The earlier 1024-wide resource
finalist loaded one exact tensor, 23 overlap-migrated
tensors, and zero skipped target tensors. The source head's extra depth is not
part of the selected target model.

## Evidence boundary

This was a quick architecture fit and trainability screen, not a converged
quality comparison. Random-initialization loss is recorded for reproducibility
but is not treated as aircraft-quality evidence. Final generated-shape quality
must be established by grounded training followed by F-16 and Cessna projection
and validity checks.

The screen held the 128 cubed grid, XYZ dimensionality, and 22-feature corpus
conditioning schema fixed. It varied every remaining ModelConfig capacity or
runtime axis one at a time around the checkpoint-recorded baseline:

- latent dimension
- encoder/decoder channels
- attention query groups
- attention KV groups
- attention-layer count
- coordinate-decoder width
- coordinate-decoder depth
- Fourier-band count
- coordinate chunk size
- diffusion gradient checkpointing

The final warmed comparison also tested coupled decoder width/depth/chunk
configurations. Dense decoding was rejected analytically: its final 2048 to
2,097,152 layer alone would contain 4,294,967,296 weights. Compile settings
were recorded but excluded because they change execution rather than learned
capacity and would dominate a quick run with compilation startup.

## Checkpoint-recorded baseline

The downloaded step-6008 run state records a different architecture than the
current repository defaults:

- converter width 3328, depth 12, chunk 8192
- total runtime parameters: 315.54 million
- converter parameters: 267.73 million
- warmed probe median: 1.7754 seconds
- peak allocated CUDA memory: 3,419.6 MiB
- observed initial generated probability standard deviation: 0.000003825

Checkpoint identity:

- path: `D:\research_paper_transfer_20260901\latest_v7_checkpoint_20260901_verified.pt`
- SHA-256: `5b88f3bcaa394df2a7a40b98932ab2b8157eb6bf9cc1c9e1a54fbdd90e970a04`
- global step: 6008

## Warmed finalists

All finalists passed finite forward/backward and gradient checks on the same
seeded 8,192-coordinate sample over a 128 cubed index domain. The ablation
process was capped to 72 percent of the local RTX 4060 allocation so the user's
separate BERT process remained running.

| Candidate | Runtime params | Converter params | Median seconds | Peak MiB | Initial probability std |
| --- | ---: | ---: | ---: | ---: | ---: |
| checkpoint baseline 3328x12, chunk 8192 | 315.54M | 267.73M | 1.7754 | 3419.6 | 0.000003825 |
| fast 512x5, chunk 4096 | 50.71M | 2.91M | 0.2327 | 448.0 | 0.000324853 |
| resource finalist 1024x5, chunk 4096 | 58.87M | 11.06M | 0.2736 | 463.5 | 0.000281755 |
| performance winner 2048x5, chunk 4096 | 90.90M | 43.10M | 0.4947 | 763.8 | 0.000363301 |

The 1024-wide resource finalist is approximately 6.5 times faster and 7.4 times
lower in peak allocated memory than the checkpoint baseline in this component
probe. The final 2048-wide selection is still substantially smaller and faster
than the collapsed 3328x12 checkpoint head, but it was selected by the real
held-out performance stage below rather than by resource fit alone.

## Real-corpus held-out occupancy perplexity

The resource sweep was followed by a model-performance ablation on the pinned
8,027-aircraft 128-cubed corpus. Every candidate used the same SHA-pinned
step-6008 warm start, deterministic train and validation aircraft, seeds,
optimizer, learning rate, and coordinate samples. The primary metric is
`exp(class-balanced Bernoulli NLL)` on held-out occupancy. Ordinary sparse-grid
perplexity is also recorded but is not the selection objective because an
almost-all-empty predictor can minimize it while producing no aircraft.

The full quick sweep varied decoder width/depth/Fourier bands, denoiser channel
widths, attention-layer count, and KV grouping after the earlier resource sweep
had covered the remaining ModelConfig axes. Each of 15 candidates received 8
identical updates. Evaluation used fixed 65,536-voxel stratified and uniform
samples from each of three held-out 128-cubed aircraft. Two candidates were
rejected by the collapse floor despite superficially near-random perplexity:
1024x8 depth and zero Fourier bands.

| Quick-sweep candidate | Balanced perplexity | Raw perplexity | Occupancy-matched recall | Probability std | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| 2048x5, Fourier 6 | 1.9993527 | 2.0105471 | 0.61952 | 0.00027281 | pass, winner |
| 1024x5, Fourier 8 | 1.9999595 | 1.9958966 | 0.34594 | 0.00018456 | pass |
| 1024x5, Fourier 6 | 1.9999826 | 1.9959333 | 0.30541 | 0.00018590 | pass |
| 1024x8, Fourier 6 | 2.0000033 | 1.9954543 | 0.31144 | 0.00002377 | collapsed |
| 1024x5, Fourier 0 | 1.9999999 | 1.9965326 | 0.26316 | 0.00004294 | collapsed |

A longer finalist confirmation used 32 identical updates, 32 train aircraft,
and 8 held-out aircraft:

| Finalist | Balanced perplexity | Raw perplexity | Occupancy-matched recall | Probability std |
| --- | ---: | ---: | ---: | ---: |
| 2048x5, Fourier 6 | 1.9955143 | 2.0632310 | 0.77081 | 0.00138721 |
| 1024x5, Fourier 8 | 1.9991375 | 2.0089397 | 0.65013 | 0.00029273 |
| 1024x5, Fourier 6 | 1.9993562 | 2.0090931 | 0.61001 | 0.00025386 |

The 2048x5 candidate therefore won both performance stages. The absolute
perplexities remain close to the uninformative Bernoulli value of 2 because
this is intentionally a very short adaptation, not converged training. Its
larger improvement and much stronger held-out ranking signal justify selecting
it for the budgeted cloud canary; complete-lattice promotion and collapse
checks remain mandatory before any quality claim.

## Provenance artifacts

Full axis sweep:

- `build/ablation_128_model_parameters_20260901T1458Z/environment.json`
- `build/ablation_128_model_parameters_20260901T1458Z/results.json`
- `build/ablation_128_model_parameters_20260901T1458Z/results.csv`
- `build/ablation_128_model_parameters_20260901T1458Z/results.partial.json`

Warmed finalist sweep:

- `build/ablation_128_model_parameters_20260901T1512Z_finalists/environment.json`
- `build/ablation_128_model_parameters_20260901T1512Z_finalists/results.json`
- `build/ablation_128_model_parameters_20260901T1512Z_finalists/results.csv`
- `build/ablation_128_model_parameters_20260901T1512Z_finalists/results.partial.json`

Real-corpus quick performance sweep:

- `build/ablation_128_heldout_perplexity_20260901T1540Z/environment.json`
- `build/ablation_128_heldout_perplexity_20260901T1540Z/results.json`
- environment SHA-256: `02f557489e6aecd7a3b033b8f5dbf2dd67640d1666e2e3a470e005d234e98b87`
- results SHA-256: `4b7e97814cf77334a50876a163c599d01323bab5fe2d65886b701d32ffc0b9ce`

Longer finalist confirmation:

- `build/ablation_128_heldout_perplexity_20260901T1545Z_finalists/environment.json`
- `build/ablation_128_heldout_perplexity_20260901T1545Z_finalists/results.json`
- environment SHA-256: `9d133b65b7cc582481bf40c4c4d35a4f8d6638609f0ef88b7f43a8798b0f198a`
- results SHA-256: `14e4818ad26537c8016e6048c0ff99257d02e2b6541b1f9ade278092775c4a09`

Harness source: `CLI/ablate_128_heldout_perplexity.py`. Failed smoke
directories are preserved separately and are not included in either ranking.

The environment files contain the exact argv, checkpoint hash, checkpoint
architecture, fixed/rejected parameters, Python/PyTorch/CUDA/GPU identity, Git
HEAD, and dirty-worktree inventory. Each candidate record contains its complete
ModelConfig, status/error, exact parameter counts, timings, losses, probability
standard deviations, gradient-finiteness result, peak CUDA allocation, and wall
time.

The first harness attempt failed before model allocation because Windows
PyTorch requires `cuda:0`, not an unindexed `cuda` device, for the process-memory
fraction API. The harness was corrected and rerun; no failed measurements were
mixed into the result table.

## Cloud budget state

The L4 was stopped before the ablation. Its brief 2026-09-01 boot ran from
20:22:11 to 20:23:59 IST, about 108 seconds. At Google's published current
g2-standard-4 Spot rate of USD 0.423956 per hour, that is approximately USD
0.0127 of VM compute. The persistent 100-GB disk remains attached with
autoDelete disabled. The next worker must use a new archive and source identity,
weights-only warm start, a whole-VM shutdown fuse, and the remaining budget
rather than resuming the earlier incompatible run state.
