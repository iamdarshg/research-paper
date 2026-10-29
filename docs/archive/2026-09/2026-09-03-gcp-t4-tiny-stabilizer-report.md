# GCP T4 16^3 stabilizer smoke test — 2026-09-03

## Scope

This was a bounded GPU-environment and stabilization probe, not a production
training run. It used a single Spot `n1-standard-2` VM with one NVIDIA Tesla
T4 in `us-central1-a`, a 100 GB temporary `pd-standard` boot disk required by
the pinned PyTorch DLVM image, no external IP, and a ten-minute automatic
termination limit. No project checkpoint was downloaded, created, or stored.

The first VM attempt failed in the harness before the optimizer loop because a
CPU `torch.Generator` was passed to a CUDA tensor factory. The GPU environment
was healthy (`Tesla T4`, driver `580.173.02`, CUDA driver 13.0); the harness
was corrected to generate on CPU and transfer the fixtures to CUDA. The first
VM and boot disk were deleted before the corrected rerun.

## Corrected run

- VM: `research-t4-16cubed-smoke-20260903`
- GPU: `Tesla T4`, 15,360 MiB
- PyTorch: `2.9.1+cu129`
- CUDA runtime reported by PyTorch: `12.9`
- Grid: `16 x 16 x 16`
- Parameters: `275,136`
- Updates per candidate: `32`
- Fixed synthetic fixtures and seed across candidates
- All five candidates: finite gradients and non-collapse gate passed
- Peak CUDA allocation: `27,740,160` bytes

The mHC-like candidate is explicitly only a small residual/RMS-normalization
probe. It is not the full production mHC architecture.

| Candidate | Std weight | Heat weight | Final BCE | PPL proxy | Probability std | Span | Collapsed |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline_std0 | 0.0 | 0.00 | 0.467581 | 1.596128 | 0.157278 | 0.929018 | no |
| baseline_std05 | 0.5 | 0.00 | 0.470495 | 1.600787 | 0.163952 | 0.938175 | no |
| heat_std05 | 0.5 | 0.10 | 0.467094 | 1.595351 | 0.157361 | 0.929547 | no |
| mhc_std05 | 0.5 | 0.00 | 0.386637 | 1.472022 | 0.182141 | 0.936930 | no |
| heat_mhc_std05 | 0.5 | 0.10 | 0.384654 | 1.469107 | 0.177941 | 0.933308 | no |

## Interpretation

On this tiny fixed probe, the mHC-like residual/RMS normalization produced the
large improvement: roughly 7.8% lower PPL proxy than the no-std baseline. The
heat-like six-neighbour diffusion term alone was effectively neutral relative
to the baseline with the same std coefficient. Adding heat to the mHC-like
variant improved the toy PPL proxy by another roughly 0.2%, but that difference
is far too small to call a real effect from one seed and 32 updates.

This does not establish that either intervention fixes the 128^3 collapse. The
probe uses a 16^3 synthetic decoder, not the production diffusion/CFD stack,
and the PPL value is an ordinary BCE-derived proxy rather than the production
held-out class-balanced metric. The result is sufficient to justify a later,
isolated multi-seed 16^3 follow-up for the mHC-like parameterization; it is not
sufficient to modify or restart the paid L4 training.

## Resource cleanup

The temporary VM and its temporary boot disk were verified absent after the
run. The pre-existing persistent disk
`research-l4-full-20260829` in `asia-south1-b` was not touched. The smoke
script is retained at
`build/gcp_t4_tiny_smoke_startup.sh` for reproducibility and contains no
checkpoint-writing path.
