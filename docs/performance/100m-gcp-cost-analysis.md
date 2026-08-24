# 100M-Param 128^3 Training: Timing and GCP Cost Analysis

Date: 2026-08-24. Branch: main.

## Selected Configuration

| Parameter | Value |
|---|---|
| Coordinate decoder width | 2432 |
| Coordinate decoder depth | 7 |
| Latent dimension | 512 |
| Voxel resolution | 128 cubed (2,097,152 voxels) |
| Decoder params | 72.3M |
| Diffusion teacher UNet | 22M |
| Consistency student | 6M |
| **Total trainable** | **100.3M** |
| EMA mirror (not trainable) | 22M |
| Precision | bf16 autocast + TF32 NN GEMMs |

## Corpus

- 8,027 unique geometries (1,069 original CAD + 4,958 perturbed + 2,000 procedural)
- Estimated information content: ~4.07 Mbit
- Capacity ratio at 100M params: ~2.4x (good balance, past grokking boundary)

## Per-Update Time Estimate (A100-80GB)

First-principles FLOP calculation:
- Decoder forward: 303 TFLOP per pass through all 2.1M voxels
- With gradient checkpointing (fwd + recompute + backward): 1,213 TFLOP
- A100 bf16 effective: ~125 TFLOPS at 40% utilization
- Decoder compute time: ~9.7s
- Solver/CFD path at 128 cubed: ~6.0s
- Consistency model overhead: ~1.0s
- **Total: ~16.7 s/update**

## Epochs to Convergence

Best guess for a diffusion model at 100M params with 8,027 shapes:
- Optimistic: 10 epochs (80k updates)
- **Best guess: 15 epochs (120k updates)**
- Conservative: 25 epochs (200k updates)

Scaled by requested 0.5x multiplier on best guess:

| Metric | Value |
|---|---|
| Scaled epochs | 8 |
| Total optimizer updates | 64,216 |
| Total A100 compute time | **298 hours (12.4 days continuous)** |

## Google Cloud Cost Quote

### Option A: A100 80GB (a2-ultragpu-1g) - RECOMMENDED

| Pricing model | Rate/hr | Hours | Total cost |
|---|---|---|---|
| On-demand | $5.07 | 298h | **$1,511** |
| Spot (~50% discount) | ~$2.50 | 298h | **$745** |
| 1yr committed use | ~$3.60 | 298h | $1,073 |
| 3yr committed use | ~$2.55 | 298h | $760 |

### Option B: RTX PRO 6000 Blackwell (g4-standard-48)

NOT recommended for this workload. The PRO 6000 has higher raw compute
(~4000 vs 312 TFLOPS bf16) but LOWER memory bandwidth (1.6 TB/s vs 2.04 TB/s).
Our coordinate decoder MLP is bandwidth-bound, not compute-bound.

| Scenario | Rate/hr | Hours | Total cost |
|---|---|---|---|
| On-demand (same speed as A100) | $4.50 | 298h | $1,341 |
| On-demand (conservative, 22% slower) | $4.50 | 382h | $1,719 |
| Spot (estimated) | ~$1.65 | 298h | $492 |
| 1yr committed | ~$3.10 | 298h | $924 |
| 3yr committed | ~$1.98 | 298h | $590 |

## Budget Reality Check

The original mission brief allocated ~$270 for compute.
The full 8-epoch training costs $745 (A100 spot) - 2.8x over budget.

| Epochs | Updates | Hours | A100 spot cost | Fits $270? |
|---|---|---|---|---|
| 1 | 8,027 | 37h | $93 | YES |
| **2** | **16,054** | **74h** | **$186** | **YES** |
| 3 | 24,081 | 112h | $279 | marginal |
| 8 (full) | 64,216 | 298h | $745 | NO |

## Bottom Line

**Within $270 budget:** Train for 2 full epochs on A100 spot ($186).
This gives the model 16k updates across 8k shapes - enough for meaningful
diffusion training at the grokking balance point, though not fully converged.

**For full convergence:** Need ~$745 on A100 spot or $1,511 on-demand.
This requires either additional credits or accepting fewer epochs.

The RTX PRO 6000 Blackwell is NOT recommended because this workload is
memory-bandwidth-bound and the PRO 6000 has less bandwidth than the A100.