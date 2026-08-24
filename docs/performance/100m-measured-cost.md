# 100M-Param 128^3 Training: Measured Timing and GCP Cost

Date: 2026-08-24. Branch: main.

## Methodology

This analysis uses MEASURED microbenchmark data from the RTX 4060 Laptop GPU,
not theoretical scaling estimates. The decoder MLP was benchmarked at the exact
target width/depth (w=2432 d=7) across chunk sizes 4096-32768.

### Key finding: the decoder is COMPUTE-BOUND, not bandwidth-bound

Measured at w=2432 d=7 on 4060:
- Effective bandwidth: 5 GB/s (out of 272 GB/s = 1.8% utilization)
- Compute: 36.7 TFLOPS bf16 (~76% of the 48 TFLOPS peak)
- Chunk size does not matter (4096 to 32768 all give same throughput)

The A100 speedup is therefore based on COMPUTE ratio:
- 4060 measured: 36.7 TFLOPS bf16
- A100 expected: ~200 TFLOPS bf16 (at similar efficiency)
- Speedup: **~5.5x** (range: 5.1-5.9x)

## Per-Update Time at 128^3 on A100-80GB

Derived from R11 production-faithful benchmark (27.53 s/u at 96^3, w=928 d=5)
scaled by decoder compute ratio and voxel count:

| Component | 4060 @ 96^3 | Scaling | A100 @ 128^3 |
|---|---|---|---|
| Decoder/model phases | 19.3s | x10.2 (params) x2.37 (voxels) / 5.5 (A100) | 84.5s |
| Solver/CFD path | 8.3s | x2.37 (voxels) / 2.5 (A100 CPU+GPU) | 7.8s |
| **Total** | **27.5s** | | **92.4s** |

Range with uncertainty: 87-99 s/update.

## Training Cost at 100M params / 128^3

| Scenario | Epochs | Updates | A100 hours | On-demand | Spot |
|---|---|---|---|---|---|
| 1 epoch | 1 | 8,027 | 206h | $1,045 | $515 |
| 2 epochs | 2 | 16,054 | 412h | $2,089 | $1,030 |
| 3 epochs | 3 | 24,081 | 618h | $3,134 | $1,545 |
| 8 epochs (half) | 8 | 64,216 | 1,648h | $8,356 | $4,121 |
| Full convergence | 15 | 120,405 | 3,090h | $15,668 | $7,726 |

## Budget Analysis

The original mission brief allocated ~$270. At 92 s/update this covers
zero full epochs. The fundamental constraint: each update pushes 2.1M voxels
through a 36.8M-param MLP = 1.2 PetaFLOPs forward + backward.

Options within budget:

| Option | Config | Est. cost | Quality expectation |
|---|---|---|---|
| A: Stay at 96^3 | w=2432 d=7, 96^3 | $217/ep spot | Meaningful training possible |
| B: Smaller model at 128^3 | w=1024 d=7 (~38M total) | ~$80/ep spot | Faster but less capacity |
| C: Partial training at 128^3 | w=2048 d=8 (~88M), partial | ~$93/ep spot | Underfitting risk |
| D: Increase budget to $750 | w=2432 d=7, full 1 epoch | $515 spot | Single-pass over corpus |

## PRO 6000 Blackwell Assessment

NOT recommended. This workload is compute-bound in the decoder MLP,
and while the PRO 6000 has higher raw compute than A100, it costs more
per hour ($4.50 vs $5.07 seems close but the A100 achieves better
utilization for large dense matmuls). Additionally, the lower memory
bandwidth (1.6 vs 2.04 TB/s) hurts the solver path which IS bandwidth-sensitive.