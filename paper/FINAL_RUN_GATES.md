# Final-Run Gating Checklist

Before presenting "final" results, the following gates must be passed to ensure the experiments map to paper claims and literature baselines.

| Gate ID | Claim / Run | Baseline Required | Metric | Status |
|---------|-------------|-------------------|--------|--------|
| G-AERO | Aerodynamically optimized | Classical Primitives; No-CFD-loss | $C_L/C_D$ | BLOCKED |
| G-STRUC | Structurally viable | No-cleanup ablation | Connectivity % | BLOCKED |
| G-DIFF | CFD-guided training | No-CFD-loss | $L/D$ convergence | BLOCKED |
| G-VAL | Publication-quality validation | OpenFOAM vs LBM | Drag coefficient error | BLOCKED |

## Gate Definitions

1. **G-AERO:** Generative candidates must be compared against swept/tapered wing baselines using consistent CFD settings.
2. **G-STRUC:** Generated geometry must pass explicit aircraft-specific validity checks (manifold integrity, non-zero wing area).
3. **G-DIFF:** Demonstrate that the CFD term measurably changes the candidate ranking compared to unguided diffusion.
4. **G-VAL:** Include evidence of grid independence or external validation (OpenFOAM) for the internal solver results.
