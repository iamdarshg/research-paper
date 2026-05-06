# Final-Run Gating Checklist (Rigorous)

Before presenting final results, each experiment must pass these gates to ensure mapping to paper claims and literature baselines.

| Gate ID | Claim | Prior-Work Bucket | Baseline / Ablation | Metric | Min. Evidence Threshold | Allowed Wording if Partial |
|---------|-------|-------------------|---------------------|--------|--------------------------|----------------------------|
| G-AERO | Aerodynamically optimized | Aero Opt; ASO | Classical Primitives; Adjoint ASO | $L/D$; Drag count | 15% better mean $L/D$ than unguided | "Demonstrates potential for optimization" |
| G-STRUC | Generates valid aircraft structures | Constraint Gen | No-cleanup ablation | Connectivity %; Manifold integrity | >95% connectivity success | "Improved physical manifold integrity" |
| G-DIFF | CFD-guided training outperforms | Diffusion | No-CFD-loss model | $L/D$ convergence rate | Measurable shift in distribution (p < 0.05) | "CFD guidance influences the latent manifold" |
| G-VAL | High-fidelity solver validation | Physics-ML | OpenFOAM (sonicFoam) | $C_d$ relative error | <10% error on canonical cube | "Code-path validation and initial calibration" |
| G-PERF | Outperforms prior generative approaches | 3D Gen | Voxel-GAN; Point-cloud diffusion | MS-SSIM; FID-equivalent | Diversity-score parity + validity advantage | "Comparable diversity with improved validity" |

## Detailed Gate Specifications

### G-AERO: Aerodynamic Optimization
- **Claim:** "enabling the optimization of aerodynamic performance as a core component".
- **Baseline:** Swept and tapered wing library + NACA 0012/4412 profiles.
- **Evidence:** $128^3$ res simulations for 50 samples.
- **Threshold:** Generated mean $L/D$ must exceed random-search baseline.

### G-STRUC: Aircraft Structure Generation
- **Claim:** "ensure the structural viability of the generated designs".
- **Baseline:** Raw diffusion output without `ConstraintProjector`.
- **Threshold:** Connectivity success rate > 95%; non-zero wing surface area.

### G-DIFF: CFD-Guided Training
- **Claim:** "CFD term measurably changes training".
- **Ablation:** Model trained with $\lambda_{aero} = 0$.
- **Threshold:** Significant difference in $C_D$ distribution of generated samples.

### G-PERF: vs Prior Approaches
- **Claim:** "maintained generative diversity... while operating at $128^3$".
- **Baseline:** Voxel-GAN trained on same synthetic dataset.
- **Metric:** Diversity measured via latent-space coverage and MS-SSIM.

## Evidence Downgrade Protocols
- **If threshold not met:** Wording must be changed from "demonstrated success" to "investigated feasibility" or "preliminary indication".
- **If baseline unavailable:** Claim must be restricted to "internal consistency check" or "code-path validation".
