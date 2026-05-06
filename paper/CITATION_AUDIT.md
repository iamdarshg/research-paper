# Citation and Claim Audit

This document tracks the audit of claims made in the research paper to ensure they are either backed by prior literature or supported by experimental evidence in this repository.

| Section | Claim | Support Type | Status | Notes |
|---------|-------|--------------|--------|-------|
| Intro | Latent diffusion for 3D aircraft structures | Literature + Repo | OK | Builds on Rombach (2022) and Luo (2021). |
| Intro | CFD integration into training loop | Repo Evidence | OK | Demonstrated via LBM loss term. |
| Intro | Hierarchical/connectivity-aware generation | Repo Evidence | OK | Demonstrated via Constraint Projector. |
| Related | Gap: Direct enforcement of hard structural connectivity | Literature | OK | Cited lack of hard constraints in standard diffusion. |
| Methodology | GPU-accelerated LBM solver efficiency | Repo Evidence | OK | Benchmarked in internal tests. |
| Results | Aerodynamic optimization success | Repo Evidence | Soften | Currently only CPU sanity runs. |
| Results | Structural viability | Repo Evidence | Soften | Needs more rigorous structural load-path checks. |

## Audit Categories
- **Prior-work citation needed:** Needs a reference to existing literature.
- **Repo evidence needed:** Needs a figure, table, or run artifact from the code.
- **Overclaim / Soften:** Claim is too strong for current evidence; needs downgrade.
- **OK:** Claim is well-supported and scoped.
