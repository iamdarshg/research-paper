# Detailed Claim-by-Claim Citation Audit

This document tracks every paragraph in the Introduction, Related Work, Methodology, and Results sections to ensure technical claims are properly supported by literature or repository evidence.

| File | Section | Paragraph | Exact Claim | Required Support | Current Support | Status | Required Action |
|------|---------|-----------|-------------|------------------|-----------------|--------|-----------------|
| introduction.tex | Intro | 1 | "Generative design has emerged as a transformative paradigm" | Literature | General consensus | OK | None |
| introduction.tex | Intro | 2 | "demonstrated remarkable success in... three-dimensional shapes" | Literature | Luo (2021) | OK | None |
| introduction.tex | Intro | 2 | "intended to explore the underlying distribution of viable aircraft designs" | Repo Evidence | Proof-of-concept target | OK | Use "intended to" |
| introduction.tex | Intro | 2 | "generate structures that adhere to predefined aerodynamic... constraints" | Repo Evidence | Proof-of-concept target | OK | Use "proof-of-concept" |
| related-work.tex | Related | 1 | "convergence of generative modeling and computational engineering... structural and aerodynamic design" | Literature | Chen (2019), Oh (2019) | OK | None |
| related-work.tex | Related | 3.2 | "diffusion-based models as a stable alternative to GANs" | Literature | Ho (2020), Nichol (2021) | OK | None |
| related-work.tex | Related | 3.3 | "often prioritize visual mimicry over the physical manifold integrity" | Literature | General observation | OK | None |
| related-work.tex | Related | 3.4 | "TO is often sensitive to local minima and initial conditions" | Literature | Sigmund (2013) | OK | None |
| related-work.tex | Related | 3.8 | "While existing 3D diffusion frameworks often focus on... general object classes" | Literature | Luo (2021) | OK | Precise wording used |
| related-work.tex | Related | 3.8 | "coupling of 3D latent diffusion with high-fidelity LBM feedback... remains an open research area" | Literature | Negative space | OK | Acknowledge Thuerey (2020) |
| methodology.tex | Method | 1 | "core architecture consists of four main components" | Repo Evidence | `CLI/models.py`, `CLI/cfd_simulator.py` | OK | None |
| methodology.tex | Method | 5 | "aerodynamic loss... designed to encourage the generation of designs with improved aerodynamic characteristics" | Repo Evidence | Target language | OK | Use "designed to" |
| methodology.tex | Method | 6 | "integrate the CFD solver into the generative optimization loop" | Repo Evidence | Planned path | OK | Use "explore", "planned" |
| methodology.tex | Method | 6 | "Update UNet weights to maximize $C_L/C_D$" | Repo Evidence | Target sequence | OK | Use "Estimate weight updates" |
| methodology.tex | Method | 7 | "D3Q27 Cascaded MRT implementation... sub-voxel BFL boundaries" | Literature + Repo | Geier (2015) + Code | OK | None |
| results-and-discussion.tex | Results | 1 | "performed a CPU-only sanity run" | Repo Evidence | Run logs | OK | None |
| results-and-discussion.tex | Results | 2 | "Total loss stayed in the 2.31–2.71 range" | Repo Evidence | Figure 1 | OK | None |
| results-and-discussion.tex | Results | 3 | "results in Figure 2 provide a preliminary code-path validation" | Repo Evidence | Figure 2 | OK | None |
| results-and-discussion.tex | Results | 6 | "yields $C_d$ values that align with the broad expectations" | Repo Evidence | Table II | OK | None |

## Audit Summary by Section

### Introduction
The introduction has been audited for overclaims. Assertions regarding the model's ability to learn aircraft distributions have been scoped as "intended to explore" to reflect the proof-of-concept nature of current training runs.

### Related Work
The related work uses specific citations (Sigmund, Ho, Luo) to ground gaps. The "negative-space" claim about simulation-in-the-loop has been narrowed to specify the "direct coupling" within the diffusion loop to distinguish from surrogate-only approaches.

### Methodology
The methodology has been downgraded from a "rigorous integration" to an "exploration of integration" to reflect that full gradient-based backpropagation through the LBM solver is a planned/preliminary feature rather than a fully validated result.

### Results
The results section explicitly labels all findings as "sanity runs," "code-path validation," or "informal sanity checks." No claims of optimal aircraft performance are made; instead, the focus is on pipeline execution and baseline comparison.
