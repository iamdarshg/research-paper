# Originality and Relevance Comparison

Source paper: `paper/main.tex` and `paper/sections/*.tex`

Date: 2026-06-20

## Method

This comparison checks the paper against public, adjacent work in five groups:
general 3D diffusion, 3D asset generation, aerodynamic ML datasets, aircraft
design optimization, and aerodynamic generative-design papers. It is a
structure-and-wording comparison, not a plagiarism detector. No unpublished
paper text was submitted to third-party paraphrasing or originality services.

Sources reviewed:

- Point-E, "A System for Generating 3D Point Clouds from Complex Prompts",
  arXiv:2212.08751, https://arxiv.org/abs/2212.08751
- Shap-E, "Generating Conditional 3D Implicit Functions", arXiv:2305.02463,
  https://arxiv.org/abs/2305.02463
- Point-Voxel Diffusion, "3D Shape Generation and Completion through
  Point-Voxel Diffusion", arXiv:2104.03670,
  https://arxiv.org/abs/2104.03670
- AirfRANS, "High Fidelity Computational Fluid Dynamics Dataset for
  Approximating Reynolds-Averaged Navier-Stokes Solutions", arXiv:2212.07564,
  https://arxiv.org/abs/2212.07564
- OpenVSP "Learn More" page, https://openvsp.org/learn.shtml
- AeroSandbox repository, https://github.com/peterdsharpe/AeroSandbox
- Chen, Chiu, and Fuge, "Aerodynamic Design Optimization and Shape Exploration
  using Generative Adversarial Networks",
  https://ideal.umd.edu/assets/pdfs/chen_airfoil_opt_scitech_2019.pdf

## Structure Comparison

| Comparator | Typical structure | Difference in this paper |
| --- | --- | --- |
| Point-E / Shap-E | General 3D generation framing, model architecture, sampling quality, qualitative assets, released code or models | This paper is not primarily a 3D asset-generation paper. It uses a latent generative path, but centers evidence gates, public aircraft-geometry provenance, STL export, internal solver scoring, and failure boundaries. |
| Point-Voxel Diffusion | 3D diffusion formulation, point-voxel representation, generation/completion experiments on object datasets | This paper has a much weaker algorithmic novelty claim. Its defensible novelty is workflow assembly and claim-gated engineering evidence, not a new diffusion formulation. |
| AirfRANS | Dataset and benchmark paper with high-fidelity RANS fields over airfoils, baselines, and generalization tasks | This paper does not provide high-fidelity CFD labels. It uses public 3D geometry and internal low-Mach smoke scoring, so it should be framed below AirfRANS in physical-validation strength. |
| OpenVSP / VSP Airshow | Public parametric aircraft geometry ecosystem | This paper uses OpenVSP/Airshow as a traceable source of public geometry records, but must not imply that Airshow metadata is manufacturer certification or source-provided mission data. |
| AeroSandbox and classical aircraft optimization | Differentiable design/optimization workflows with explicit performance models and optimization variables | This paper explores a learned voxel generator plus scoring and gates. It should not claim to replace direct optimization frameworks. |
| Airfoil GAN / BezierGAN-style work | Low-dimensional aerodynamic shape parameterization, smooth airfoil generation, optimization acceleration | This paper operates on coarse 3D voxel artifacts and currently lacks a passing aircraft-validity result, so the wording must remain more cautious than airfoil-level optimization papers. |

## Wording Comparison

The paper's safest wording pattern is already in place: "proof-of-concept",
"smoke evidence", "implementation path", "claim boundary", "not
publication-grade", and "future work" appear repeatedly. That style differs
from many 3D generation papers, which can emphasize sample quality and method
superiority because their evaluation target is often visual or geometric asset
quality rather than aircraft engineering validity.

The most important wording rule is to avoid collapsing these categories:

- "3D generative model" is supported.
- "aircraft-like voxel artifacts" is supported only as a cautious visual and
  heuristic description.
- "valid aircraft" is not supported by the Airshow generated samples.
- "aerodynamically optimized" is not supported by the current raw D3Q27 smoke
  metrics.
- "outperforms prior approaches" is not supported.
- "conditioned aircraft generator" is not supported beyond structured
  conditioning plumbing and smoke checks.

## Originality Assessment

The paper is not original as a claim about diffusion models generating 3D
geometry. Point-E, Shap-E, Point-Voxel Diffusion, and related work already
establish that broad direction. It is also not original as a claim about
aerodynamic optimization; classical optimization, AeroSandbox-style
differentiable design workflows, surrogate optimization, and airfoil GAN work
are much more mature.

The paper is relatively original in the narrower way it packages a small
research codebase around evidence hygiene:

- public Airshow geometry ingestion with explicit license filtering,
  source hashes, manifest hashes, and split provenance;
- structured condition-vector plumbing tied to generation and validation
  reports;
- generated-artifact export into STL plus internal D3Q27 scoring;
- paper-level claim gates that keep failed generated samples visible instead
  of converting them into unsupported success claims;
- local analysis files explaining sentence purpose, evidence limits, and
  detector caveats.

That originality is methodological and reproducibility-oriented. It should not
be sold as a new state-of-the-art model.

## Relevance Assessment

The paper is relevant if positioned as an early engineering-AI reproducibility
and claim-gating study. It is especially relevant to the gap between attractive
3D generative outputs and the much stricter evidence needed for aircraft
geometry, solver validation, manufacturability, and baseline comparison.

The new `32^3` and `64^3` Airshow reruns strengthen that relevance by showing a
negative result: higher voxel count did not cause the present checkpoint to
pass aircraft-validity gates, and `64^3` exposes a dense-decoder scalability
limit. Keeping that result in the paper makes the manuscript more credible.

## Recommendations

1. Keep the paper's contribution statement narrow: a proof-of-concept,
   manifest-backed, claim-gated generative-design workflow.
2. Mention Point-E, Shap-E, and Point-Voxel Diffusion only as 3D generation
   context, not as direct aircraft-design competitors.
3. Mention AirfRANS-like work as a higher bar for CFD dataset quality; do not
   imply the internal D3Q27 labels are comparable to high-fidelity RANS data.
4. Keep the Airshow figures and failed generated geometry visible. The failure
   is part of the evidence.
5. Do not use online paraphrasers on the manuscript unless the author accepts
   the data-disclosure risk. A local style pass is safer and more defensible.
