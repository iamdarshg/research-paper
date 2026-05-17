# Citation And Claim Audit

This audit was created to close GitHub issues `#24`, `#25`, and `#26`.

## Scope

The current repository supports:
- a proof-of-concept latent generative pipeline,
- synthetic voxel training data,
- a reduced freeform-object sanity experiment,
- an internal D3Q27/OpenFOAM benchmarking path.

The current repository does **not** yet support, at publication quality:
- mission-conditioned aircraft generation,
- manufacturing-conditioned generation,
- structural validation,
- aerodynamic superiority claims,
- differentiable end-to-end CFD training claims,
- benchmarked comparison against prior aircraft-design methods.

## Audit Labels

- `OK`
- `prior-work citation needed`
- `repo evidence needed`
- `overclaim / soften`
- `baseline needed`
- `definition needed`

## Introduction

| Section | Status | Notes |
| --- | --- | --- |
| Paragraph 1 | `prior-work citation needed` | Generative design in aerospace, structural and manufacturing tradeoffs, and iterative baseline methods need citations or narrower wording. |
| Paragraph 2 | `overclaim / soften` | Current repo does not show a validated distribution of viable aircraft designs; it shows synthetic/freeform generation with limited conditioning. |
| Contribution list item 1 | `overclaim / soften` | "Generating 3D aircraft structures" must be reduced to proof-of-concept voxel geometry generation unless aircraft-specific validity checks exist. |
| Contribution list item 2 | `repo evidence needed` | The code contains a CFD loss path, but the paper should not claim demonstrated aerodynamic optimization from the current sanity run. |
| Contribution list item 3 | `overclaim / soften` | Connectivity penalties do not by themselves establish structural viability. |
| Closing roadmap paragraph | `OK` | Keep as structure-only text. |

## Related Work

| Section | Status | Notes |
| --- | --- | --- |
| Paragraph 1 | `prior-work citation needed` | Early 3D generative-model discussion needs at least one 3D diffusion reference and one adjacent design-method reference. |
| Paragraph 2 | `prior-work citation needed` | Diffusion claims should cite 3D diffusion work, not only image diffusion work. |
| Paragraph 3 | `definition needed` | "Hierarchical representation" needs a precise scoped meaning in this repo or should be reduced to voxel decoding/training stages. |
| Paragraph 4 | `overclaim / soften` | Direct-in-training CFD integration is currently stronger in code intent than in validated evidence. |
| Paragraph 5 | `prior-work citation needed` | Transformer motivation is fine, but should not imply novelty. |
| Paragraph 6 | `overclaim / soften` | TiDAR/HRM language is too idiosyncratic and reads as unsupported novelty framing. |
| New positioning subsection | `required` | Must distinguish the repo from 3D diffusion, topology optimization, classical CFD shape optimization, neural surrogate work, and differentiable simulation. |

## Methodology

| Section | Status | Notes |
| --- | --- | --- |
| Opening paragraph | `overclaim / soften` | The "thinking/speaking" framing is colorful but not technically grounded for the paper. |
| Architecture figure caption | `repo evidence needed` | Should describe a proof-of-concept score-guided path, not a proven differentiable training loop. |
| Noise scheduling | `OK` | Generic method description is fine with diffusion citations. |
| Latent diffusion UNet | `OK` | Keep factual and implementation-scoped. |
| Latent-to-3D converter | `OK` | Keep factual and implementation-scoped. |
| CFD simulator | `definition needed` | Must say "internal LBM-based evaluator" and note that external validation currently comes from benchmark cross-checks. |
| Loss function | `OK` | Keep as implemented objective, not as proof of real-world performance. |
| "Rigorous Integration of CFD in Diffusion Training" | `overclaim / soften` | Current text implies differentiating through the solver; current evidence does not support that claim. Rewrite as implementation intent and present limitations. |
| TODO graph note | `remove` | Replace with scoped future work or delete. |

## Results And Discussion

| Section | Status | Notes |
| --- | --- | --- |
| Opening paragraph | `OK` | Already scopes the run as narrow; keep that tone. |
| Training convergence paragraph | `OK` | Fine if explicitly described as sanity evidence. |
| Sweep discussion | `OK` | Fine if "freeform-object" language is retained. |
| Geometry summary | `overclaim / soften` | Avoid implying aircraft realism or manufacturability. |
| Hyperparameter notes | `OK` | Keep as pilot-study framing. |
| Discussion paragraph 1 | `repo evidence needed` | End-to-end path claim is okay; any claim beyond code-path validation should be removed. |
| Discussion paragraph 2 | `baseline needed` | Solver mismatch statements need benchmark framing, not paper-level performance claims. |
| Discussion paragraph 3 | `OK` | Keep as benchmark-history note. |

## Validation And Conclusion

| Section | Status | Notes |
| --- | --- | --- |
| Validation standards | `OK` | Current future-work framing is appropriate. |
| Conclusion paragraph 1 | `overclaim / soften` | Best-sample `L/D` should stay explicitly tied to the reduced sanity setup. |
| Conclusion paragraph 2 | `OK` | Solver calibration and larger ablations are correctly framed as next steps. |
| Final conclusion sentence | `required` | Add an explicit statement that flight-profile/manufacturing conditioning and publication-grade aircraft validation remain future work. |

## Required Follow-Up

1. Add a `Positioning and Novelty` subsection to `paper/sections/related-work.tex`.
2. Rewrite the contribution list in `paper/sections/introduction.tex`.
3. Replace differentiable-CFD wording in `paper/sections/methodology.tex` with implementation-accurate wording.
4. Keep all current results labeled as sanity or pilot evidence.
5. Gate all future final runs with `paper/FINAL_RUN_GATES.md`.
