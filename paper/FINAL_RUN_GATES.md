# Final Run Gates

This checklist blocks the paper from using a final run to support claims that the current repository does not yet validate.

## Current Allowed Summary

With the repo and hardware available on 2026-05-17, the strongest currently supportable claim is:

> The repository contains a proof-of-concept latent generative pipeline with a CFD-informed scoring path, STL export, a reproducible sanity benchmark on synthetic/freeform data, and partial structured-conditioning plumbing that is not yet scientifically validated.

## Claim Gates

| Claim | Required run or artifact | Required baseline | Required metric | Minimum gate to pass | Fallback wording if gate fails | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| `Generates aircraft structures` | Generated samples evaluated with aircraft-specific geometric checks | Hand-built aircraft-like template or curated aircraft corpus | Connectedness, symmetry, planform plausibility, tail/wing/body checks | Multiple generated samples pass `CLI/aircraft_validity.py` checks | `Generates freeform voxel geometries with some aircraft-like motifs` | Gate implemented / claim evidence blocked |
| `Aerodynamically optimized` | Controlled comparison of generated candidates under fixed CFD settings | Baseline geometry plus ablations | `C_L`, `C_D`, `L/D`, reference area normalization | Generated candidates outperform or consistently match baseline under the same setup | `Produces candidates that can be scored by the current CFD path` | Failed / incomplete |
| `Structurally viable` | Structural or load-path analysis | At least one explicit structural baseline | Connectivity plus structural metric | Structural/manufacturing condition gates pass and structural reports exist | `Uses connectivity penalties as a heuristic proxy only` | Feasibility gate implemented / structural evidence blocked |
| `CFD-guided training` | Ablation with and without CFD term | Same architecture and seed budget | Training loss terms plus ranking change in generated outputs | CFD term measurably changes learning dynamics or candidate ranking | `Contains an aerodynamic loss term in the implementation` | Failed / not demonstrated |
| `Outperforms prior approaches` | Reproduced baseline comparisons | Named prior methods or strong internal baselines | Same evaluation metrics across methods | Statistically defensible comparison | `We do not claim superiority over prior approaches` | Failed / not attempted |
| `Publication-quality validation` | Convergence, sensitivity, or external validation study | External solver and/or experimental references | Grid convergence, timestep sensitivity, solver agreement | Validation plan executed and reported | `Current evidence is limited to sanity checks and code-path validation` | Failed / not attempted |
| `Conditioned on flight profile and manufacturing method` | Conditioned dataset, schema, inference examples, and condition-response evaluation | Unconditioned model or prompt-free baseline | Constraint satisfaction and conditional consistency | Model consumes structured conditions and `CLI/run_condition_benchmark.py` passes under a grounded aircraft-like evaluation corpus | `The repo has structured-conditioning plumbing, but scientific validation remains incomplete` | Benchmark gate implemented / grounded evidence blocked |

## Required Final-Run Inputs Before Claim Expansion

1. A real aircraft or aircraft-like dataset with a documented schema.
2. A condition schema for mission/flight profile and manufacturing constraints plus public CLI/config exposure for the intended controls.
3. At least one aircraft-specific validity test suite.
4. A named baseline for aerodynamic comparison.
5. A structural proxy stronger than simple connectivity.
6. A fixed 8 GB GPU run protocol or access to larger hardware.
7. A passing final evidence package from `CLI/final_evidence.py`.

## Final Run Decision Rule

Do **not** present a final training run as a paper result unless:
- the relevant claim gate above passes, and
- the run can be reproduced in the current environment or on explicitly named external hardware.

If a smoke run is executed only to verify the training code path or the conditioning plumbing, report it as:

> a bounded implementation smoke run on synthetic/freeform data that exercises partial conditioning plumbing, not a publication-grade aircraft-design result.
