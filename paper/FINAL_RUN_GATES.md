# Final Run Gates

This checklist blocks the paper from using a final run to support claims that the current repository does not yet validate.

Readiness language here is intentionally separate from evidence language:
mapped, implemented, and scaffolded describe gate readiness, while `pass` is
reserved for executable reports or evidence outcomes. A passing reduced evidence
package unlocks only the reduced evidence-bundle claim, not the stronger
aircraft-level claims listed below.

## Current Allowed Summary

With the repo and hardware available on 2026-06-20, the strongest currently supportable claim is:

> The repository contains a proof-of-concept latent generative pipeline with a CFD-informed scoring path, STL export, deterministic reference evidence, structured-conditioning plumbing, and public VSP Airshow smoke-training runs on hundreds of traceable geometries. The generated Airshow-checkpoint samples currently fail aircraft validity checks, and the `64^3` run did not produce a checkpoint under the local run ceiling, so aircraft-level generation and performance claims remain blocked.

**Updated 2026-11-05 (generator recovery pass; evidence
`evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`):** a checkpoint
retrained on this host at `32^3` (dense decoder, 45 epochs, CRM corpus) produces voxel
geometries that pass every aircraft-specific check on a pre-registered held-out split
with an uncalibrated 0.5 materialization threshold: ground truth 9/9, reconstruction 9/9,
generated 45/45 (4-step consistency) and 45/45 (teacher DDIM), independently re-screened
by `CLI/aircraft_validity.py`. The same checkpoint emits **one** geometry for every
condition and seed (2 distinct arrays among 45 draws; mean IoU across the nine held-out
conditions 1.000), and its condition vector is unnormalized. The strongest supportable
claim is therefore:

> The repository can generate aircraft-valid voxel geometries at `32^3` from a checkpoint
> trained and evaluated on this 8 GB host, verified on a declared held-out split with an
> uncalibrated screen. The generator does not yet demonstrate conditioned or diverse
> generation: a single aircraft-valid geometry is emitted regardless of condition or seed.
> Aerodynamic, structural, manufacturing and superiority claims remain blocked, and the
> Airshow `16^3`/`32^3`/`96^3` checkpoints are not reproduced here.


## Claim Gates

| Claim | Required run or artifact | Required baseline | Required metric | Minimum gate to pass | Fallback wording if gate fails | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| `Generates aircraft structures` | Generated samples evaluated with aircraft-specific geometric checks | Hand-built aircraft-like template or curated aircraft corpus | Connectedness, symmetry, planform plausibility, tail/wing/body checks | Multiple generated samples pass `CLI/aircraft_validity.py` checks | `Generates freeform voxel geometries with some aircraft-like motifs` | **Literal form passed 2026-11-05** (9/9 reconstruction, 45/45 generated on a pre-registered held-out split, uncalibrated threshold 0.5, re-screened by the shipped CLI). Scientific form **blocked**: one distinct geometry per condition/seed (2 distinct arrays in 45), conditioning unnormalized. See `evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`. || `Aerodynamically optimized` | Controlled comparison of generated candidates under fixed CFD settings | Baseline geometry plus ablations | `C_L`, `C_D`, `L/D`, reference area normalization | Generated candidates outperform or consistently match baseline under the same setup | `Produces candidates that can be scored by the current CFD path` | Solver provenance implemented / claim evidence blocked |
| `Structurally viable` | Structural or load-path analysis | At least one explicit structural baseline | Connectivity plus structural metric | Structural/manufacturing condition gates pass and structural reports exist | `Uses connectivity penalties as a heuristic proxy only` | Feasibility gate implemented / structural evidence blocked |
| `CFD-guided training` | Ablation with and without CFD/scoring feedback | Same architecture and seed budget | Optimization-loss history plus candidate-ranking change in generated outputs | Solver feedback measurably changes learning dynamics or candidate ranking | `Contains an aerodynamic scoring diagnostic in the implementation` | Ablation scaffold implemented / claim evidence blocked |
| `Outperforms prior approaches` | Reproduced baseline comparisons | Named prior methods or strong internal baselines | Same evaluation metrics across methods | Statistically defensible comparison | `We do not claim superiority over prior approaches` | Comparison scaffold implemented / claim evidence blocked |
| `Publication-quality validation` | Convergence, sensitivity, or external validation study | External solver and/or experimental references | Grid convergence, timestep sensitivity, solver agreement | Validation plan executed and reported | `Current evidence is limited to sanity checks and code-path validation` | Validation scaffold implemented / claim evidence blocked |
| `Conditioned on flight profile and manufacturing method` | Conditioned dataset, schema, inference examples, and condition-response evaluation | Unconditioned model or prompt-free baseline | Constraint satisfaction and conditional consistency | Model consumes structured conditions and generated samples pass aircraft-specific validity plus `CLI/run_condition_benchmark.py` passes under a grounded aircraft-like evaluation corpus | `The repo has structured-conditioning plumbing and public-corpus smoke evidence, but scientific validation remains incomplete` | Benchmark gate implemented / generated validity evidence blocked |
| `Validated compressible/high-Mach internal solver` | Thermal/compressible LBM implementation or external compressible reference comparison under fixed geometries | OpenFOAM compressible solver such as `sonicFoam`/`rhoPimpleFoam` with residual, Courant, latest-time, Cd/Cl history, and force-stability records | Mach-specific Cd/Cl agreement, positive rho/T/p, stable force history, documented boundary conditions | All focused compressibility tests pass and high-Mach comparisons are finite, converged, and documented | `Internal D3Q27 high-Mach runs are experimental and unvalidated` | Gate implemented / claim evidence blocked |

## Required Final-Run Inputs Before Claim Expansion

1. Generated outputs from the documented aircraft-like dataset must pass aircraft-specific validity screens.
2. A condition schema for mission/flight profile and manufacturing constraints plus public CLI/config exposure for the intended controls.
3. At least one aircraft-specific validity test suite.
4. A named baseline for aerodynamic comparison.
5. A structural proxy stronger than simple connectivity.
6. A fixed 8 GB GPU run protocol, an architecture that scales beyond dense voxel decoding, or access to larger hardware.
7. A passing final evidence package from `CLI/final_evidence.py`.
8. Compressibility audit and evidence artifacts under `build/solver_diagnostics/compressibility_*`.
9. For high-Mach claims, an external compressible OpenFOAM comparison or a validated thermal/compressible LBM path with shock/steep-gradient and boundary-condition tests.

## Final Run Decision Rule

Do **not** present a final training run as a paper result unless:
- the relevant claim gate above has a passing executable/evidence report, and
- the run can be reproduced in the current environment or on explicitly named external hardware.

If a smoke run is executed only to verify the training code path, public-corpus path, or conditioning plumbing, report it as:

> a bounded implementation smoke run on public/synthetic/freeform data that exercises the pipeline, not a publication-grade aircraft-design result.

## Publish-Readiness Addendum (2026-11-01)

No claim gate below is marked passed by the publish-readiness pass. The pass
fixed a measurement defect and quantified statistical reality; it did not
create new evidence where none existed:

- **Reference-area convention fixed (code + tests).** All LBM-vs-OpenFOAM
  comparisons now renormalize onto a single documented reference area
  (`lbm_renormalized_to_openfoam_reference_area`); see
  `CLAIMS_EVIDENCE_MATRIX.md` C11. Corrected frozen-corpus agreement under the
  repo's own <5% gate: sonicFoam 1/16 rows, simpleFoam 0/14 rows. No
  pipeline-level <5% claim is supportable in either the old or new convention.
- **Condition-response statistics (n=12, non-significant).** The smoke
  checkpoint shows no statistically significant condition-response signal; the
  paper reports the null result with CIs (see `CLAIMS_EVIDENCE_MATRIX.md` C12).
- **`Aerodynamically optimized` gate:** the reference-area term in this gate's
  required-metrics column is now correctly computed by the comparison scripts;
  the gate itself remains `claim evidence blocked` (no converged comparison
  against baselines).
- **`Validated compressible/high-Mach internal solver` gate:** remains
  `claim evidence blocked`. The internal D3Q27 is a low-Mach sanity path only;
  the paper claims no high-Mach validation and no 2.2x mixed-precision speedup
  as a measured result.
- **8 GB smoke protocol (#32):** satisfied by the canonical
  `CLI/run_protocols/smoke_8gb.yaml`; the cloud-only final-evaluation path
  remains a documented requirement for the strongest claims.
