# Claims Evidence Matrix

This matrix records the strongest claims that appeared in the paper and the wording currently allowed by the available evidence.

| claim_id | file | claim_type | earlier wording risk | allowed wording now | required citation or evidence | status |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | `paper/main.tex` | abstract scope | implied validated aircraft design | proof-of-concept pipeline for aircraft-like voxel geometries with partial structured-conditioning plumbing | current repo architecture + sanity run | revised |
| C02 | `paper/main.tex` | abstract performance | implied structural viability and aerodynamic efficiency | CFD-informed scoring path with limited sanity evidence | stronger validation study for upgrade | revised |
| C03 | `paper/sections/introduction.tex` | contribution | implied 3D aircraft generation | latent generative pipeline for aircraft-like voxel artifacts | aircraft-specific validity tests for upgrade | revised |
| C04 | `paper/sections/introduction.tex` | contribution | implied demonstrated CFD-guided optimization | implementation path for CFD-informed evaluation | ablation with and without CFD term | revised |
| C05 | `paper/sections/introduction.tex` | contribution | implied structural viability | connectivity heuristics and validation framework | structural or load-path analysis | revised |
| C06 | `paper/sections/related-work.tex` | novelty | unclear novelty relative to 3D diffusion | novelty is the assembled proof-of-concept workflow and validation discipline | related-work citations already added | revised |
| C07 | `paper/sections/related-work.tex` | comparison | unclear distinction from topology optimization and CFD shape optimization | learned generative prior plus post hoc scoring rather than direct parameter optimization | topology/CFD optimization citations | revised |
| C08 | `paper/sections/methodology.tex` | methods | implied differentiating through CFD solver | current code routes geometries through an aerodynamic score path and now documents public Airshow corpus construction separately from model validation | `docs/dataset/airshow_corpus_addition_report_20260620.md`; differentiable-CFD experiment if upgraded | revised |
| C09 | `paper/sections/results-and-discussion.tex` | empirical | risk of aircraft-level interpretation | public Airshow-corpus smoke training plus synthetic/freeform/context runs supports code-path execution only; generated Airshow-checkpoint samples currently fail aircraft span-sanity validity | larger controlled benchmark with grounded aircraft-like data whose generated samples pass validity and external validation gates | revised |
| C10 | `paper/sections/conclusion.tex` | project scope | implied fully AI-driven conditioned airplane generator | schema and code plumbing plus a public-corpus smoke run exist, but validated mission/manufacturing-conditioned aircraft generation remains future work | generated Airshow-checkpoint samples passing validity, grounded condition-response benchmark, ablations, and full CLI/config exposure | revised |
| C11 | `CLI/advanced_lbm_solver.py`, `CLI/cascaded_lbm.py`, `CLI/GROUND_TRUTH_SPEC.md` | internal solver compressibility | implied that accepting Mach > 0.3 means validated compressible CFD | internal D3Q27 is validated only as a low-Mach weakly compressible/isothermal sanity path; Mach > 0.3 is experimental unless external compressible validation exists | `build/solver_diagnostics/compressibility_audit_20260612/solver_compressibility_audit.md` and focused regime tests | revised |
| C12 | `paper/sections/results-and-discussion.tex`, `docs/benchmarks/airshow_grounded_training_20260620.md`, `docs/dataset/airshow_corpus_addition_report_20260620.md`, `docs/dataset/airshow_corpus_replication_20260620.md` | public-corpus training | implied hundreds of public models unlock validated aircraft generation | 355 public VSP Airshow records were converted and used for a smoke training run; the resulting three generated flight-path checks exercise the pipeline but fail `span_sanity` and are not claim-bearing CFD evidence | generated samples passing aircraft-validity screens, matched baselines, converged/external CFD validation, and structural/load-path evidence | revised |
| C13 | `paper/sections/results-and-discussion.tex`, `paper/sections/conclusion.tex`, `docs/benchmarks/airshow_resolution_sweep_20260620.md` | higher-resolution Airshow rerun | implied increasing voxel count might itself clear aircraft-generation gates | `32^3` training completed but all three generated cases failed validity checks; `64^3` corpus validation passed but no checkpoint was produced within the local run ceiling | architecture/runtime changes plus generated samples that pass aircraft-validity and external validation gates | revised |

## Upgrade Rule

Do not strengthen any `allowed wording now` entry until the required evidence exists and the corresponding gate in `paper/FINAL_RUN_GATES.md` is marked as passed.

## Conditioning-Specific Evidence Boundary

- The repo now includes a documented condition schema and code plumbing through dataset generation, latent construction, model conditioning, and generator inference.
- That code plumbing is not enough to claim validated mission-conditioned or manufacturing-conditioned aircraft generation.
- The public Airshow corpus now provides hundreds of traceable public geometry records for smoke training, and the paper includes corpus, training, metric, and generated-geometry figures; however, the generated Airshow-checkpoint flight-path checks fail the current aircraft-specific validity screens at the completed resolutions.
- The higher-resolution addendum is negative evidence: `32^3` generated samples still fail, and `64^3` did not produce a checkpoint under the local run ceiling.
- This means grounded data exists for code-path evidence; it still does not support aircraft-level conditioned-generation claims.

## Compressibility-Specific Evidence Boundary

- Internal LBM low-Mach raw outputs are separate from calibrated/surrogate/training labels.
- Internal LBM high-Mach outputs are executable diagnostics labeled `experimental_high_mach_unvalidated`.
- OpenFOAM incompressible and OpenFOAM compressible references must be recorded as distinct evidence sources.
- Paper text must not describe the internal solver as validated transonic or supersonic CFD until a thermal/compressible solver path and external validation gates pass.
