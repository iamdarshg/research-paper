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
| C08 | `paper/sections/methodology.tex` | methods | implied differentiating through CFD solver | current code separates differentiable model-weight training from measured black-box candidate optimization; connectivity, validity, and CFD scores can drive the sequential candidate optimizer but are not autograd terms | `docs/dataset/airshow_corpus_addition_report_20260620.md`; `docs/benchmarks/airshow_loss_debug_20260620.md`; benchmark evidence that sequential optimization reliably improves generated candidates if upgraded | revised |
| C09 | `paper/sections/results-and-discussion.tex` | empirical | risk of aircraft-level interpretation | A `32^3` checkpoint produces aircraft-valid voxel geometries on a pre-registered held-out CRM split (reconstruction 9/9; generation 45/45 at an uncalibrated threshold), independently re-screened by the repository's CLI; the 45 draws contain only 2 distinct arrays, so this demonstrates geometry-validity emission, not conditioned or diverse generation. The earlier public Airshow checkpoints remain as reported and are not re-run here. | larger controlled benchmark with diverse generated aircraft, grounded condition-response, external validation and baselines | revised |
| C10 | `paper/sections/conclusion.tex` | project scope | implied fully AI-driven conditioned airplane generator | schema and code plumbing plus a public-corpus smoke run exist, but validated mission/manufacturing-conditioned aircraft generation remains future work | generated Airshow-checkpoint samples passing validity, grounded condition-response benchmark, ablations, and full CLI/config exposure | revised |
| C11 | `CLI/advanced_lbm_solver.py`, `CLI/cascaded_lbm.py`, `CLI/GROUND_TRUTH_SPEC.md` | internal solver compressibility | implied that accepting Mach > 0.3 means validated compressible CFD | internal D3Q27 is validated only as a low-Mach weakly compressible/isothermal sanity path; Mach > 0.3 is experimental unless external compressible validation exists | `build/solver_diagnostics/compressibility_audit_20260612/solver_compressibility_audit.md` and focused regime tests | revised |
| C12 | `paper/sections/results-and-discussion.tex`, `docs/benchmarks/airshow_grounded_training_20260620.md`, `docs/dataset/airshow_corpus_addition_report_20260620.md`, `docs/dataset/airshow_corpus_replication_20260620.md` | public-corpus training | implied hundreds of public models unlock validated aircraft generation | 355 public VSP Airshow records were converted and used for a smoke training run; the resulting three generated flight-path checks exercise the pipeline but fail `span_sanity` and are not claim-bearing CFD evidence | generated samples passing aircraft-validity screens, matched baselines, converged/external CFD validation, and structural/load-path evidence | revised |
| C13 | `paper/sections/results-and-discussion.tex`, `paper/sections/conclusion.tex`, `docs/benchmarks/airshow_resolution_sweep_20260620.md` | higher-resolution Airshow rerun | implied increasing voxel count might itself clear aircraft-generation gates | `32^3` training completed but all three generated cases failed validity checks; `64^3` corpus validation passed but no dense-decoder checkpoint was produced within the local run ceiling; `96^3` coordinate-decoder training completed and one of three calibrated top-k exports passed the heuristic validity screen | reliable generated samples passing aircraft-validity across the suite plus external validation gates | revised |

## Upgrade Rule

Do not strengthen any `allowed wording now` entry until the required evidence exists and the corresponding gate in `paper/FINAL_RUN_GATES.md` is marked as passed.

## Conditioning-Specific Evidence Boundary

- The repo now includes a documented condition schema and code plumbing through dataset generation, latent construction, model conditioning, and generator inference.
- That code plumbing is not enough to claim validated mission-conditioned or manufacturing-conditioned aircraft generation.
- The public Airshow corpus now provides hundreds of traceable public geometry records for smoke training, and the paper includes corpus, training, metric, and generated-geometry figures; however, the generated Airshow-checkpoint flight-path checks fail the current aircraft-specific validity screens at the completed resolutions.
- The higher-resolution addendum is mixed but still non-claim-bearing evidence: `32^3` generated samples still fail, `64^3` did not produce a dense-decoder checkpoint under the local run ceiling, and `96^3` coordinate-decoder training produced one heuristic-screen pass out of three calibrated top-k exports.
- This means grounded data exists for code-path evidence; it still does not support aircraft-level conditioned-generation claims.

## Compressibility-Specific Evidence Boundary

- Internal LBM low-Mach raw outputs are separate from calibrated/surrogate/training labels.
- Internal LBM high-Mach outputs are executable diagnostics labeled `experimental_high_mach_unvalidated`.
- OpenFOAM incompressible and OpenFOAM compressible references must be recorded as distinct evidence sources.
- Paper text must not describe the internal solver as validated transonic or supersonic CFD until a thermal/compressible solver path and external validation gates pass.

## Agent D Evidence Addendum (2026-10-31, Workstream C pipeline audit)

Independent pipeline audit + bounded re-validation on RTX 4060 Laptop
(torch 2.9.1+cu130, Python 3.12.2). Full report:
`evidence/research-paper/D_PIPELINE_AUDIT.md`. These rows update the *evidence
comments* only; no `allowed wording now` is strengthened and no FINAL_RUN_GATES
entry is marked passed.

| claim_id | validated evidence comment (2026-10-31) |
| --- | --- |
| C08 | Audit confirms connectivity/validity/CFD losses are detached diagnostics, not autograd terms (tested: gradient probe + architecture); no differentiation-through-CFD claim. Pipeline fixes F6/F7 make the canonical `smoke_8gb.yaml` protocol reproducible end-to-end on this host (train→evaluate-baselines→validate-conditions→multi-seed-eval all produce artifacts) but do not change the evidence tier. |
| C09 | Fresh validity counts reproduced: frozen NASA CRM whole-aircraft voxels 15/15 pass the heuristic screen; bundled root STLs voxelized at 96³ (biplane, F-18, cylinder, 20mm cube) 0/4 pass. This re-confirms the frozen-docs statement that the heuristic screen is a first-pass plausibility filter, not structural/aerodynamic validation. |
| C11 | Reference-area normalization mismatch quantified in the frozen low-Mach sweep: OpenFOAM Cd normalized by fixed param `0.27734375` while internal LBM Cd normalized by voxel-projected frontal area `0.285020` (g32) / `0.238549` (g64); `cd_error_percent` rows are therefore contaminated by up to ±16% and the 0.25% passing row (M0.3/of-g64 LBM-g64) is a normalization coincidence. Frozen corpus agreement under repo gate (<5%): sonicFoam 2/16 rows, simpleFoam 0/14 rows. Low-Mach-only boundary (Mach ≤ 0.3, `low_mach_sanity_only`, `weakly_compressible_isothermal_lbm`) reconfirmed; no high-Mach claim. |
| C12 | Reproduced condition-response correlations from the fresh smoke checkpoint (n=4 seeds): target_speed↔drag r=−0.806 (p=0.19), wingspan↔occupancy r=−0.621 (p=0.38); all p ≫ 0.05, n=4. `multi_seed_eval` self-report status=pass but the enclosing `baseline_statistics` gate is fail-closed `blocked` (missing named baseline_set). Consistent with no-upgrade. |

## Publish-Readiness Addendum (2026-11-01, reference-area convention fix + statistical substance)

Prepared as part of the publish-readiness pass (run log:
`evidence/research-paper-publish/PUBLISH_READY.md`). These rows update the
*evidence comments* only; no `allowed wording now` is strengthened and no
FINAL_RUN_GATES entry is marked passed.

| claim_id | validated evidence comment (2026-11-01) |
| --- | --- |
| C11 | **Reference-area convention fixed in code (test-first).** The frozen low-Mach comparison corpus normalized OpenFOAM coefficients on a fixed sweep area (`0.27734375`) while the internal LBM solver normalized on its own resolution-dependent voxel-projected frontal area (`0.2850196` @ g32 / `0.2385490` @ g64 / `0.3965490` @ g16 / `0.3084270` @ g24); the old `cd_error_percent` was contaminated up to ±16% and the 0.25% "passing" row was a normalization coincidence. `CLI/solver_grid_speed_study.py::compare_cases` and `CLI/run_simplefoam_low_mach_suite.py::make_row` now renormalize the LBM coefficient onto the OpenFOAM area (`C_corr = C_raw * Aref_LBM / Aref_OF`), recording both areas per row. 9 new regression tests (`tests/test_reference_area_regression.py`) pin the convention; the old 0.25% row now evaluates to 14.2% Cd error and the old code (raw coefficient division) fails the new test. Corrected frozen-corpus gate table (`scripts/recompute_corpus_consistent_reference_area.py`; tables in `evidence/research-paper-publish/corrected_corpus/`): sonicFoam **1/16** rows < 5% (was 2/16), simpleFoam **0/14** (unchanged). SimpleFoam agreement worsens under correction (LBM frontal areas at g16/g24 exceed the sweep area). Low-Mach-only boundary reconfirmed; no high-Mach claim. |
| C12 | **n=12 condition-response sweep executed** (canonical `validate-conditions`, grid 32, 100 steps, smoke checkpoint; artifact `build/protocol_smoke/condition_validation_n12.json`). Two of twelve seeds produced non-physical D3Q27 blow-ups (drag ≈1e22, 1e33) — documented solver-instability events, not physical values; identical seed conditions differ from the October n=4 artifact on the same CUDA checkpoint (run-to-run variation, not bit-reproducible). Outcome: **non-significant**. Raw n=12 target_speed↔drag r=−0.410 (p=0.185, CI [−0.797, 0.214]); excluding the two blow-up seeds (n=10) r=−0.198 (p=0.583); wingspan↔lift r=0.603 (p=0.065, CI [−0.043, 0.893]) is the closest but still null. No cherry-picked seeds; the paper reports a null/indeterminate result for the smoke checkpoint and keeps condition-response signal an open gate. Power: n=12 ≈ 80% power for |ρ|≈0.78 (two-sided α=0.05). |
| C09 | Validity screen re-audited 2026-11-01 (harness `scripts/agent_d_validity_harness.py`): frozen NASA CRM whole-aircraft voxels **15/15 pass**; bundled root STLs **0/4 fail** (biplane: symmetry, normalization_margin, tail_body_plausibility; F-18: normalization_margin; cylinder/cube: 6 checks each). Screen remains a first-pass shape-plausibility heuristic with quantified failure modes; not structural/aerodynamic/manufacturing validation. |
| C08 | No change to the detached-diagnostics contract; the reference-area fix does not alter the autograd boundary. The canonical smoke protocol (smoke_8gb.yaml) remains the documented #32-consistent 8 GB smoke path; the cloud-only final-eval path (#32 second half) remains a documented requirement for the strongest claims. |

## Generator Recovery Addendum (2026-11-05, aircraft-valid generation on a declared held-out split)

Full report: `evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`;
protocol fixed before evaluation: `repos/research-paper/build/recovery_2026-11-05/PREREGISTERED_PROTOCOL.md`;
raw artifacts: `repos/research-paper/build/recovery_2026-11-05/`.

These rows update the *evidence comments*. C03's "allowed wording now" is unchanged, and
no aerodynamic, structural, conditioning or superiority claim is strengthened. The one
claim whose gate status changes is gate 1 of `FINAL_RUN_GATES.md`, and it changes in its
literal form only.

| claim_id | validated evidence comment (2026-11-05) |
| --- | --- |
| C09 | **Pre-registered held-out generation validity.** Split from the corpus's own `split` labels: 6 NASA CRM records (`test`/`val`) for training, 9 (`holdout`) held out and never used for gradients, threshold selection, early stopping or checkpoint selection. Screen unchanged (`evaluate_aircraft_validity`, `canonicalize=True`); materialization threshold fixed and uncalibrated at 0.5. Measured: ground truth 9/9, reconstruction 9/9 (IoU 0.441), generated **45/45** with the 4-step consistency sampler and **45/45** with a teacher-DDIM sampler (45 draws each; empty failed-check histogram). Independent re-screen through `CLI/aircraft_validity.py --input-dir` reports `status: pass`, 45/45, no failed indices. Before-state on the same records with the shipped 16³ checkpoint: 0/9 and 0/45. **Caveat recorded in the same artifact**: the 45 generated voxel arrays contain 2 distinct geometries (mean IoU across the nine held-out conditions 1.000), so this is aircraft-valid *geometry emission*, not conditioned generation; held-out latent codes are near-degenerate (pairwise L2 0.0027 vs norm 0.497). |
| C13 | Root causes for the earlier resolution-sweep failures, measured: (i) the procedural corpus generator's private gate admitted samples the declared screen rejects (47 admitted, 1 passing, before the fix); (ii) the shipped 16³ checkpoint's decoder emits a near-constant probability field (min/max 0.435–0.573, occupancy 45–46 % at threshold 0.5); (iii) `--learning-rate` never reached the decoder or consistency student, whose rates are fixed at 2e-5 in `CLI/config.yaml` (decoder loss flat at 2e-5, falling at 2e-4); (iv) generation sampled only the under-trained consistency student; (v) latent capacity and corpus scale. On the 32³ path the decisive repair was dense decoder supervision density (67.1 M-parameter final layer emitting a logit per voxel) rather than the coordinate decoder's 8 192 sampled coordinates per step. The 16³/32³/96³ Airshow checkpoints cited in the earlier rows are not present in the tree and were not re-run. |
| C10 | No change to the conditioning claim, and one measured reason is now on record: `build_condition_vector` emits raw physical units, so the 22-slot vector's pairwise L2 distance spans 9–1.6e5 across the held-out records; `normalize_condition_vector_tensor` is applied only inside the latent-code builder. Condition-response evidence therefore remains blocked, and the recovery result must not be presented as conditioned generation. |
| C03 | Unchanged: the allowed wording stays "latent generative pipeline for aircraft-like voxel artifacts". The recovery adds an executable, held-out-validated path to aircraft-valid 32³ voxel geometry but no diversity or conditioning evidence, so no upgrade is claimed. |
