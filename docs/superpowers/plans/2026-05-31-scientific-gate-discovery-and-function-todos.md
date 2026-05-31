# Scientific Gate Discovery And Function TODOs

> Discovery artifact for PR #37 scientific gates. This is not an implementation plan for weakening gates. Claim-bearing results require real corpus records, real checkpoints, real baseline runs, and real validation artifacts.

**Goal:** Map every scientific gate that must pass before claim-bearing paper results can be used, and identify the function-level TODOs needed to move from current scaffolding to scientifically accurate implementation.

**Scope boundary:** This document was produced by reading `paper/FINAL_RUN_GATES.md`, `paper/FINAL_EVIDENCE_PACKAGE.md`, `CLI/run_protocol.py`, `CLI/run_protocols/final_cloud.yaml`, `CLI/validate_manifest.py`, `CLI/run_condition_benchmark.py`, `CLI/aircraft_validity.py`, `CLI/condition_feasibility.py`, `CLI/multi_seed_eval.py`, `CLI/aircraft_diffusion_cfd.py`, and relevant tests under `tests/`.

---

## Gate Inventory

Claim-bearing results are blocked unless all applicable gates below pass.

1. Manifest validation gate: `manifest_validation.json` from `CLI/validate_manifest.py --level claim-bearing`.
2. Aircraft validity gate: `aircraft_validity.json` from `CLI/aircraft_validity.py`, with multiple generated samples passing aircraft-specific validity checks.
3. Condition response gate: `condition_benchmark.json` from `CLI/run_condition_benchmark.py`, with fixed sweeps passing on grounded records.
4. Manufacturing and structural condition gate: `manufacturing_constraints.json` from `CLI/condition_feasibility.py` and `DesignSpec` validation.
5. Baseline statistics gate: `baseline_statistics.json` from `CLI/multi_seed_eval.py` policy/statistical helpers.
6. Final evidence package gate: `final_evidence_package.json` from `CLI/final_evidence.py`, with every required report status equal to `pass`.
7. Generates aircraft structures claim gate from `paper/FINAL_RUN_GATES.md`.
8. Aerodynamically optimized claim gate from `paper/FINAL_RUN_GATES.md`.
9. Structurally viable claim gate from `paper/FINAL_RUN_GATES.md`.
10. CFD-guided training claim gate from `paper/FINAL_RUN_GATES.md`.
11. Outperforms prior approaches claim gate from `paper/FINAL_RUN_GATES.md`.
12. Publication-quality validation claim gate from `paper/FINAL_RUN_GATES.md`.
13. Conditioned on flight profile and manufacturing method claim gate from `paper/FINAL_RUN_GATES.md`.

The evidence gates overlap the claim gates, but they are not interchangeable. For example, a passing manifest gate does not prove aircraft generation; it only proves the input corpus contract.

---

## Current Protocol Shape

`CLI/run_protocols/final_cloud.yaml` currently builds a guarded final protocol:

- `validate_manifest` runs claim-bearing manifest validation against `docs/dataset/minimal_grounded_manifest.jsonl`.
- `train` runs `CLI/aircraft_diffusion_cfd.py train` with `run_class: final`, `baseline_config`, and `claim_gates`.
- `evaluate_baselines` runs bundled STL baseline evaluation.
- `validate_conditions` runs checkpoint condition-correlation smoke/current-checkpoint evaluation.
- `condition_benchmark` runs grounded condition-response benchmarking.
- `multi_seed_eval` runs the multi-seed CLI wrapper.

This protocol is conservative but not yet publication-grade. The checked-in minimal manifest has two records and is expected by tests to pass `basic` validation but block `claim-bearing` validation. The checked-in baseline config names only `bundled_grounded_stl`, while the baseline policy requires `retrieval`, `unconditional_checkpoint`, and `bundled_grounded_stl`.

---

## Gate 1: Manifest Validation

**Required to pass:** `manifest_validation.json` must report `status: pass` at `--level claim-bearing`.

### TODO Prototype

- File/function: `CLI/validate_manifest.py::validate_manifest_records`
- Current behavior: Checks existence of `geometry_path` or `stl_path`, split membership, and claim-bearing metadata fields. It requires all schema-listed `design_spec` fields but does not validate units semantics, provenance resolvability, split balance, duplicate samples, response metric provenance, or geometry type/scale beyond file existence.
- Scientific gap: A manifest can pass with a single complete record, fake provenance strings, arbitrary units, and no traceable source corpus. That is sufficient for contract tests but not enough for claim-bearing grounded aircraft evaluation.
- Proposed implementation: Add a strict claim-bearing validator layer that checks record uniqueness, corpus-level minimum counts by split and design family, unit vocabulary and scale metadata, source/provenance URI or citation resolvability, preprocessing hash/version, geometry file type, geometry dimensionality, nonzero geometry occupancy, and optional `response_metrics` provenance when condition-response claims are requested. Keep `basic` unchanged for smoke wiring.
- Tests needed: Add tests with duplicate `source_id`, unsupported units, missing provenance URI/hash, invalid split distribution, zero-occupancy geometry, and a complete multi-record corpus fixture that passes. Add a regression test that `docs/dataset/minimal_grounded_manifest.jsonl` remains blocked at claim-bearing level.
- Evidence artifact needed: A real aircraft or aircraft-like corpus manifest with documented provenance, preprocessing metadata, units, split assignments, design families, and geometry artifacts.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::AircraftDesignDataset._load_manifest_dataset`
- Current behavior: Loads manifest records directly into geometry tensors, `DesignSpec`, condition vectors, and deterministic latent codes. It does not require claim-bearing validation before final training; `_validate_run_class_inputs` only checks that the manifest exists and is non-empty.
- Scientific gap: `run_class: final` can train on a non-empty but scientifically incomplete manifest if the separate manifest validation command is skipped or ignored.
- Proposed implementation: Add a final-run dataset guard that calls the strict manifest validator at `claim-bearing` level before constructing a final-run dataset, or requires a validated manifest report path whose manifest hash matches the loaded manifest.
- Tests needed: Add CLI/unit tests showing final training rejects the minimal manifest, accepts a complete claim-bearing fixture, and fails when a manifest report does not match the loaded manifest hash.
- Evidence artifact needed: `build/protocol_final/manifest_validation.json` with `status: pass`, record count, corpus hash, geometry hash inventory, and validation level.

---

## Gate 2: Aircraft Validity

**Required to pass:** Multiple generated samples must pass aircraft-specific geometric checks, reported in `aircraft_validity.json`.

### TODO Prototype

- File/function: `CLI/aircraft_validity.py::evaluate_aircraft_validity`
- Current behavior: Converts voxels to a binary grid, then checks occupancy bounds, bilateral symmetry, span and length sanity, center/wing balance, and tail fraction. Tests cover a minimal hand-built voxel aircraft and an asymmetric blob.
- Scientific gap: The gate is a first-pass heuristic. It does not compute connectedness, fuselage/wing/tail segmentation, planform geometry, airfoil-like section plausibility, component topology, scale normalization, or orientation invariance. It can be fooled by symmetric voxel arrangements.
- Proposed implementation: Replace or extend the heuristic suite with explicit aircraft geometry analysis: canonical orientation detection, connected-component metrics, fuselage skeleton extraction, wing plane and tail plane detection, left/right symmetry after canonical alignment, planform area/aspect ratio, body/wing/tail attachment checks, and per-sample failure reasons. Preserve fail-closed status when orientation or segmentation cannot be resolved.
- Tests needed: Add fixtures for plausible monoplane, plausible biplane, disconnected wing, no tail, no fuselage, mirrored but non-aircraft slab, orientation-rotated aircraft, and noisy generated artifact. Tests should assert specific failed checks, not just global fail.
- Evidence artifact needed: `build/protocol_final/aircraft_validity.json` containing per-sample reports for a fixed generated sample set, sample IDs, checkpoint hash, generation seeds, geometry artifact paths, and aggregate pass count.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::OptimizedAircraftGenerator.generate`
- Current behavior: Loads a checkpoint, builds a condition vector, runs the consistency model, converts latent to voxels, and returns sigmoid voxel probabilities. It does not save raw voxel artifacts, thresholded artifacts, seed metadata, or validity reports.
- Scientific gap: Aircraft validity evidence cannot be reproduced from the generator alone unless sample seeds, conditions, checkpoint identity, voxel thresholds, and output files are recorded.
- Proposed implementation: Add a claim-eval generation path that accepts a sample manifest of conditions and seeds, writes raw probabilities and thresholded voxels, records checkpoint/config hashes, and invokes `evaluate_aircraft_validity` per sample.
- Tests needed: Mock generator output and assert deterministic artifact naming, seed capture, condition capture, threshold capture, checkpoint hash capture, and validity report aggregation.
- Evidence artifact needed: Generated voxel artifacts plus a validity report linking each artifact to condition payload, checkpoint hash, and generation seed.

---

## Gate 3: Condition Response

**Required to pass:** `condition_benchmark.json` must report `status: pass` for all fixed sweeps on grounded records.

### TODO Prototype

- File/function: `CLI/run_condition_benchmark.py::build_condition_benchmark_report`
- Current behavior: Loads manifest records, requires minimum grounded records, requires `design_spec` and `response_metrics`, checks checkpoint existence, then sorts records by each fixed condition and compares high-group vs low-group response means. It does not run the checkpoint to generate outputs, and the `seeds` field is currently metadata only.
- Scientific gap: A passing report can be based solely on manifest metadata if `response_metrics` are present. It does not prove the checkpoint consumes conditions or that generated outputs respond to condition changes. It also lacks statistical confidence, split isolation, paired comparisons, and provenance checks for response metrics.
- Proposed implementation: Split the benchmark into two explicit modes: `manifest-grounded-response` for corpus records and `checkpoint-conditioned-response` for generated samples. The checkpoint mode should generate paired samples under controlled seeds and conditions, evaluate response metrics from generated artifacts, compare against unconditioned or prompt-free baselines, and report effect sizes with confidence intervals. Manifest-grounded mode should require response metric provenance.
- Tests needed: Add tests for paired seed use, checkpoint invocation, unconditioned baseline comparison, effect-size thresholding, confidence interval reporting, response metric provenance rejection, and fail-closed behavior when generated artifact evaluation is missing.
- Evidence artifact needed: `build/protocol_final/condition_benchmark.json` with grounded corpus records, generated paired samples, seeds, checkpoint hash, unconditioned baseline results, metric provenance, effect sizes, and per-sweep statuses.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::validate_conditions`
- Current behavior: Generates samples for random speeds/spans, runs short CFD, and reports Pearson correlations between input conditions and measured drag/lift/occupancy. It prints that the report is current-checkpoint evidence only, not grounded aircraft validation.
- Scientific gap: This is not the final condition-response gate. It uses random condition sweeps, short CFD, no grounded corpus, no baseline comparison, no aircraft validity prefilter, and no fixed acceptance criteria.
- Proposed implementation: Rename or mark this as smoke/current-checkpoint only in report schema, and add a separate claim-bearing validator that reuses generated artifacts and response metrics from `run_condition_benchmark.py` rather than ad hoc correlations.
- Tests needed: Assert the report carries `status: smoke_only` or equivalent non-claim-bearing boundary, and assert final evidence packaging cannot accept this report as `condition_benchmark`.
- Evidence artifact needed: The claim-bearing condition benchmark report described above, not the current `condition_validation.json`.

---

## Gate 4: Manufacturing And Structural Condition Feasibility

**Required to pass:** Impossible payloads must be rejected, target payloads must pass, and structural reports must exist for structural-viability claims.

### TODO Prototype

- File/function: `CLI/condition_feasibility.py::validate_condition_feasibility`
- Current behavior: Checks simple scalar consistency: engine counts, thrust nonzero with engines, payload bounds, part-count bounds, wall-thickness bounds, method minimum wall thickness, nonnegative thrust-to-weight, positive speed, and nonnegative turn rate.
- Scientific gap: These are feasibility guards, not structural or manufacturing validation. They do not evaluate load paths, material properties, manufacturing process constraints beyond minimum wall thickness, volume/mass consistency, engine packaging, payload bay capacity, assembly constraints, or failure modes.
- Proposed implementation: Add a richer `validate_manufacturing_constraints` function that accepts geometry plus condition payload and checks manufacturability against method-specific rules: minimum feature size, unsupported spans/overhangs for FDM, sheet thickness/tab constraints, composite layup thickness, part count extraction, enclosed voids, engine/payload envelope fit, and mass estimate bounds.
- Tests needed: Add geometry fixtures for too-thin wing, impossible enclosed void, FDM overhang violation, too many disconnected printable parts, valid sheet-balsa tabbed fixture, and geometry/payload mass inconsistency.
- Evidence artifact needed: `build/protocol_final/manufacturing_constraints.json` with payload-only feasibility results and geometry-aware manufacturability results for generated and baseline samples.

### TODO Prototype

- File/function: new function candidate `CLI/aircraft_diffusion_cfd.py::evaluate_structural_load_paths`
- Current behavior: No function-level structural analysis exists in the inspected files. `ConnectivityLoss` only penalizes disconnected voxel components during training, and `condition_feasibility` checks condition bounds.
- Scientific gap: The `Structurally viable` gate requires a structural or load-path analysis and at least one explicit structural baseline. Connectivity is only a proxy and cannot support structural viability wording.
- Proposed implementation: Add a structural analysis entry point that estimates load paths from wing/tail/body segmentation, material assumptions, payload/engine loads, spar continuity, bending moment proxies, and safety factor proxies. The function should report `blocked` unless material model, load cases, geometry scale, and baseline are provided.
- Tests needed: Unit tests for blocked missing material/load cases, failing disconnected spar, failing overloaded wing, passing hand-built baseline under a simple declared load case, and report schema stability.
- Evidence artifact needed: Structural reports for generated samples and baselines, including material assumptions, load cases, safety-factor proxy outputs, and explicit baseline comparison.

---

## Gate 5: Baseline Statistics

**Required to pass:** Required baselines and minimum seeds must be present in `baseline_statistics.json`.

### TODO Prototype

- File/function: `CLI/multi_seed_eval.py::validate_baseline_policy`
- Current behavior: Requires `baseline_set` and a `baseline_name`; by default it requires `retrieval`, `unconditional_checkpoint`, and `bundled_grounded_stl`. Tests verify missing baseline sets block.
- Scientific gap: The helper validates names but not that the baselines exist, were run under identical evaluation settings, have comparable metrics, or correspond to published prior methods.
- Proposed implementation: Extend the policy validator to resolve each baseline entry to a concrete artifact/config, verify geometry/checkpoint hashes, evaluation protocol identity, metric availability, solver settings, seed counts, and corpus split usage. Add a separate category for named prior-method reproductions.
- Tests needed: Add tests for missing baseline artifact, mismatched solver settings, missing metric, insufficient seed count per baseline, successful complete baseline package, and prior-method baseline metadata.
- Evidence artifact needed: Baseline package with retrieval baseline outputs, unconditional checkpoint outputs, bundled grounded STL results, and any prior-method reproduction results under the same metrics.

### TODO Prototype

- File/function: `CLI/multi_seed_eval.py::build_statistical_summary`
- Current behavior: Computes mean, sample standard deviation, count, and blocks when seed or metric counts are below `min_seeds`.
- Scientific gap: Mean/std alone do not support superiority claims. The gate needs effect sizes, confidence intervals, paired/unpaired test selection, multiple-baseline comparisons, and explicit uncertainty reporting.
- Proposed implementation: Add statistical comparison helpers that compute paired differences where seeds are matched, bootstrap confidence intervals, nonparametric tests where appropriate, correction for multiple comparisons, and a fail-closed interpretation field that distinguishes "no superiority" from "insufficient evidence".
- Tests needed: Add deterministic tests for paired superiority, no significant difference, insufficient paired seeds, NaN metric rejection, and multiple baseline correction.
- Evidence artifact needed: `build/protocol_final/baseline_statistics.json` with per-baseline metrics, seeds, confidence intervals, effect sizes, p-values where justified, and a clear status.

---

## Gate 6: Final Evidence Package

**Required to pass:** `CLI/final_evidence.py` must report `status: pass` only when all required evidence reports pass.

### TODO Prototype

- File/function: `CLI/final_evidence.py::evaluate_final_evidence_package`
- Current behavior: Checks five report dictionaries by key and requires each `status` to equal `pass`. It does not verify report schema versions, artifact hashes, run IDs, protocol identity, timestamps, or cross-report consistency.
- Scientific gap: Reports can be mixed from different checkpoints, manifests, seeds, solver settings, or dates and still pass if each has `status: pass`.
- Proposed implementation: Require a common `run_id`, checkpoint hash, manifest hash, protocol hash, solver settings, timestamp, and schema version across all reports. Fail closed on missing or inconsistent identifiers. Include a `claim_unlocks` map that states which paper claims are supported by which gates.
- Tests needed: Add tests for mismatched checkpoint hashes, mismatched manifest hashes, missing run IDs, unsupported schema versions, all-consistent pass, and a report that passes individually but is stale relative to the protocol hash.
- Evidence artifact needed: `build/protocol_final/final_evidence_package.json` produced from a single consistent final run bundle.

---

## Gate 7: Generates Aircraft Structures

**Required to pass:** Generated samples must pass aircraft-specific validity checks against a hand-built aircraft-like template or curated aircraft corpus.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::AircraftDesignDataset._generate_geometries`
- Current behavior: Synthetic procedural geometries are generated, with occasional bundled STLs mixed in when available. This is useful for smoke training but is not a curated aircraft corpus.
- Scientific gap: Training/evaluation on procedural/freeform geometry cannot substantiate that generated structures are aircraft structures. The current function can make the model learn procedural scaffold bias rather than grounded aircraft structure.
- Proposed implementation: Add a separate dataset path for curated aircraft corpus records and keep procedural data as smoke/bootstrap only. Final runs should fail unless the dataset metadata declares `data_source: grounded_manifest` and passes claim-bearing manifest validation.
- Tests needed: Assert final run rejects procedural-only datasets, accepts claim-bearing manifest datasets, and labels procedural output as smoke-only in metadata.
- Evidence artifact needed: Curated aircraft-like corpus with geometry, conditions, provenance, train/holdout split, and generated samples evaluated by the aircraft validity suite.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::OptimizedAircraftGenerator.voxels_to_stl`
- Current behavior: Converts voxels to STL using marching cubes or voxel cubes, with optional simplification. It does not validate mesh watertightness, scale, normals, self-intersections, or component labeling before export.
- Scientific gap: STL export does not prove the generated structure is an aircraft or a valid evaluable mesh. Mesh defects can invalidate downstream CFD/structural reports.
- Proposed implementation: Add a mesh validation/reporting function invoked during claim-eval export: watertightness, connected components, face orientation, nonmanifold edges, physical scale bounds, and link back to voxel validity report.
- Tests needed: Add tests for empty mesh fallback, nonmanifold mesh rejection, watertight cube fixture, aircraft fixture, and scale metadata.
- Evidence artifact needed: Mesh QA report for each generated STL included in the aircraft validity package.

---

## Gate 8: Aerodynamically Optimized

**Required to pass:** Generated candidates must outperform or consistently match a named baseline under fixed CFD settings with `C_L`, `C_D`, `L/D`, and reference-area normalization.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::AdvancedCFDSimulator.simulate_aerodynamics`
- Current behavior: Runs the D3Q27 solver, optionally runs a higher-resolution AMR pass, then optionally blends in `_run_fluidx3d_validation`. The FluidX3D method is currently a simplified approximation returning values from a hardcoded volume proxy.
- Scientific gap: Blended heuristic/approximate CFD outputs cannot support aerodynamic optimization claims. There is no convergence criterion, reference-area normalization report, flow condition report, solver agreement report, or external solver validation.
- Proposed implementation: Add an explicit `label_tier` and `solver_provenance` report. For claim-bearing aerodynamic metrics, require a high-fidelity solver path or validated D3Q27 path with convergence checks, reference area computation, freestream settings, mesh/grid details, residual/convergence data, and no heuristic FluidX3D approximation blending. Return `blocked` if external validation is unavailable where required.
- Tests needed: Add tests that heuristic FluidX3D approximation is never labeled claim-bearing, missing convergence data blocks, reference area is included, solver settings are included, and external solver outputs are not silently blended without provenance.
- Evidence artifact needed: CFD reports for generated candidates and baselines under identical solver settings, including `C_L`, `C_D`, `L/D`, reference area, convergence, and solver provenance.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::evaluate_baselines`
- Current behavior: Looks for `F-18_Hornet.stl` and `biplane.stl`, voxelizes them, runs CFD, and writes raw results keyed by filename. It does not read `baseline_config.yaml`, enforce required baseline set, normalize against generated candidates, or emit report status.
- Scientific gap: The current report is a smoke baseline report, not a controlled comparison. It cannot establish that generated candidates outperform or match baselines.
- Proposed implementation: Make baseline evaluation config-driven, require fixed solver settings shared with generated evaluation, include baseline IDs and artifact hashes, compute reference area consistently, include `status`, and feed the results into `baseline_statistics.json`.
- Tests needed: Add tests for config parsing, missing required baselines, identical solver settings across baseline/generated runs, output schema with `status`, and blocked status when no grounded baselines exist.
- Evidence artifact needed: Controlled baseline report with named baselines, exact CFD settings, normalized metrics, and generated-vs-baseline comparison.

---

## Gate 9: Structurally Viable

**Required to pass:** Structural/manufacturing condition gates pass and structural reports exist against an explicit structural baseline.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::ConnectivityLoss.forward`
- Current behavior: Penalizes disconnected voxel components during training by largest-component fraction.
- Scientific gap: Connectivity is not structural viability. It does not account for loads, material strength, stiffness, fatigue, buckling, spar continuity, control surface loads, or manufacturability.
- Proposed implementation: Keep this loss as a training heuristic only. Add report metadata and docs that it is not accepted as a structural gate. Use the proposed `evaluate_structural_load_paths` or a future FEA/external structural solver for claim-bearing reports.
- Tests needed: Add tests ensuring structural evidence packaging rejects connectivity-only reports, and that training logs label connectivity as heuristic.
- Evidence artifact needed: Structural analysis reports, not connectivity loss logs.

### TODO Prototype

- File/function: `CLI/condition_feasibility.py::validate_condition_feasibility`
- Current behavior: Rejects impossible condition payloads but does not inspect geometry.
- Scientific gap: A condition payload can be feasible while the generated geometry is structurally impossible.
- Proposed implementation: Add geometry-aware manufacturing/structural companion functions and require both payload and geometry reports for structural claims.
- Tests needed: Payload passes but geometry fails; payload fails before geometry; both pass for a validated fixture.
- Evidence artifact needed: Combined structural/manufacturing report linked to each generated sample and baseline.

---

## Gate 10: CFD-Guided Training

**Required to pass:** An ablation with and without the CFD term must show that the CFD term measurably changes learning dynamics or candidate ranking.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::OptimizedDiffusionTrainer.train_epoch`
- Current behavior: Computes MSE, geometry reconstruction, consistency, connectivity, and an aerodynamic loss every 10 batches. The aerodynamic loss is added to total loss, but no ablation machinery records with-CFD vs without-CFD training dynamics.
- Scientific gap: Presence of an aerodynamic loss term is not evidence of CFD-guided training. The loss may be too sparse, heuristic, noisy, nondifferentiable through CFD, or ineffective. No ranking-change or training-dynamics report exists.
- Proposed implementation: Add an explicit ablation runner that trains matched seeds/configs with `aero_loss_weight=0` and `aero_loss_weight>0`, records loss terms separately, evaluates generated candidates under the same CFD pipeline, and compares ranking changes and metric distributions.
- Tests needed: Add tests for ablation config generation, identical seed/config enforcement except aero weight, report schema, blocked status when one arm is missing, and ranking-change computation on synthetic records.
- Evidence artifact needed: `cfd_guided_training_ablation.json` with matched seed runs, checkpoints, training curves, candidate rankings, CFD metrics, and statistical comparison.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::AerodynamicLoss.forward`
- Current behavior: For each batch item, thresholds voxels, runs CFD, combines volume, drag coefficient, and lift coefficient into a scalar. CFD results are detached from gradient flow because the solver path is not differentiable through the voxel threshold and external simulation.
- Scientific gap: This may act as an expensive scalar penalty without meaningful gradient guidance. It does not report whether candidate ranking changes or whether optimization improves physically meaningful metrics.
- Proposed implementation: Rename/report this as `aerodynamic_scoring_loss` unless a differentiable surrogate or reinforcement-style update is implemented. For claim-bearing CFD-guided training, add either a differentiable surrogate validated against CFD or an explicit sample-rank-update loop with logged acceptance/ranking changes.
- Tests needed: Add tests that the report distinguishes differentiable loss, scoring penalty, and post-hoc ranking. Add a synthetic surrogate test showing gradients reach generator parameters when differentiable mode is claimed.
- Evidence artifact needed: Ablation report plus surrogate validation or ranking-update trace.

---

## Gate 11: Outperforms Prior Approaches

**Required to pass:** A statistically defensible comparison against named prior methods or strong internal baselines using the same evaluation metrics.

### TODO Prototype

- File/function: `CLI/multi_seed_eval.py::validate_baseline_policy`
- Current behavior: Names required internal baseline categories but does not represent prior methods.
- Scientific gap: No prior method reproduction, literature baseline, or externally comparable dataset is present.
- Proposed implementation: Add a `prior_methods` section to the baseline policy schema with required citation, implementation source, version/commit, evaluation protocol, metric mapping, and reproduction status. Block superiority wording if prior methods are absent or not comparable.
- Tests needed: Missing prior method blocks superiority claim; prior method without citation blocks; prior method with mismatched metric blocks; complete prior method package passes policy only when metric compatibility is proven.
- Evidence artifact needed: Reproduced prior-method results or defensible external baseline results under the same corpus, conditions, metrics, and statistical analysis.

### TODO Prototype

- File/function: new function candidate `CLI/multi_seed_eval.py::compare_methods_for_superiority`
- Current behavior: No function computes cross-method superiority.
- Scientific gap: The repo cannot substantiate "outperforms prior approaches" with current helpers.
- Proposed implementation: Implement a method-comparison report that ingests per-method metric tables, validates common sample sets or matched evaluation protocols, computes uncertainty/effect sizes, and returns `pass` only when the superiority criterion is met without lowering thresholds.
- Tests needed: Synthetic matched-method pass, synthetic no-difference fail, missing method blocked, mismatched sample set blocked, insufficient seeds blocked.
- Evidence artifact needed: `prior_method_comparison.json` with method definitions, metrics, seeds, statistical tests, and result interpretation.

---

## Gate 12: Publication-Quality Validation

**Required to pass:** A convergence, sensitivity, or external validation study must be executed and reported.

### TODO Prototype

- File/function: `CLI/advanced_lbm_solver.py::D3Q27CascadedSolver.compute_aerodynamic_coefficients`
- Current behavior: Computes drag/lift coefficient outputs and rich diagnostic fields from the internal LBM path, including projected pressure-drag proxy and shape-drag correction. Tests currently check existence, finite values, and stability behavior, not publication-grade validation.
- Scientific gap: There is no grid convergence index, timestep sensitivity, solver-to-solver agreement, uncertainty estimate, or experimental/reference comparison. Shape-drag correction coefficients also need provenance/calibration documentation before claim-bearing use.
- Proposed implementation: Add validation-study helpers around the solver rather than altering smoke solver behavior: run multiple grid resolutions, multiple timesteps/Mach/Re settings, compare against external solver or reference cases, report observed order/convergence, and mark outputs as `blocked` when validation is missing.
- Tests needed: Add tests for convergence report schema, blocked missing resolution ladder, blocked nonmonotonic/nonconverged study, passing synthetic convergence fixture, and external solver agreement fixture.
- Evidence artifact needed: `solver_validation_study.json` with grid convergence, timestep sensitivity, external solver agreement, and reference-case comparison.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::OptimizedAircraftGenerator.export_openfoam_case`
- Current behavior: Writes a minimal OpenFOAM case directory and returns whether OpenFOAM binaries appear available. It does not run OpenFOAM, parse residuals, parse forces, or compare results.
- Scientific gap: Exporting a case is not external validation. Publication-quality validation requires executing the external solver or documenting why an external reference is used.
- Proposed implementation: Add a separate external validation runner that executes `blockMesh`, `snappyHexMesh`, solver, and force extraction when available; otherwise returns `blocked`. It should parse residuals/forces and compare to internal solver metrics.
- Tests needed: Mock command execution for successful external run, mesh failure, solver failure, missing force output, and comparison report generation.
- Evidence artifact needed: OpenFOAM or other external solver logs, mesh stats, residuals, force coefficients, and agreement report.

---

## Gate 13: Conditioned On Flight Profile And Manufacturing Method

**Required to pass:** The model consumes structured conditions and `CLI/run_condition_benchmark.py` passes under a grounded aircraft-like evaluation corpus.

### TODO Prototype

- File/function: `CLI/aircraft_diffusion_cfd.py::build_condition_vector`
- Current behavior: Converts `DesignSpec` scalar fields and categorical manufacturing method into the documented condition vector. Tests verify shape, dtype, layout, normalization, and generator propagation.
- Scientific gap: Correct vectorization and propagation do not prove conditioning effectiveness. The model may ignore conditions, and the procedural latent builder can leak deterministic condition signatures in ways that are not scientifically equivalent to learned conditioning.
- Proposed implementation: Add condition-effect diagnostics that compare generated outputs under controlled condition changes with identical noise seeds, quantify effect sizes, and compare against unconditioned/prompt-free baselines. Ensure final evidence uses trained checkpoint outputs, not procedural latent construction.
- Tests needed: Mock paired generation to verify same seed/different condition pairing, same condition/same seed reproducibility, unconditioned baseline inclusion, and blocked status when checkpoint outputs are unavailable.
- Evidence artifact needed: Paired generated artifacts, condition payloads, unconditioned baseline outputs, and grounded response metrics.

### TODO Prototype

- File/function: `CLI/run_condition_benchmark.py::FIXED_SWEEPS`
- Current behavior: Defines four sweeps: payload, thrust, maneuverability, and wall thickness, each expecting higher condition to produce higher response metric.
- Scientific gap: The sweeps are useful but incomplete. They do not cover speed, wingspan, takeoff distance, engine count, method-specific manufacturing response, or tradeoff expectations such as higher speed potentially reducing drag at fixed geometry assumptions only under well-defined metrics.
- Proposed implementation: Expand sweeps only when corresponding grounded response metrics and expected directions are scientifically defined. Add sweep metadata describing physical rationale, allowed splits, baseline comparison, and metric provenance requirements.
- Tests needed: Add tests for scientifically defined sweep metadata, blocked sweep with missing provenance, and no pass when expected direction is undefined.
- Evidence artifact needed: Grounded manifest response metrics for every enabled sweep and generated checkpoint evaluation results for those sweeps.

---

## Strict Non-Solutions

Do not make a gate pass by:

- lowering thresholds without scientific justification and a versioned protocol change;
- adding placeholder `response_metrics`;
- treating the two-record minimal manifest as a claim-bearing corpus;
- relabeling smoke outputs as final evidence;
- using bundled STLs alone as publication-grade baselines;
- blending heuristic CFD approximations into claim-bearing external validation;
- treating connectivity loss as structural viability;
- claiming conditioning from vector propagation tests alone;
- claiming superiority without named prior/internal baselines and statistical comparison.

---

## Evidence Blockers

Current blockers that require real data or external runs:

- A real aircraft or aircraft-like grounded corpus with documented schema, provenance, preprocessing, units, and splits is missing.
- A claim-bearing checkpoint trained under the final protocol on a valid corpus is missing.
- Multiple generated sample artifacts linked to seeds, conditions, and checkpoint hash are missing.
- Grounded response metrics for condition sweeps are missing.
- Structural/load-path analysis reports are missing.
- Controlled generated-vs-baseline CFD reports with convergence/reference-area normalization are missing.
- Required internal baselines are incomplete; prior-method comparisons are not attempted.
- Publication-quality convergence, sensitivity, and external solver validation studies are not present.

Until those blockers are resolved, the correct claim boundary remains the one in `paper/FINAL_RUN_GATES.md`: proof-of-concept latent generative pipeline with CFD-informed scoring path and partial structured-conditioning plumbing, not scientifically validated aircraft-design results.
