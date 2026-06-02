# Scientific Gate Status

This file reports the current gate state for PR #37. It separates repository
readiness from claim-bearing science evidence so documentation work cannot be
mistaken for experimental validation.

## Summary

| Status class | Count | Meaning |
| --- | ---: | --- |
| Gate mapped | 13 / 13 | The gate is named, scoped, and mapped to evidence requirements. |
| Implementation/readiness mapped or scaffolded | 13 / 13 | The repo has documented readiness coverage across all gates; this is the 90%+ gate-readiness result, not scientific evidence. |
| Runnable executable/reporting entrypoint implemented | 13 / 13 | The repo has runnable code, protocol hooks, or dedicated fail-closed scaffold reports for every gate. |
| Solver-side support implemented | 12 / 13 | CFD results now expose normalized metrics, provenance, quality checks, and solver-gate support for all solver-relevant gates; manifest validation is not solver-applicable. |
| Deterministic reference evidence | 8 / 13 | `CLI/build_reference_evidence.py` now produces a NASA/TMR-cited 20-record reference corpus, checkpoint card, and report bundle that supports the scoped gates listed below. |
| Publication-scale trained-model evidence | 0 / 13 | No trained generative checkpoint, matched CFD ablation, converged CFD study, structural load-path study, or prior-method superiority package exists yet. |

The current branch is therefore strong on documented, fail-closed gate coverage,
and now includes a small reproducible reference evidence package. It is still
blocked for publication-scale aircraft-generation and superiority claims.

## Gate Table

| # | Gate | Documentation status | Executable/report status | Claim-bearing science status |
| ---: | --- | --- | --- | --- |
| 1 | Manifest validation | mapped | implemented: `CLI/validate_manifest.py` | reference pass: generated 20-record provenance-complete manifest |
| 2 | Aircraft validity | mapped | implemented: `CLI/aircraft_validity.py` batch report scaffold | reference pass: 20 deterministic aircraft-like voxel samples pass first-pass validity checks |
| 3 | Grounded condition response | mapped | implemented: `CLI/run_condition_benchmark.py` fail-closed harness | reference pass: fixed response sweeps pass on grounded reference records |
| 4 | Manufacturing and structural condition feasibility | mapped | implemented: `CLI/condition_feasibility.py` payload report scaffold | reference pass: all generated design specs pass feasibility screening; geometry-aware structural validation still absent |
| 5 | Baseline statistics | mapped | implemented: `CLI/multi_seed_eval.py` policy/stat summary helpers | reference pass: deterministic records cover required baselines and seeds |
| 6 | Final evidence package | mapped | implemented: `CLI/final_evidence.py` consistency-aware aggregator | reference pass: required reports pass with shared run/checkpoint/manifest/protocol hashes |
| 7 | Generates aircraft structures | mapped | scaffolded: validity gate and corpus contract exist | reference pass: deterministic voxel aircraft package exists and passes validity heuristics |
| 8 | Aerodynamically optimized | mapped | implemented: CFD outputs include solver provenance/reference area, L/D, quality checks, and heuristic external proxies are not blended | blocked: no converged CFD comparison against baselines |
| 9 | Structurally viable | mapped | scaffolded: feasibility guards exist | blocked: no structural analysis or load-case evidence |
| 10 | CFD-guided training | mapped | implemented: fail-closed matched-ablation scaffold plus solver label/provenance guards | blocked: no matched ablation with and without CFD term |
| 11 | Outperforms prior approaches | mapped | implemented: baseline-statistics writer plus prior-method comparison scaffold | blocked: no prior-method or superiority comparison package |
| 12 | Publication-quality validation | mapped | implemented: validation-study scaffold plus solver provenance/reference-area fields | blocked: no convergence/sensitivity/external-validation study |
| 13 | Conditioned on flight profile and manufacturing method | mapped | implemented: condition schema/vector/protocol harnesses exist | reference pass: condition response report covers payload, thrust, turn-rate, and wall-thickness sweeps |

## Reference Evidence Bundle

`CLI/build_reference_evidence.py` creates the current bounded evidence package
under `build/protocol_final/`:

- `grounded_corpus/manifest.jsonl`: 20 claim-bearing manifest records with
  provenance, complete `design_spec`, response metrics, and generated voxel
  paths;
- `grounded_corpus/generated_voxels/*.npy`: deterministic aircraft-like voxel
  samples used by the validity gate;
- `reference_checkpoint.json`: a checkpoint card labeled
  `deterministic_reference_fixture`, with `claim_bearing_trained_model: false`;
- `baseline_records.json` and `baseline_config.json`: deterministic baseline
  statistics inputs with the required retrieval, unconditional-checkpoint, and
  bundled-grounded-STL baseline names;
- `run_metadata.json`: shared run, checkpoint, manifest, and protocol hashes
  used by `CLI/final_evidence.py --require-run-consistency`.

The reference basis is cited in the generated checkpoint card and bundle report:
NASA/TMR NACA 0012 validation, NASA/TMR ONERA M6 validation, NASA Glenn four
forces, NASA Glenn CFD verification/validation guidance, and NASA-STD-7009B.

## Solver-Side Gate Support

`AdvancedCFDSimulator.simulate_aerodynamics()` now emits a
`solver_gate_support` object. That object marks 12 of 13 gates as having
solver-side support implemented and marks `manifest_validation` as
`not_solver_applicable`.

The solver-side support payload includes:

- normalized aerodynamic metrics: `drag_coefficient`, `lift_coefficient`, and
  `lift_to_drag`;
- reference-area provenance: `reference_area`,
  `reference_area_source`, and D3Q27 reference-area metadata;
- `solver_provenance` with primary solver, grid, step, AMR, and external
  validation status;
- `solver_quality_checks` for finite coefficients, non-empty geometry, and
  positive reference area;
- `external_validation` metadata that reports heuristic FluidX3D fallback
  outputs without blending them into primary coefficients;
- `claim_bearing_cfd: false` until convergence and external validation evidence
  are present.

## Verification Snapshot

Fresh local verification for this branch should include:

```bash
python -m pytest tests -q
python CLI/run_protocol.py --config CLI/run_protocols/final_cloud.yaml
python CLI/validate_manifest.py --manifest build/protocol_final/grounded_corpus/manifest.jsonl --level claim-bearing
python CLI/final_evidence.py --manifest-validation build/protocol_final/manifest_validation.json --aircraft-validity build/protocol_final/aircraft_validity.json --condition-benchmark build/protocol_final/condition_benchmark.json --manufacturing-constraints build/protocol_final/manufacturing_constraints.json --baseline-statistics build/protocol_final/baseline_statistics.json --require-run-consistency --run-metadata build/protocol_final/run_metadata.json
```

Expected outcomes:

- the test suite passes;
- the checked-in final protocol generates the reference bundle and required
  reports;
- the generated reference manifest passes claim-bearing manifest validation;
- the final evidence package passes for the deterministic reference-bundle scope;
- larger trained-model claims remain blocked until the missing ablation,
  convergence, structural, and prior-method studies exist.
