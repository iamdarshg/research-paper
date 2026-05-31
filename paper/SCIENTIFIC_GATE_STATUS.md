# Scientific Gate Status

This file reports the current gate state for PR #37. It separates repository
readiness from claim-bearing science evidence so documentation work cannot be
mistaken for experimental validation.

## Summary

| Status class | Count | Meaning |
| --- | ---: | --- |
| Gate documented | 13 / 13 | The gate is named, scoped, and mapped to evidence requirements. |
| Executable/reporting scaffold present | 8 / 13 | The repo has runnable code or protocol hooks for the gate. |
| Claim-bearing scientific evidence passed | 0 / 13 | No publication-grade corpus/checkpoint/report bundle exists yet. |

The current branch is therefore strong on documented, fail-closed gate coverage,
but still blocked for scientific aircraft-generation claims.

## Gate Table

| # | Gate | Documentation status | Executable/report status | Claim-bearing science status |
| ---: | --- | --- | --- | --- |
| 1 | Manifest validation | pass | pass: `CLI/validate_manifest.py` | blocked: minimal manifest intentionally lacks claim-bearing provenance |
| 2 | Aircraft validity | pass | pass: `CLI/aircraft_validity.py` batch report scaffold | blocked: no generated claim-eval voxel set |
| 3 | Grounded condition response | pass | pass: `CLI/run_condition_benchmark.py` fail-closed harness | blocked: no grounded response metrics/corpus |
| 4 | Manufacturing and structural condition feasibility | pass | pass: `CLI/condition_feasibility.py` payload report scaffold | blocked: no geometry-aware structural/load-path report |
| 5 | Baseline statistics | pass | pass: `CLI/multi_seed_eval.py` policy/stat summary helpers | blocked: required baselines and final metric tables missing |
| 6 | Final evidence package | pass | pass: `CLI/final_evidence.py` consistency-aware aggregator | blocked: required report bundle missing |
| 7 | Generates aircraft structures | pass | partial: validity gate and corpus contract exist | blocked: no claim-bearing generated sample package |
| 8 | Aerodynamically optimized | pass | partial: baseline/CFD smoke paths exist | blocked: no converged CFD comparison against baselines |
| 9 | Structurally viable | pass | partial: feasibility guards exist | blocked: no structural analysis or load-case evidence |
| 10 | CFD-guided training | pass | partial: training path contains CFD-informed scoring | blocked: no matched ablation with and without CFD term |
| 11 | Outperforms prior approaches | pass | partial: baseline policy names required internal baselines | blocked: no prior-method or superiority comparison package |
| 12 | Publication-quality validation | pass | partial: source map and protocol requirements documented | blocked: no convergence/sensitivity/external-validation study |
| 13 | Conditioned on flight profile and manufacturing method | pass | pass: condition schema/vector/protocol harnesses exist | blocked: no grounded generated-output response evidence |

## Verification Snapshot

Fresh local verification for this branch should include:

```bash
python -m pytest tests -q
python CLI/run_protocol.py --config CLI/run_protocols/final_cloud.yaml --dry-run
python CLI/validate_manifest.py --manifest docs/dataset/minimal_grounded_manifest.jsonl --level claim-bearing
python CLI/final_evidence.py
```

Expected outcomes:

- the test suite passes;
- the final protocol dry run prints the fail-closed evidence sequence;
- the minimal manifest blocks at claim-bearing level;
- the final evidence package blocks until all required reports exist.

