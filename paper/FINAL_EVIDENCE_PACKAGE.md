# Final Evidence Package

This artifact maps each claim-bearing surface to the executable report required
before the paper, README, or CLI docs can strengthen their wording.

The current per-gate implementation/evidence status lives in
`paper/SCIENTIFIC_GATE_STATUS.md`.

Current gate readiness is 90%+ only in the mapped/implemented/scaffolded
sense. Claim-bearing scientific evidence remains 0/13 until the reports below
exist and return evidence-level outcomes.

## Required Reports

| Gate | Required report | Current executable source | Claim unlocked only if |
| --- | --- | --- | --- |
| Manifest validation | `manifest_validation.json` | `CLI/validate_manifest.py --level claim-bearing` | status is `pass` |
| Aircraft validity | `aircraft_validity.json` | `CLI/aircraft_validity.py` | multiple generated samples pass |
| Condition response | `condition_benchmark.json` | `CLI/run_condition_benchmark.py` | all fixed sweeps pass on grounded records |
| Manufacturing constraints | `manufacturing_constraints.json` | `CLI/condition_feasibility.py` / `DesignSpec` validation | impossible payloads are rejected and target payloads pass |
| Baseline statistics | `baseline_statistics.json` | `CLI/multi_seed_eval.py` policy helpers | required baselines and minimum seeds are present |

Each passing report should also carry the same `run_id`, `checkpoint_hash`,
`manifest_hash`, and `protocol_hash`. Mixing reports from different runs can
make every individual JSON file look green while the assembled evidence package
does not describe one reproducible experiment.

## Package Evaluator

Use `CLI/final_evidence.py` to combine report statuses:

```bash
python CLI/final_evidence.py \
  --manifest-validation build/protocol_final/manifest_validation.json \
  --aircraft-validity build/protocol_final/aircraft_validity.json \
  --condition-benchmark build/protocol_final/condition_benchmark.json \
  --manufacturing-constraints build/protocol_final/manufacturing_constraints.json \
  --baseline-statistics build/protocol_final/baseline_statistics.json \
  --require-run-consistency \
  --output build/protocol_final/final_evidence_package.json
```

## Decision Rule

If any required report is missing, blocked, or failed, claim-bearing wording must
stay blocked. Successful smoke or wiring checks are not enough.

The current repository still lacks a publication-grade grounded corpus and
claim-bearing result reports, so paper-level aircraft-generation claims remain
blocked at 0/13 until those artifacts exist.
