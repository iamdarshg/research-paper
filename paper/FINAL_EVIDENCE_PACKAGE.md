# Final Evidence Package

This artifact maps each claim-bearing surface to the executable report required
before the paper, README, or CLI docs can strengthen their wording.

## Required Reports

| Gate | Required report | Current executable source | Claim unlocked only if |
| --- | --- | --- | --- |
| Manifest validation | `manifest_validation.json` | `CLI/validate_manifest.py --level claim-bearing` | status is `pass` |
| Aircraft validity | `aircraft_validity.json` | `CLI/aircraft_validity.py` | multiple generated samples pass |
| Condition response | `condition_benchmark.json` | `CLI/run_condition_benchmark.py` | all fixed sweeps pass on grounded records |
| Manufacturing constraints | `manufacturing_constraints.json` | `CLI/condition_feasibility.py` / `DesignSpec` validation | impossible payloads are rejected and target payloads pass |
| Baseline statistics | `baseline_statistics.json` | `CLI/multi_seed_eval.py` | required baselines and minimum seeds are present |

Each passing report should also carry the same `run_id`, `checkpoint_hash`,
`manifest_hash`, and `protocol_hash`. Mixing reports from different runs can
make every individual JSON file look green while the assembled evidence package
does not describe one reproducible experiment. The checked-in protocol now
re-validates the manifest after training so the manifest report can share the
same checkpoint lineage fields as the later gate reports.

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
stay blocked. Passing smoke or wiring checks is not enough.

The latest local final protocol run writes a passing
`build/protocol_final/final_evidence_package.json` with aligned lineage fields.
That unlocks reporting the reduced evidence bundle itself. It does not unlock
publication-grade claims about aerodynamic optimality, structural viability, or
superiority over mature optimization baselines; those still require stronger
solver validation, structural analysis, larger grounded data, and ablations.

## Airshow Public-Corpus Smoke Addendum

The 2026-06-20 Airshow run is a supplemental smoke package, not a replacement
for the reduced final protocol above. It builds
`build/airshow_grounded_corpus_20260620/manifest.jsonl` from 355 public VSP
Airshow geometry records, validates that manifest at claim-bearing level, trains
a three-epoch D3Q27 checkpoint at
`build/airshow_training_20260620/checkpoints/final_optimized_model.pt`, and
runs three generated flight-path checks through STL export, aircraft validity,
and internal D3Q27 CFD.

The addendum supports only a public-corpus code-path claim. All three generated
flight-path checks are nonempty and have finite raw D3Q27 outputs, but all three
fail the current aircraft-specific `span_sanity` validity check. The generated
results must therefore remain non-claim-bearing for aircraft validity,
aerodynamic prediction, structural viability, and method superiority.

Tracked Airshow documentation and paper figures:

- `docs/benchmarks/airshow_grounded_training_20260620.md`
- `docs/benchmarks/airshow_resolution_sweep_20260620.md`
- `docs/dataset/airshow_corpus_addition_report_20260620.md`
- `docs/dataset/airshow_corpus_replication_20260620.md`
- `paper/figures/airshow_corpus_summary.png`
- `paper/figures/airshow_training_losses.png`
- `paper/figures/airshow_flight_path_metrics.png`
- `paper/figures/airshow_generated_geometry.png`
- `paper/figures/airshow_flight_path_metrics_g32.png`
- `paper/figures/airshow_generated_geometry_g32.png`

The higher-resolution addendum does not unlock stronger claims. The `32^3`
Airshow checkpoint still produced three generated cases that failed
aircraft-validity checks, and the `64^3` run validated the corpus but did not
produce a checkpoint within the local run ceiling.
