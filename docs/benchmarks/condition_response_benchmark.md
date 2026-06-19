# Grounded Condition-Response Benchmark

This benchmark is the first fail-closed layer for checking whether grounded
condition changes have directional response evidence. It is not a training
recipe, and it is not publication-grade proof by itself.

## Command

```bash
python CLI/run_condition_benchmark.py \
  --checkpoint checkpoints/final_optimized_model.pt \
  --manifest docs/dataset/grounded_condition_manifest.jsonl \
  --output build/condition_benchmark_report.json \
  --seeds 0-4
```

The runner exits with:

- `0` when every fixed sweep passes
- `1` when grounded evidence exists but at least one directional check fails
- `2` when the benchmark is blocked before claim-bearing evidence can be read

## Required Manifest Metadata

Each record must include:

- `split`
- `design_spec`
- `response_metrics`

The initial fixed sweeps require these `design_spec` fields:

- `payload_mass_max_g`
- `required_static_thrust_n`
- `turn_rate_min_deg_s`
- `wall_thickness_min_mm`

The initial fixed sweeps require these `response_metrics` fields:

- `payload_response`
- `thrust_response`
- `maneuverability_response`
- `structural_response`

These response metrics must come from grounded evaluation artifacts or generated
artifact evaluations that are traceable back to the checkpoint and manifest
record. Placeholder values should keep the benchmark blocked rather than being
treated as evidence.

## Fixed Sweeps

The runner sorts records by the named condition field, splits them into low and
high condition groups, and compares mean response metrics.

| Sweep | Condition field | Response metric | Expected direction |
| --- | --- | --- | --- |
| `payload_increase` | `payload_mass_max_g` | `payload_response` | high > low |
| `thrust_increase` | `required_static_thrust_n` | `thrust_response` | high > low |
| `maneuverability_increase` | `turn_rate_min_deg_s` | `maneuverability_response` | high > low |
| `wall_thickness_increase` | `wall_thickness_min_mm` | `structural_response` | high > low |

## Report Contract

The JSON report includes:

- `status`: `pass`, `fail`, or `blocked`
- `record_count`
- `seeds`
- `checkpoint_checked`
- `blockers`
- `sweeps`
- `claim_boundary`

The benchmark checks manifest sufficiency before checking that the checkpoint
exists. This preserves the intended fail-closed behavior for minimal wiring
manifests: a weak manifest should be reported as blocked without implying that a
checkpoint was scientifically evaluated.

## Current Claim Boundary

A passing report means the checked manifest metadata has directional response
support for the fixed sweeps. It does not establish aircraft-level validity,
structural feasibility, statistical superiority, or publication-grade conditioned
generation. Those remain separate final-run gates.
