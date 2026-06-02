# Grounded Corpus Specification

This document defines the difference between:

- a **basic runnable manifest** used for smoke wiring and protocol validation
- a **claim-bearing grounded manifest** that is strict enough to support future conditioned-aircraft evaluation

## Validation Levels

### `basic`

Purpose:
- keep the current repo-level grounded path executable
- validate file resolution and split bookkeeping
- support smoke and protocol wiring

Minimum requirements per record:
- `geometry_path` or `stl_path`
- `split`
- referenced geometry file exists

This level is intentionally permissive. The checked-in `minimal_grounded_manifest.jsonl` is expected to pass `basic`.

### `claim-bearing`

Purpose:
- define the minimum contract for future grounded aircraft-like evaluation
- fail closed when provenance or conditioning metadata are incomplete

Required per record:
- `geometry_path` or `stl_path`
- `split`
- `source_id`
- `geometry_provenance`
- `preprocessing_version`
- `units`
- `design_family`
- `design_spec`

Required `design_spec` fields:
- `target_speed_mps`
- `wingspan_limit_m`
- `thrust_to_weight_min`
- `turn_rate_min_deg_s`
- `required_static_thrust_n`
- `engine_diameter_mm`
- `engine_length_mm`
- `engine_count_min`
- `engine_count_max`
- `payload_mass_min_g`
- `payload_mass_max_g`
- `takeoff_distance_min_m`
- `takeoff_distance_max_m`
- `wall_thickness_min_mm`
- `wall_thickness_max_mm`
- `part_count_min`
- `part_count_max`
- `manufacturing_method`

## Design Notes

- `basic` is not a scientific endorsement. It only says the manifest is usable by the current repo wiring.
- `claim-bearing` is not sufficient by itself for publication-grade results. It is only the entry contract for future grounded evaluation.
- The checked-in minimal manifest should remain a wiring artifact and should currently fail `claim-bearing` validation.

## Current Validator

Use:

```bash
python CLI/validate_manifest.py --manifest docs/dataset/minimal_grounded_manifest.jsonl --level basic
python CLI/validate_manifest.py --manifest docs/dataset/minimal_grounded_manifest.jsonl --level claim-bearing
```

Expected current behavior:
- `basic`: pass
- `claim-bearing`: blocked

## Deterministic Reference Bundle

For PR-level evidence plumbing, the repo now includes a bounded reference-bundle
builder:

```bash
python CLI/build_reference_evidence.py --output-root build/protocol_final --protocol CLI/run_protocols/final_cloud.yaml
```

This produces a 20-record manifest under
`build/protocol_final/grounded_corpus/manifest.jsonl`. The records include
complete claim-bearing manifest fields, deterministic aircraft-like voxel paths,
condition-response metrics, and public NASA/TMR source URLs. The paired
`reference_checkpoint.json` is a checkpoint card for a deterministic fixture,
not a trained diffusion checkpoint.
