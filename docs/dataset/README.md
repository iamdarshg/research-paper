# Dataset Status And Requirements

This repository does not yet ship a grounded aircraft-like training corpus. The current data path is procedural and synthetic: generated voxel geometries, synthetic `DesignSpec` samples, and offline densification artifacts used to exercise the conditioning seam and training pipeline.

## What The Current Data Supports

- Smoke-testing the dataset, model, and generator conditioning interfaces
- Verifying that condition vectors can be serialized, loaded, normalized, and consumed end to end
- Running bounded synthetic/freeform experiments for implementation debugging

## What The Current Data Does Not Support

- Claims about real aircraft geometry distributions
- Claims about mission-conditioned aircraft generation
- Claims about manufacturing-conditioned aircraft generation
- Claims about aerodynamic or structural realism beyond synthetic heuristics
- Condition-response benchmarks that would support paper-level conclusions

## What A Grounded Aircraft-Like Corpus Must Contain

A claim-bearing corpus needs examples that are recognizably aircraft-like and traceable to a consistent data source. At minimum:

- Aircraft-like geometry assets with documented provenance
- One canonical preprocessing path from mesh or CAD to voxel grid
- Stable orientation, scale, and unit conventions
- Enough variation to cover the intended design family rather than a single template
- Explicit mission and manufacturing metadata attached to every example
- Train, validation, and holdout splits that prevent near-duplicate leakage

## Required Metadata Fields

Every example in a grounded corpus should include, at minimum, the fields already represented in the documented condition schema:

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

Recommended additional metadata:

- Geometry provenance or source identifier
- Aircraft family or configuration tag
- Units and preprocessing version
- Voxelization resolution and occupancy threshold
- Any CFD or structural annotations used for evaluation

## Split Rules

To support honest evaluation, a grounded dataset should follow these split rules:

- Split by source design or aircraft family before augmentation so near-duplicates cannot land in multiple splits.
- Keep the holdout split untouched until final evaluation.
- Preserve coverage across manufacturing categories and mission envelopes in train and validation.
- If procedural augmentation is used, record the parent example so leakage can be audited.
- Freeze the split manifest before reporting any benchmark numbers.

## Claims Blocked Until This Exists

The following claims remain blocked until a grounded corpus and evaluation protocol exist:

- Mission-conditioned aircraft generation
- Manufacturing-conditioned aircraft generation
- Aircraft-level geometric validity at useful rates
- Condition-response claims for payload, takeoff, wingspan, wall-thickness, part-count, or manufacturing inputs
- Publication-grade aerodynamic comparison claims tied to conditioned generation

## Practical Reading Of The Current Repo State

The current repo has partial conditioning plumbing, not dataset-backed conditioned validation. The condition vector is real and used by the code path, but the present procedural/synthetic data cannot justify scientific claims about aircraft-like conditioned generation.
