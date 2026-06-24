# Dataset Status And Requirements

This repository now contains partial public-source and generated local geometry corpora, but it still does not ship a publication-grade, split-controlled aircraft training corpus. The older procedural/synthetic path remains useful for wiring tests; the newer Airshow, NASA CRM, NASA UAM, and HiLiftAeroML lanes add exact-geometry provenance but still need leakage control, validation splits, and external solver checks before paper-level claims.

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

## Checked-In Minimal Manifest

The repo now includes a minimal runnable manifest at `docs/dataset/minimal_grounded_manifest.jsonl`.

Its purpose is limited:

- exercise the grounded-manifest code path end to end,
- make final-run guardrails point at a real non-empty dataset input,
- provide a versioned template for richer manifests.

It is **not** a scientifically adequate aircraft corpus. It contains only the repository's bundled STL examples and should be treated as wiring validation, not dataset completion.

The repo now also includes an executable validator:

```bash
python CLI/validate_manifest.py --manifest docs/dataset/minimal_grounded_manifest.jsonl --level basic
python CLI/validate_manifest.py --manifest docs/dataset/minimal_grounded_manifest.jsonl --level claim-bearing
```

Current expected behavior:

- `basic` should pass for the checked-in minimal manifest.
- `claim-bearing` should return `blocked`, because the file is intentionally a wiring artifact and not a publication-grade corpus.

See `docs/dataset/GROUNDED_CORPUS_SPEC.md` for the stricter claim-bearing contract and
`docs/dataset/manifest_schema.example.json` for a machine-readable example record shape.

## Active Geometry Corpus Artifacts

The old checked-in NACA airfoil-section corpus has been removed from the active
dataset package because it was not whole-aircraft geometry and was easy to
misread as aircraft-level evidence.

The active whole-aircraft geometry sources are now:

- `docs/dataset/nasa_crm_whole_aircraft_manifest.jsonl`
- `docs/dataset/nasa_crm_whole_aircraft_provenance.json`
- `docs/dataset/NASA_CRM_WHOLE_AIRCRAFT_REPORT.md`
- generated Airshow/NASA expanded geometry corpus:
  `build/expanded_aircraft_corpus_20260622/manifest.jsonl`
- generated Airshow/NASA/HiLift expanded geometry corpus:
  `build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl`

The Airshow/NASA/HiLift local manifest contains `752` 96^3 records after the
HiLiftAeroML STL streaming pass. The HiLift slice contributes AoA-labeled exact
STL records with force/moment labels, but repeated AoA files for the same
`geometry_variant_id` share geometry. Treat them as flow-labeled repeated
geometry records, not as hundreds of independent aircraft designs.

The FAA/OpenSky flight-regime path now has a generated geometry-bearing case
manifest:

- `build/faa_geometry_case_corpus_20260624/geometry_case_manifest_5k.jsonl`
- `build/faa_geometry_case_corpus_20260624/geometry_case_report_5k.json`

This 5,000-record manifest assigns each FAA/OpenSky observed flight case a
whole-aircraft geometry proxy from the 370-record Airshow+NASA `96^3` geometry
pool. It is suitable for geometry-bearing conditioning experiments. It is not a
claim that every FAA/OpenSky row has exact aircraft-type CAD.

## Public Whole-Aircraft Package

The repository now also contains a separate public-source whole-aircraft package
built from official NASA Common Research Model STEP assets plus local geometry
and CFD proxy analysis:

- `docs/dataset/nasa_crm_whole_aircraft_manifest.jsonl`
- `docs/dataset/nasa_crm_whole_aircraft_provenance.json`
- `docs/dataset/NASA_CRM_WHOLE_AIRCRAFT_REPORT.md`
- `docs/dataset/nasa_crm_source_catalog.json`
- `docs/dataset/nasa_crm_whole_aircraft/`

This package is useful for whole-aircraft validity evidence and public-source
geometry provenance, but it is still bounded:

- it is entirely NASA CRM-family geometry, so family leakage cannot be fully removed,
- some design-spec fields are explicit inferences rather than published flight data,
- it contains fewer than the final protocol minimum of 20 grounded records,
- it does not by itself establish external CFD agreement.

The whole-aircraft builder is now catalog-driven. New ready-to-ingest NASA CRM
entries should be added to `docs/dataset/nasa_crm_source_catalog.json`, while
broader discovery work can live in a separate candidate sweep file before those
entries are promoted into the ready catalog.

The broader discovery lane now also has dedicated artifacts:

- `docs/dataset/NASA_CRM_SOURCE_SWEEP.md`
- `docs/dataset/nasa_crm_source_candidates.json`

The exact-CAD discovery lane now has a separate catalog and report:

- `docs/dataset/exact_cad_source_catalog_20260624.json`
- `docs/dataset/exact_cad_source_sweep_20260624.md`
- `docs/benchmarks/hiliftaeroml_96_training_20260624.md`

This catalog records exact OpenVSP/STEP source URLs from VSP Airshow, NASA UAM
reference vehicles, HiLiftAeroML, HiLiftAeroML exact STL surface-run URLs, and
the existing local NASA CRM ready catalog. It is a source catalog, not a binary
CAD mirror or proof that every discovered model is a validated aircraft.

These files are intentionally wider than the ready catalog. They can include
format-conversion-needed, reference-only, or not-yet-promoted candidates that
still need an honest ingestion decision before they are fed into the builder.

Each record may include:

- `geometry_path` or `stl_path`
- `design_spec`
- `condition_vector`
- `latent_path`
- `split`

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
