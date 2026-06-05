# NASA CRM Whole-Aircraft Corpus Report

- Generated: `2026-06-05`
- Manifest: `D:\CodeProjects\research-paper\docs\dataset\nasa_crm_whole_aircraft_manifest.jsonl`
- Provenance ledger: `D:\CodeProjects\research-paper\docs\dataset\nasa_crm_whole_aircraft_provenance.json`
- Record count: `15`

## Sources

- CRM-HL assembled geometry page: `https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/assembled-geometry/`
- CRM-HL reference geometry page: `https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/reference-geometry/`
- CRM-HL model-specific geometry page: `https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/model-specific-geometry/`
- CRM-HL bare model page: `https://commonresearchmodel.larc.nasa.gov/crm-hl-reference-geometry/crm-hl-bare-cad-model/`
- High-speed CRM STP page: `https://commonresearchmodel.larc.nasa.gov/geometry/stp-files/`
- DPW6 geometry page: `https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/`
- NASA data-use policy context: `https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy`

## Included Corpus

- All records are public NASA CRM whole-aircraft or semispan aircraft-like CAD assets converted to STL and voxelized locally.
- Semispan geometries were mirrored into full-aircraft STL artifacts before aircraft-validity analysis.
- Scale-model NTF assets were retained, but represented design-envelope fields were scale-corrected from the official 2.7% or 5.2% factors.
- One exception was required: the `NASA5p2` landing STEP extents matched full-scale CRM-HL dimensions, so it was treated as full-scale geometry rather than scale-corrected a second time.
- Split counts: `{"holdout": 9, "test": 3, "val": 3}`
- Design-family counts: `{"nasa_crm_hl_transport": 6, "nasa_crm_hs_transport": 9}`
- Source-page counts: `{"https://commonresearchmodel.larc.nasa.gov/crm-hl-reference-geometry/crm-hl-bare-cad-model/": 1, "https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/": 8, "https://commonresearchmodel.larc.nasa.gov/geometry/stp-files/": 1, "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/assembled-geometry/": 2, "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/model-specific-geometry/": 3}`

## Preprocessing

- Intake is driven by the checked-in `docs/dataset/nasa_crm_source_catalog.json` source catalog so new ready entries can be added without modifying builder code.
- Official STEP files were downloaded from NASA CRM pages, hashed, and extracted locally from zip archives.
- CAD triangulation used CadQuery/OpenCascade with fixed STL export tolerances.
- Full-aircraft STL artifacts were voxelized at `32^3` with `AircraftDesignDataset._voxelize_stl`.
- Local analysis reports used the repo internal `D3Q27` solver on CPU with fixed settings.

## Validation

- Every manifest record has a local aircraft-validity report and grounded response metrics derived from reproducible local analysis.
- Representative grid-refinement report: `D:\CodeProjects\research-paper\docs\dataset\nasa_crm_whole_aircraft\reports\refinement\grid_refinement.json`
- These reports are internal consistency evidence, not publication-grade aerodynamic coefficient validation.

## Exclusions And Boundaries

- A broken CRM65 icing-page zip was excluded because the downloaded `.stp` payload was only web-metadata text, not geometry.
- DPW7 IGES-only assets were not included because this builder currently standardizes on STEP-based conversion for reproducibility on this host.
- Manufacturing fields are mapped to the repo's supported conditioning schema categories; they are not claims about real transport-aircraft factory processes.
- Target speed, payload, thrust, and takeoff fields are bounded inferences from configuration family and represented dimensions, not imported flight-test values.

## Gate Support

- Supports `validate_manifest.py --level claim-bearing`.
- Supports `CLI/aircraft_validity.py`-style whole-aircraft geometry checks on public-source records.
- Supports `condition_feasibility.py` because every record contains complete manufacturing fields.
- Does not by itself satisfy the final protocol `min_grounded_records >= 20` requirement, because this package contains fewer than 20 records.
- Does not by itself establish external CFD validation or paper-level aerodynamic accuracy claims.

## Split Limits

- This package is deterministic, but source-family leakage cannot be fully removed because every record comes from the NASA CRM ecosystem.
- The split therefore separates high-lift reference, wind-tunnel model, and high-speed DPW contexts rather than pretending they are unrelated aircraft families.

