# Grounded Aircraft Corpus Report

- Manifest: `D:\CodeProjects\research-paper\docs\dataset\grounded_aircraft_manifest.jsonl`
- Provenance ledger: `D:\CodeProjects\research-paper\docs\dataset\grounded_aircraft_provenance.json`
- Record count: `20`

## Sources

- Primary geometry source: public `Extrality/NACA_simulation` NACA generator code under `ODbL-1.0`.
- AirfRANS dataset context: https://airfrans.readthedocs.io/en/latest/notes/introduction.html
- AirfRANS library page: https://github.com/Extrality/airfrans_lib
- NASA CRM benchmark context: https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/
- NASA TMR benchmark context: https://tmbwg.github.io/turbmodels/naca0012numerics_grids.html and https://tmbwg.github.io/turbmodels/onerawingnumerics_grids.html

## Included Corpus

- This manifest is airfoil-section-heavy rather than a full-aircraft corpus.
- All 20 records are watertight 3D extrusions of public NACA 4-digit or 5-digit section profiles.
- Split counts: `{"holdout": 3, "test": 5, "train": 8, "val": 4}`
- Design-family counts: `{"airfoil_section_4digit": 16, "airfoil_section_5digit": 4}`
- Manufacturing-method counts: `{"composite_wet_layup": 7, "foam_core_hotwire": 4, "sheet_balsa_tabbed": 9}`

## Exclusions

- No opaque local STL smoke fixtures were promoted into this manifest.
- NASA CRM STEP assets were kept as validation context only; they were not required for the 20-record claim-bearing manifest because local STEP triangulation was too expensive on this host for this turn.
- No flight-test, wind-tunnel payload, or propulsion claims were imported. Design-spec bounds beyond geometry scale are explicitly inferred and marked as such in provenance.

## Preprocessing

- Unit-chord NACA profiles were generated from the public source code, scaled to meters, and extruded into watertight STL solids.
- STL files were voxelized with the repo's `AircraftDesignDataset._voxelize_stl` path at `32^3` resolution.
- Local analysis reports use the repo's internal `D3Q27` solver on CPU with fixed settings.

## Validation

- Per-record local geometry/CFD reports were generated and used to populate `response_metrics`.
- Representative refinement study: `D:\CodeProjects\research-paper\docs\dataset\grounded_aircraft\reports\refinement\grid_refinement.json`
- Response metrics are grounded local proxies, not published aerodynamic coefficients or structural certification data.
- Claim-bearing manifest validation passed on `2026-06-03` for `20` records.
- Grounded condition benchmark passed on `2026-06-03` with seeds `0-7`.
- Manufacturing feasibility passed on `2026-06-03` for all `20` records.
- Aircraft validity failed on `2026-06-03` with `0/20` passing because the corpus contains airfoil-section extrusions rather than fuselage-wing-tail aircraft bodies.

## Gate Support

- Supports `validate_manifest.py --level claim-bearing`.
- Supports `run_condition_benchmark.py` at the current manifest-grounded contract because all records contain explicit grounded response metrics.
- Supports `condition_feasibility.py` because every record has complete manufacturing fields.
- Does not by itself unlock whole-aircraft validity claims; `CLI/aircraft_validity.py` failed on all `20` corpus voxels because the heuristic suite assumes fuselage-wing-tail structure.
- Does not unlock publication-grade aerodynamic optimization or external solver validation; the local reports are bounded internal-solver evidence only.

## Limits

- Whole-aircraft evidence: absent from the manifest; still needed for aircraft-structure and planform-claim upgrades.
- Airfoil-only evidence: present and reproducible.
- Solver-validation evidence: limited to internal consistency and refinement trends, with NASA CRM/TMR pages recorded as benchmark context rather than reproduced coefficient agreement.
