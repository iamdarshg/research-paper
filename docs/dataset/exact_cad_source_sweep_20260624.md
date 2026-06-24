# Exact CAD Source Sweep

Generated: `2026-06-24T10:18:16.156081+00:00`

## Summary

- Exact CAD catalog records: `2363`
- This is a source catalog, not a checked-in binary CAD mirror.
- Large assets, especially HiLiftAeroML STEP/STL/flow fields, are intentionally referenced by URL.
- Third-party generic CAD marketplaces are excluded from the claim-bearing lane until license and provenance review.

| Source collection | Records |
| --- | ---: |
| `hiliftaeroml_crm_hl_surface_runs` | 1800 |
| `hiliftaeroml_crm_hl_variants` | 180 |
| `local_nasa_crm_ready_catalog` | 15 |
| `nasa_uam_reference_vehicles` | 9 |
| `vsp_airshow_public_models` | 359 |

| CAD format | Records |
| --- | ---: |
| `step_zip` | 15 |
| `stl` | 1800 |
| `stp` | 180 |
| `vsp3` | 368 |

## Source Lanes

### VSP Airshow

The live Airshow sweep catalogs license-qualified exact OpenVSP `.vsp3` URLs. These are exact parametric model source files, but they remain community/user-contributed models rather than certified aircraft CAD.

- Public model documents observed: `382`
- Documents with exact VSP URLs: `382`
- License-qualified exact VSP records: `359`

### NASA UAM Reference Vehicles

The NASA UAM lane inspects official NASA OpenVSP ZIP archives and records each `.vsp3` member. These are representative public reference vehicles, not production vehicle drawings.

- OpenVSP archives inspected: `7`
- Archive fetch failures recorded: `1`
- `.vsp3` members found: `9`

### HiLiftAeroML

The HiLiftAeroML lane catalogs one canonical STEP URL per CRM-HL geometry variant. The dataset also provides STL surfaces and force/moment CSVs for ten angles of attack per variant.

- Geometry variants cataloged: `180`
- Exact STL surface run records: `1800`
- Canonical AoA used for CAD URLs: `4 deg`

### NASA CRM Local And Candidate Sources

The local CRM lane mirrors the repository's existing ready source catalog. The candidate sweep is kept separate because some entries are component libraries or format-conversion candidates rather than already-promoted training examples.

- Ready local CRM catalog records: `15`
- CRM candidate groups already tracked: `41`
- Ready CRM candidate groups: `31`
- Apparent candidate CAD records across groups: `1873`

## Recommended Ingestion Order

1. Pull HiLiftAeroML selected STEP, STL, force/moment CSV, and geometry-values CSV files first; it is the strongest exact-CAD-plus-flow-label source.
2. Pull NASA UAM `.vsp3` archives next; they add configuration diversity with official NASA provenance and small payloads.
3. Pull license-qualified Airshow `.vsp3` files in batches, preserving upload metadata and license IDs.
4. Promote CRM candidate-sweep component libraries only after deciding whether component-level STEP files should train whole-aircraft generation or remain validation/context assets.

## Ground-Truth Boundary

This catalog can support exact geometry provenance. It does not by itself prove that every model is a physically valid aircraft, that Airshow community models match real aircraft dimensions, or that inferred mission metadata is factual. Those claims still require solver validation, unit normalization, duplicate/family split control, and source-specific metadata review.

## Catalog Files

- JSON catalog: `docs\dataset\exact_cad_source_catalog_20260624.json`
- Machine report: `docs\dataset\exact_cad_source_report_20260624.json`

## Source References

- NASA UAM reference vehicles: https://www.nasa.gov/reference/uam-refs/
- VSP Airshow: https://airshow.openvsp.org/
- OpenVSP Airshow announcement: https://openvsp.org/blogs/announcements/2024/08/22/openvsp-airshow-is-live
- HiLiftAeroML dataset: https://huggingface.co/datasets/nvidia/HiLiftAeroML
- HiLiftAeroML overview: https://caemldatasets.org/hiliftaeroml/
- NASA Common Research Model original CAD: https://commonresearchmodel.larc.nasa.gov/geometry/original-cad-files/
- AIAA DPW-6 CRM CAD geometry notes: https://www.aiaa-dpw.org/Workshop6/DPW6-geom.html

## First Records

- `hiliftaeroml_surface_geo_LHC001_AoA_10` `stl` from `hiliftaeroml_crm_hl_surface_runs`
- `hiliftaeroml_surface_geo_LHC001_AoA_12` `stl` from `hiliftaeroml_crm_hl_surface_runs`
- `hiliftaeroml_surface_geo_LHC001_AoA_14` `stl` from `hiliftaeroml_crm_hl_surface_runs`
- `hiliftaeroml_surface_geo_LHC001_AoA_16` `stl` from `hiliftaeroml_crm_hl_surface_runs`
- `hiliftaeroml_surface_geo_LHC001_AoA_18` `stl` from `hiliftaeroml_crm_hl_surface_runs`
