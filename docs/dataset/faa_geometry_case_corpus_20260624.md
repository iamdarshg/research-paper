# FAA/OpenSky Geometry-Case Corpus Report

Date: 2026-06-24

This report documents the generated FAA/OpenSky geometry-case corpus. The goal
is to give each selected FAA/OpenSky flight-regime case a geometry reference
while excluding the older airfoil-section corpus from active training evidence.

## Inputs

- Flight-regime case manifest:
  `build/faa_flight_regime_corpus_20260623/observed_case_manifest_opensky_202004_50k.jsonl`
- Whole-aircraft geometry manifest:
  `build/expanded_aircraft_corpus_20260622/manifest.jsonl`
- Geometry sources in that expanded manifest:
  - 355 OpenVSP Airshow `96^3` records
  - 15 NASA CRM whole-aircraft `96^3` records

## Command

```powershell
python CLI\build_faa_geometry_case_manifest.py `
  --flight-case-manifest build\faa_flight_regime_corpus_20260623\observed_case_manifest_opensky_202004_50k.jsonl `
  --geometry-manifest build\expanded_aircraft_corpus_20260622\manifest.jsonl `
  --output-manifest build\faa_geometry_case_corpus_20260624\geometry_case_manifest_5k.jsonl `
  --report build\faa_geometry_case_corpus_20260624\geometry_case_report_5k.json `
  --target-records 5000 `
  --run-id faa-geometry-case-corpus-5k-20260624
```

## Output Summary

- Output manifest:
  `build/faa_geometry_case_corpus_20260624/geometry_case_manifest_5k.jsonl`
- Output report:
  `build/faa_geometry_case_corpus_20260624/geometry_case_report_5k.json`
- Records: 5,000
- Records with `geometry_path`: 5,000
- Eligible whole-aircraft geometry records: 370
- Unique geometry associations used: 370
- Airfoil-section geometry references: 0
- Splits:
  - train: 4,000
  - validation: 500
  - holdout: 500

Validation:

```powershell
python CLI\validate_manifest.py `
  --manifest build\faa_geometry_case_corpus_20260624\geometry_case_manifest_5k.jsonl `
  --level claim-bearing `
  --output build\faa_geometry_case_corpus_20260624\manifest_validation_claim_bearing.json
```

Result:

- status: `pass`
- record count: 5,000
- manifest SHA-256: `fb92c0f05c9c3d67cda8c0a3f7841af76f0ca54440acf7d08e8f7d2fa4781fca`

## Claim Boundary

Each row has a real geometry reference, but the assigned geometry is a
deterministic whole-aircraft proxy selected from the Airshow+NASA geometry pool.
The geometry is not exact CAD for the observed FAA/OpenSky aircraft registration
or route.

This corpus supports geometry-bearing conditioning experiments at 5,000-record
scale. It does not establish 5,000 unique aircraft geometries or type-exact FAA
geometry ground truth.
