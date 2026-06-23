# FAA/OpenSky Flight-Regime Corpus Report

Date: 2026-06-23

This report documents the FAA/OpenSky flight-regime corpus builder added for
aircraft-type operating envelopes and per-observed-flight conditioning cases.
It is a regime/conditioning corpus, not a geometry corpus.

## Sources

- FAA Aircraft Characteristics Database:
  <https://www.faa.gov/airports/engineering/aircraft_char_database>
- FAA aircraft data workbook:
  <https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data>
- OpenSky scientific datasets page:
  <https://opensky-network.org/data/scientific>
- OpenSky/Zenodo COVID-era flight-list record:
  <https://zenodo.org/records/7923702>
- OpenSky aircraft metadata URL recorded for provenance:
  <https://opensky-network.org/datasets/metadata/aircraftDatabase.csv>
- Official military/public envelope seed:
  `docs/dataset/military_flight_specs_seed.json`

The OpenSky input used for the first observed build was:

`flightlist_20200401_20200430.csv.gz`

downloaded from the Zenodo record above into:

`build/source_cache/opensky_flightlist_202004.csv.gz`

## Commands

```powershell
python CLI\build_faa_flight_regime_corpus.py `
  --faa-source build\source_cache\faa_aircraft_data.xlsx `
  --observed-flights-csv build\source_cache\opensky_flightlist_202004.csv.gz `
  --military-specs-json docs\dataset\military_flight_specs_seed.json `
  --output-manifest build\faa_flight_regime_corpus_20260623\regime_manifest_with_opensky_202004.jsonl `
  --report build\faa_flight_regime_corpus_20260623\regime_report_with_opensky_202004.json `
  --output-case-manifest build\faa_flight_regime_corpus_20260623\observed_case_manifest_opensky_202004_50k.jsonl `
  --case-report build\faa_flight_regime_corpus_20260623\observed_case_report_opensky_202004_50k.json `
  --run-id faa-opensky-flight-regime-corpus-20260623 `
  --min-observed-flights 25 `
  --max-observed-cases 50000
```

## Outputs

Type-level regime manifest:

- Path: `build/faa_flight_regime_corpus_20260623/regime_manifest_with_opensky_202004.jsonl`
- Records: 388 FAA aircraft types
- Provenance counts:
  - `observed_adsb`: 225
  - `observed_adsb_and_published_spec`: 1
  - `published_spec`: 4
  - `faa_characteristics_only`: 158

Per-flight observed case manifest:

- Path: `build/faa_flight_regime_corpus_20260623/observed_case_manifest_opensky_202004_50k.jsonl`
- Records: 50,000 observed flight cases
- Provenance counts:
  - `observed_flight_case`: 49,915
  - `observed_flight_case_and_published_spec`: 85
- Largest type buckets in this 50k cap:
  - B738: 6,750
  - B737: 4,284
  - A320: 4,028
  - A319: 1,895
  - A321: 1,854
  - B752: 1,699
  - B763: 1,616
  - E75L: 1,575

## Claim Boundary

FAA-only rows are bounded estimates from FAA characteristics fields. They are
not observed flight telemetry.

OpenSky flight-list rows are previous public flight records. In this build they
provide route duration, great-circle endpoint distance, and route-average speed.
They do not provide full cruise-state trajectories, so route-average speed is
recorded as `route_average_speed_mps`, not `cruise_speed_mps`.

Military rows with official fact-sheet data use published envelope caps such as
maximum speed, ceiling, range, or combat radius. These are not observed
operational distributions.

The 50,000-case manifest reaches the requested scale for flight-regime
conditioning cases. It does not add 50,000 unique aircraft geometries.

## Next Data Step

To replace route-average cases with true trajectory-derived regimes, use
OpenSky Trino/state-vector exports or equivalent ADS-B state CSVs containing
time-series altitude and groundspeed by ICAO type. The builder already accepts
state rows with `groundspeed_mps`, `altitude_m`, and `vertical_rate_mps`; those
will populate direct observed speed and altitude regime statistics.
