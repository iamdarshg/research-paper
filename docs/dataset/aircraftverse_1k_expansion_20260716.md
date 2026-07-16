# AircraftVerse Expansion To A 1,069-Aircraft Combined Corpus

Generated: `2026-07-16`

## What Was Built

`build/grounded_combined_1k_20260716/manifest.jsonl`: 1,069 unique canonical
`96^3` aircraft geometries, claim-level validation **pass** (unique-geometry
target 1,000).

| Source | Records |
| --- | ---: |
| `build/aircraftverse_geometry_only_20260716/aircraftverse_manifest.jsonl` | 720 |
| `build/grounded_combined_exact_20260714/manifest.jsonl` (canonical fixed-wing + OpenVSP airshow exact) | 349 |

Splits: 758 train / 100 val / 103 test / 108 holdout. All 1,069 geometry
files verified present on disk (25-record random sample re-checked at combine
time; combine tool enforces hash uniqueness).

## AircraftVerse Ingestion

All three Zenodo shards of record 6525446 (v1.0.0) were examined with the
fail-closed builder (`CLI/build_aircraftverse_corpus.py`, seed 20260715,
grid 96, `--allow-unavailable-performance`; conditioning fields remain null
and masked — source metadata is never converted into mission labels):

| Stage | Accepted (cumulative) |
| --- | ---: |
| Shard 1 (previous 20260715 build, target 200) | 200 |
| Shard 1 remainder (`--skip-examined` resume) | 285 |
| Shard 2 (full) | 538 |
| Shard 3 (stopped at target) | 720 |

Acceptance rate was ~2.7-3.1% per shard; the dominant rejection is
`source_interference` (self-intersecting source CAD), followed by
`aircraft_validity_failed`. The complete rejection ledger with per-design
codes is `build/aircraftverse_geometry_only_20260716/rejections.jsonl`
(9,000+ entries, cumulative across resumes).

Because local disk was constrained (~5 GB working budget), shards 2 and 3
(~4.1 GB each) were downloaded, ingested, and deleted sequentially; only the
compact voxel/manifest outputs were retained. Archive provenance (URL, size,
MD5/SHA-256, member hashes) is embedded per record, so any shard can be
re-fetched and re-verified.

## Claim Boundary

Records certify manifest completeness, provenance, and distinct canonical CAD
identities, not flightworthiness or experimental validation. AircraftVerse
records carry no validated performance labels in this corpus; their
conditioning availability masks say so explicitly.
