# AircraftVerse Grounded Corpus Expansion Design

## Purpose

Expand the training corpus past 600 unique grounded whole-vehicle geometries
without counting repeated flow conditions, component-only CAD, visual meshes, or
failed designs as independent aircraft. AircraftVerse is the primary expansion
source because each design includes STL and STEP CAD, a symbolic design tree,
low-level parameters, and physics-derived performance output.

The resulting manifest must distinguish direct source data, measured geometry
data, derived values, and unavailable values. It must not invent generic
aircraft labels merely to satisfy the conditioning schema.

## Source Boundary

The pinned source is Zenodo record `6525446`, version `1.0.0`, published
2023-06-08. The record contains 27,714 aerial-vehicle designs in three ZIP
shards. The first shard is 4,065,179,292 bytes and contains 9,238 design
directories. Each complete directory contains:

- `cadfile.stl`
- `Geom.stp`
- `design_tree.json`
- `design_low_level.json`
- `design_seq.json`
- `output.json`
- `pointCloud.npy`
- `trims.npy`

The Zenodo API reports `CC BY 4.0`; the accompanying paper describes the
dataset as `CC BY-SA`. The manifest and corpus report will preserve both
statements and apply the stricter attribution/share-alike handling. The Zenodo
record URL, archive URL, archive MD5, downloaded archive SHA-256, member names,
and member SHA-256 values are retained.

## Acquisition

The default path is a resumable download of `AircraftVerse_1.zip`. Additional
shards are downloaded sequentially only if validated designs from prior shards
do not clear the combined-corpus target. Each download is accepted only when
its size and Zenodo MD5 match the record. The builder reads members directly
from the ZIP and never extracts a full shard. This requires about 4.1 GB per
active source archive plus accepted voxel artifacts.

HTTP range reads remain available for catalog inspection and small probes, but
not for corpus construction. Zenodo returned HTTP 429 under member-wise range
concurrency, so the production builder must not turn one archive into thousands
of remote requests.

Design IDs are ordered by a pinned SHA-256 selection key rather than archive
order. This gives deterministic coverage across the shard and avoids bias
toward low numeric IDs.

## Admission Gates

Admission is fail-closed. A design is accepted only when every gate passes.

### Completeness

- All six CAD, specification, sequence, and performance members required for
  training and provenance are present.
- JSON members parse as objects or arrays of the documented kind.
- The archive directory key and source-native design name are both present and
  retained. They are not required to be numerically equal because the published
  shard maps container keys to independently named generated designs.
- STL and STEP members are non-empty and have stable content hashes.

### Source Performance

- `Interferences` is exactly zero.
- `Mass`, `Max_Distance`, `Hover_Time`, `Max_Speed`, `Power_MFD`,
  `Power_MxSpd`, and `Speed_MFD` are finite and strictly positive.
- Battery-current, motor-current, motor-power, and control-utilization ratios
  are finite and within the source feasibility interval `[0, 1]`.
- A zero-filled performance record is rejected, even when CAD is present.

These tests establish that the source evaluation considered the design
flyable; they do not claim flight-test validation or certify the fidelity of
the source physics models.

### Geometry

- The STL parses as a non-empty whole assembly with finite vertices and faces.
- All three extents are positive, and the assembly contains no non-finite,
  degenerate, or zero-area geometry after conservative mesh processing.
- The STL hash and normalized voxel hash are new to the combined corpus.
- Voxelization at `96^3` succeeds with nonzero occupancy.
- Existing aircraft-validity, occupancy, component-support, and connectivity
  screens pass. Multi-part assemblies are allowed, but isolated fragments and
  collapsed or near-spherical outputs are rejected.
- Source dimensions recovered from `design_low_level.json` are compared with
  STL extents in source-native units. The maximum mesh extent must be between
  `0.5x` and `50x` the largest positive declared linear component dimension.
  This broad assembly-envelope check catches unit and scale corruption without
  pretending that a local arm, fuselage, or propeller dimension is the complete
  vehicle span. Designs without a recoverable declared linear dimension are
  rejected from the primary lane.

### Specification Integrity

The source-native design tree, component types, parameter values, and
performance output are preserved. Compatibility fields are populated only as
follows:

- Direct fields use source values, such as `Max_Speed`, `Mass`, cargo mass,
  motor count, and component types.
- Measured fields use mesh or voxel measurements, such as assembly extents and
  occupancy.
- Derived fields contain a formula and input provenance in the record.
- Unavailable fixed-wing concepts, such as runway takeoff distance for a VTOL
  design, remain null and masked.

No seeded pseudo-specification is permitted for AircraftVerse records.

## Visual-Model Policy

Visual-only models are disabled in the primary ingestion path. They may enter a
separate supplemental lane only when:

- an authoritative source supplies design dimensions and configuration data;
- mesh dimensions agree with those specifications within a documented
  tolerance;
- the same geometry, connectivity, occupancy, and solver screens pass; and
- the record is labeled as a visual reconstruction rather than engineering
  CAD.

The supplemental lane is unnecessary when validated AircraftVerse records
clear the unique-geometry target and must not be used merely to increase count.

## Manifest And Splits

Each accepted record includes source collection, Zenodo record and member URLs,
license statements, archive and member hashes, source design ID, generator
version, source-native specification, source performance output, mesh metrics,
voxel hash, rejection-gate version, and field-level provenance.

Splits are grouped by AircraftVerse design ID and assigned deterministically.
Content-hash duplicates cannot cross splits. The combined corpus report records
accepted, rejected, duplicate, missing-member, invalid-performance,
invalid-geometry, and invalid-specification counts.

The build target is at least 625 unique combined geometries, providing a buffer
above the 600 claim gate. The validator computes uniqueness from source CAD
hashes, with voxel hashes used as an additional duplicate check rather than as
a substitute for source identity.

## Failure And Resume Behavior

The archive download uses a partial file and resumes only when the server
honors byte ranges. A checksum mismatch deletes neither the archive nor prior
reports; it marks the build blocked and requires an explicit retry.

Per-design failures are recorded with stable rejection codes and do not abort
the scan. The builder checkpoints accepted IDs and hashes so an interrupted
voxelization run can resume without changing deterministic selection order.
The final combined manifest is written atomically only after validation.

## Verification

Tests cover ZIP indexing, deterministic selection, checksum enforcement,
performance rejection, specification provenance, content deduplication,
geometry rejection, grouped splits, resume behavior, and the 600-unique gate.

The corpus build produces:

- the AircraftVerse-only manifest and rejection ledger;
- the combined manifest;
- basic and claim-bearing validation reports;
- aircraft-validity and split-leakage reports;
- exact accepted, rejected, duplicate, family, source, and split counts; and
- a replication document with archive checksums and commands.

Downloaded ZIP, STEP, STL, and voxel artifacts remain under ignored `build/`.
Only code, tests, reports containing bounded metadata, and documentation are
committed.
