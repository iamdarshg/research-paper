# Aircraft Validity Suite

This suite adds first-pass aircraft-specific checks beyond generic voxel
connectivity. It is a claim gate, not a claim by itself.

## Executable Checks

`CLI/aircraft_validity.py` reports:

- non-empty but bounded occupancy
- bilateral symmetry
- span and length sanity
- center-body plus left/right wing occupancy balance
- tail/body plausibility proxy

The output JSON uses `status: pass` or `status: fail`, with per-check booleans,
failed check names, and metrics.

## Current Boundary

Passing this suite only means a generated voxel artifact clears a lightweight
aircraft-shape plausibility filter. It does not prove aerodynamic performance,
structural viability, manufacturability, or publication-grade aircraft design.

Any future claim that the repo generates aircraft structures must also include
grounded corpus evidence, baseline comparison, structural/manufacturing gates,
and final evidence package status.
