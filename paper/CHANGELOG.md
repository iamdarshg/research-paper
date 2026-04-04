# Changelog

## Unreleased

- Added the centered-cube validation object to the benchmark script and documented the exact geometry in `paper/sections/validation-and-testing-standards.tex`.
- Consolidated local verification into `run_internal_benchmark.py`, which now runs the internal D3Q27 solver and the OpenFOAM sonicFoam case on the same geometry.
- Switched the OpenFOAM extraction path from `forceCoeffs` to a lower-level `forces` function object, with a manual pressure-integration fallback if the function-object output is unavailable.
- Documented the current benchmark gap: the latest verified run still showed a large Cd mismatch, so the remaining work is focused on force accounting, sampling-window consistency, and lattice-to-physical conversion.
- Moved the remaining extraction/workaround details out of the main paper and into this changelog so the paper stays focused on the validation result itself.
