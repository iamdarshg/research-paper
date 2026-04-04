# Changelog

## Unreleased

- Added the centered cube validation object to the benchmark script and documented a one-command local verification path.
- Switched the OpenFOAM export fallback from `forceCoeffs` to a lower-level `forces` function object, then normalized the result manually for coefficient comparison.
- Kept the paper text focused on the validation result itself; the implementation detail above lives here instead of in the main narrative.
