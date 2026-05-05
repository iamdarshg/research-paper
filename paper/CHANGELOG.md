# Changelog

## Unreleased

- Made `run_internal_benchmark.py` cross-platform by sourcing OpenFOAM through either Linux `bash` or Windows WSL, added an optional `--install-openfoam` bootstrap path, and switched the benchmark sweep to `20mm_cube.stl` plus any other STL files in the repo root.
- Added the centered-cube validation object to the benchmark script and documented the exact geometry in `paper/sections/validation-and-testing-standards.tex`.
- Consolidated local verification into `run_internal_benchmark.py`, which now runs the internal D3Q27 solver and the OpenFOAM sonicFoam case on the same geometry.
- Switched the OpenFOAM extraction path from `forceCoeffs` to a lower-level `forces` function object, with a manual pressure-integration fallback if the function-object output is unavailable.
- Documented the current benchmark gap: the latest verified run still showed a large Cd mismatch, so the remaining work is focused on force accounting, sampling-window consistency, and lattice-to-physical conversion.
- Retroactively recorded the next physics pass: `CLI/cascaded_lbm.py` now uses the lattice-consistent D3Q27 freestream scaling (`u = Ma / sqrt(3)`) instead of the earlier arbitrary freestream speed, and that fix is part of the ongoing benchmark history.
- Added a GitHub Actions workflow to compile `paper/main.tex` into a PDF automatically and publish the artifact on push/PR.
- Implemented a rigorous, literature-driven baseline selection strategy in the research paper, covering classical, generative, and optimization-based families. (Issue #22)
- Overhauled and audited the project bibliography (references.bib), adding missing categories for SDEs, 3D diffusion, and CFD validation standards. (Issue #23)
- Optimized D3Q27 solver performance (geometry hashing, BFL padding) and improved stability via MRT parameter mapping and a relative L2 convergence monitor.
- Moved the remaining extraction/workaround details out of the main paper and into this changelog so the paper stays focused on the validation result itself.
