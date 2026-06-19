# OpenFOAM and Solver Sweep TODO

Current state:
- Reusable OpenFOAM tooling exists in `CLI/openfoam_case_utils.py` and `CLI/openfoam_mach_sweep.py`.
- Grid/speed comparison tooling exists in `CLI/solver_grid_speed_study.py`.
- A 32^3 repaired biplane OpenFOAM sweep from Mach 0.1 to Mach 2 completed, but the low-Mach `simpleFoam` branch was not converged.
- A 128^3 OpenFOAM / 192^3 internal-LBM comparison attempt blocked on the first Mach 0.1 `simpleFoam` case after a 7200 second timeout.
- Internal LBM high-Mach values must remain labeled `experimental_high_mach_unvalidated` unless a real compressible/thermal validation path is added.

## Pending Sweep Work

- [ ] Make OpenFOAM sweep execution resumable per case, including timeout recovery and case copy-back after solver timeout.
- [ ] Add Cd/Cl history extraction instead of using only the latest OpenFOAM time directory.
- [ ] Diagnose low-Mach 128^3 `simpleFoam` non-convergence before increasing wall time:
  - residual trend
  - continuity trend
  - relaxation factors
  - boundary condition suitability
  - mesh/cell count and decomposition cost
- [ ] Evaluate whether `rhoSimpleFoam`, `rhoPimpleFoam`, or `sonicFoam` should replace `simpleFoam` for low/subsonic comparison consistency.
- [ ] Run a small solver-routing matrix before high resolution:
  - Mach 0.1 at 32^3 and 64^3
  - Mach 0.3 at 32^3 and 64^3
  - Mach 1.0 at 32^3 and 64^3
- [ ] If the routing matrix is stable, run the first high-resolution comparison:
  - OpenFOAM 128^3
  - internal LBM 192^3
  - Mach values: 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0
- [ ] Treat 256^3 and 512^3 OpenFOAM sweeps as separate scheduled jobs, not interactive smoke tests.
- [ ] Add parallel OpenFOAM decomposition support before attempting 256^3 or 512^3.
- [ ] Generate the requested 3D plot only after finite paired data exists:
  - x-axis: Mach
  - y-axis: cubic grid resolution
  - z-axis: Cd/Cl error percentage
  - color/style: validated low-Mach vs experimental high-Mach
- [ ] Store final sweep artifacts under `build/solver_diagnostics/grid_speed_study_<date>/`.

## Claim Boundary

- Low-Mach internal LBM raw force behavior is improved and separately tested.
- Whole-aircraft convergence and high-Mach compressibility remain open evidence gates.
- Do not use high-Mach internal LBM output as validated compressible CFD until thermal/compressible physics and benchmark validation are implemented.
