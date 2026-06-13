# simpleFoam Low-Mach Error Suite

This artifact is a convergence-gated, resumable `simpleFoam` comparison suite
for the repaired biplane geometry.

## Requested Matrix

- Mach values: `0.05` to `0.50` in `0.05` increments
- Grid values: `16^3` to `96^3` in increments of `8`
- OpenFOAM route: `simpleFoam`
- Internal solver: D3Q27 LBM at matching grid values

## Current Coverage

The current pushed artifact is a bounded partial run:

- `16^3`: Mach `0.05` through `0.50` completed and passed the rough OpenFOAM convergence gate.
- `24^3`: Mach `0.05` through `0.20` completed, but only Mach `0.15` passed the rough convergence gate.
- Higher grids remain pending because the run was intentionally bounded and internal LBM timing dominates the wall clock.

## Rough Convergence Gate

A `simpleFoam` case is marked rough-converged when the final log tail satisfies:

- max final `Ux/Uy/Uz` initial residual <= `1e-2`
- final pressure initial residual <= `5e-2`
- final local continuity <= `1e-1`

This is a pragmatic diagnostic gate, not a publication-grade CFD convergence
criterion.

## Files

- `suite_rows.csv` - compact table of OpenFOAM/LBM values, timing, error, and convergence flags.
- `suite_rows.json` - full per-case data including raw solver summaries.
- `simplefoam_error_heatmaps.html` - interactive Plotly heatmaps for Cd error, Cl error, OpenFOAM time, and cell status.

## Command

```powershell
python CLI\run_simplefoam_low_mach_suite.py --stl build\solver_diagnostics\repaired_biplane_voxel_openfoam_20260610\openfoam_case_happy_mesh_grid32_flow\constant\triSurface\implicit_snappy_g32_snap0_lvl00.stl --output-dir build\solver_diagnostics\simplefoam_low_mach_suite_20260613_converged --grid-range 16:96:8 --mach-values 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 --simple-iterations 5000 --lbm-steps 1000 --case-timeout 1200 --wall-time-limit 1200 --device cuda

python CLI\render_simplefoam_heatmaps.py --input-csv build\solver_diagnostics\simplefoam_low_mach_suite_20260613_converged\suite_rows.csv --output-html build\solver_diagnostics\simplefoam_low_mach_suite_20260613_converged\simplefoam_error_heatmaps.html
```

The suite is resumable: rerun the first command with the same output directory
to continue filling the requested grid/Mach matrix.
