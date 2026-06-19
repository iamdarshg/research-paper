# Low-Mach OpenFOAM vs Internal LBM Sweep

This artifact records a low-Mach-only comparison between OpenFOAM and the
internal D3Q27 LBM solver on the repaired biplane geometry.

## Scope

- Mach values: `0.05`, `0.1`, `0.2`, `0.3`
- OpenFOAM grids: `32^3`, `64^3`
- Internal LBM grids: `32^3`, `64^3`
- OpenFOAM solver route: `sonicFoam` for all points
- Internal LBM steps: `1000`
- Reference area: `0.27734375`

OpenFOAM is treated as the reference for the plotted error. This is still a
solver-comparison diagnostic, not final publication-grade validation.

## Files

- `comparison_rows.csv` - paired OpenFOAM/LBM Cd, Cl, timing, and error rows.
- `grid_speed_study_summary.json` - full structured sweep summary.
- `cd_error_surface.png` - static Cd error plot from the sweep driver.
- `low_mach_error_plotly.html` - interactive Plotly view for Cd/Cl error, Cd
  values, and compute time.

## Command

```powershell
python CLI\solver_grid_speed_study.py --stl build\solver_diagnostics\repaired_biplane_voxel_openfoam_20260610\openfoam_case_happy_mesh_grid32_flow\constant\triSurface\implicit_snappy_g32_snap0_lvl00.stl --output-dir build\solver_diagnostics\low_mach_openfoam_lbm_sweep_20260613 --mach-values 0.05,0.1,0.2,0.3 --openfoam-grids 32,64 --lbm-grids 32,64 --reference-area 0.27734375 --simple-threshold 0 --compressible-body-transits 1.0 --lbm-steps 1000 --timeout 1800 --device cuda

python CLI\render_low_mach_plotly.py --input-csv docs\benchmarks\low_mach_openfoam_lbm_sweep_20260613\comparison_rows.csv --output-html docs\benchmarks\low_mach_openfoam_lbm_sweep_20260613\low_mach_error_plotly.html
```

## Notes

- Cl percentage errors are large because the OpenFOAM reference Cl values are
  close to zero; small absolute differences become large relative errors.
- The previous `simpleFoam` low-Mach route was avoided here because it had
  already timed out/nonconverged at higher resolution.
- Mach values above `0.3` are intentionally excluded from this artifact.
