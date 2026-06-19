# GPU Thermal LBM Attachment Plan

## Objective

Create a staged thermodynamic solver path that attaches to the existing Torch D3Q27 solver without replacing the validated low-Mach raw path. The new path must keep all working tensors on the same device as the flow solver, expose pressure-density-temperature metadata, and remain explicitly experimental for Mach > 0.3 until shock-capable physics and independent validation exist.

This pass does **not** claim validated Mach 2 CFD. It creates the fast GPU thermal scaffold needed for that future work.

## Claim Boundary

- Existing `D3Q27CascadedSolver` remains the raw isothermal low-Mach solver.
- New thermal path is `experimental_thermal_lbm_unvalidated`.
- New thermal pressure is a thermodynamic diagnostic field using `p = rho * R * T`.
- Existing aerodynamic force extraction remains the raw D3Q27 momentum-exchange path unless a later validated compressible force model replaces it.
- Mach 2 runs remain executable but cannot set `pinn_ready`, `claim_grade` above experimental, or training-truth labels without external validation.
- Shock capturing, normal-shock validation, and compressible wall/inlet/outlet replacement are future gates, not hidden in this pass.

## Files

Implementation:
- `CLI/thermal_lbm_solver.py`
  - Add `ThermalLBMConfig`.
  - Add `ThermalBGKSolver` using a D3Q7 thermal distribution stored as Torch tensors.
  - Add `ThermalD3Q27Solver` that composes `advanced_lbm_solver.D3Q27CascadedSolver`.
  - Add `create_thermal_lbm_solver(config, device, phys_config)` factory.
  - Add thermal metadata and pressure-density-temperature diagnostics.
- `CLI/lbm_utils.py`
  - Add thermal/staged regime constants and a helper for thermal metadata if needed by multiple callers.

Tests:
- `tests/test_thermal_lbm_solver.py`
  - Verify thermal tensors are on the requested device.
  - Verify uniform temperature remains uniform after thermal update.
  - Verify pressure obeys `p = rho * R * T`.
  - Verify positivity guards clamp invalid temperature/density/pressure.
  - Verify thermal solver output remains `experimental_thermal_lbm_unvalidated` and `pinn_ready` false at Mach 2.
  - Verify existing raw D3Q27 low-Mach mapping is not changed.

Documentation/evidence:
- `build/solver_diagnostics/thermal_lbm_20260612/summary.json`
  - Generated after validation with command outcomes and explicit claim boundary.
- `build/solver_diagnostics/thermal_lbm_20260612/thermal_lbm_report.md`
  - Generated after validation with what was implemented, what remains gated, and why Mach 2 remains experimental.

## TDD Sequence

1. Add `tests/test_thermal_lbm_solver.py` that imports `thermal_lbm_solver`.
2. Run:
   ```powershell
   python -m unittest tests.test_thermal_lbm_solver -v
   ```
   Expected: fail with `ModuleNotFoundError` or missing classes.
3. Implement `CLI/thermal_lbm_solver.py` with only the API needed by tests.
4. Run:
   ```powershell
   python -m unittest tests.test_thermal_lbm_solver -v
   ```
   Expected: pass.
5. Run regression tests:
   ```powershell
   python -m py_compile CLI\thermal_lbm_solver.py CLI\advanced_lbm_solver.py CLI\lbm_utils.py
   python -m unittest tests.test_lbm_mach_mapping -v
   python -m unittest tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata tests.test_thermal_lbm_solver -v
   ```
   Expected: pass.
6. Run canonical low-Mach validation:
   ```powershell
   python -m unittest tests.test_canonical_validation -v
   ```
   Expected: pass with canonical low-Mach sanity unchanged.
7. Run full discovery:
   ```powershell
   $env:PYTHONIOENCODING='utf-8'; python -m unittest discover tests -v
   ```
   Expected based on current repo state: may still fail on the pre-existing unrelated aircraft validity assertion. Record exact outcome.

## Implementation Details

### ThermalBGKSolver

- Velocity set: D3Q7 `(0,0,0)`, axis-positive and axis-negative directions.
- Weights: rest `0.25`, each axis direction `0.125`.
- Distribution tensor: `g` shape `[7, N, N, N]`.
- Equilibrium:
  ```python
  g_eq_i = w_i * T * (1 + 3 * e_i dot u)
  ```
- Streaming: `torch.roll` per thermal direction on the same device.
- Collision: BGK relaxation with `omega = 1 / tau`.
- Thermal diffusivity relation:
  ```python
  alpha_lattice = (tau - 0.5) / 3
  tau = 0.5 + 3 * alpha_lattice
  ```
- Positivity guards:
  - `temperature >= min_temperature`
  - `density >= min_density`
  - `pressure >= min_pressure`
- Solid cells: no-flux staged behavior by restoring pre-stream values in solid cells. This is a scaffold, not a validated compressible wall condition.

### ThermalD3Q27Solver

- Composes `D3Q27CascadedSolver` as `flow_solver`.
- Uses the flow solver's velocity and density tensors for thermal advection and thermodynamic pressure diagnostics.
- Keeps thermal tensors on `flow_solver.device`.
- Proxies `collide_stream(geometry_mask, steps, ext_force=None)`:
  - Run flow solver.
  - Run thermal BGK updates for `thermal_steps = max(1, min(steps, max_thermal_steps_per_call))`.
  - Refresh `temperature`, `density`, and `pressure` fields.
- Proxies `compute_aerodynamic_coefficients(geometry_mask)`:
  - Start with raw flow-solver coefficients.
  - Add thermal diagnostic metadata.
  - Force `validity_regime = experimental_thermal_lbm_unvalidated`.
  - Force `claim_grade = no_claim_experimental`.
  - Force `training_drag_source = none_thermal_internal_lbm_unvalidated`.
  - Force `pinn_ready = False`.

## Validation Artifacts

After tests, write:

```powershell
New-Item -ItemType Directory -Force build\solver_diagnostics\thermal_lbm_20260612
```

Then create:
- `summary.json`
- `thermal_lbm_report.md`

These artifacts must include:
- exact commands run
- pass/fail outcomes
- whether CUDA was available during tests
- thermal model name
- pressure model
- Mach 2 status
- remaining physics gates

## Commit Points

1. Commit `add gpu thermal lbm tests`
   - `tests/test_thermal_lbm_solver.py`
2. Commit `add staged gpu thermal lbm solver`
   - `CLI/thermal_lbm_solver.py`
   - `CLI/lbm_utils.py` if touched
3. Commit `add thermal lbm evidence artifacts`
   - only lightweight evidence docs/JSON and this plan

Do not commit unrelated dirty files already present in the worktree.
