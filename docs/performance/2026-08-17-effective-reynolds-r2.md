# R2 — Effective Reynolds / realized relaxation time (PR 41 review, item 2)

Date: 2026-08-17
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)

## The reviewer finding

> The solver clamps tau at 0.52 (`tau_min_d3q27`), so the requested `Re = 1e6`
> is not what the solver runs at. Effective Reynolds is ~2,494, not 1e6. The
> paper must report the realized value.

The reviewer's number checks out exactly (below). This was the single most
important review item alongside R1 (TF32 scope), because every drag/lift
coefficient produced by the D3Q27 solver is the output of a flow field evolved
at **Re ≈ 2,494**, not Re = 1e6 — and the paper's methods section, if it cited
Re = 1e6, would be describing a simulation that did not happen.

## The physics

The D3Q27 solver uses a BGK/MRT collision kernel. The relaxation time `tau`
relates to the lattice kinematic viscosity by

```
nu = (tau - 0.5) / 3
```

`tau > 0.5` is required for stability (positive viscosity); the configurable
floor `tau_min_d3q27 = 0.52` is the lowest usable value. The configured
Reynolds number is converted to a *requested* viscosity:

```
nu_req = u_lu * L_lu / Re_req
```

But the realized viscosity is set by the *clamped* tau:

```
tau_actual   = max(3 * nu_req + 0.5, tau_min)
nu_effective = (tau_actual - 0.5) / 3
Re_effective = u_lu * L_lu / nu_effective
```

When the requested Re is high enough to imply `tau < tau_min`, the requested Re
is **not realized** — the solver runs at the floor tau and at a lower effective
Reynolds number.

## The numbers at the paper operating point

Grid 96³, Mach 0.3 (lattice freestream `u_lu = 0.3/sqrt(3) = 0.173205`),
`L_lu = 96`, requested `Re = 1e6`:

| Quantity | Value |
|---|---|
| `nu_req` | 1.662768e-05 |
| `tau` if unclamped | 0.500050 |
| `tau_min_d3q27` | **0.52** (clamped) |
| `tau_actual` | 0.520000 |
| `nu_effective = (0.52-0.5)/3` | 6.666667e-03 |
| **`Re_effective`** | **2,494.2** |

At 96³ the configured Re = 1e6 implies a viscosity ~400× below the stability
floor; the solver can only resolve 1/400 of it. This is a fundamental
grid-resolution × stability coupling, not a config bug: a real Re = 1e6 flow
needs sub-grid turbulence modeling or far higher resolution.

## What changed

`CLI/advanced_lbm_solver.py`:

1. **`_resolve_relaxation_time(nu_requested)`** — new helper on
   `D3Q27CascadedSolver` mapping requested viscosity → `(tau_actual,
   nu_effective)`, documenting that the request may not be realized.
2. **`collide_stream` / `collide_stream_batch`** — the nu/tau block now sets
   `self.tau_actual`, `self.nu_effective`, `self.nu_requested`, and sets
   `self.nu = self.nu_effective` (the **realized** viscosity). Both the eager
   (`compute_aerodynamic_coefficients`, passes `self.nu`) and deferred
   (`compute_aerodynamic_coefficients_deferred`, freezes `self.nu`) coefficient
   paths now read the realized viscosity automatically.
3. **`_aerodynamic_coefficients_from_raw`** — reports the realized physics in
   every coefficient dict:
   - `requested_reynolds` (config value, e.g. 1e6)
   - `effective_reynolds` (`u_lu * L_lu / nu_effective`, ≈ 2,494)
   - `reynolds_clamped` (bool — request was clamped away)
   - `tau_actual`, `effective_laminar_viscosity`
   - `reynolds_number_turbulent` fixed to lattice-consistent units
     (`u_lu * L_lu / (nu + nu_turb_mean)`); the old `v_inf * h * L / nu` mixed
     physical freestream speed with lattice viscosity (dimensionally wrong).
4. **Batch path** (`compute_aerodynamic_coefficients_batch`) exposes the same
   five laminar telemetry keys (only `nu_turb_mean` stays `None`, as before).

`CLI/config.yaml` — the `cfd.reynolds_number` entry now carries a NOTE that the
value is clamped and the paper must cite the realized ~2,494.

`tests/test_solver.py` — `test_effective_reynolds_and_tau_actual`:
- Clamped case (8³/Mach 0.1, Re=100): asserts `tau_actual == 0.52`,
  `self.nu == nu_effective`, `reynolds_clamped is True`, and
  `effective_reynolds == u_lu * 8 / nu_eff` (~69.3, not 100).
- Unclamped case (Re=50): asserts `reynolds_clamped is False` and the realized
  Re equals the requested Re.

## Evidence

- `tests/test_solver.py` — 7 passed (incl. the new R2 test)
- `tests/test_direct_solver_fused_parity.py`, `tests/test_cfd_solver_contract.py`,
  `tests/test_d3q27_kernel_parity.py` — 19 passed, 1 skipped (deferred materialize
  and batch paths still reproduce the eager dict).

## Paper implication

The methods section must report the **realized** operating point: the D3Q27
LBM solver at 96³ / Mach 0.3 / `Re_requested = 1e6` runs at **tau = 0.52 and
effective Reynolds ≈ 2,494**. Any statement "solved at Re = 1e6" is incorrect.
The `effective_reynolds` / `tau_actual` / `reynolds_clamped` keys in the
solver's coefficient dicts are the machine-verifiable evidence of the realized
value.
