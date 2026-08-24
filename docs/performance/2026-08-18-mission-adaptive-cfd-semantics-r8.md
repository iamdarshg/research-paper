# R8 — Mission-adaptive CFD semantics (PR 41 review, item 8)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)

## The reviewer finding

> The pipeline claims "mission-adaptive" aircraft design (the generator is
> conditioned on per-mission design specs like `target_speed`), but the CFD
> solve appears to run at a single fixed flow condition. Decide what
> "mission-adaptive CFD" means and make the semantics explicit and enforced.

## The decision

**The aerodynamic *solve* is mission-independent by design. Mission-adaptivity
is delivered where it can be measured — the conditioning and evaluation
paths — not in the flow regime of the solve.**

1. **The solve regime is fixed and global.** Every sample's D3Q27 solve runs at
   the same flow conditions from `CFDConfig`: `mach_number` (config `cfd.mach_number`,
   paper operating point Mach 0.3) and `reynolds_number` (1e6 requested, realized as
   effective Re ≈ 2,494 after the `tau_min_d3q27` stability floor at 96³ — see
   [[R2]](../performance/2026-08-17-effective-reynolds-r2.md)). This keeps the
   aerodynamic loss a **stable, cross-sample-comparable objective** and every
   reported drag/lift coefficient reproducible.

2. **Mission-adaptivity lives in conditioning and evaluation:**
   - `design_spec.target_speed` is normalized into the **condition vector**, so
     the diffusion generator learns a geometry ↔ mission mapping.
   - Per-mission flight paths (climb / cruise / maneuver / descent segments) are
     synthesized at evaluation time from the design_spec
     (`build_aircraft_flight_path_manifest.py`).
   - Physical-unit aero conversions use the solver's fixed Mach against a fixed
     `drag_reference_speed` reference — internally consistent with the solve, and
     **not** a per-mission speed.

3. **Per-sample flow conditions are intentionally NOT propagated into the
   solver.** Doing so would silently change the objective per sample, break
   exact-resume continuity (the R4 fingerprint), and invalidate the documented
   effective-Re operating point. Any future change that derives solver flow
   conditions from a per-sample `design_spec` is a **training-objective change**
   and must be reviewed as such, not a refactor.

## Code evidence

- `CLI/aircraft_diffusion_cfd.py`:
  - `SOLVER_MISSION_INDEPENDENT_FIELDS` — the flow fields that define a solve
    regime: `mach_number`, `reynolds_number`, `tau_min_d3q27`,
    `drag_reference_speed`.
  - `_mission_independent_solver_conditions(config)` — returns the frozen
    flow-condition tuple derived from `config` alone.
  - `AdvancedCFDSimulator.__init__` — **enforces** the contract at every
    construction site: the tuple is built from config and must be positive
    scalars (a zero/negative Mach or Re solve is degenerate and rejected).
  - The training/validation simulators are each constructed **once** from a
    global `CFDConfig` (`self.lbm_solver`/`amr_solver`, `self.cfd_simulator`,
    `val_cfd_simulator`) — no code path passes a per-sample `design_spec`
    value into the solver.
- `CLI/advanced_lbm_solver.py`:
  - `_resolve_relaxation_time` and the realized `tau_actual` / `nu_effective`
    (R2) derive from the config flow fields only.
  - The SPSA / direct-solver gradient path perturbs **geometry**, not flow:
    every perturbed solve runs at the same fixed regime.

## What changed

1. **`CLI/aircraft_diffusion_cfd.py`** — added the semantics decision block,
   `SOLVER_MISSION_INDEPENDENT_FIELDS`, `_mission_independent_solver_conditions`,
   and a construction-time enforcement in `AdvancedCFDSimulator.__init__`
   (asserts positive, config-derived flow conditions).
2. **`tests/test_solver.py`** — `test_flow_regime_is_mission_independent_fixed_global`:
   two different geometries solved with the same config realize the identical
   `tau_actual` / `nu_effective` regime, pinned to the config flow fields.
3. **`tests/test_cfd_solver_contract.py`** —
   `test_mission_independent_solver_conditions_are_global_scalars`: the flow
   tuple is global-scalar derived, unaffected by mission-flavored config
   fields, and degrades optional LBM fields to `None` (never fabricated).

## Paper implication

The methods section must describe the semantics precisely:

- **Conditioning is mission-adaptive** — the generator is conditioned on
  per-mission `design_spec` values (`target_speed`, takeoff/turn-rate
  constraints), and evaluation synthesizes per-mission flight paths.
- **The CFD solve is mission-independent** — every geometry is evaluated in the
  same flow regime (Mach 0.3, effective Re ≈ 2,494 at 96³), so aero-loss
  comparisons across missions are apples-to-apples.

A claim that "flow conditions adapt to the mission" would be incorrect and is
explicitly disallowed by the enforced contract.
