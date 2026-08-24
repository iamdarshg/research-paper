# R9 — Config source-of-truth (PR 41 review, item 9)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)

## The reviewer finding

> `CFDConfig` hardcodes `reynolds_number = 1e6` and `simulation_steps = 1000`
> while `config.yaml` declares `reynolds_number: 1000000.0` and
> `simulation_steps: 500`. The YAML values are silently ignored, so the "source
> of truth" for the CFD operating point is ambiguous. There are also two
> near-duplicate `CFDConfig` classes (`config.py` and
> `aircraft_diffusion_cfd.py`).

## The fix

1. **Every flow-condition field now reads `config.yaml`** via `config_value(...)`,
   exactly as `mach_number` already did, in **both** `CFDConfig` classes:
   - `reynolds_number = float(config_value("cfd", "reynolds_number", 1e6))`
   - `simulation_steps = int(config_value("cfd", "simulation_steps", 1000))`

   `config.yaml` is now the single source of truth for the CFD operating point.

2. **`simulation_steps` aligned to 1000 in `config.yaml`** with a note. This field
   is **advisory only** — no training / validation path consumes
   `CFDConfig.simulation_steps` (the direct solver is governed by
   `training.direct_solver_steps`, and the corpus builders use their own
   `SIMULATION_STEPS` constants). The YAML value is kept at 1000 — the value
   pre-existing exact-resume fingerprints recorded — so aligning the code to the
   YAML (and the YAML to the legacy fingerprint) does **not** change any
   fingerprint or block resuming a run-state created before this change.

3. **`reynolds_number` value is unchanged** (YAML 1000000.0 == code 1e6), so
   this is a pure source-of-truth correction with zero behavioral or fingerprint
   change.

## Resume-compatibility analysis (R4 interplay)

The exact-resume fingerprint (`_build_objective_configuration_fingerprint`)
records `cfd_config` via a full dataclass fingerprint. Before this change an
old run-state recorded `mach=0.3 / re=1e6 / simulation_steps=1000`. After this
change the same construction yields the identical values (YAML now declares
`mach=0.3 / re=1000000.0 / simulation_steps=1000`), so:

- **No fingerprint value changed** → existing run-states (e.g.
  `build/recovery_continuation_20260815`) still pass exact-resume compatibility.
- **A future YAML edit now propagates into the fingerprint**, which is exactly
  what R4's "no silent configuration drift" contract wants: a change to the
  declared operating point becomes a reviewable, flagged event.

## The duplicate-family note

There are two `CFDConfig` classes (`config.CFDConfig` and
`aircraft_diffusion_cfd.CFDConfig`). Fully unifying them is a larger refactor
beyond a review-clean fix; this change makes the **flow-condition fields they
share** read the same YAML source and adds a parity test that pins they cannot
silently diverge on `mach_number` / `reynolds_number` / `simulation_steps`.
A full unification is tracked for the cleanup pass (task #87).

## What changed

1. `CLI/config.yaml` — `simulation_steps: 1000` + advisory note.
2. `CLI/config.py` and `CLI/aircraft_diffusion_cfd.py` — both `CFDConfig`
   classes read `reynolds_number` / `simulation_steps` from `config.yaml`.
3. `tests/test_config.py` — pins the YAML-sourced values and the parity between
   the two `CFDConfig` classes.
