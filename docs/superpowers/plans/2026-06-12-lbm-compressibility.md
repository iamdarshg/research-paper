# LBM Compressibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the internal D3Q27 solver's compressibility claims physically honest by gating the current solver as low-Mach weakly compressible/isothermal, preserving high-Mach exploratory runs as experimental, and producing tests, diagnostics, evidence, and documentation that enforce that boundary.

**Architecture:** Use Path B for this pass. Add one reusable regime/metadata contract in `CLI/lbm_utils.py`, consume it from both internal solver wrappers and evidence drivers, then generate audit/evidence artifacts under `build/solver_diagnostics/...`. Do not add a fake thermal/compressible solver; high-Mach internal LBM remains executable but cannot become claim-grade without independent external validation.

**Tech Stack:** Python 3, PyTorch, unittest, existing D3Q27 LBM modules, existing OpenFOAM utility scripts, JSON/Markdown/CSV evidence artifacts.

---

## File Structure

- Modify `CLI/lbm_utils.py`: add reusable Mach mapping, low/high-Mach regime classification, and metadata helpers used by solver and evidence code.
- Modify `CLI/advanced_lbm_solver.py`: use `mach_to_lattice_velocity()` for the D3Q27 wrapper, attach regime metadata to aerodynamic coefficient outputs, and keep raw momentum-exchange force separate from surrogate diagnostics.
- Modify `CLI/cascaded_lbm.py`: align legacy D3Q27 Mach mapping with `u_lattice = Ma / sqrt(3)` and attach the same metadata to its coefficient output.
- Modify `CLI/cfd_simulator.py`: preserve solver metadata through simulator outputs and ensure internal high-Mach runs cannot set `pinn_ready` or a claim-grade label.
- Modify `CLI/solver_grid_speed_study.py`: replace the local validity string with `classify_lbm_regime()` and emit required metadata in LBM summaries and comparison rows.
- Create `CLI/solver_compressibility_audit.py`: reusable audit generator that writes the required Markdown table with file/line references.
- Create `CLI/compressibility_evidence_report.py`: reusable evidence/report generator that summarizes tests, optional OpenFOAM/LBM comparison artifacts, and plot gating without placeholders.
- Create `tests/test_lbm_mach_mapping.py`: focused tests for D3Q27 low-Mach mapping and legacy path alignment.
- Create `tests/test_lbm_compressibility_regime.py`: focused tests for regime classification and claim gates.
- Create `tests/test_lbm_compressibility_metadata.py`: focused tests for solver/simulator metadata propagation and high-Mach training source policy.
- Modify `CLI/GROUND_TRUTH_SPEC.md`: document raw low-Mach LBM, experimental high-Mach LBM, OpenFOAM incompressible, OpenFOAM compressible, and calibrated/surrogate label boundaries.
- Modify `paper/CLAIMS_EVIDENCE_MATRIX.md`: add an explicit compressibility claim boundary row.
- Modify `paper/FINAL_RUN_GATES.md`: add gates blocking compressible/high-Mach solver claims until external validation and physics implementation exist.
- Generate `build/solver_diagnostics/compressibility_audit_20260612/solver_compressibility_audit.md`.
- Generate `build/solver_diagnostics/compressibility_evidence_20260612/summary.json`.
- Generate `build/solver_diagnostics/compressibility_evidence_20260612/compressibility_report.md`.
- Generate `build/solver_diagnostics/compressibility_evidence_20260612/comparison_rows.csv` only if comparison data is available.
- Generate `build/solver_diagnostics/compressibility_evidence_20260612/cd_error_surface.png` only if at least one finite Cd error row exists; otherwise the report must state which gate blocked the plot.

## Commit Points

1. `add lbm compressibility regime tests`
   - Commit only `tests/test_lbm_mach_mapping.py`, `tests/test_lbm_compressibility_regime.py`, and `tests/test_lbm_compressibility_metadata.py`.
2. `add lbm compressibility regime metadata`
   - Commit only `CLI/lbm_utils.py`, `CLI/advanced_lbm_solver.py`, `CLI/cascaded_lbm.py`, `CLI/cfd_simulator.py`, and `CLI/solver_grid_speed_study.py`.
3. `add solver compressibility evidence utilities`
   - Commit only `CLI/solver_compressibility_audit.py` and `CLI/compressibility_evidence_report.py`.
4. `document lbm compressibility claim boundary`
   - Commit only `CLI/GROUND_TRUTH_SPEC.md`, `paper/CLAIMS_EVIDENCE_MATRIX.md`, and `paper/FINAL_RUN_GATES.md`.

Generated artifacts under `build/solver_diagnostics/...` are evidence outputs and should not be committed unless the repository's existing convention demands generated evidence in git. The current convention has `build/` untracked for diagnostics, so leave these artifacts uncommitted and cite their paths in the final response.

---

### Task 1: Regime Contract Tests

**Files:**
- Create: `tests/test_lbm_mach_mapping.py`
- Create: `tests/test_lbm_compressibility_regime.py`
- Create: `tests/test_lbm_compressibility_metadata.py`

- [ ] **Step 1: Write the failing Mach mapping tests**

Create `tests/test_lbm_mach_mapping.py`:

```python
import math
import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from advanced_lbm_solver import D3Q27CascadedSolver as AdvancedD3Q27CascadedSolver
from cascaded_lbm import D3Q27CascadedSolver as LegacyD3Q27CascadedSolver
from config import CFDConfig, LBMPhysicsConfig
from lbm_utils import D3Q27_LATTICE_SOUND_SPEED, mach_to_lattice_velocity


class TestLBMMachMapping(unittest.TestCase):
    def test_helper_maps_mach_to_d3q27_lattice_sound_speed(self):
        self.assertAlmostEqual(mach_to_lattice_velocity(0.3), 0.3 / math.sqrt(3.0), places=12)
        self.assertAlmostEqual(D3Q27_LATTICE_SOUND_SPEED, 1.0 / math.sqrt(3.0), places=12)

    def test_advanced_solver_uses_mach_over_sqrt_three_when_not_clipped(self):
        cfg = CFDConfig(base_grid_resolution=8, mach_number=0.12, reynolds_number=100.0, simulation_steps=1)
        cfg.lbm_config.target_lattice_velocity = 1.0
        solver = AdvancedD3Q27CascadedSolver(cfg, torch.device("cpu"), LBMPhysicsConfig)
        self.assertAlmostEqual(solver.inlet_velocity_lu, 0.12 / math.sqrt(3.0), places=7)

    def test_legacy_solver_uses_same_lattice_mapping(self):
        cfg = CFDConfig(base_grid_resolution=8, mach_number=0.12, reynolds_number=100.0, simulation_steps=1)
        solver = LegacyD3Q27CascadedSolver(cfg, torch.device("cpu"), LBMPhysicsConfig)
        self.assertAlmostEqual(solver.inlet_velocity_lu, 0.12 / math.sqrt(3.0), places=7)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Write the failing regime classification tests**

Create `tests/test_lbm_compressibility_regime.py`:

```python
import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from lbm_utils import classify_lbm_regime


class TestLBMCompressibilityRegime(unittest.TestCase):
    def test_low_mach_is_claim_grade_only_for_current_weakly_compressible_model(self):
        regime = classify_lbm_regime(0.3)
        self.assertEqual(regime["validity_regime"], "validated_low_mach_envelope")
        self.assertEqual(regime["claim_grade"], "low_mach_sanity_only")
        self.assertIsNone(regime["high_mach_warning"])
        self.assertEqual(regime["compressibility_model"], "weakly_compressible_isothermal_lbm")
        self.assertEqual(regime["thermal_model"], "none_isothermal")

    def test_high_mach_internal_lbm_is_experimental_and_not_claim_grade(self):
        regime = classify_lbm_regime(0.31)
        self.assertEqual(regime["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(regime["claim_grade"], "no_claim_experimental")
        self.assertIn("not validated compressible CFD", regime["high_mach_warning"])
        self.assertFalse(regime["claim_grade"].startswith("low_mach"))

    def test_external_validation_can_be_recorded_without_upgrading_internal_physics(self):
        regime = classify_lbm_regime(0.8, external_validation="openfoam_compressible_converged")
        self.assertEqual(regime["validity_regime"], "external_compressible_reference_available")
        self.assertEqual(regime["claim_grade"], "external_reference_only")
        self.assertIn("Internal D3Q27", regime["high_mach_warning"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Write the failing metadata propagation tests**

Create `tests/test_lbm_compressibility_metadata.py`:

```python
import os
import sys
import unittest

import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from advanced_lbm_solver import D3Q27CascadedSolver
from cfd_simulator import AdvancedCFDSimulator
from config import CFDConfig, LBMPhysicsConfig
from lbm_utils import build_lbm_compressibility_metadata


class TestLBMCompressibilityMetadata(unittest.TestCase):
    def test_metadata_helper_emits_required_fields_for_high_mach(self):
        metadata = build_lbm_compressibility_metadata(
            mach_number=0.8,
            u_lattice=0.12,
            lbm_converged=True,
            force_stability=0.02,
        )
        for key in (
            "mach_number",
            "lattice_mach",
            "u_lattice",
            "sound_speed_model",
            "compressibility_model",
            "thermal_model",
            "validity_regime",
            "claim_grade",
            "high_mach_warning",
            "lbm_converged",
            "force_stability",
            "training_drag_source",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(metadata["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")

    def test_solver_coefficients_include_regime_metadata(self):
        cfg = CFDConfig(base_grid_resolution=8, mach_number=0.5, reynolds_number=100.0, simulation_steps=1)
        cfg.lbm_config.target_lattice_velocity = 1.0
        solver = D3Q27CascadedSolver(cfg, torch.device("cpu"), LBMPhysicsConfig)
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0
        results = solver.compute_aerodynamic_coefficients(geometry)
        self.assertEqual(results["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["label_tier"], "lbm_raw")
        self.assertEqual(results["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")

    def test_simulator_does_not_promote_high_mach_internal_lbm_to_pinn_ready(self):
        cfg = CFDConfig(base_grid_resolution=8, mach_number=0.8, reynolds_number=100.0, simulation_steps=1)
        cfg.lbm_config.target_lattice_velocity = 1.0
        simulator = AdvancedCFDSimulator(cfg, torch.device("cpu"))
        geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
        geometry[3:5, 3:5, 3:5] = 1.0
        results = simulator.simulate_aerodynamics(geometry, steps=1)
        self.assertFalse(results["pinn_ready"])
        self.assertEqual(results["validity_regime"], "experimental_high_mach_unvalidated")
        self.assertEqual(results["claim_grade"], "no_claim_experimental")
        self.assertEqual(results["training_drag_source"], "none_high_mach_internal_lbm_unvalidated")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run focused tests to confirm they fail for missing contract**

Run:

```powershell
python -m unittest tests.test_lbm_mach_mapping tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata -v
```

Expected outcome:

```text
ImportError: cannot import name 'D3Q27_LATTICE_SOUND_SPEED'
```

or:

```text
ImportError: cannot import name 'classify_lbm_regime'
```

The failure is expected because the reusable regime contract has not been implemented yet.

- [ ] **Step 5: Commit tests**

Run:

```powershell
git -C D:\CodeProjects\research-paper add tests\test_lbm_mach_mapping.py tests\test_lbm_compressibility_regime.py tests\test_lbm_compressibility_metadata.py
git -C D:\CodeProjects\research-paper commit -m "add lbm compressibility regime tests"
```

Expected outcome: one commit containing only the three new tests.

---

### Task 2: Solver Regime Metadata Implementation

**Files:**
- Modify: `CLI/lbm_utils.py`
- Modify: `CLI/advanced_lbm_solver.py`
- Modify: `CLI/cascaded_lbm.py`
- Modify: `CLI/cfd_simulator.py`
- Modify: `CLI/solver_grid_speed_study.py`
- Test: `tests/test_lbm_mach_mapping.py`
- Test: `tests/test_lbm_compressibility_regime.py`
- Test: `tests/test_lbm_compressibility_metadata.py`

- [ ] **Step 1: Add reusable Mach/regime helpers**

Append these definitions near the top of `CLI/lbm_utils.py`, after imports:

```python
import math
from typing import Any


D3Q27_LATTICE_SOUND_SPEED = 1.0 / math.sqrt(3.0)
LOW_MACH_VALIDATED_LIMIT = 0.3
LOW_MACH_VALIDITY_REGIME = "validated_low_mach_envelope"
HIGH_MACH_EXPERIMENTAL_REGIME = "experimental_high_mach_unvalidated"


def mach_to_lattice_velocity(mach_number: float) -> float:
    """Map physical Mach number to D3Q27 lattice velocity for the current isothermal model."""
    return float(mach_number) * D3Q27_LATTICE_SOUND_SPEED


def classify_lbm_regime(mach_number: float, external_validation: str | None = None) -> dict[str, Any]:
    """Classify the current internal LBM result without upgrading the underlying physics."""
    mach = float(mach_number)
    base = {
        "sound_speed_model": "D3Q27 isothermal cs=1/sqrt(3); physical scaling uses a=343 m/s",
        "compressibility_model": "weakly_compressible_isothermal_lbm",
        "thermal_model": "none_isothermal",
    }
    if external_validation:
        return {
            **base,
            "validity_regime": "external_compressible_reference_available",
            "claim_grade": "external_reference_only",
            "high_mach_warning": (
                "Internal D3Q27 remains weakly compressible/isothermal; high-Mach claim support comes only "
                f"from external validation: {external_validation}."
            ),
        }
    if mach <= LOW_MACH_VALIDATED_LIMIT:
        return {
            **base,
            "validity_regime": LOW_MACH_VALIDITY_REGIME,
            "claim_grade": "low_mach_sanity_only",
            "high_mach_warning": None,
        }
    return {
        **base,
        "validity_regime": HIGH_MACH_EXPERIMENTAL_REGIME,
        "claim_grade": "no_claim_experimental",
        "high_mach_warning": (
            f"Mach {mach:.3g} exceeds the current internal D3Q27 low-Mach validation envelope; "
            "this run is not validated compressible CFD."
        ),
    }


def build_lbm_compressibility_metadata(
    *,
    mach_number: float,
    u_lattice: float,
    lbm_converged: bool,
    force_stability: float | None,
    external_validation: str | None = None,
) -> dict[str, Any]:
    """Build the standard compressibility metadata block for solver outputs and evidence JSON."""
    regime = classify_lbm_regime(mach_number, external_validation=external_validation)
    lattice_mach = float(u_lattice) / D3Q27_LATTICE_SOUND_SPEED
    if regime["validity_regime"] == HIGH_MACH_EXPERIMENTAL_REGIME:
        training_drag_source = "none_high_mach_internal_lbm_unvalidated"
    elif regime["validity_regime"] == "external_compressible_reference_available":
        training_drag_source = "external_validated_reference"
    else:
        training_drag_source = "internal_lbm_raw_low_mach"
    return {
        "mach_number": float(mach_number),
        "lattice_mach": lattice_mach,
        "u_lattice": float(u_lattice),
        **regime,
        "lbm_converged": bool(lbm_converged),
        "force_stability": None if force_stability is None else float(force_stability),
        "training_drag_source": training_drag_source,
    }
```

- [ ] **Step 2: Update `advanced_lbm_solver.py` imports and velocity mapping**

Change the import:

```python
from lbm_utils import (
    D3Q27Lattice,
    _compute_force_coefficients,
    build_lbm_compressibility_metadata,
    mach_to_lattice_velocity,
)
```

Change `_estimate_lattice_freestream_velocity()` to compute:

```python
mach = getattr(self.config, "mach_number", 0.0)
u_lattice = mach_to_lattice_velocity(mach)
```

Keep the existing clipping logic, because the clipping itself is a stability guard that metadata will expose through `u_lattice` and `lattice_mach`.

- [ ] **Step 3: Attach metadata in `advanced_lbm_solver.py`**

Before the `return { ... } | shape_drag_metrics` block in `compute_aerodynamic_coefficients()`, compute:

```python
compressibility_metadata = build_lbm_compressibility_metadata(
    mach_number=getattr(self.config, "mach_number", 0.0),
    u_lattice=self.inlet_velocity_lu,
    lbm_converged=lbm_converged,
    force_stability=force_stability,
)
```

Inside the returned dict, add:

```python
            **compressibility_metadata,
```

after `force_stability` so required fields override no existing key except `lbm_converged` with the same boolean value.

- [ ] **Step 4: Update legacy `cascaded_lbm.py` mapping and metadata**

Change imports to:

```python
from lbm_utils import (
    D3Q27Lattice,
    _compute_force_coefficients,
    build_lbm_compressibility_metadata,
    mach_to_lattice_velocity,
)
```

In `D3Q27CascadedSolver._setup_physics_constants()`, replace:

```python
u_lattice = self.config.mach_number / 3.0
```

with:

```python
self.inlet_velocity_lu = mach_to_lattice_velocity(self.config.mach_number)
u_lattice = self.inlet_velocity_lu
```

In `_initialize_equilibrium()`, replace:

```python
u_lattice = self.config.mach_number / 3.0
```

with:

```python
u_lattice = getattr(self, "inlet_velocity_lu", mach_to_lattice_velocity(self.config.mach_number))
```

In `collide_stream()`, replace:

```python
u_lattice = self.config.mach_number * 0.10
```

with:

```python
u_lattice = getattr(self, "inlet_velocity_lu", mach_to_lattice_velocity(self.config.mach_number))
```

In `compute_aerodynamic_coefficients()`, compute:

```python
force_stability = None
if self.force_samples > 20:
    avg_fx = float(self.force_x_accum.item()) / self.force_samples
    last_fx = float(self.force_x_last.item())
    force_stability = abs(last_fx - avg_fx) / (abs(avg_fx) + 1e-6)
lbm_converged = bool(self.force_samples > 0 and not torch.isnan(self.f).any())
compressibility_metadata = build_lbm_compressibility_metadata(
    mach_number=self.config.mach_number,
    u_lattice=getattr(self, "inlet_velocity_lu", mach_to_lattice_velocity(self.config.mach_number)),
    lbm_converged=lbm_converged,
    force_stability=force_stability,
)
```

Add to the returned dict:

```python
            "label_source": "lbm_d3q27",
            "label_tier": "lbm_raw",
            **compressibility_metadata,
```

- [ ] **Step 5: Preserve simulator gates in `cfd_simulator.py`**

After:

```python
results['pinn_ready'] = False
```

add:

```python
        if results.get("validity_regime") == "experimental_high_mach_unvalidated":
            results["pinn_ready"] = False
            results["claim_grade"] = "no_claim_experimental"
```

Inside the `if external_results:` promotion block, after setting `label_tier`, add:

```python
            if results.get("validity_regime") == "experimental_high_mach_unvalidated":
                results["claim_grade"] = "external_reference_only" if results["label_tier"] == "external_pde" else "no_claim_experimental"
```

This preserves the requirement that internal high-Mach LBM cannot become claim-grade by itself, while allowing external PDE validation to be recorded separately.

- [ ] **Step 6: Use shared regime metadata in `solver_grid_speed_study.py`**

Change imports to include:

```python
from lbm_utils import classify_lbm_regime
```

In `run_lbm_case()`, replace:

```python
"validity": "validated_low_mach_envelope" if mach <= 0.3 else "experimental_high_mach_unvalidated",
```

with:

```python
"validity": coeffs.get("validity_regime", classify_lbm_regime(mach)["validity_regime"]),
"validity_regime": coeffs.get("validity_regime"),
"claim_grade": coeffs.get("claim_grade"),
"high_mach_warning": coeffs.get("high_mach_warning"),
"u_lattice": coeffs.get("u_lattice"),
"lattice_mach": coeffs.get("lattice_mach"),
"sound_speed_model": coeffs.get("sound_speed_model"),
"compressibility_model": coeffs.get("compressibility_model"),
"thermal_model": coeffs.get("thermal_model"),
```

In `compare_cases()`, add fields to each row:

```python
                    "lbm_claim_grade": lbm_case.get("claim_grade"),
                    "lbm_compressibility_model": lbm_case.get("compressibility_model"),
                    "lbm_thermal_model": lbm_case.get("thermal_model"),
```

Update `write_csv()` field list with:

```python
        "lbm_claim_grade",
        "lbm_compressibility_model",
        "lbm_thermal_model",
```

In `write_plot()`, replace the color test with:

```python
        colors = np.array([0.0 if row.get("lbm_validity") == "validated_low_mach_envelope" else 1.0 for row in plot_rows])
```

leaving the string unchanged because it now comes from the shared helper.

- [ ] **Step 7: Run focused tests to verify implementation**

Run:

```powershell
python -m unittest tests.test_lbm_mach_mapping tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata -v
```

Expected outcome:

```text
Ran 9 tests
OK
```

- [ ] **Step 8: Run low-Mach regression tests**

Run:

```powershell
python -m unittest tests.test_canonical_validation -v
```

Expected outcome:

```text
Ran 2 tests
OK
```

If the second placeholder test is reported as `ok` because it contains `pass`, record that honestly in the evidence report.

- [ ] **Step 9: Commit implementation**

Run:

```powershell
git -C D:\CodeProjects\research-paper add CLI\lbm_utils.py CLI\advanced_lbm_solver.py CLI\cascaded_lbm.py CLI\cfd_simulator.py CLI\solver_grid_speed_study.py
git -C D:\CodeProjects\research-paper commit -m "add lbm compressibility regime metadata"
```

Expected outcome: one commit containing only solver/evidence-path code changes.

---

### Task 3: Audit and Evidence Utilities

**Files:**
- Create: `CLI/solver_compressibility_audit.py`
- Create: `CLI/compressibility_evidence_report.py`
- Generate: `build/solver_diagnostics/compressibility_audit_20260612/solver_compressibility_audit.md`
- Generate: `build/solver_diagnostics/compressibility_evidence_20260612/summary.json`
- Generate: `build/solver_diagnostics/compressibility_evidence_20260612/compressibility_report.md`

- [ ] **Step 1: Create audit generator**

Create `CLI/solver_compressibility_audit.py`:

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditRow:
    assumption: str
    location: str
    physical_consequence: str
    valid_regime: str
    required_fix_or_gate: str


AUDIT_ROWS = [
    AuditRow(
        "Second-order isothermal D3Q27 equilibrium",
        "CLI/advanced_lbm_solver.py:compute_equilibrium; CLI/cascaded_lbm.py:_initialize_equilibrium",
        "No energy distribution and no temperature evolution; density-pressure relation is isothermal.",
        "Low-Mach weakly compressible flows where density fluctuations stay small.",
        "Gate internal results above Mach 0.3 as experimental until a thermal/compressible model is implemented and validated.",
    ),
    AuditRow(
        "Raw/tensor-product moment basis with fixed isothermal cs2",
        "CLI/advanced_lbm_solver.py:compute_moment_equilibrium; CLI/cascaded_lbm.py:equilibrium_raw_moments",
        "Collision relaxes moments toward isothermal raw-moment equilibria, not compressible energy moments.",
        "Low-Mach weakly compressible/isothermal D3Q27 operation.",
        "Add thermal moments or coupled energy distribution before making compressible LBM claims.",
    ),
    AuditRow(
        "Pressure model p = rho / 3 in lattice units",
        "CLI/advanced_lbm_solver.py:collide_stream; CLI/cascaded_lbm.py:collide_stream",
        "Pressure lacks perfect-gas p=rho R T coupling and cannot model compressible thermodynamics.",
        "Isothermal LBM with fixed lattice sound speed.",
        "Document as weakly compressible only; external compressible OpenFOAM is required for high-Mach reference data.",
    ),
    AuditRow(
        "Mach-to-lattice mapping u_lattice = Ma / sqrt(3)",
        "CLI/lbm_utils.py:mach_to_lattice_velocity; CLI/advanced_lbm_solver.py:_estimate_lattice_freestream_velocity",
        "The internal solver preserves the requested Mach only while the stability clip does not reduce u_lattice.",
        "Claim-grade only through Mach 0.3 and current low-Mach tests.",
        "Emit u_lattice and lattice_mach in every output so clipping is visible.",
    ),
    AuditRow(
        "Viscosity from low-Mach LBM relaxation relation",
        "CLI/advanced_lbm_solver.py:_estimate_kinematic_viscosity; CLI/cascaded_lbm.py:_setup_physics_constants",
        "Relaxation time sets lattice viscosity for the isothermal solver; it does not encode high-speed gas transport.",
        "Under-resolved low-Mach sanity/regression cases.",
        "For compressible work, add Mach/Re nondimensionalization with temperature-dependent viscosity or state an explicit gate.",
    ),
    AuditRow(
        "Momentum-exchange wall force",
        "CLI/advanced_lbm_solver.py:_accumulate_momentum_exchange_force; CLI/cascaded_lbm.py:collide_stream",
        "Force is raw bounce-back momentum exchange, not a compressible pressure/shear surface integration.",
        "Low-Mach voxelized wall-force sanity checks.",
        "Keep raw force separate from calibrated/surrogate force and require external compressible validation for high-Mach force claims.",
    ),
    AuditRow(
        "Far-field and wall boundaries are low-Mach/simple bounce-back style",
        "CLI/advanced_lbm_solver.py:collide_and_stream; CLI/cascaded_lbm.py:collide_stream",
        "No subsonic/supersonic inlet/outlet characteristic treatment and no shock-aware boundary handling.",
        "Low-Mach exploratory internal runs.",
        "Gate transonic/supersonic internal runs as experimental until compressible-aware boundaries exist.",
    ),
    AuditRow(
        "LES/turbulence diagnostics are not a validated compressible turbulence closure",
        "CLI/advanced_lbm_solver.py:_refresh_flow_diagnostics; CLI/config.py:LBMPhysicsConfig",
        "Smagorinsky-like diagnostics may stabilize/diagnose but do not validate compressible turbulent physics.",
        "Qualitative low-Mach diagnostics.",
        "Keep turbulence outputs as diagnostics and do not use them to upgrade claim grade.",
    ),
    AuditRow(
        "Convergence gate uses finite fields and force stability",
        "CLI/advanced_lbm_solver.py:compute_aerodynamic_coefficients",
        "A numerically stable internal run can still be physically invalid for high Mach.",
        "Internal low-Mach sanity gate only.",
        "Emit lbm_converged separately from validity_regime and claim_grade.",
    ),
    AuditRow(
        "Training label source selection is tiered",
        "CLI/cfd_simulator.py:simulate_aerodynamics; CLI/data_utils.py:GroundTruthExporter.export_sample",
        "Internal LBM labels can train surrogates, but cannot be silently treated as external ground truth.",
        "Low-Mach raw LBM for internal/surrogate training; external PDE for PINN-ready labels.",
        "High-Mach internal LBM must set training_drag_source to none_high_mach_internal_lbm_unvalidated.",
    ),
]


def render_markdown() -> str:
    lines = [
        "# Solver Compressibility Audit",
        "",
        "This audit documents where the current internal D3Q27 LBM path remains low-Mach, weakly compressible, and isothermal.",
        "",
        "| assumption | location | physical consequence | valid regime | required fix or gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in AUDIT_ROWS:
        lines.append(
            f"| {row.assumption} | `{row.location}` | {row.physical_consequence} | "
            f"{row.valid_regime} | {row.required_fix_or_gate} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Internal D3Q27 low-Mach raw outputs are bounded sanity/regression evidence, not production CFD.",
            "- Internal D3Q27 Mach > 0.3 outputs are executable exploratory runs labeled `experimental_high_mach_unvalidated`.",
            "- Compressible/high-Mach claim support must come from external compressible OpenFOAM evidence or a future thermal/compressible LBM implementation with validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "build" / "solver_diagnostics" / "compressibility_audit_20260612")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "solver_compressibility_audit.md"
    output.write_text(render_markdown(), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Create evidence report generator**

Create `CLI/compressibility_evidence_report.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except Exception:
        return False


def _copy_comparison_rows(source: Path | None, destination: Path) -> tuple[int, str]:
    if source is None or not source.exists():
        destination.write_text("", encoding="utf-8")
        return 0, "No comparison CSV was available; Cd-error plot is blocked by missing paired OpenFOAM/LBM rows."
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")
    finite_rows = sum(1 for row in rows if _finite_number(row.get("cd_error_percent")))
    if finite_rows == 0:
        return 0, "Cd-error plot is blocked because no finite paired Cd error rows exist."
    return finite_rows, "Cd-error rows are available for plotting."


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Compressibility Evidence Report",
        "",
        "## Solver Status",
        "",
        "- Internal D3Q27 status: low-Mach weakly compressible/isothermal.",
        "- High-Mach internal D3Q27 status: experimental and unvalidated.",
        "- Compressible LBM implementation in this pass: not implemented, because the current solver lacks thermal state, perfect-gas EOS coupling, compressible boundary conditions, and shock/steep-gradient validation.",
        "",
        "## Commands Recorded",
        "",
    ]
    for item in summary.get("commands", []):
        lines.append(f"- `{item['command']}`: {item['outcome']}")
    lines.extend(
        [
            "",
            "## Evidence Gates",
            "",
            f"- Audit artifact: `{summary['audit_artifact']}`",
            f"- Comparison rows: `{summary['comparison_csv']}`",
            f"- Plot status: {summary['plot_status']}",
            "",
            "## Claim Boundary",
            "",
            "Raw internal low-Mach LBM remains separate from calibrated/surrogate/training paths. Internal Mach > 0.3 results must not be cited as validated compressible CFD.",
        ]
    )
    (output_dir / "compressibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "build" / "solver_diagnostics" / "compressibility_evidence_20260612")
    parser.add_argument("--audit-artifact", type=Path, default=REPO / "build" / "solver_diagnostics" / "compressibility_audit_20260612" / "solver_compressibility_audit.md")
    parser.add_argument("--grid-speed-summary", type=Path, default=None)
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--command", action="append", default=[], help="Recorded command/outcome pair as command :: outcome")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "comparison_rows.csv"
    finite_rows, plot_status = _copy_comparison_rows(args.comparison_csv, comparison_csv)
    summary = {
        "status": "path_b_gated_low_mach_internal_solver",
        "audit_artifact": str(args.audit_artifact),
        "grid_speed_summary": str(args.grid_speed_summary) if args.grid_speed_summary else None,
        "comparison_csv": str(comparison_csv),
        "finite_cd_error_rows": finite_rows,
        "plot_status": plot_status,
        "commands": [
            {"command": item.split(" :: ", 1)[0], "outcome": item.split(" :: ", 1)[1] if " :: " in item else "recorded"}
            for item in args.command
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Generate audit artifact**

Run:

```powershell
python CLI\solver_compressibility_audit.py --output-dir build\solver_diagnostics\compressibility_audit_20260612
```

Expected outcome:

```text
build\solver_diagnostics\compressibility_audit_20260612\solver_compressibility_audit.md
```

- [ ] **Step 4: Probe OpenFOAM availability without launching a large run**

Run:

```powershell
python CLI\openfoam_mach_sweep.py --help
```

Expected outcome: CLI usage prints successfully.

Run:

```powershell
wsl -d Ubuntu-24.04 bash -lc "source /usr/lib/openfoam/openfoam*/etc/bashrc 2>/dev/null || source /opt/openfoam*/etc/bashrc 2>/dev/null || true; command -v checkMesh; command -v simpleFoam; command -v sonicFoam"
```

Expected outcome: either command paths print, or no command paths print. If no command paths print, record OpenFOAM as unavailable in the evidence report and do not launch mesh/solver jobs.

- [ ] **Step 5: Run modest LBM-only evidence check**

Run:

```powershell
python CLI\solver_grid_speed_study.py --stl test_cylinder.stl --output-dir build\solver_diagnostics\compressibility_evidence_20260612\grid_speed_lbm_only --mach-values 0.1,0.5 --lbm-grids 16 --lbm-steps 5 --device cpu --skip-openfoam
```

Expected outcome:

```text
build\solver_diagnostics\compressibility_evidence_20260612\grid_speed_lbm_only\grid_speed_study_summary.json
```

The run must show Mach 0.1 as `validated_low_mach_envelope` and Mach 0.5 as `experimental_high_mach_unvalidated`. It is not a validation study because OpenFOAM is skipped.

- [ ] **Step 6: Generate evidence report**

Run:

```powershell
python CLI\compressibility_evidence_report.py --output-dir build\solver_diagnostics\compressibility_evidence_20260612 --audit-artifact build\solver_diagnostics\compressibility_audit_20260612\solver_compressibility_audit.md --grid-speed-summary build\solver_diagnostics\compressibility_evidence_20260612\grid_speed_lbm_only\grid_speed_study_summary.json --command "python CLI\solver_compressibility_audit.py --output-dir build\solver_diagnostics\compressibility_audit_20260612 :: wrote audit markdown" --command "python CLI\solver_grid_speed_study.py --stl test_cylinder.stl --output-dir build\solver_diagnostics\compressibility_evidence_20260612\grid_speed_lbm_only --mach-values 0.1,0.5 --lbm-grids 16 --lbm-steps 5 --device cpu --skip-openfoam :: completed LBM-only metadata probe"
```

Expected outcome:

```text
build\solver_diagnostics\compressibility_evidence_20260612
```

- [ ] **Step 7: Commit evidence utilities**

Run:

```powershell
git -C D:\CodeProjects\research-paper add CLI\solver_compressibility_audit.py CLI\compressibility_evidence_report.py
git -C D:\CodeProjects\research-paper commit -m "add solver compressibility evidence utilities"
```

Expected outcome: one commit containing only reusable utility scripts.

---

### Task 4: Documentation Claim Boundary

**Files:**
- Modify: `CLI/GROUND_TRUTH_SPEC.md`
- Modify: `paper/CLAIMS_EVIDENCE_MATRIX.md`
- Modify: `paper/FINAL_RUN_GATES.md`

- [ ] **Step 1: Update ground-truth spec**

In `CLI/GROUND_TRUTH_SPEC.md`, under `## Label Tiers and Sources`, replace the current tier table with:

```markdown
| Tier | Source | Logic | Use Case | Compressibility boundary |
| :--- | :--- | :--- | :--- | :--- |
| `lbm_raw` | `lbm_d3q27` | Pure internal D3Q27 momentum exchange. | Fast low-Mach internal/surrogate training diagnostics. | Claim-grade only inside the current `validated_low_mach_envelope`; Mach > 0.3 is `experimental_high_mach_unvalidated`. |
| `lbm_calibrated` | `lbm_d3q27` | Heuristically corrected value derived from internal LBM diagnostics. | Stable surrogate/training proxy when explicitly labeled as calibrated. | Never upgrades raw high-Mach internal LBM to validated compressible CFD. |
| `external_pde` | `OpenFOAM` or analytic reference | Independent solver/reference with strict gates. | High-fidelity PINN ground truth when fields and residual gates pass. | Incompressible OpenFOAM and compressible OpenFOAM must be labeled separately. |
```

Add this section after `## PDE Target`:

```markdown
## Compressibility Boundary

The current internal D3Q27 solver is a low-Mach, weakly compressible, isothermal LBM path. It uses a fixed lattice sound speed, second-order isothermal equilibrium, and no evolved thermal state. Internal Mach values above 0.3 may be executed for exploratory diagnostics, but they are labeled `experimental_high_mach_unvalidated` and cannot set `pinn_ready: true` or support compressible-CFD claims without independent external validation.

Required solver metadata fields are:
- `mach_number`
- `lattice_mach`
- `u_lattice`
- `sound_speed_model`
- `compressibility_model`
- `thermal_model`
- `validity_regime`
- `claim_grade`
- `high_mach_warning`
- `lbm_converged`
- `force_stability`
- `training_drag_source`
```

- [ ] **Step 2: Update claims matrix**

Append to `paper/CLAIMS_EVIDENCE_MATRIX.md` table:

```markdown
| C11 | `CLI/advanced_lbm_solver.py`, `CLI/cascaded_lbm.py`, `CLI/GROUND_TRUTH_SPEC.md` | internal solver compressibility | implied that accepting Mach > 0.3 means validated compressible CFD | internal D3Q27 is validated only as a low-Mach weakly compressible/isothermal sanity path; Mach > 0.3 is experimental unless external compressible validation exists | `build/solver_diagnostics/compressibility_audit_20260612/solver_compressibility_audit.md` and focused regime tests | revised |
```

Add after `## Conditioning-Specific Evidence Boundary`:

```markdown
## Compressibility-Specific Evidence Boundary

- Internal LBM low-Mach raw outputs are separate from calibrated/surrogate/training labels.
- Internal LBM high-Mach outputs are executable diagnostics labeled `experimental_high_mach_unvalidated`.
- OpenFOAM incompressible and OpenFOAM compressible references must be recorded as distinct evidence sources.
- Paper text must not describe the internal solver as validated transonic or supersonic CFD until a thermal/compressible solver path and external validation gates pass.
```

- [ ] **Step 3: Update final run gates**

Append to the `paper/FINAL_RUN_GATES.md` Claim Gates table:

```markdown
| `Validated compressible/high-Mach internal solver` | Thermal/compressible LBM implementation or external compressible reference comparison under fixed geometries | OpenFOAM compressible solver such as `sonicFoam`/`rhoPimpleFoam` with residual, Courant, latest-time, Cd/Cl history, and force-stability records | Mach-specific Cd/Cl agreement, positive rho/T/p, stable force history, documented boundary conditions | All focused compressibility tests pass and high-Mach comparisons are finite, converged, and documented | `Internal D3Q27 high-Mach runs are experimental and unvalidated` | Gate implemented / claim evidence blocked |
```

Add to `## Required Final-Run Inputs Before Claim Expansion`:

```markdown
8. Compressibility audit and evidence artifacts under `build/solver_diagnostics/compressibility_*`.
9. For high-Mach claims, an external compressible OpenFOAM comparison or a validated thermal/compressible LBM path with shock/steep-gradient and boundary-condition tests.
```

- [ ] **Step 4: Commit documentation**

Run:

```powershell
git -C D:\CodeProjects\research-paper add CLI\GROUND_TRUTH_SPEC.md paper\CLAIMS_EVIDENCE_MATRIX.md paper\FINAL_RUN_GATES.md
git -C D:\CodeProjects\research-paper commit -m "document lbm compressibility claim boundary"
```

Expected outcome: one commit containing only documentation changes.

---

### Task 5: Verification and Final Evidence

**Files:**
- Read: all files changed in Tasks 1-4
- Generate or update: `build/solver_diagnostics/compressibility_evidence_20260612/summary.json`
- Generate or update: `build/solver_diagnostics/compressibility_evidence_20260612/compressibility_report.md`

- [ ] **Step 1: Run required compile command**

Run:

```powershell
python -m py_compile CLI\advanced_lbm_solver.py CLI\cascaded_lbm.py CLI\lbm_utils.py CLI\d3q27_kernels.py CLI\solver_grid_speed_study.py CLI\openfoam_mach_sweep.py CLI\solver_compressibility_audit.py CLI\compressibility_evidence_report.py
```

Expected outcome: exit code 0 with no syntax errors.

- [ ] **Step 2: Run focused Mach mapping tests**

Run:

```powershell
python -m unittest tests.test_lbm_mach_mapping -v
```

Expected outcome:

```text
Ran 3 tests
OK
```

- [ ] **Step 3: Run focused compressibility tests**

Run:

```powershell
python -m unittest tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata -v
```

Expected outcome:

```text
Ran 6 tests
OK
```

- [ ] **Step 4: Run canonical low-Mach regression**

Run:

```powershell
python -m unittest tests.test_canonical_validation -v
```

Expected outcome:

```text
Ran 2 tests
OK
```

- [ ] **Step 5: Run full unittest discovery**

Run:

```powershell
python -m unittest discover tests -v
```

Expected outcome: record exact result. If unrelated dirty-worktree tests fail, capture the failure names and do not claim the full suite passes.

- [ ] **Step 6: Regenerate audit/evidence after verification**

Run:

```powershell
python CLI\solver_compressibility_audit.py --output-dir build\solver_diagnostics\compressibility_audit_20260612
python CLI\compressibility_evidence_report.py --output-dir build\solver_diagnostics\compressibility_evidence_20260612 --audit-artifact build\solver_diagnostics\compressibility_audit_20260612\solver_compressibility_audit.md --grid-speed-summary build\solver_diagnostics\compressibility_evidence_20260612\grid_speed_lbm_only\grid_speed_study_summary.json --command "python -m py_compile CLI\advanced_lbm_solver.py CLI\cascaded_lbm.py CLI\lbm_utils.py CLI\d3q27_kernels.py CLI\solver_grid_speed_study.py CLI\openfoam_mach_sweep.py CLI\solver_compressibility_audit.py CLI\compressibility_evidence_report.py :: passed" --command "python -m unittest tests.test_lbm_mach_mapping -v :: passed" --command "python -m unittest tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata -v :: passed" --command "python -m unittest tests.test_canonical_validation -v :: record exact outcome" --command "python -m unittest discover tests -v :: record exact outcome"
```

Expected outcome: summary and report exist and match the fresh command outcomes.

- [ ] **Step 7: Inspect git history and working tree**

Run:

```powershell
git -C D:\CodeProjects\research-paper log --oneline -4
git -C D:\CodeProjects\research-paper status --short
```

Expected outcome: four new commits at the top if all commit points succeeded, generated `build/solver_diagnostics/...` artifacts untracked or ignored, and no unrelated user changes reverted.

---

## Self-Review

Spec coverage:
- Physics audit: Task 3 creates `solver_compressibility_audit.md` with assumption/location/consequence/regime/fix table and explicit file references.
- Regime classification: Tasks 1-2 add shared helpers and tests for Mach <= 0.3 and Mach > 0.3.
- Metadata: Tasks 1-2 add and test every requested metadata field.
- No fake compressible LBM: The architecture explicitly selects Path B and documents why Path A is not implemented.
- Low-Mach preservation: Task 5 runs canonical validation and Mach mapping tests.
- OpenFOAM/evidence: Task 3 probes availability and runs a modest LBM-only grid/speed metadata check; no huge 512^3 jobs are launched.
- Documentation: Task 4 updates the ground-truth spec, claims matrix, and final run gates.
- Validation command set: Task 5 includes the required py_compile, focused tests, canonical test, and full unittest discovery.
- Git discipline: Commit points stage exact files only.

Placeholder scan:
- There are no `TODO`, `TBD`, or "fill in" items.
- Plot generation is explicitly gated when paired finite Cd-error data is absent.

Type consistency:
- `classify_lbm_regime()` and `build_lbm_compressibility_metadata()` are defined in `CLI/lbm_utils.py` and used consistently by tests and implementation.
- `validity_regime`, `claim_grade`, and `training_drag_source` use exact string values checked by tests.
