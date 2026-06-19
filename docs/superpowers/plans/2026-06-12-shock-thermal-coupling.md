# Shock Thermal Coupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add staged shock sensing/stabilization, thermal inlet/outlet boundary hooks, and pressure-gradient thermal-to-flow coupling to the GPU thermal LBM wrapper without claiming validated Mach 2 CFD.

**Architecture:** Keep `D3Q27CascadedSolver` as the raw isothermal flow solver. Extend `ThermalBGKSolver` with D3Q7 thermal shock sensing, local artificial thermal diffusivity, and fixed/zero-gradient thermal boundaries. Extend `ThermalD3Q27Solver` with a pressure-gradient Guo forcing hook that couples the ideal-gas thermal pressure field into the next flow solve while metadata remains experimental.

**Tech Stack:** Python, Torch tensors on CPU/CUDA, `unittest`, existing D3Q27 Torch solver.

---

## Claim Boundary

- This pass may add a `shock_stabilization_model`, but `shock_capable` remains `false`.
- This pass may add `thermal_boundary_model`, but it is not a validated characteristic compressible boundary condition for the isothermal D3Q27 flow populations.
- Pressure-gradient coupling may feed thermal pressure into the existing Guo forcing path, but force coefficients remain raw internal LBM and `pinn_ready` remains `false`.
- Mach 2 remains `experimental_thermal_lbm_unvalidated` until shock tube/normal shock/OpenFOAM evidence exists.

## Files

- Modify: `CLI/thermal_lbm_solver.py`
  - Add config fields for boundary temperatures, shock sensor, artificial diffusivity, and opt-in pressure coupling.
  - Add `compute_shock_sensor`, `compute_effective_omega`, `apply_thermal_boundaries`, and `compute_thermal_pressure_force`.
  - Update metadata in `compute_aerodynamic_coefficients`.
- Modify: `tests/test_thermal_lbm_solver.py`
  - Add failing tests first for shock sensor, boundary hooks, and pressure coupling.
- Create: `build/solver_diagnostics/shock_thermal_coupling_20260612/summary.json`
- Create: `build/solver_diagnostics/shock_thermal_coupling_20260612/shock_thermal_coupling_report.md`

## Task 1: Shock Sensor And Artificial Thermal Diffusion

**Files:**
- Modify: `tests/test_thermal_lbm_solver.py`
- Modify: `CLI/thermal_lbm_solver.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_shock_sensor_is_low_for_uniform_temperature_and_high_at_jump(self):
    device = torch.device("cpu")
    solver = ThermalBGKSolver((8, 4, 4), device, ThermalLBMConfig(shock_sensor_threshold=0.02))
    uniform = torch.full((8, 4, 4), 300.0, device=device)
    jump = uniform.clone()
    jump[4:, :, :] = 600.0

    uniform_sensor = solver.compute_shock_sensor(uniform)
    jump_sensor = solver.compute_shock_sensor(jump)

    self.assertLess(float(uniform_sensor.max().item()), 1e-6)
    self.assertGreater(float(jump_sensor.max().item()), 0.5)

def test_shock_stabilization_lowers_local_omega_near_jump(self):
    device = torch.device("cpu")
    config = ThermalLBMConfig(
        reference_temperature=300.0,
        shock_stabilization_enabled=True,
        shock_diffusivity_multiplier=4.0,
    )
    solver = ThermalBGKSolver((8, 4, 4), device, config)
    temperature = torch.full((8, 4, 4), 300.0, device=device)
    temperature[4:, :, :] = 600.0

    sensor = solver.compute_shock_sensor(temperature)
    omega_field = solver.compute_effective_omega(sensor)

    self.assertLess(float(omega_field[sensor > 0.5].mean().item()), float(solver.omega))
```

- [ ] **Step 2: Run tests red**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: fail with missing `compute_shock_sensor`, missing `compute_effective_omega`, or missing config fields.

- [ ] **Step 3: Implement minimal production code**

Add config fields:

```python
shock_stabilization_enabled: bool = True
shock_sensor_threshold: float = 0.02
shock_diffusivity_multiplier: float = 3.0
shock_sensor_epsilon: float = 1e-6
```

Add methods:

```python
def compute_shock_sensor(self, scalar_field: torch.Tensor) -> torch.Tensor:
    field = self._coerce_field(scalar_field)
    gx = 0.5 * (torch.roll(field, -1, 0) - torch.roll(field, 1, 0))
    gy = 0.5 * (torch.roll(field, -1, 1) - torch.roll(field, 1, 1))
    gz = 0.5 * (torch.roll(field, -1, 2) - torch.roll(field, 1, 2))
    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz)
    normalized = grad_mag / (torch.abs(field) + float(self.config.shock_sensor_epsilon))
    threshold = max(float(self.config.shock_sensor_threshold), 1e-12)
    return torch.clamp(normalized / threshold, min=0.0, max=1.0)

def compute_effective_omega(self, shock_sensor: torch.Tensor) -> torch.Tensor:
    sensor = self._coerce_field(shock_sensor).clamp(0.0, 1.0)
    if not bool(self.config.shock_stabilization_enabled):
        return torch.full(self.shape, float(self.omega), dtype=self.dtype, device=self.device)
    base_alpha = max(float(self.config.thermal_diffusivity_lattice), 1e-12)
    multiplier = max(float(self.config.shock_diffusivity_multiplier), 1.0)
    alpha_field = base_alpha * (1.0 + (multiplier - 1.0) * sensor)
    tau_field = 0.5 + 3.0 * alpha_field
    return 1.0 / tau_field.clamp_min(0.500001)
```

In `collide_stream`, replace scalar `self.omega` with `omega_field.unsqueeze(0)` for the collision.

- [ ] **Step 4: Run tests green**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: pass.

## Task 2: Thermal Inlet/Outlet Boundary Hooks

**Files:**
- Modify: `tests/test_thermal_lbm_solver.py`
- Modify: `CLI/thermal_lbm_solver.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_thermal_inlet_outlet_boundaries_set_inlet_and_extrapolate_outlet(self):
    device = torch.device("cpu")
    config = ThermalLBMConfig(
        reference_temperature=300.0,
        inlet_temperature=350.0,
        outlet_temperature=None,
        thermal_boundary_model="fixed_temperature_inlet_zero_gradient_outlet",
    )
    solver = ThermalBGKSolver((6, 4, 4), device, config)
    geometry = torch.zeros((6, 4, 4), dtype=torch.float32, device=device)
    zero_velocity = tuple(torch.zeros_like(geometry) for _ in range(3))
    solver.temperature[-2, :, :] = 325.0
    solver.g.copy_(solver.compute_equilibrium(solver.temperature, zero_velocity))

    solver.collide_stream(zero_velocity, geometry, steps=1)

    self.assertTrue(torch.allclose(solver.temperature[0], torch.full_like(solver.temperature[0], 350.0)))
    self.assertTrue(torch.allclose(solver.temperature[-1], solver.temperature[-2]))

def test_boundary_metadata_marks_supersonic_as_experimental_not_validated(self):
    cfg = make_config(2.0, thermal_enabled=True)
    cfg.thermal_lbm_config = ThermalLBMConfig(
        reference_temperature=310.0,
        inlet_temperature=350.0,
        thermal_boundary_model="fixed_temperature_inlet_zero_gradient_outlet",
    )
    solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
    geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
    geometry[3:5, 3:5, 3:5] = 1.0

    solver.collide_stream(geometry, steps=1)
    results = solver.compute_aerodynamic_coefficients(geometry)

    self.assertEqual(results["thermal_boundary_model"], "fixed_temperature_inlet_zero_gradient_outlet")
    self.assertEqual(results["compressible_boundary_status"], "staged_thermal_boundary_not_characteristic_validated")
    self.assertEqual(results["inlet_outlet_regime"], "supersonic_experimental")
```

- [ ] **Step 2: Run tests red**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: fail because boundary config fields and metadata are absent.

- [ ] **Step 3: Implement minimal production code**

Add config fields:

```python
inlet_temperature: float | None = None
outlet_temperature: float | None = None
thermal_boundary_model: str = "fixed_temperature_inlet_zero_gradient_outlet"
```

Add method:

```python
def apply_thermal_boundaries(self, velocity):
    if self.config.thermal_boundary_model == "none":
        return
    inlet_temperature = self.config.reference_temperature if self.config.inlet_temperature is None else self.config.inlet_temperature
    self.temperature[0, :, :] = self._clamp_temperature_value(float(inlet_temperature))
    if self.config.outlet_temperature is None:
        self.temperature[-1, :, :] = self.temperature[-2, :, :]
    else:
        self.temperature[-1, :, :] = self._clamp_temperature_value(float(self.config.outlet_temperature))
    self.g[:, 0, :, :] = self.compute_equilibrium(self.temperature, velocity)[:, 0, :, :]
    self.g[:, -1, :, :] = self.compute_equilibrium(self.temperature, velocity)[:, -1, :, :]
```

Call it after each thermal stream step.

- [ ] **Step 4: Run tests green**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: pass.

## Task 3: Pressure-Gradient Thermal Coupling Into Flow Solver

**Files:**
- Modify: `tests/test_thermal_lbm_solver.py`
- Modify: `CLI/thermal_lbm_solver.py`

- [ ] **Step 1: Write failing tests**

Add tests:

```python
def test_pressure_gradient_coupling_produces_clipped_guo_force(self):
    cfg = make_config(0.5, thermal_enabled=True)
    cfg.thermal_lbm_config = ThermalLBMConfig(
        reference_temperature=300.0,
        pressure_coupling_strength=0.1,
        pressure_gradient_clip=0.02,
    )
    solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
    ramp = torch.linspace(300.0, 600.0, solver.resolution).view(-1, 1, 1).expand(
        solver.resolution, solver.resolution, solver.resolution
    )
    solver.thermal_solver.set_temperature(ramp)
    force = solver.compute_thermal_pressure_force()

    self.assertEqual(force.shape, (3, solver.resolution, solver.resolution, solver.resolution))
    self.assertGreater(float(torch.abs(force[0]).max().item()), 0.0)
    self.assertLessEqual(float(torch.abs(force).max().item()), 0.020001)

def test_pressure_gradient_coupling_metadata_replaces_diagnostic_only_label(self):
    cfg = make_config(0.5, thermal_enabled=True)
    cfg.thermal_lbm_config = ThermalLBMConfig(reference_temperature=310.0, pressure_coupling_strength=0.1)
    solver = create_thermal_lbm_solver(cfg, torch.device("cpu"), TestPhysicsConfig)
    geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
    geometry[3:5, 3:5, 3:5] = 1.0

    solver.collide_stream(geometry, steps=1)
    results = solver.compute_aerodynamic_coefficients(geometry)

    self.assertEqual(results["thermal_force_coupling"], "pressure_gradient_guo_forcing_experimental")
    self.assertFalse(results["pinn_ready"])
```

- [ ] **Step 2: Run tests red**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: fail because pressure coupling fields/methods are absent.

- [ ] **Step 3: Implement minimal production code**

Add config fields:

```python
pressure_coupling_strength: float = 0.0
pressure_gradient_clip: float = 0.02
```

Add method on `ThermalD3Q27Solver`:

```python
def compute_thermal_pressure_force(self) -> torch.Tensor:
    state = self._refresh_thermodynamic_state()
    pressure = state.pressure
    gx = 0.5 * (torch.roll(pressure, -1, 0) - torch.roll(pressure, 1, 0))
    gy = 0.5 * (torch.roll(pressure, -1, 1) - torch.roll(pressure, 1, 1))
    gz = 0.5 * (torch.roll(pressure, -1, 2) - torch.roll(pressure, 1, 2))
    scale = float(self.thermal_config.pressure_coupling_strength) / pressure.mean().clamp_min(float(self.thermal_config.min_pressure))
    force = -scale * torch.stack([gx, gy, gz])
    clip = max(float(self.thermal_config.pressure_gradient_clip), 0.0)
    return torch.clamp(force, min=-clip, max=clip)
```

In `collide_stream`, compute `thermal_force = compute_thermal_pressure_force()` before flow solve and add it to `ext_force` if provided.

- [ ] **Step 4: Run tests green**

Run:

```powershell
python -m unittest tests.test_thermal_lbm_solver -v
```

Expected: pass.

## Task 4: Evidence And Regression Verification

**Files:**
- Create: `build/solver_diagnostics/shock_thermal_coupling_20260612/summary.json`
- Create: `build/solver_diagnostics/shock_thermal_coupling_20260612/shock_thermal_coupling_report.md`

- [ ] **Step 1: Run focused verification**

Run:

```powershell
python -m py_compile CLI\thermal_lbm_solver.py CLI\advanced_lbm_solver.py CLI\lbm_utils.py
python -m unittest tests.test_thermal_lbm_solver -v
python -m unittest tests.test_lbm_compressibility_regime tests.test_lbm_compressibility_metadata tests.test_thermal_lbm_solver -v
python -m unittest tests.test_lbm_mach_mapping -v
python -m unittest tests.test_canonical_validation -v
```

Expected: pass.

- [ ] **Step 2: Run full discovery**

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'; python -m unittest discover tests -v
```

Expected current repo state: may fail the pre-existing `tests/test_aircraft_validity.py` symmetry assertion. Record exact result.

- [ ] **Step 3: Write evidence artifacts**

Write `summary.json` and `shock_thermal_coupling_report.md` with:
- commands and outcomes
- CUDA availability
- shock sensor max for a synthetic jump
- boundary metadata for Mach 2
- coupling force max for a synthetic temperature ramp
- explicit statement that `shock_capable` remains false

## Commit Points

1. `add thermal shock stabilization tests`
2. `add staged shock and boundary thermal coupling`
3. `add shock thermal coupling evidence`

Generated files under `build/` may remain untracked if ignored by repo convention.
