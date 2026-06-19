from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Iterable

import torch

from advanced_lbm_solver import D3Q27CascadedSolver


THERMAL_MODEL_NAME = "coupled_d3q7_temperature_bgk"
THERMAL_VALIDITY_REGIME = "experimental_thermal_lbm_unvalidated"
THERMAL_COMPRESSIBILITY_MODEL = "staged_thermal_lbm_experimental"
THERMAL_PRESSURE_MODEL = "ideal_gas_diagnostic_p_equals_rho_R_T"
THERMAL_TRAINING_SOURCE = "none_thermal_internal_lbm_unvalidated"
THERMODYNAMIC_SOLVER_NAME = "staged_d3q7_thermal_bgk_attached_to_d3q27"


@dataclass(frozen=True)
class ThermalLBMConfig:
    """Parameters for the staged GPU thermal LBM attachment."""

    reference_temperature: float = 300.0
    gas_constant: float = 287.05
    thermal_diffusivity_lattice: float = 0.01
    min_temperature: float = 1.0
    max_temperature: float = 5000.0
    min_density: float = 1e-6
    min_pressure: float = 1e-6
    equilibrium_cu_limit: float = 0.25
    max_thermal_steps_per_call: int = 64
    shock_stabilization_enabled: bool = True
    shock_sensor_threshold: float = 0.02
    shock_diffusivity_multiplier: float = 3.0
    shock_sensor_epsilon: float = 1e-6
    inlet_temperature: float | None = None
    outlet_temperature: float | None = None
    thermal_boundary_model: str = "fixed_temperature_inlet_zero_gradient_outlet"
    pressure_coupling_strength: float = 0.0
    pressure_gradient_clip: float = 0.02
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class ThermodynamicState:
    density: torch.Tensor
    temperature: torch.Tensor
    pressure: torch.Tensor


def _coerce_thermal_config(value: Any) -> ThermalLBMConfig:
    if value is None:
        return ThermalLBMConfig()
    if isinstance(value, ThermalLBMConfig):
        return value
    if isinstance(value, dict):
        return ThermalLBMConfig(**{k: v for k, v in value.items() if k in {f.name for f in fields(ThermalLBMConfig)}})

    defaults = ThermalLBMConfig()
    kwargs = {}
    for field in fields(ThermalLBMConfig):
        kwargs[field.name] = getattr(value, field.name, getattr(defaults, field.name))
    return ThermalLBMConfig(**kwargs)


def _shape_tuple(shape: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(shape, int):
        return (int(shape), int(shape), int(shape))
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"Thermal LBM shape must have three dimensions, got {values!r}")
    if min(values) <= 0:
        raise ValueError(f"Thermal LBM shape dimensions must be positive, got {values!r}")
    return values


class ThermalBGKSolver:
    """D3Q7 temperature transport solver using vectorized Torch operations."""

    _DIRECTIONS = (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    _WEIGHTS = (0.25, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)

    def __init__(self, shape: int | Iterable[int], device: torch.device | str, config: ThermalLBMConfig | None = None):
        self.shape = _shape_tuple(shape)
        self.device = torch.device(device)
        self.config = _coerce_thermal_config(config)
        self.dtype = self.config.dtype

        self.directions = torch.tensor(self._DIRECTIONS, dtype=torch.long, device=self.device)
        self.weights = torch.tensor(self._WEIGHTS, dtype=self.dtype, device=self.device)
        self.ex = self.directions[:, 0].to(dtype=self.dtype).view(7, 1, 1, 1)
        self.ey = self.directions[:, 1].to(dtype=self.dtype).view(7, 1, 1, 1)
        self.ez = self.directions[:, 2].to(dtype=self.dtype).view(7, 1, 1, 1)
        self._stream_shifts = [tuple(int(v) for v in direction) for direction in self._DIRECTIONS]

        alpha = max(float(self.config.thermal_diffusivity_lattice), 1e-12)
        self.tau = max(0.500001, 0.5 + 3.0 * alpha)
        self.omega = 1.0 / self.tau

        self.temperature = torch.full(
            self.shape,
            self._clamp_temperature_value(self.config.reference_temperature),
            dtype=self.dtype,
            device=self.device,
        )
        zero_velocity = tuple(torch.zeros(self.shape, dtype=self.dtype, device=self.device) for _ in range(3))
        self.g = self.compute_equilibrium(self.temperature, zero_velocity)
        self._stream_buffer = torch.empty_like(self.g)
        self.shock_sensor = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        self.effective_omega = torch.full(self.shape, float(self.omega), dtype=self.dtype, device=self.device)
        self.shock_stabilization_mask = torch.zeros(self.shape, dtype=torch.bool, device=self.device)

    def _nonperiodic_gradient(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gx = torch.zeros_like(field)
        gy = torch.zeros_like(field)
        gz = torch.zeros_like(field)

        gx[1:-1, :, :] = 0.5 * (field[2:, :, :] - field[:-2, :, :])
        gx[0, :, :] = field[1, :, :] - field[0, :, :]
        gx[-1, :, :] = field[-1, :, :] - field[-2, :, :]

        gy[:, 1:-1, :] = 0.5 * (field[:, 2:, :] - field[:, :-2, :])
        gy[:, 0, :] = field[:, 1, :] - field[:, 0, :]
        gy[:, -1, :] = field[:, -1, :] - field[:, -2, :]

        gz[:, :, 1:-1] = 0.5 * (field[:, :, 2:] - field[:, :, :-2])
        gz[:, :, 0] = field[:, :, 1] - field[:, :, 0]
        gz[:, :, -1] = field[:, :, -1] - field[:, :, -2]
        return gx, gy, gz

    def _clamp_temperature_value(self, value: float) -> float:
        return float(min(max(float(value), float(self.config.min_temperature)), float(self.config.max_temperature)))

    def _coerce_field(self, field: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return field.to(device=self.device, dtype=dtype or self.dtype, non_blocking=True)

    def _coerce_velocity(self, velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(velocity) != 3:
            raise ValueError("Thermal velocity must contain ux, uy, and uz tensors")
        return tuple(self._coerce_field(component) for component in velocity)

    def set_temperature(self, temperature: torch.Tensor | float, velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None):
        if isinstance(temperature, torch.Tensor):
            next_temperature = self._coerce_field(temperature).clamp(
                min=float(self.config.min_temperature),
                max=float(self.config.max_temperature),
            )
        else:
            next_temperature = torch.full(
                self.shape,
                self._clamp_temperature_value(float(temperature)),
                dtype=self.dtype,
                device=self.device,
            )
        if tuple(next_temperature.shape) != self.shape:
            raise ValueError(f"Temperature shape {tuple(next_temperature.shape)!r} does not match solver shape {self.shape!r}")

        if velocity is None:
            velocity = tuple(torch.zeros(self.shape, dtype=self.dtype, device=self.device) for _ in range(3))
        with torch.no_grad():
            self.temperature.copy_(next_temperature)
            self.g.copy_(self.compute_equilibrium(self.temperature, self._coerce_velocity(velocity)))

    def compute_equilibrium(
        self,
        temperature: torch.Tensor,
        velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        ux, uy, uz = self._coerce_velocity(velocity)
        temperature = self._coerce_field(temperature).clamp(
            min=float(self.config.min_temperature),
            max=float(self.config.max_temperature),
        )
        cu = self.ex * ux + self.ey * uy + self.ez * uz
        cu_limit = max(float(self.config.equilibrium_cu_limit), 1e-6)
        cu = torch.clamp(cu, min=-cu_limit, max=cu_limit)
        return self.weights.view(7, 1, 1, 1) * temperature.unsqueeze(0) * (1.0 + 3.0 * cu)

    def compute_shock_sensor(self, scalar_field: torch.Tensor) -> torch.Tensor:
        field = self._coerce_field(scalar_field)
        gx, gy, gz = self._nonperiodic_gradient(field)
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
        tau_field = (0.5 + 3.0 * alpha_field).clamp_min(0.500001)
        return 1.0 / tau_field

    def build_thermodynamic_state(self, density: torch.Tensor, temperature: torch.Tensor | None = None) -> ThermodynamicState:
        if temperature is None:
            temperature = self.temperature
        rho = self._coerce_field(density).clamp_min(float(self.config.min_density))
        temp = self._coerce_field(temperature).clamp(
            min=float(self.config.min_temperature),
            max=float(self.config.max_temperature),
        )
        pressure = (rho * float(self.config.gas_constant) * temp).clamp_min(float(self.config.min_pressure))
        return ThermodynamicState(density=rho, temperature=temp, pressure=pressure)

    def apply_thermal_boundaries(self, velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        if str(self.config.thermal_boundary_model).lower() == "none":
            return

        inlet_temperature = (
            float(self.config.reference_temperature)
            if self.config.inlet_temperature is None
            else float(self.config.inlet_temperature)
        )
        self.temperature[0, :, :] = self._clamp_temperature_value(inlet_temperature)

        if self.config.outlet_temperature is None:
            self.temperature[-1, :, :] = self.temperature[-2, :, :]
        else:
            self.temperature[-1, :, :] = self._clamp_temperature_value(float(self.config.outlet_temperature))

        boundary_equilibrium = self.compute_equilibrium(self.temperature, velocity)
        self.g[:, 0, :, :] = boundary_equilibrium[:, 0, :, :]
        self.g[:, -1, :, :] = boundary_equilibrium[:, -1, :, :]

    def collide_stream(
        self,
        velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        geometry_mask: torch.Tensor,
        steps: int = 1,
    ) -> torch.Tensor:
        geometry = self._coerce_field(geometry_mask) > 0.5
        velocity = self._coerce_velocity(velocity)
        num_steps = max(0, int(steps))

        for _ in range(num_steps):
            temperature = torch.sum(self.g, dim=0).clamp(
                min=float(self.config.min_temperature),
                max=float(self.config.max_temperature),
            )
            self.shock_sensor = self.compute_shock_sensor(temperature)
            self.effective_omega = self.compute_effective_omega(self.shock_sensor)
            self.shock_stabilization_mask = self.shock_sensor > 0.5
            geq = self.compute_equilibrium(temperature, velocity)
            post_collision = self.g + self.effective_omega.unsqueeze(0) * (geq - self.g)

            for i, shift in enumerate(self._stream_shifts):
                streamed = torch.roll(post_collision[i], shifts=shift, dims=(0, 1, 2))
                source_solid = torch.roll(geometry, shifts=shift, dims=(0, 1, 2))
                blocked = geometry | source_solid
                self._stream_buffer[i].copy_(torch.where(blocked, self.g[i], streamed))

            self.g.copy_(self._stream_buffer)
            self.temperature.copy_(
                torch.sum(self.g, dim=0).clamp(
                    min=float(self.config.min_temperature),
                    max=float(self.config.max_temperature),
                )
            )
            self.apply_thermal_boundaries(velocity)

        return self.temperature


class ThermalD3Q27Solver:
    """Staged thermodynamic wrapper around the existing D3Q27 flow solver."""

    def __init__(self, config: Any, device: torch.device | str, phys_config: Any):
        self.config = config
        self.device = torch.device(device)
        self.flow_solver = D3Q27CascadedSolver(config, self.device, phys_config)
        self.thermal_config = _coerce_thermal_config(getattr(config, "thermal_lbm_config", None))
        self.thermal_solver = ThermalBGKSolver(
            self.flow_solver.resolution,
            self.device,
            self.thermal_config,
        )
        self._sync_flow_fields()
        self.temperature = self.thermal_solver.temperature
        self.thermodynamic_state = self.thermal_solver.build_thermodynamic_state(self.rho, self.temperature)
        self.thermodynamic_pressure = self.thermodynamic_state.pressure
        self.thermal_pressure_gradient_force = torch.zeros(
            (3, self.flow_solver.resolution, self.flow_solver.resolution, self.flow_solver.resolution),
            dtype=self.thermal_config.dtype,
            device=self.device,
        )

    def __getattr__(self, name: str):
        flow_solver = self.__dict__.get("flow_solver")
        if flow_solver is not None:
            return getattr(flow_solver, name)
        raise AttributeError(name)

    def _sync_flow_fields(self):
        for name in (
            "resolution",
            "inlet_velocity_lu",
            "f",
            "f_temp",
            "f_pre_stream",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "pressure",
            "rho",
            "force_x_accum",
            "force_z_accum",
            "force_x_last",
            "force_z_last",
            "force_samples",
            "projected_drag_accum",
            "projected_drag_last",
            "nu_turb",
            "vorticity",
            "q_criterion",
            "nu",
        ):
            if hasattr(self.flow_solver, name):
                setattr(self, name, getattr(self.flow_solver, name))

    def _thermal_velocity(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.velocity_x, self.velocity_y, self.velocity_z

    def _refresh_thermodynamic_state(self):
        self.thermodynamic_state = self.thermal_solver.build_thermodynamic_state(self.rho, self.thermal_solver.temperature)
        self.temperature = self.thermodynamic_state.temperature
        self.thermodynamic_pressure = self.thermodynamic_state.pressure
        return self.thermodynamic_state

    def _nonperiodic_gradient(self, field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.thermal_solver._nonperiodic_gradient(field)

    def compute_thermal_pressure_force(self) -> torch.Tensor:
        state = self._refresh_thermodynamic_state()
        pressure = state.pressure
        strength = float(self.thermal_config.pressure_coupling_strength)
        if strength == 0.0:
            self.thermal_pressure_gradient_force = torch.zeros(
                (3, *pressure.shape),
                dtype=pressure.dtype,
                device=self.device,
            )
            return self.thermal_pressure_gradient_force

        gx, gy, gz = self._nonperiodic_gradient(pressure)
        pressure_scale = pressure.mean().clamp_min(float(self.thermal_config.min_pressure))
        force = -(strength / pressure_scale) * torch.stack([gx, gy, gz])
        clip = max(float(self.thermal_config.pressure_gradient_clip), 0.0)
        if clip > 0.0:
            force = torch.clamp(force, min=-clip, max=clip)
        self.thermal_pressure_gradient_force = force.nan_to_num(0.0, posinf=clip, neginf=-clip)
        return self.thermal_pressure_gradient_force

    def collide_stream(self, geometry_mask: torch.Tensor, steps: int = 100, ext_force=None):
        geometry = geometry_mask.to(self.device, non_blocking=True)
        pressure_coupled = float(self.thermal_config.pressure_coupling_strength) != 0.0
        thermal_force = self.compute_thermal_pressure_force()
        if not pressure_coupled:
            coupled_force = ext_force
        elif ext_force is None:
            coupled_force = thermal_force
        else:
            coupled_force = ext_force.to(self.device, dtype=thermal_force.dtype, non_blocking=True) + thermal_force

        result = self.flow_solver.collide_stream(geometry, steps=steps, ext_force=coupled_force)
        self._sync_flow_fields()

        requested_steps = max(0, int(steps))
        thermal_steps = min(requested_steps, max(1, int(self.thermal_config.max_thermal_steps_per_call)))
        if thermal_steps > 0:
            self.thermal_solver.collide_stream(self._thermal_velocity(), geometry, steps=thermal_steps)
        self._refresh_thermodynamic_state()
        return result

    def compute_aerodynamic_coefficients(self, geometry_mask: torch.Tensor) -> dict[str, Any]:
        self._sync_flow_fields()
        state = self._refresh_thermodynamic_state()
        geometry = geometry_mask.to(self.device, non_blocking=True)
        results = dict(self.flow_solver.compute_aerodynamic_coefficients(geometry))

        temperature = state.temperature
        pressure = state.pressure
        density = state.density
        flow_isothermal_pressure_sum = results.get("pressure_sum")
        shock_sensor = self.thermal_solver.shock_sensor
        shock_mask = self.thermal_solver.shock_stabilization_mask
        thermal_force = self.thermal_pressure_gradient_force
        thermal_stats = torch.stack(
            [
                temperature.min(),
                temperature.max(),
                temperature.mean(),
                density.min(),
                density.max(),
                density.mean(),
                pressure.sum(),
                pressure.min(),
                pressure.max(),
                pressure.mean(),
                shock_sensor.max(),
                shock_mask.to(dtype=temperature.dtype).sum(),
                torch.linalg.vector_norm(thermal_force),
                torch.abs(thermal_force).max(),
            ]
        ).detach().cpu().tolist()
        (
            temperature_min,
            temperature_max,
            temperature_mean,
            density_min,
            density_max,
            density_mean,
            thermodynamic_pressure_sum,
            thermodynamic_pressure_min,
            thermodynamic_pressure_max,
            thermodynamic_pressure_mean,
            shock_sensor_max,
            shock_cell_count,
            thermal_pressure_gradient_force_norm,
            thermal_pressure_gradient_force_max,
        ) = (float(value) for value in thermal_stats)
        mach_number = float(getattr(self.config, "mach_number", 0.0))
        mach_magnitude = abs(mach_number)
        if mach_magnitude >= 1.0:
            inlet_outlet_regime = "supersonic_experimental"
        elif mach_magnitude > 0.3:
            inlet_outlet_regime = "high_mach_experimental"
        else:
            inlet_outlet_regime = "low_mach_staged"
        pressure_coupled = float(self.thermal_config.pressure_coupling_strength) != 0.0
        shock_stabilization_enabled = bool(self.thermal_config.shock_stabilization_enabled)

        results.update(
            {
                "compressibility_model": THERMAL_COMPRESSIBILITY_MODEL,
                "thermal_model": THERMAL_MODEL_NAME,
                "pressure_model": THERMAL_PRESSURE_MODEL,
                "validity_regime": THERMAL_VALIDITY_REGIME,
                "claim_grade": "no_claim_experimental",
                "high_mach_warning": (
                    f"Mach {abs(mach_number):.3g} is using the staged thermal LBM attachment. "
                    "The thermal field and ideal-gas pressure are diagnostic; shock-capable compressible "
                    "boundaries and force validation are not complete."
                ),
                "training_drag_source": THERMAL_TRAINING_SOURCE,
                "pinn_ready": False,
                "shock_capable": False,
                "thermodynamic_solver": THERMODYNAMIC_SOLVER_NAME,
                "thermal_force_coupling": (
                    "pressure_gradient_guo_forcing_experimental"
                    if pressure_coupled
                    else "diagnostic_pressure_not_force_coupled"
                ),
                "thermal_boundary_model": str(self.thermal_config.thermal_boundary_model),
                "compressible_boundary_status": "staged_thermal_boundary_not_characteristic_validated",
                "inlet_outlet_regime": inlet_outlet_regime,
                "shock_capture_model": "sensor_artificial_thermal_diffusion_not_flow_shock_capture",
                "shock_stabilization_model": (
                    "local_artificial_thermal_diffusivity"
                    if shock_stabilization_enabled
                    else "disabled"
                ),
                "shock_sensor_max": shock_sensor_max,
                "shock_cell_count": shock_cell_count,
                "thermal_pressure_gradient_force_norm": thermal_pressure_gradient_force_norm,
                "thermal_pressure_gradient_force_max": thermal_pressure_gradient_force_max,
                "thermal_lattice_tau": float(self.thermal_solver.tau),
                "thermal_lattice_omega": float(self.thermal_solver.omega),
                "thermal_diffusivity_lattice": float(self.thermal_config.thermal_diffusivity_lattice),
                "gas_constant": float(self.thermal_config.gas_constant),
                "temperature_min": temperature_min,
                "temperature_max": temperature_max,
                "temperature_mean": temperature_mean,
                "density_min": density_min,
                "density_max": density_max,
                "density_mean": density_mean,
                "flow_isothermal_pressure_sum": flow_isothermal_pressure_sum,
                "thermodynamic_pressure_sum": thermodynamic_pressure_sum,
                "thermodynamic_pressure_min": thermodynamic_pressure_min,
                "thermodynamic_pressure_max": thermodynamic_pressure_max,
                "thermodynamic_pressure_mean": thermodynamic_pressure_mean,
                "thermal_solver_device": str(self.device),
            }
        )
        return results


def _thermal_enabled(config: Any) -> bool:
    if bool(getattr(config, "thermal_enabled", False)):
        return True
    if bool(getattr(config, "enable_thermal_lbm", False)):
        return True
    if bool(getattr(config, "use_thermal_lbm", False)):
        return True

    solver_type = str(getattr(config, "solver_type", "")).lower()
    if solver_type in {"thermal_lbm", "d3q27_thermal", "thermal_d3q27"}:
        return True

    thermal_model = str(getattr(config, "thermal_model", "")).lower()
    return thermal_model in {THERMAL_MODEL_NAME, "thermal_lbm", "d3q7_temperature_bgk"}


def create_thermal_lbm_solver(config: Any, device: torch.device | str, phys_config: Any):
    """Create either the raw D3Q27 solver or the staged thermal wrapper."""

    if _thermal_enabled(config):
        return ThermalD3Q27Solver(config, device, phys_config)
    return D3Q27CascadedSolver(config, torch.device(device), phys_config)
