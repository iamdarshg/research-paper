from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch

from thermal_lbm_solver import ThermalLBMConfig, create_thermal_lbm_solver


DEFAULT_RAW_PROFILE_FIELDS = (
    "case",
    "step",
    "x_index",
    "density_mean",
    "density_min",
    "density_max",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "pressure_mean",
    "pressure_min",
    "pressure_max",
    "flow_pressure_lu_mean",
    "flow_pressure_lu_min",
    "flow_pressure_lu_max",
    "ux_lattice_mean",
    "ux_lattice_min",
    "ux_lattice_max",
    "uy_lattice_mean",
    "uz_lattice_mean",
    "shock_sensor_mean",
    "shock_sensor_max",
    "thermal_force_x_mean",
    "thermal_force_x_min",
    "thermal_force_x_max",
)


class DiagnosticPhysicsConfig:
    max_mach = 2.0
    target_lattice_velocity = 0.2
    tau_min_d3q27 = 0.52
    s_e_d3q27 = 1.2
    s_h_d3q27 = 1.6
    drag_link_metric_exponent = None
    use_triton_streaming = False
    convergence_tolerance = 1e-7
    check_convergence_every = 1000
    smagorinsky_constant = 0.17
    q_threshold = 0.0
    use_shape_drag_correction = False


def _as_shape(shape: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(shape, int):
        return (int(shape), int(shape), int(shape))
    values = tuple(int(v) for v in shape)
    if len(values) != 3:
        raise ValueError(f"Expected a 3D shape, got {values!r}")
    if min(values) <= 1:
        raise ValueError(f"Shape dimensions must be greater than one, got {values!r}")
    return values


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _tensor_stats_1d(field: torch.Tensor, x_index: int) -> dict[str, float]:
    plane = field[x_index].detach().float().cpu()
    return {
        "mean": _safe_float(plane.mean().item()),
        "min": _safe_float(plane.min().item()),
        "max": _safe_float(plane.max().item()),
    }


def build_shock_tube_initial_fields(
    *,
    shape: int | Iterable[int],
    left_density: float,
    right_density: float,
    left_pressure: float,
    right_pressure: float,
    gas_constant: float,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Build a raw left/right Riemann state with pressure = rho * R * T."""

    shape_tuple = _as_shape(shape)
    device = torch.device(device)
    split = shape_tuple[0] // 2
    density = torch.empty(shape_tuple, dtype=dtype, device=device)
    pressure = torch.empty_like(density)
    density[:split, :, :] = float(left_density)
    density[split:, :, :] = float(right_density)
    pressure[:split, :, :] = float(left_pressure)
    pressure[split:, :, :] = float(right_pressure)

    gas_constant = max(float(gas_constant), 1e-30)
    temperature = pressure / density.clamp_min(1e-30) / gas_constant
    zero = torch.zeros_like(density)
    return {
        "density": density,
        "temperature": temperature,
        "pressure": pressure,
        "ux_lattice": zero.clone(),
        "uy_lattice": zero.clone(),
        "uz_lattice": zero.clone(),
    }


def line_profile(fields: dict[str, torch.Tensor], *, case_name: str, step: int | None = None) -> list[dict[str, Any]]:
    """Return x-line means/min/max of raw fields averaged across y-z planes."""

    density = fields["density"]
    rows: list[dict[str, Any]] = []
    for x_index in range(int(density.shape[0])):
        density_stats = _tensor_stats_1d(fields["density"], x_index)
        temperature_stats = _tensor_stats_1d(fields["temperature"], x_index)
        pressure_stats = _tensor_stats_1d(fields["pressure"], x_index)
        flow_pressure_stats = _tensor_stats_1d(fields.get("flow_pressure_lu", torch.full_like(density, math.nan)), x_index)
        ux_stats = _tensor_stats_1d(fields.get("ux_lattice", torch.full_like(density, math.nan)), x_index)
        uy_stats = _tensor_stats_1d(fields.get("uy_lattice", torch.full_like(density, math.nan)), x_index)
        uz_stats = _tensor_stats_1d(fields.get("uz_lattice", torch.full_like(density, math.nan)), x_index)
        shock_stats = _tensor_stats_1d(fields.get("shock_sensor", torch.full_like(density, math.nan)), x_index)
        force_x_stats = _tensor_stats_1d(fields.get("thermal_force_x", torch.full_like(density, math.nan)), x_index)
        rows.append(
            {
                "case": case_name,
                "step": step if step is not None else "",
                "x_index": x_index,
                "density_mean": density_stats["mean"],
                "density_min": density_stats["min"],
                "density_max": density_stats["max"],
                "temperature_mean": temperature_stats["mean"],
                "temperature_min": temperature_stats["min"],
                "temperature_max": temperature_stats["max"],
                "pressure_mean": pressure_stats["mean"],
                "pressure_min": pressure_stats["min"],
                "pressure_max": pressure_stats["max"],
                "flow_pressure_lu_mean": flow_pressure_stats["mean"],
                "flow_pressure_lu_min": flow_pressure_stats["min"],
                "flow_pressure_lu_max": flow_pressure_stats["max"],
                "ux_lattice_mean": ux_stats["mean"],
                "ux_lattice_min": ux_stats["min"],
                "ux_lattice_max": ux_stats["max"],
                "uy_lattice_mean": uy_stats["mean"],
                "uz_lattice_mean": uz_stats["mean"],
                "shock_sensor_mean": shock_stats["mean"],
                "shock_sensor_max": shock_stats["max"],
                "thermal_force_x_mean": force_x_stats["mean"],
                "thermal_force_x_min": force_x_stats["min"],
                "thermal_force_x_max": force_x_stats["max"],
            }
        )
    return rows


def summarize_profile(rows: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for field in ("pressure", "density", "temperature", "ux_lattice", "shock_sensor", "thermal_force_x"):
        values = [_safe_float(row.get(f"{field}_mean")) for row in rows]
        values = [value for value in values if math.isfinite(value)]
        if not values:
            continue
        summary[f"{field}_min"] = min(values)
        summary[f"{field}_max"] = max(values)
        summary[f"{field}_mean"] = sum(values) / len(values)
        denom = max(abs(summary[f"{field}_min"]), 1e-30)
        summary[f"{field}_max_to_min_ratio"] = summary[f"{field}_max"] / denom
    return summary


def summarize_instability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ux_values = []
    for row in rows:
        ux_values.extend(
            [
                _safe_float(row.get("ux_lattice_mean")),
                _safe_float(row.get("ux_lattice_min")),
                _safe_float(row.get("ux_lattice_max")),
            ]
        )
    finite_ux = [abs(value) for value in ux_values if math.isfinite(value)]
    density_mins = [_safe_float(row.get("density_min")) for row in rows]
    pressure_mins = [_safe_float(row.get("pressure_min")) for row in rows]
    max_abs_ux = max(finite_ux) if finite_ux else math.nan
    min_density = min((value for value in density_mins if math.isfinite(value)), default=math.nan)
    min_pressure = min((value for value in pressure_mins if math.isfinite(value)), default=math.nan)

    flags = []
    if math.isfinite(max_abs_ux) and max_abs_ux > 0.3:
        flags.append("lattice_velocity_exceeds_low_mach_envelope")
    if math.isfinite(max_abs_ux) and max_abs_ux > 1.0:
        flags.append("lattice_velocity_exceeds_lattice_sound_scale")
    if math.isfinite(min_density) and min_density < 1e-4:
        flags.append("density_near_positivity_floor")
    if math.isfinite(min_pressure) and min_pressure < 1e-6:
        flags.append("pressure_near_positivity_floor")

    return {
        "max_abs_ux_lattice": max_abs_ux,
        "min_density": min_density,
        "min_pressure": min_pressure,
        "flags": flags,
    }


def write_profile_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEFAULT_RAW_PROFILE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in DEFAULT_RAW_PROFILE_FIELDS})


def make_solver_config(
    *,
    resolution: int,
    mach_number: float,
    reynolds_number: float,
    thermal_config: ThermalLBMConfig,
) -> SimpleNamespace:
    lbm_config = SimpleNamespace(
        grid_spacing=1.0 / max(int(resolution), 1),
        time_step=0.001,
        physical_length_scale=1.0,
    )
    return SimpleNamespace(
        base_grid_resolution=int(resolution),
        resolution=int(resolution),
        mach_number=float(mach_number),
        reynolds_number=float(reynolds_number),
        simulation_steps=1,
        lbm_config=lbm_config,
        thermal_enabled=True,
        thermal_model="coupled_d3q7_temperature_bgk",
        thermal_lbm_config=thermal_config,
    )


def initialize_solver_from_fields(solver: Any, fields: dict[str, torch.Tensor]) -> None:
    flow_solver = solver.flow_solver if hasattr(solver, "flow_solver") else solver
    core = flow_solver._solver
    density = fields["density"].to(flow_solver.device, dtype=core.f.dtype)
    ux = fields["ux_lattice"].to(flow_solver.device, dtype=core.f.dtype)
    uy = fields["uy_lattice"].to(flow_solver.device, dtype=core.f.dtype)
    uz = fields["uz_lattice"].to(flow_solver.device, dtype=core.f.dtype)
    with torch.no_grad():
        feq = core.compute_equilibrium(density, ux, uy, uz)
        core.f.copy_(feq)
        core.f_temp.copy_(feq)
        core.velocity_x.copy_(ux)
        core.velocity_y.copy_(uy)
        core.velocity_z.copy_(uz)
        core.rho.copy_(density)
        core.pressure.copy_(density / 3.0)
        flow_solver.velocity_x = core.velocity_x
        flow_solver.velocity_y = core.velocity_y
        flow_solver.velocity_z = core.velocity_z
        flow_solver.rho = core.rho
        flow_solver.pressure = core.pressure
        flow_solver.f = core.f
        flow_solver.f_temp = core.f_temp

        if hasattr(solver, "thermal_solver"):
            velocity = (ux, uy, uz)
            solver.thermal_solver.set_temperature(fields["temperature"], velocity)
            solver._sync_flow_fields()
            solver._refresh_thermodynamic_state()


def collect_raw_fields(solver: Any) -> dict[str, torch.Tensor]:
    if hasattr(solver, "_sync_flow_fields"):
        solver._sync_flow_fields()
    if hasattr(solver, "_refresh_thermodynamic_state"):
        state = solver._refresh_thermodynamic_state()
        if hasattr(solver, "compute_thermal_pressure_force"):
            solver.compute_thermal_pressure_force()
        pressure = state.pressure
        temperature = state.temperature
        shock_sensor = solver.thermal_solver.shock_sensor
        thermal_force_x = solver.thermal_pressure_gradient_force[0]
    else:
        pressure = solver.pressure
        temperature = torch.full_like(solver.rho, math.nan)
        shock_sensor = torch.full_like(solver.rho, math.nan)
        thermal_force_x = torch.full_like(solver.rho, math.nan)

    return {
        "density": solver.rho,
        "temperature": temperature,
        "pressure": pressure,
        "flow_pressure_lu": solver.pressure,
        "ux_lattice": solver.velocity_x,
        "uy_lattice": solver.velocity_y,
        "uz_lattice": solver.velocity_z,
        "shock_sensor": shock_sensor,
        "thermal_force_x": thermal_force_x,
    }


def create_cylinder_mask(resolution: int, device: torch.device, *, radius_fraction: float = 0.12) -> torch.Tensor:
    coords = torch.arange(resolution, dtype=torch.float32, device=device)
    x = coords.view(resolution, 1, 1)
    y = coords.view(1, resolution, 1)
    cx = 0.36 * float(resolution - 1)
    cy = 0.50 * float(resolution - 1)
    radius = max(2.0, float(resolution) * float(radius_fraction))
    cylinder = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2
    return cylinder.expand(resolution, resolution, resolution).clone().to(dtype=torch.float32)


def plot_shock_tube(path: Path, final_rows: list[dict[str, Any]], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([row["x_index"] for row in final_rows], dtype=float)
    fields = [
        ("density_mean", "Density"),
        ("pressure_mean", "Thermal pressure p=rhoRT"),
        ("temperature_mean", "Temperature"),
        ("ux_lattice_mean", "Ux lattice"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (field, label) in zip(axes.flat, fields):
        y = np.array([_safe_float(row.get(field)) for row in final_rows], dtype=float)
        axis.plot(x, y, linewidth=1.8)
        axis.set_xlabel("x index")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cylinder_pressure(path: Path, fields: dict[str, torch.Tensor], geometry: torch.Tensor, *, mach: float, actual_u: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid_z = int(fields["pressure"].shape[2] // 2)
    therm_p = fields["pressure"][:, :, mid_z].detach().float().cpu().numpy().T
    flow_p = fields["flow_pressure_lu"][:, :, mid_z].detach().float().cpu().numpy().T
    geom = geometry[:, :, mid_z].detach().float().cpu().numpy().T

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    panels = [
        (flow_p, "Raw flow pressure rho/3 [lattice]"),
        (therm_p, "Raw thermal pressure rhoRT"),
    ]
    for axis, (image, label) in zip(axes, panels):
        masked = np.ma.array(image, mask=geom > 0.5)
        im = axis.imshow(masked, origin="lower", cmap="viridis", interpolation="nearest")
        axis.contour(geom, levels=[0.5], colors="white", linewidths=0.8)
        axis.set_title(label)
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
        fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(f"Cylinder raw pressure diagnostic, requested Mach {mach:g}, actual u_lattice {actual_u:.6g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_shock_tube(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thermal_config = ThermalLBMConfig(
        reference_temperature=1.0,
        gas_constant=1.0,
        thermal_diffusivity_lattice=float(args.thermal_diffusivity),
        min_temperature=1e-6,
        max_temperature=10.0,
        min_density=1e-8,
        min_pressure=1e-9,
        max_thermal_steps_per_call=int(args.thermal_steps_per_call),
        shock_stabilization_enabled=not args.disable_shock_stabilization,
        shock_sensor_threshold=float(args.shock_sensor_threshold),
        shock_diffusivity_multiplier=float(args.shock_diffusivity_multiplier),
        thermal_boundary_model=str(args.thermal_boundary_model),
        pressure_coupling_strength=float(args.pressure_coupling_strength),
        pressure_gradient_clip=float(args.pressure_gradient_clip),
    )
    config = make_solver_config(
        resolution=int(args.resolution),
        mach_number=0.0,
        reynolds_number=float(args.reynolds_number),
        thermal_config=thermal_config,
    )
    solver = create_thermal_lbm_solver(config, device, DiagnosticPhysicsConfig)
    fields0 = build_shock_tube_initial_fields(
        shape=int(args.resolution),
        left_density=float(args.left_density),
        right_density=float(args.right_density),
        left_pressure=float(args.left_pressure),
        right_pressure=float(args.right_pressure),
        gas_constant=float(thermal_config.gas_constant),
        device=device,
    )
    initialize_solver_from_fields(solver, fields0)

    geometry = torch.zeros((args.resolution, args.resolution, args.resolution), dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    case_name = "sod_raw_pressure_coupled" if args.pressure_coupling_strength else "sod_raw_uncoupled"
    rows.extend(line_profile(collect_raw_fields(solver), case_name=case_name, step=0))
    final_step = 0
    started = time.perf_counter()
    for final_step in range(int(args.sample_every), int(args.steps) + 1, int(args.sample_every)):
        solver.collide_stream(geometry, steps=int(args.sample_every))
        rows.extend(line_profile(collect_raw_fields(solver), case_name=case_name, step=final_step))
    if final_step < int(args.steps):
        remaining = int(args.steps) - final_step
        solver.collide_stream(geometry, steps=remaining)
        final_step = int(args.steps)
        rows.extend(line_profile(collect_raw_fields(solver), case_name=case_name, step=final_step))
    elapsed = time.perf_counter() - started

    profile_csv = output_dir / "shock_tube_raw_profiles.csv"
    write_profile_csv(profile_csv, rows)
    final_rows = [row for row in rows if row["step"] == final_step]
    summary = {
        "case": case_name,
        "resolution": int(args.resolution),
        "steps": int(args.steps),
        "sample_every": int(args.sample_every),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "left_state": {
            "density": float(args.left_density),
            "pressure": float(args.left_pressure),
            "temperature": float(args.left_pressure) / max(float(args.left_density), 1e-30),
        },
        "right_state": {
            "density": float(args.right_density),
            "pressure": float(args.right_pressure),
            "temperature": float(args.right_pressure) / max(float(args.right_density), 1e-30),
        },
        "raw_profile_summary_final": summarize_profile(final_rows),
        "raw_instability_summary_final": summarize_instability(final_rows),
        "csv": str(profile_csv),
        "npz": str(output_dir / "shock_tube_final_raw_fields.npz"),
        "plot": str(output_dir / "shock_tube_raw_profiles.png"),
        "normalization_status": "raw_fields_no_profile_normalization",
        "claim_grade": "no_claim_experimental",
        "validity_regime": "experimental_thermal_lbm_unvalidated",
        "shock_capable": False,
        "notes": [
            "This is a raw internal staged thermal-LBM diagnostic, not a validated Sod solver.",
            "The D3Q27 flow populations remain isothermal; thermal pressure coupling is experimental.",
            "x coordinate is an integer lattice index, not a normalized physical coordinate.",
        ],
    }
    raw = collect_raw_fields(solver)
    np.savez_compressed(
        output_dir / "shock_tube_final_raw_fields.npz",
        density=raw["density"].detach().float().cpu().numpy(),
        temperature=raw["temperature"].detach().float().cpu().numpy(),
        thermal_pressure=raw["pressure"].detach().float().cpu().numpy(),
        flow_pressure_lu=raw["flow_pressure_lu"].detach().float().cpu().numpy(),
        ux_lattice=raw["ux_lattice"].detach().float().cpu().numpy(),
        shock_sensor=raw["shock_sensor"].detach().float().cpu().numpy(),
        thermal_force_x=raw["thermal_force_x"].detach().float().cpu().numpy(),
    )
    plot_shock_tube(Path(summary["plot"]), final_rows, title=f"Raw shock-tube profile at step {final_step}")
    (output_dir / "shock_tube_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_cylinder_pressure(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [float(item) for item in str(args.mach).split(",")]
    summaries: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for mach in cases:
        thermal_config = ThermalLBMConfig(
            reference_temperature=300.0,
            gas_constant=287.05,
            thermal_diffusivity_lattice=float(args.thermal_diffusivity),
            max_thermal_steps_per_call=int(args.thermal_steps_per_call),
            inlet_temperature=300.0,
            outlet_temperature=None,
            pressure_coupling_strength=float(args.pressure_coupling_strength),
            pressure_gradient_clip=float(args.pressure_gradient_clip),
        )
        config = make_solver_config(
            resolution=int(args.resolution),
            mach_number=mach,
            reynolds_number=float(args.reynolds_number),
            thermal_config=thermal_config,
        )
        solver = create_thermal_lbm_solver(config, device, DiagnosticPhysicsConfig)
        geometry = create_cylinder_mask(int(args.resolution), device, radius_fraction=float(args.radius_fraction))
        started = time.perf_counter()
        solver.collide_stream(geometry, steps=int(args.steps))
        elapsed = time.perf_counter() - started
        fields = collect_raw_fields(solver)
        coeffs = solver.compute_aerodynamic_coefficients(geometry)
        safe_mach = str(mach).replace(".", "p").replace("-", "m")
        image_path = output_dir / f"cylinder_mach_{safe_mach}_raw_pressure_res{args.resolution}.png"
        plot_cylinder_pressure(image_path, fields, geometry, mach=mach, actual_u=float(solver.inlet_velocity_lu))
        summary = {
            "mach_requested": mach,
            "resolution": int(args.resolution),
            "steps": int(args.steps),
            "device": str(device),
            "elapsed_seconds": elapsed,
            "actual_u_lattice": float(solver.inlet_velocity_lu),
            "flow_pressure_lu_min": _safe_float(fields["flow_pressure_lu"].min().item()),
            "flow_pressure_lu_max": _safe_float(fields["flow_pressure_lu"].max().item()),
            "thermodynamic_pressure_min": _safe_float(fields["pressure"].min().item()),
            "thermodynamic_pressure_max": _safe_float(fields["pressure"].max().item()),
            "density_min": _safe_float(fields["density"].min().item()),
            "density_max": _safe_float(fields["density"].max().item()),
            "temperature_min": _safe_float(fields["temperature"].min().item()),
            "temperature_max": _safe_float(fields["temperature"].max().item()),
            "shock_sensor_max": _safe_float(fields["shock_sensor"].max().item()),
            "thermal_force_x_min": _safe_float(fields["thermal_force_x"].min().item()),
            "thermal_force_x_max": _safe_float(fields["thermal_force_x"].max().item()),
            "validity_regime": coeffs.get("validity_regime"),
            "claim_grade": coeffs.get("claim_grade"),
            "shock_capable": coeffs.get("shock_capable"),
            "thermal_force_coupling": coeffs.get("thermal_force_coupling"),
            "image": str(image_path),
        }
        summaries.append(summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "resolution": int(args.resolution),
        "steps": int(args.steps),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started_all,
        "normalization_status": "raw_pressure_images_no_field_normalization",
        "cases": summaries,
    }
    (output_dir / "cylinder_pressure_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw staged LBM diagnostics for shock tubes and cylinder pressure fields.")
    parser.add_argument("--output-dir", default="build/solver_diagnostics/raw_solver_outputs_20260612")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    shock = subparsers.add_parser("shock-tube", help="Run a raw Sod-style shock-tube diagnostic.")
    shock.add_argument("--resolution", type=int, default=96)
    shock.add_argument("--steps", type=int, default=40)
    shock.add_argument("--sample-every", type=int, default=10)
    shock.add_argument("--left-density", type=float, default=1.0)
    shock.add_argument("--right-density", type=float, default=0.125)
    shock.add_argument("--left-pressure", type=float, default=1.0)
    shock.add_argument("--right-pressure", type=float, default=0.1)
    shock.add_argument("--reynolds-number", type=float, default=1000.0)
    shock.add_argument("--thermal-diffusivity", type=float, default=0.005)
    shock.add_argument("--thermal-steps-per-call", type=int, default=8)
    shock.add_argument("--shock-sensor-threshold", type=float, default=0.02)
    shock.add_argument("--shock-diffusivity-multiplier", type=float, default=3.0)
    shock.add_argument("--disable-shock-stabilization", action="store_true")
    shock.add_argument("--thermal-boundary-model", default="fixed_temperature_inlet_zero_gradient_outlet")
    shock.add_argument("--pressure-coupling-strength", type=float, default=0.1)
    shock.add_argument("--pressure-gradient-clip", type=float, default=0.02)
    shock.set_defaults(func=run_shock_tube)

    cylinder = subparsers.add_parser("cylinder-pressure", help="Render high-resolution raw cylinder pressure images.")
    cylinder.add_argument("--resolution", type=int, default=128)
    cylinder.add_argument("--steps", type=int, default=40)
    cylinder.add_argument("--mach", default="0.3,2.0")
    cylinder.add_argument("--reynolds-number", type=float, default=1000.0)
    cylinder.add_argument("--radius-fraction", type=float, default=0.12)
    cylinder.add_argument("--thermal-diffusivity", type=float, default=0.01)
    cylinder.add_argument("--thermal-steps-per-call", type=int, default=8)
    cylinder.add_argument("--pressure-coupling-strength", type=float, default=0.05)
    cylinder.add_argument("--pressure-gradient-clip", type=float, default=0.02)
    cylinder.set_defaults(func=run_cylinder_pressure)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = args.func(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
