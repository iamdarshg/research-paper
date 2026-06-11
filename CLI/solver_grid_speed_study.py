"""Run OpenFOAM/LBM grid-speed comparisons and plot error surfaces.

This is an evidence driver, not a calibration path. It preserves raw solver
outputs, timing, convergence metadata, and validity caveats for each case.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CLI_DIR = REPO / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import run_internal_benchmark as rib
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig
from advanced_lbm_solver import D3Q27CascadedSolver
from lbm_utils import classify_lbm_regime
from CLI.openfoam_mach_sweep import SPEED_OF_SOUND, parse_float_list, run_case as run_openfoam_case


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def run_lbm_case(
    stl: Path,
    mesh,
    *,
    grid: int,
    mach: float,
    reynolds: float,
    steps: int,
    domain_scale: float,
    device_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    domain_min, domain_max, domain_size, max_extent = rib.compute_geometry_frame(mesh, domain_scale)
    geometry_started = time.perf_counter()
    geometry_mask = rib.mesh_to_geometry_mask(mesh, grid, domain_min, domain_size)
    geometry_seconds = time.perf_counter() - geometry_started

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    cfg = CFDConfig(
        base_grid_resolution=grid,
        mach_number=mach,
        reynolds_number=reynolds,
        simulation_steps=steps,
    )
    cfg.lbm_config.physical_length_scale = domain_size
    cfg.lbm_config.grid_spacing = domain_size / grid

    setup_started = time.perf_counter()
    solver = D3Q27CascadedSolver(cfg, device, LBMPhysicsConfig)
    geometry_mask = geometry_mask.to(device, non_blocking=True)
    sync_device(device)
    setup_seconds = time.perf_counter() - setup_started

    solve_started = time.perf_counter()
    solver.collide_stream(geometry_mask, steps=steps)
    sync_device(device)
    solve_seconds = time.perf_counter() - solve_started

    coeff_started = time.perf_counter()
    coeffs = solver.compute_aerodynamic_coefficients(geometry_mask)
    sync_device(device)
    coeff_seconds = time.perf_counter() - coeff_started

    return {
        "solver": "internal_lbm_d3q27",
        "stl": str(stl),
        "grid": grid,
        "mach": mach,
        "speed_mps": mach * SPEED_OF_SOUND,
        "reynolds": reynolds,
        "steps": steps,
        "domain_scale": domain_scale,
        "domain_size": float(domain_size),
        "max_extent": float(max_extent),
        "device": str(device),
        "cd": float(coeffs.get("drag_coefficient", float("nan"))),
        "cl": float(coeffs.get("lift_coefficient", float("nan"))),
        "lbm_converged": bool(coeffs.get("lbm_converged", False)),
        "training_drag_source": coeffs.get("training_drag_source"),
        "force_stability": coeffs.get("force_stability"),
        "validity": coeffs.get("validity_regime", classify_lbm_regime(mach)["validity_regime"]),
        "validity_regime": coeffs.get("validity_regime"),
        "claim_grade": coeffs.get("claim_grade"),
        "high_mach_warning": coeffs.get("high_mach_warning"),
        "u_lattice": coeffs.get("u_lattice"),
        "lattice_mach": coeffs.get("lattice_mach"),
        "sound_speed_model": coeffs.get("sound_speed_model"),
        "compressibility_model": coeffs.get("compressibility_model"),
        "thermal_model": coeffs.get("thermal_model"),
        "coefficients": coeffs,
        "timings": {
            "geometry_seconds": geometry_seconds,
            "setup_seconds": setup_seconds,
            "solve_seconds": solve_seconds,
            "coefficients_seconds": coeff_seconds,
            "total_seconds": time.perf_counter() - started,
        },
    }


def summarize_openfoam(result: dict[str, Any], grid: int) -> dict[str, Any]:
    force = result.get("force") or {}
    timings = result.get("commands") or {}
    total_seconds = sum(
        float(value.get("seconds", 0.0))
        for value in timings.values()
        if isinstance(value, dict)
    )
    return {
        "solver": result.get("solver"),
        "grid": grid,
        "mach": result.get("mach"),
        "speed_mps": result.get("speed_mps"),
        "cd": force.get("cd_total"),
        "cl": force.get("cl_total"),
        "mesh_ok": result.get("mesh_ok"),
        "failed": result.get("failed"),
        "latest_time": result.get("latest_time"),
        "case_dir": result.get("case_dir"),
        "timings": {"total_seconds": total_seconds, "commands": timings},
        "raw": result,
    }


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def compare_cases(openfoam_cases: list[dict[str, Any]], lbm_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for of_case in openfoam_cases:
        for lbm_case in lbm_cases:
            if abs(float(of_case["mach"]) - float(lbm_case["mach"])) > 1e-12:
                continue
            of_cd = of_case.get("cd")
            lbm_cd = lbm_case.get("cd")
            of_cl = of_case.get("cl")
            lbm_cl = lbm_case.get("cl")
            cd_error = None
            cl_error = None
            if finite(of_cd) and finite(lbm_cd) and abs(float(of_cd)) > 1e-12:
                cd_error = abs(float(lbm_cd) - float(of_cd)) / abs(float(of_cd)) * 100.0
            if finite(of_cl) and finite(lbm_cl) and abs(float(of_cl)) > 1e-12:
                cl_error = abs(float(lbm_cl) - float(of_cl)) / abs(float(of_cl)) * 100.0
            rows.append(
                {
                    "mach": of_case["mach"],
                    "speed_mps": of_case["speed_mps"],
                    "openfoam_grid": of_case["grid"],
                    "lbm_grid": lbm_case["grid"],
                    "openfoam_solver": of_case["solver"],
                    "openfoam_cd": of_cd,
                    "openfoam_cl": of_cl,
                    "openfoam_seconds": of_case["timings"]["total_seconds"],
                    "lbm_cd": lbm_cd,
                    "lbm_cl": lbm_cl,
                    "lbm_seconds": lbm_case["timings"]["total_seconds"],
                    "lbm_converged": lbm_case.get("lbm_converged"),
                    "lbm_validity": lbm_case.get("validity"),
                    "lbm_claim_grade": lbm_case.get("claim_grade"),
                    "lbm_compressibility_model": lbm_case.get("compressibility_model"),
                    "lbm_thermal_model": lbm_case.get("thermal_model"),
                    "cd_error_percent": cd_error,
                    "cl_error_percent": cl_error,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mach",
        "speed_mps",
        "openfoam_grid",
        "lbm_grid",
        "openfoam_solver",
        "openfoam_cd",
        "openfoam_cl",
        "openfoam_seconds",
        "lbm_cd",
        "lbm_cl",
        "lbm_seconds",
        "lbm_converged",
        "lbm_validity",
        "lbm_claim_grade",
        "lbm_compressibility_model",
        "lbm_thermal_model",
        "cd_error_percent",
        "cl_error_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [row for row in rows if finite(row.get("cd_error_percent"))]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    if plot_rows:
        x = np.array([float(row["mach"]) for row in plot_rows])
        y = np.array([float(row["openfoam_grid"]) for row in plot_rows])
        z = np.array([float(row["cd_error_percent"]) for row in plot_rows])
        colors = np.array([0.0 if row.get("lbm_validity") == "validated_low_mach_envelope" else 1.0 for row in plot_rows])
        scatter = ax.scatter(x, y, z, c=colors, cmap="coolwarm", s=48)
        legend = ax.legend(*scatter.legend_elements(), title="LBM validity", loc="upper left")
        ax.add_artist(legend)
    ax.set_xlabel("Mach")
    ax.set_ylabel("OpenFOAM grid resolution")
    ax.set_zlabel("Cd error (%)")
    ax.set_title("OpenFOAM vs internal LBM Cd error")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/solver_diagnostics/grid_speed_study"))
    parser.add_argument("--mach-values", default="0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0")
    parser.add_argument("--openfoam-grids", default="128")
    parser.add_argument("--lbm-grids", default="192")
    parser.add_argument("--reference-area", type=float, default=0.27734375)
    parser.add_argument("--reference-length", type=float, default=1.0)
    parser.add_argument("--domain-scale", type=float, default=2.0)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--lbm-steps", type=int, default=500)
    parser.add_argument("--simple-iterations", type=int, default=3000)
    parser.add_argument("--compressible-body-transits", type=float, default=3.0)
    parser.add_argument("--simple-threshold", type=float, default=0.3)
    parser.add_argument("--max-co", type=float, default=0.25)
    parser.add_argument("--max-delta-t", type=float, default=1e-4)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--skip-openfoam", action="store_true")
    parser.add_argument("--skip-lbm", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stl = Path(args.stl)
    mesh = trimesh.load_mesh(stl)
    domain_min, domain_max, domain_size, _ = rib.compute_geometry_frame(mesh, args.domain_scale)
    mach_values = parse_float_list(args.mach_values)
    openfoam_grids = parse_int_list(args.openfoam_grids)
    lbm_grids = parse_int_list(args.lbm_grids)

    openfoam_cases: list[dict[str, Any]] = []
    lbm_cases: list[dict[str, Any]] = []

    if not args.skip_openfoam:
        for grid in openfoam_grids:
            for mach in mach_values:
                of_args = argparse.Namespace(**vars(args))
                of_args.grid = grid
                of_args.refinement_level = "0 0"
                of_args.snap = False
                of_args.output_dir = args.output_dir / "openfoam_cases" / f"grid_{grid}"
                of_args.stl = str(stl)
                result = run_openfoam_case(of_args, mach, mesh, domain_min, domain_max, domain_size, args.reference_area)
                summary = summarize_openfoam(result, grid)
                openfoam_cases.append(summary)
                (args.output_dir / "openfoam_cases.json").write_text(json.dumps(openfoam_cases, indent=2), encoding="utf-8")

    if not args.skip_lbm:
        for grid in lbm_grids:
            for mach in mach_values:
                result = run_lbm_case(
                    stl,
                    mesh,
                    grid=grid,
                    mach=mach,
                    reynolds=args.reynolds,
                    steps=args.lbm_steps,
                    domain_scale=args.domain_scale,
                    device_name=args.device,
                )
                lbm_cases.append(result)
                (args.output_dir / "lbm_cases.json").write_text(json.dumps(lbm_cases, indent=2), encoding="utf-8")

    rows = compare_cases(openfoam_cases, lbm_cases)
    write_csv(args.output_dir / "comparison_rows.csv", rows)
    write_plot(args.output_dir / "cd_error_surface.png", rows)
    summary = {
        "stl": str(stl),
        "mach_values": mach_values,
        "openfoam_grids": openfoam_grids,
        "lbm_grids": lbm_grids,
        "reference_area": args.reference_area,
        "reference_length": args.reference_length,
        "reynolds": args.reynolds,
        "openfoam_cases": openfoam_cases,
        "lbm_cases": lbm_cases,
        "comparisons": rows,
    }
    (args.output_dir / "grid_speed_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
