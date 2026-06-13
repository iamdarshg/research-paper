"""Bounded low-Mach simpleFoam/LBM grid-speed error suite.

The suite is intentionally resumable. Each finished or failed case is appended
to a JSON/CSV row file so a long OpenFOAM case does not discard previous data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_internal_benchmark as rib
from CLI.openfoam_mach_sweep import SPEED_OF_SOUND, parse_float_list, run_case as run_openfoam_case
from CLI.solver_grid_speed_study import run_lbm_case


FIELDS = [
    "mach",
    "speed_mps",
    "grid",
    "openfoam_solver",
    "openfoam_status",
    "openfoam_mesh_ok",
    "openfoam_latest_time",
    "openfoam_cd",
    "openfoam_cl",
    "openfoam_seconds",
    "openfoam_failed_stage",
    "of_u_initial_residual_max",
    "of_p_initial_residual",
    "of_continuity_local",
    "of_continuity_global",
    "of_rough_converged",
    "lbm_cd",
    "lbm_cl",
    "lbm_seconds",
    "lbm_converged",
    "cd_error_percent",
    "cl_error_percent",
]


def parse_grid_range(value: str) -> list[int]:
    parts = [int(part.strip()) for part in value.split(":")]
    if len(parts) != 3:
        raise ValueError("--grid-range must be start:stop:step")
    start, stop, step = parts
    return list(range(start, stop + 1, step))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "suite_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[int, float]]:
    keys = set()
    for row in rows:
        if row.get("openfoam_status") in {"completed", "failed", "timeout"}:
            keys.add((int(row["grid"]), float(row["mach"])))
    return keys


def parse_simplefoam_convergence(case_dir: Path) -> dict[str, Any]:
    log_path = case_dir / "log.simpleFoam"
    if not log_path.exists():
        return {
            "of_u_initial_residual_max": None,
            "of_p_initial_residual": None,
            "of_continuity_local": None,
            "of_continuity_global": None,
            "of_rough_converged": False,
        }
    text = log_path.read_text(errors="ignore")
    u_matches = re.findall(r"Solving for U[xyz], Initial residual = ([0-9.eE+-]+)", text)
    p_matches = re.findall(r"Solving for p, Initial residual = ([0-9.eE+-]+)", text)
    c_matches = re.findall(
        r"time step continuity errors : sum local = ([0-9.eE+-]+), global = ([0-9.eE+-]+), cumulative = ([0-9.eE+-]+)",
        text,
    )
    u_tail = [float(value) for value in u_matches[-3:]] if len(u_matches) >= 3 else []
    p_initial = float(p_matches[-1]) if p_matches else None
    continuity_local = float(c_matches[-1][0]) if c_matches else None
    continuity_global = float(c_matches[-1][1]) if c_matches else None
    u_max = max(u_tail) if u_tail else None
    rough = (
        u_max is not None
        and p_initial is not None
        and continuity_local is not None
        and u_max <= 1e-2
        and p_initial <= 5e-2
        and continuity_local <= 1e-1
    )
    return {
        "of_u_initial_residual_max": u_max,
        "of_p_initial_residual": p_initial,
        "of_continuity_local": continuity_local,
        "of_continuity_global": continuity_global,
        "of_rough_converged": rough,
    }


def command_failed_stage(result: dict[str, Any]) -> str | None:
    commands = result.get("commands") or {}
    for name, details in commands.items():
        if isinstance(details, dict) and details.get("returncode") not in (0, None):
            return name
    return None


def make_row(of_result: dict[str, Any], lbm_result: dict[str, Any] | None, grid: int, mach: float) -> dict[str, Any]:
    force = of_result.get("force") or {}
    commands = of_result.get("commands") or {}
    of_seconds = sum(float(v.get("seconds", 0.0)) for v in commands.values() if isinstance(v, dict))
    status = "completed"
    failed_stage = command_failed_stage(of_result)
    if of_result.get("failed"):
        status = "timeout" if failed_stage and (commands.get(failed_stage) or {}).get("returncode") == "timeout" else "failed"

    of_cd = force.get("cd_total")
    of_cl = force.get("cl_total")
    lbm_cd = lbm_result.get("cd") if lbm_result else None
    lbm_cl = lbm_result.get("cl") if lbm_result else None
    cd_error = None
    cl_error = None
    if finite(of_cd) and finite(lbm_cd) and abs(float(of_cd)) > 1e-12:
        cd_error = abs(float(lbm_cd) - float(of_cd)) / abs(float(of_cd)) * 100.0
    if finite(of_cl) and finite(lbm_cl) and abs(float(of_cl)) > 1e-12:
        cl_error = abs(float(lbm_cl) - float(of_cl)) / abs(float(of_cl)) * 100.0

    convergence = parse_simplefoam_convergence(Path(of_result.get("case_dir", "")))
    return {
        "mach": mach,
        "speed_mps": mach * SPEED_OF_SOUND,
        "grid": grid,
        "openfoam_solver": of_result.get("solver"),
        "openfoam_status": status,
        "openfoam_mesh_ok": of_result.get("mesh_ok"),
        "openfoam_latest_time": of_result.get("latest_time"),
        "openfoam_cd": of_cd,
        "openfoam_cl": of_cl,
        "openfoam_seconds": of_seconds,
        "openfoam_failed_stage": failed_stage,
        **convergence,
        "lbm_cd": lbm_cd,
        "lbm_cl": lbm_cl,
        "lbm_seconds": (lbm_result or {}).get("timings", {}).get("total_seconds"),
        "lbm_converged": (lbm_result or {}).get("lbm_converged"),
        "cd_error_percent": cd_error,
        "cl_error_percent": cl_error,
        "openfoam_case_dir": of_result.get("case_dir"),
        "lbm_result": lbm_result,
        "openfoam_result": of_result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-range", default="16:96:8")
    parser.add_argument("--mach-values", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--reference-area", type=float, default=0.27734375)
    parser.add_argument("--reference-length", type=float, default=1.0)
    parser.add_argument("--domain-scale", type=float, default=2.0)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--simple-iterations", type=int, default=1000)
    parser.add_argument("--lbm-steps", type=int, default=1000)
    parser.add_argument("--case-timeout", type=int, default=1200)
    parser.add_argument("--wall-time-limit", type=int, default=1200)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--rerun-unconverged", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.output_dir / "suite_rows.json")
    done = completed_keys(rows)
    if args.rerun_unconverged:
        rows = [row for row in rows if bool(row.get("of_rough_converged")) or row.get("openfoam_status") in {"failed", "timeout"}]
        done = completed_keys(rows)

    stl = Path(args.stl)
    mesh = trimesh.load_mesh(stl)
    domain_min, domain_max, domain_size, _ = rib.compute_geometry_frame(mesh, args.domain_scale)
    grids = parse_grid_range(args.grid_range)
    mach_values = parse_float_list(args.mach_values)

    for grid in grids:
        for mach in mach_values:
            if (grid, mach) in done:
                continue
            if time.perf_counter() - started > args.wall_time_limit:
                write_rows(args.output_dir, rows)
                return 0

            of_args = argparse.Namespace(**vars(args))
            of_args.grid = grid
            of_args.refinement_level = "0 0"
            of_args.snap = False
            of_args.simple_threshold = 999.0
            of_args.timeout = args.case_timeout
            of_args.output_dir = args.output_dir / "openfoam_cases" / f"grid_{grid}"
            of_args.stl = str(stl)
            of_result = run_openfoam_case(of_args, mach, mesh, domain_min, domain_max, domain_size, args.reference_area)

            lbm_result = None
            if not of_result.get("failed") and of_result.get("force"):
                lbm_result = run_lbm_case(
                    stl,
                    mesh,
                    grid=grid,
                    mach=mach,
                    reynolds=args.reynolds,
                    steps=args.lbm_steps,
                    domain_scale=args.domain_scale,
                    device_name=args.device,
                )
            rows.append(make_row(of_result, lbm_result, grid, mach))
            write_rows(args.output_dir, rows)

    summary = {
        "stl": str(stl),
        "grids": grids,
        "mach_values": mach_values,
        "rows": rows,
        "wall_time_limit": args.wall_time_limit,
        "case_timeout": args.case_timeout,
    }
    (args.output_dir / "suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
