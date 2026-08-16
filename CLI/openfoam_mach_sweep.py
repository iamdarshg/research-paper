"""Run OpenFOAM Mach sweeps with low/high-speed solver routing.

The script is meant for diagnostic sweeps, not final validation. It builds a
snappyHexMesh case, routes low-Mach cases to an incompressible solver, routes
higher-Mach cases to a compressible solver, and records force coefficients and
failure modes in JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_internal_benchmark as rib
from CLI.openfoam_case_utils import (
    copy_windows_dir_to_wsl,
    copy_wsl_dir_to_windows,
    latest_numeric_time_dir,
    restore_hidden_force_files,
    run_openfoam_wsl,
    temporarily_hide_force_files,
)


from lbm_utils import REFERENCE_SPEED_OF_SOUND_MPS as SPEED_OF_SOUND
RHO_INF = 1.225


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def sanitize_patch_name(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", stem)
    return cleaned[:60].strip("_") or "body"


def patch_snappy_strategy(case: Path, patch_name: str, *, snap: bool, refinement_level: str) -> None:
    snappy_path = case / "system" / "snappyHexMeshDict"
    text = snappy_path.read_text(encoding="utf-8")
    text = re.sub(r"snap\s+true;", f"snap {str(snap).lower()};", text)
    text = re.sub(
        r"refinementSurfaces \{ .*? \{ level \([^)]+\); \} \}",
        f"refinementSurfaces {{ {patch_name} {{ level ({refinement_level}); }} }}",
        text,
    )
    if snap:
        text = re.sub(
            r"snapControls \{[^}]+\}",
            "snapControls { nSmoothPatch 5; tolerance 0.5; nSolveIter 50; nRelaxIter 8; }",
            text,
        )
    snappy_path.write_text(text, encoding="utf-8")


def patch_sonic_case(case: Path, *, end_time: float, max_delta_t: float) -> None:
    control = (case / "system" / "controlDict").read_text(encoding="utf-8")
    write_interval = max(end_time / 25.0, max_delta_t)
    control = re.sub(r"application\s+\w+;", "application sonicFoam;", control)
    control = re.sub(r"endTime\s+[^;]+;", f"endTime {end_time:.8g};", control)
    control = re.sub(r"deltaT\s+[^;]+;", f"deltaT {max_delta_t:.8g};", control)
    control = re.sub(r"adjustTimeStep\s+[^;]+;", "adjustTimeStep yes;", control)
    control = re.sub(r"maxCo\s+[^;]+;", "maxCo 0.35;", control)
    control = re.sub(r"maxDeltaT\s+[^;]+;", f"maxDeltaT {max_delta_t:.8g};", control)
    control = re.sub(r"writeControl\s+[^;]+;", "writeControl adjustableRunTime;", control)
    control = re.sub(r"writeInterval\s+[^;]+;", f"writeInterval {write_interval:.8g};", control)
    (case / "system" / "controlDict").write_text(control, encoding="utf-8")


def patch_simple_case(
    case: Path,
    patch_name: str,
    *,
    freestream_speed: float,
    reynolds_number: float,
    reference_length: float,
    iterations: int,
) -> None:
    nu = freestream_speed * reference_length / max(reynolds_number, 1e-12)
    write(
        case / "system" / "controlDict",
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {int(iterations)};
deltaT 1;
writeControl timeStep;
writeInterval {max(1, int(iterations // 5))};
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable true;
""",
    )
    write(
        case / "constant" / "transportProperties",
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object transportProperties;
}}
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.8e};
""",
    )
    write(
        case / "constant" / "turbulenceProperties",
        """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object turbulenceProperties;
}
simulationType laminar;
""",
    )
    write(
        case / "0" / "U",
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({freestream_speed:.8g} 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform ({freestream_speed:.8g} 0 0); }}
    outlet {{ type zeroGradient; }}
    top {{ type slip; }}
    bottom {{ type slip; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type noSlip; }}
}}
""",
    )
    write(
        case / "0" / "p",
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type zeroGradient; }}
}}
""",
    )
    write(
        case / "system" / "fvSchemes",
        """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""",
    )
    write(
        case / "system" / "fvSolution",
        """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers
{
    p
    {
        solver GAMG;
        tolerance 1e-8;
        relTol 0.01;
        smoother GaussSeidel;
    }
    U
    {
        solver smoothSolver;
        smoother symGaussSeidel;
        tolerance 1e-9;
        relTol 0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""",
    )
    write_forces_dict(case, patch_name, rho_mode="rhoInf")


def write_forces_dict(case: Path, patch_name: str, *, rho_mode: str) -> None:
    rho_line = "rho rhoInf;" if rho_mode == "rhoInf" else "rho rho;"
    write(
        case / "system" / "forces",
        f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object forces;
}}
type forces;
libs ("libforces.so");
patches ({patch_name});
{rho_line}
rhoInf {RHO_INF};
p p;
U U;
CofR (0 0 0);
writeControl writeTime;
""",
    )


def choose_solver(mach: float, threshold: float) -> str:
    return "simpleFoam" if mach < threshold else "sonicFoam"


def pressure_force_latest(case: Path, patch_name: str, reference_area: float, speed: float, solver: str) -> dict[str, Any]:
    if solver == "simpleFoam":
        return rib.pressure_force_from_case(
            case,
            patch_name,
            reference_area=reference_area,
            pressure_reference=0.0,
            density=RHO_INF,
            freestream_speed=speed,
        )
    moved = temporarily_hide_force_files(case)
    try:
        pressure_reference = rib.OPENFOAM_PRESSURE_REFERENCE if solver == "sonicFoam" else 0.0
        force = rib.pressure_force_from_case(
            case,
            patch_name,
            reference_area=reference_area,
            pressure_reference=pressure_reference,
            density=RHO_INF,
            freestream_speed=speed,
        )
    finally:
        restore_hidden_force_files(moved)
    return force


def run_case(args: argparse.Namespace, mach: float, mesh, domain_min, domain_max, domain_size, reference_area: float) -> dict[str, Any]:
    speed = mach * SPEED_OF_SOUND
    solver = choose_solver(mach, args.simple_threshold)
    patch_name = sanitize_patch_name(f"{Path(args.stl).stem}_ma{mach:g}_{solver}")
    case = rib.make_case(
        Path(args.stl),
        patch_name=patch_name,
        grid_resolution=args.grid,
        domain_min=domain_min,
        domain_max=domain_max,
        freestream_speed=speed,
        reynolds_number=args.reynolds,
    )
    patch_snappy_strategy(case, patch_name, snap=args.snap, refinement_level=args.refinement_level)
    if solver == "simpleFoam":
        patch_simple_case(
            case,
            patch_name,
            freestream_speed=speed,
            reynolds_number=args.reynolds,
            reference_length=args.reference_length,
            iterations=args.simple_iterations,
        )
    else:
        dx = domain_size / args.grid
        max_delta_t = min(args.max_delta_t, args.max_co * dx / max(speed, 1e-12))
        end_time = args.compressible_body_transits * args.reference_length / max(speed, 1e-12)
        patch_sonic_case(case, end_time=end_time, max_delta_t=max_delta_t)
        write_forces_dict(case, patch_name, rho_mode="rho")

    wsl_case = f"/tmp/{case.name}"
    copy_windows_dir_to_wsl(case, wsl_case, distro=args.distro)
    stages = [
        ("surfaceCheck", f"surfaceCheck constant/triSurface/{patch_name}.stl > log.surfaceCheck 2>&1", True),
        ("blockMesh", "blockMesh > log.blockMesh 2>&1", True),
        ("surfaceFeatureExtract", "surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1", True),
        ("snappyHexMesh", "snappyHexMesh -overwrite > log.snappyHexMesh 2>&1", True),
        ("checkMesh", "checkMesh -allTopology -allGeometry > log.checkMesh 2>&1", False),
        (solver, f"{solver} > log.{solver} 2>&1", True),
        ("forces", "postProcess -dict system/forces -latestTime > log.forces 2>&1", False),
    ]
    commands: dict[str, Any] = {}
    failed = False
    for name, command, required in stages:
        started = time.perf_counter()
        try:
            proc = run_openfoam_wsl(wsl_case, command, distro=args.distro, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            commands[name] = {
                "returncode": "timeout",
                "seconds": time.perf_counter() - started,
                "stdout_tail": (exc.stdout or "")[-1000:] if isinstance(exc.stdout, str) else "",
                "stderr_tail": (exc.stderr or "")[-1000:] if isinstance(exc.stderr, str) else "",
                "timeout_seconds": args.timeout,
            }
            failed = True
            break
        commands[name] = {
            "returncode": proc.returncode,
            "seconds": time.perf_counter() - started,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
        }
        if proc.returncode != 0 and required:
            failed = True
            break
    try:
        copy_wsl_dir_to_windows(wsl_case, case, distro=args.distro)
    except Exception as exc:
        commands["copy_back"] = {"returncode": 1, "error": repr(exc)}

    force = None
    if not failed:
        try:
            force = pressure_force_latest(case, patch_name, reference_area, speed, solver)
        except Exception as exc:  # keep sweep moving
            commands["force_parse"] = {"returncode": 1, "error": repr(exc)}

    check_log = (case / "log.checkMesh").read_text(errors="ignore") if (case / "log.checkMesh").exists() else ""
    solver_log = (case / f"log.{solver}").read_text(errors="ignore") if (case / f"log.{solver}").exists() else ""
    latest_time = None
    try:
        latest_time = latest_numeric_time_dir(case).name
    except Exception:
        latest_time = None

    case_out = args.output_dir / f"mach_{mach:g}_{solver}"
    if case_out.exists():
        shutil.rmtree(case_out)
    shutil.copytree(case, case_out)

    return {
        "mach": mach,
        "speed_mps": speed,
        "solver": solver,
        "case_dir": str(case_out),
        "failed": failed,
        "latest_time": latest_time,
        "mesh_ok": "Mesh OK." in check_log and "Failed" not in check_log,
        "checkmesh_line": next((line.strip() for line in check_log.splitlines() if "Mesh OK" in line or "Failed" in line), ""),
        "fatal_error_tail": "\n".join(solver_log.splitlines()[-30:]) if failed else "",
        "force": force,
        "commands": commands,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True, help="STL to run through OpenFOAM.")
    parser.add_argument("--output-dir", type=Path, default=Path("build/solver_diagnostics/openfoam_mach_sweep"))
    parser.add_argument("--mach-values", default="0.1,0.2,0.3,0.5,0.8,1.0,1.5,2.0")
    parser.add_argument("--simple-threshold", type=float, default=0.3)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--domain-scale", type=float, default=2.0)
    parser.add_argument("--reference-length", type=float, default=1.0)
    parser.add_argument("--reference-area", type=float, default=None)
    parser.add_argument("--reynolds", type=float, default=100000.0)
    parser.add_argument("--simple-iterations", type=int, default=300)
    parser.add_argument("--compressible-body-transits", type=float, default=1.0)
    parser.add_argument("--max-co", type=float, default=0.25)
    parser.add_argument("--max-delta-t", type=float, default=1e-4)
    parser.add_argument("--refinement-level", default="0 0")
    parser.add_argument("--snap", action="store_true")
    parser.add_argument("--distro", default="Ubuntu-24.04")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = Path(args.stl)
    mesh = trimesh.load_mesh(stl_path)
    domain_min, domain_max, domain_size, max_extent = rib.compute_geometry_frame(mesh, args.domain_scale)
    reference_area = args.reference_area
    if reference_area is None:
        mask = rib.mesh_to_geometry_mask(mesh, args.grid, domain_min, domain_size).numpy() > 0.5
        reference_area = float(np.any(mask, axis=0).sum() * (domain_size / args.grid) ** 2)

    results = []
    for mach in parse_float_list(args.mach_values):
        result = run_case(args, mach, mesh, domain_min, domain_max, domain_size, reference_area)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "stl": str(stl_path),
        "grid": args.grid,
        "domain_scale": args.domain_scale,
        "reference_area": reference_area,
        "reference_length": args.reference_length,
        "simple_threshold": args.simple_threshold,
        "results": results,
    }
    (args.output_dir / "mach_sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
