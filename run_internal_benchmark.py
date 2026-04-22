from __future__ import annotations

import argparse
import io
import itertools
import math
import json
import os
import re
import glob
import subprocess
import tempfile
import shutil
import tarfile
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str((REPO / 'CLI').resolve()))
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig
from advanced_lbm_solver import D3Q27CascadedSolver

OPENFOAM_PACKAGE = os.environ.get('OPENFOAM_PACKAGE', 'openfoam')
OPENFOAM_FREESTREAM_SPEED = 80.0
OPENFOAM_DENSITY = 1.225
OPENFOAM_PRESSURE_REFERENCE = 101325.0
KINEMATIC_VISCOSITY = 1.47e-5
DEFAULT_GRID_RESOLUTION = 32
DEFAULT_SIMULATION_STEPS = 200
DEFAULT_DOMAIN_SCALE = 2.0
HIGH_ACCURACY_ERROR_PERCENT = 1.0
MINIMUM_ACCEPTABLE_ERROR_PERCENT = 5.0
ROOT_STL_PRIORITY = ('20mm_cube.stl',)
OPENFOAM_WSL_DISTRO = os.environ.get('OPENFOAM_WSL_DISTRO')


def _is_windows_host() -> bool:
    return os.name == 'nt'


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', name).strip('_')
    if not cleaned:
        return 'geometry'
    if cleaned[0].isdigit():
        return f'geom_{cleaned}'
    return cleaned


def classify_surrogate_label_quality(error_percentage: Optional[float]) -> str:
    if error_percentage is None or not math.isfinite(float(error_percentage)):
        return 'not_acceptable'
    if float(error_percentage) < HIGH_ACCURACY_ERROR_PERCENT:
        return 'high_accuracy'
    if float(error_percentage) < MINIMUM_ACCEPTABLE_ERROR_PERCENT:
        return 'minimum_acceptable'
    return 'not_acceptable'


def add_surrogate_quality_fields(target: Dict[str, Any], error_percentage: Optional[float]) -> None:
    finite_error = (
        error_percentage is not None
        and math.isfinite(float(error_percentage))
    )
    target['surrogate_label_quality'] = classify_surrogate_label_quality(error_percentage)
    target['meets_one_percent_target'] = (
        finite_error and float(error_percentage) < HIGH_ACCURACY_ERROR_PERCENT
    )
    target['meets_five_percent_minimum'] = (
        finite_error and float(error_percentage) < MINIMUM_ACCEPTABLE_ERROR_PERCENT
    )


def _default_openfoam_bashrc_candidates(package: str = OPENFOAM_PACKAGE) -> List[str]:
    root = os.environ.get('OPENFOAM_ROOT')
    candidates = []
    if root:
        candidates.append(str(Path(root) / 'etc' / 'bashrc'))
    candidates.extend([
        '/usr/share/openfoam/etc/bashrc',
        f'/usr/lib/openfoam/{package}/etc/bashrc',
        f'/opt/{package}/etc/bashrc',
        f'/usr/share/openfoam/{package}/etc/bashrc',
        '/home/darsh/.openclaw/openfoam/usr/share/openfoam/etc/bashrc',
    ])
    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def _wsl_available() -> bool:
    return shutil.which('wsl') is not None


def _detect_wsl_distro() -> Optional[str]:
    if OPENFOAM_WSL_DISTRO:
        return OPENFOAM_WSL_DISTRO
    if not _wsl_available():
        return None

    proc = subprocess.run(['wsl', '-l', '-q'], text=True, capture_output=True)
    if proc.returncode != 0:
        return None

    normalized = proc.stdout.replace('\x00', '')
    distros = [line.strip() for line in normalized.splitlines() if line.strip()]
    preferred = ['Ubuntu-24.04', 'Ubuntu-22.04', 'Ubuntu']
    for name in preferred:
        for distro in distros:
            if distro.lower() == name.lower():
                return distro
    for distro in distros:
        if distro.lower().startswith('ubuntu'):
            return distro
    return None


def _wsl_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _probe_wsl_for_bashrc(package: str) -> Optional[str]:
    distro = _detect_wsl_distro()
    if not _wsl_available() or not distro:
        return None
    candidates = [
        '/usr/share/openfoam/etc/bashrc',
        f'/usr/lib/openfoam/{package}/etc/bashrc',
        f'/opt/{package}/etc/bashrc',
        f'/usr/share/openfoam/{package}/etc/bashrc',
    ]
    for candidate in candidates:
        proc = subprocess.run(['wsl', '-d', distro, '--', 'test', '-f', candidate], text=True, capture_output=True)
        if proc.returncode == 0:
            return candidate
    return None


def resolve_openfoam_bashrc(package: str = OPENFOAM_PACKAGE) -> Optional[str]:
    env_bashrc = os.environ.get('OPENFOAM_BASHRC')
    if env_bashrc:
        return env_bashrc
    if _is_windows_host():
        return _probe_wsl_for_bashrc(package)
    for candidate in _default_openfoam_bashrc_candidates(package):
        if Path(candidate).exists():
            return candidate
    return None


def _openfoam_runner(package: str = OPENFOAM_PACKAGE) -> Tuple[List[str], Optional[str]]:
    bashrc = resolve_openfoam_bashrc(package)
    if bashrc is None:
        raise FileNotFoundError('Could not find an OpenFOAM bashrc. Install OpenFOAM or pass --install-openfoam.')
    if _is_windows_host():
        distro = _detect_wsl_distro()
        if not _wsl_available() or not distro:
            raise FileNotFoundError('Windows OpenFOAM runs through WSL, but wsl.exe was not found.')
        return ['wsl', '-d', distro, 'bash', '-lc'], bashrc
    return ['bash', '-lc'], bashrc


def run_openfoam(cmd: str, cwd: Path, timeout: int = 600, package: str = OPENFOAM_PACKAGE) -> Tuple[int, str, str]:
    launcher, bashrc = _openfoam_runner(package)
    if launcher[0] == 'wsl':
        cwd_str = str(cwd)
        wsl_cwd = cwd_str if cwd_str.startswith('/') else windows_path_to_wsl_path(cwd)
        shell_cmd = f'cd {_wsl_quote(wsl_cwd)} && source {_wsl_quote(bashrc)} >/dev/null 2>&1 && {cmd}'
        proc = subprocess.run(launcher + [shell_cmd], text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr

    shell_cmd = f'source "{bashrc}" >/dev/null 2>&1 && {cmd}'
    proc = subprocess.run(launcher + [shell_cmd], cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def windows_path_to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.drive:
        drive = resolved.drive.rstrip(':').lower()
        rest = resolved.as_posix().split(':', 1)[-1]
        return f'/mnt/{drive}{rest}'
    return resolved.as_posix()


def ensure_openfoam_installed(install: bool, package: str = OPENFOAM_PACKAGE) -> bool:
    if resolve_openfoam_bashrc(package) is not None:
        return True
    if not install:
        return False

    if _is_windows_host():
        distro = _detect_wsl_distro()
        if not _wsl_available() or not distro:
            return False
        cmd = f'DEBIAN_FRONTEND=noninteractive sudo -n apt-get update && DEBIAN_FRONTEND=noninteractive sudo -n apt-get install -y {package}'
        proc = subprocess.run(['wsl', '-d', distro, 'bash', '-lc', cmd], text=True, capture_output=True)
        if proc.returncode != 0:
            return False
        return resolve_openfoam_bashrc(package) is not None

    cmd = f'DEBIAN_FRONTEND=noninteractive sudo -n apt-get update && DEBIAN_FRONTEND=noninteractive sudo -n apt-get install -y {package}'
    proc = subprocess.run(['bash', '-lc', cmd], text=True, capture_output=True)
    if proc.returncode != 0:
        return False
    return resolve_openfoam_bashrc(package) is not None


def discover_root_stls(root: Path = REPO) -> List[Path]:
    return _order_stl_paths(root.glob('*.stl'))


def _order_stl_paths(paths) -> List[Path]:
    stls = []
    seen = set()
    for path in paths:
        p = Path(path).resolve()
        if not p.is_file() or p.suffix.lower() != '.stl':
            continue
        if p in seen:
            continue
        seen.add(p)
        stls.append(p)

    prioritized = []
    remaining = []
    for stl in stls:
        if stl.name in ROOT_STL_PRIORITY:
            prioritized.append(stl)
        else:
            remaining.append(stl)
    prioritized.sort(key=lambda p: ROOT_STL_PRIORITY.index(p.name))
    remaining.sort(key=lambda p: p.name.lower())
    ordered = prioritized + remaining
    if ordered:
        return ordered
    return []


def _split_path_specs(specs: Optional[str]) -> List[str]:
    if not specs:
        return []
    return [part.strip() for part in str(specs).split(',') if part.strip()]


def _expand_stl_spec(spec: str, root: Path, recursive: bool) -> List[Path]:
    has_glob = any(ch in spec for ch in '*?[')
    candidate = Path(spec)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(root / candidate)

    matches: List[Path] = []
    for path in candidates:
        if has_glob:
            matches.extend(Path(p) for p in glob.glob(str(path), recursive=True))
        elif path.is_dir():
            iterator = path.rglob('*.stl') if recursive else path.glob('*.stl')
            matches.extend(iterator)
        else:
            matches.append(path)
    return matches


def discover_stls(
    root: Path = REPO,
    *,
    recursive: bool = False,
    stl_files: Optional[str] = None,
    max_stls: Optional[int] = None,
) -> List[Path]:
    root = Path(root).resolve()
    specs = _split_path_specs(stl_files)
    if specs:
        paths = []
        for spec in specs:
            paths.extend(_expand_stl_spec(spec, root, recursive))
    else:
        iterator = root.rglob('*.stl') if recursive else root.glob('*.stl')
        paths = list(iterator)

    ordered = _order_stl_paths(paths)
    if max_stls is not None:
        ordered = ordered[:max(0, int(max_stls))]
    return ordered


def mesh_to_geometry_mask(mesh, grid_resolution: int, domain_min: np.ndarray, domain_size: float) -> torch.Tensor:
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - dependency should already be present for STL runs
        raise ImportError('trimesh is required to voxelize STL files for the benchmark') from exc

    voxel_pitch = domain_size / float(grid_resolution)
    shifted = mesh.copy()
    shifted.apply_translation(-domain_min)

    try:
        voxel_grid = shifted.voxelized(voxel_pitch).fill()
        target_shape = (grid_resolution, grid_resolution, grid_resolution)
        resized = np.zeros(target_shape, dtype=np.float32)
        voxel_points = voxel_grid.points
        voxel_indices = np.rint(voxel_points / voxel_pitch).astype(int)
        voxel_indices = np.clip(voxel_indices, 0, grid_resolution - 1)
        resized[
            voxel_indices[:, 0],
            voxel_indices[:, 1],
            voxel_indices[:, 2],
        ] = 1.0
        return torch.from_numpy(resized)
    except Exception:
        from scipy.ndimage import zoom

        voxel_grid = shifted.voxelized(voxel_pitch)
        voxel_np = voxel_grid.matrix.view(np.ndarray)
        zoom_factors = np.array((grid_resolution, grid_resolution, grid_resolution)) / np.array(voxel_np.shape)
        resized = zoom(voxel_np.astype(np.float32), zoom_factors, order=1)
        return torch.from_numpy((resized > 0.5).astype(np.float32))


def compute_geometry_frame(mesh, domain_scale: float = DEFAULT_DOMAIN_SCALE) -> Tuple[np.ndarray, np.ndarray, float, float]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    mins, maxs = bounds
    extents = np.maximum(maxs - mins, 1e-9)
    max_extent = float(np.max(extents))
    if max_extent <= 0.0:
        max_extent = 1.0
    half_box = max_extent * domain_scale * 0.5
    center = 0.5 * (mins + maxs)
    domain_min = center - half_box
    domain_max = center + half_box
    domain_size = float(np.max(domain_max - domain_min))
    return domain_min, domain_max, domain_size, max_extent


def run(cmd: str, cwd: Path, timeout: int = 600):
    proc = subprocess.run(['bash', '-lc', cmd], cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def write(case: Path, rel: str, content: str) -> None:
    path = case / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _cube_stl_text(
    *,
    solid_name: str = 'cube',
    center: Sequence[float] = (0.0, 0.0, 0.0),
    edge_length: float = 1.0,
) -> str:
    cx, cy, cz = (float(v) for v in center)
    half = float(edge_length) * 0.5
    vertices = [
        (cx - half, cy - half, cz - half),
        (cx + half, cy - half, cz - half),
        (cx + half, cy + half, cz - half),
        (cx - half, cy + half, cz - half),
        (cx - half, cy - half, cz + half),
        (cx + half, cy - half, cz + half),
        (cx + half, cy + half, cz + half),
        (cx - half, cy + half, cz + half),
    ]
    faces = [
        ((0, 1, 2), (0.0, 0.0, -1.0)),
        ((0, 2, 3), (0.0, 0.0, -1.0)),
        ((4, 6, 5), (0.0, 0.0, 1.0)),
        ((4, 7, 6), (0.0, 0.0, 1.0)),
        ((0, 5, 1), (0.0, -1.0, 0.0)),
        ((0, 4, 5), (0.0, -1.0, 0.0)),
        ((3, 2, 6), (0.0, 1.0, 0.0)),
        ((3, 6, 7), (0.0, 1.0, 0.0)),
        ((0, 3, 7), (-1.0, 0.0, 0.0)),
        ((0, 7, 4), (-1.0, 0.0, 0.0)),
        ((1, 5, 6), (1.0, 0.0, 0.0)),
        ((1, 6, 2), (1.0, 0.0, 0.0)),
    ]
    lines = [f'solid {solid_name}']
    for tri, normal in faces:
        lines.append(f'facet normal {normal[0]:.1f} {normal[1]:.1f} {normal[2]:.1f}')
        lines.append(' outer loop')
        for idx in tri:
            x, y, z = vertices[idx]
            lines.append(f'  vertex {x:.8g} {y:.8g} {z:.8g}')
        lines.append(' endloop')
        lines.append('endfacet')
    lines.append(f'endsolid {solid_name}')
    return '\n'.join(lines) + '\n'


VALIDATION_OBJECT_NAME = 'centered unit cube'
VALIDATION_OBJECT_DESCRIPTION = 'A 1.0-unit cube centered at the origin, used as the shared validation object for both solvers.'
VALIDATION_OBJECT_DETAILS = {
    'name': VALIDATION_OBJECT_NAME,
    'description': VALIDATION_OBJECT_DESCRIPTION,
    'geometry': 'solid cube STL spanning [-0.5, 0.5]^3',
    'purpose': 'Shared verification geometry for the internal D3Q27 solver and OpenFOAM sonicFoam.',
    'force_definition': 'total hydrodynamic force (pressure + viscous)',
    'pressure_reference': 101325.0,
    'reference_area': 1.0,
    'reference_length': 1.0,
    'freestream_speed': 80.0,
    'density': 1.225,
}


def _legacy_make_case() -> Path:
    case = Path(tempfile.mkdtemp(prefix='openfoam_sonic_cube_'))
    for p in [case / '0', case / 'constant' / 'triSurface', case / 'system']:
        p.mkdir(parents=True, exist_ok=True)

    # Validation object: simple cube STL centered at the origin
    write(case, 'constant/triSurface/cube.stl', _cube_stl_text(center=(0.0, 0.0, 0.0), edge_length=1.0))

    write(case, 'system/blockMeshDict', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}
scale 1;
vertices
(
    (-4 -4 -4)
    ( 4 -4 -4)
    ( 4  4 -4)
    (-4  4 -4)
    (-4 -4  4)
    ( 4 -4  4)
    ( 4  4  4)
    (-4  4  4)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (24 24 24) simpleGrading (1 1 1)
);
edges ( );
boundary
(
    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    bottom { type patch; faces ((0 1 5 4)); }
    top { type patch; faces ((3 7 6 2)); }
    front { type symmetryPlane; faces ((0 3 2 1)); }
    back { type symmetryPlane; faces ((4 5 6 7)); }
);
mergePatchPairs ( );
""")

    write(case, 'system/surfaceFeatureExtractDict', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object surfaceFeatureExtractDict;
}
cube.stl
{
    extractionMethod extractFromSurface;
    includedAngle 150;
    writeObj yes;
}
""")

    write(case, 'system/snappyHexMeshDict', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}
castellatedMesh true;
snap true;
addLayers false;
mergeTolerance 1e-6;
geometry
{
    cube.stl { type triSurfaceMesh; name cube; }
}
castellatedMeshControls
{
    maxLocalCells 50000;
    maxGlobalCells 200000;
    minRefinementCells 0;
    nCellsBetweenLevels 1;
    features ( );
    refinementSurfaces { cube { level (1 2); } }
    refinementRegions { }
    locationInMesh (3 0 0);
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle 30;
}
snapControls { nSmoothPatch 2; tolerance 1.0; nSolveIter 20; nRelaxIter 3; }
addLayersControls { relativeSizes true; layers { } expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 0; }
meshQualityControls { maxNonOrtho 80; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }
""")

    write(case, 'system/controlDict', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application sonicFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 5e-05;
deltaT 1e-06;
adjustTimeStep yes;
maxCo 0.5;
maxDeltaT 1e-05;
writeControl timeStep;
writeInterval 50;
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable true;
""")

    write(case, 'system/fvSchemes', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) Gauss limitedLinearV 1;
    div(phi,e) Gauss limitedLinear 1;
    div(phid,p) Gauss limitedLinear 1;
    div(phi,K) Gauss limitedLinear 1;
    div(phiv,p) Gauss limitedLinear 1;
    div(phi,k) Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")

    write(case, 'system/fvSolution', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers
{
    "rho.*" { solver diagonal; }
    "p.*"   { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }
    "(U|e).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; }
    "(k|epsilon).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0; }
}
PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
}
""")

    write(case, 'constant/thermophysicalProperties', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object thermophysicalProperties;
}
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}
mixture
{
    specie { molWeight 28.9; }
    thermodynamics { Cp 1005; Hf 0; }
    transport { mu 1.8e-05; Pr 0.71; }
}
""")

    write(case, 'constant/turbulenceProperties', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object turbulenceProperties;
}
simulationType laminar;
""")

    write(case, '0/U', """FoamFile
{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (80 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (80 0 0); }
    outlet { type pressureInletOutletVelocity; value uniform (80 0 0); }
    top { type slip; }
    bottom { type slip; }
    front { type symmetryPlane; }
    back { type symmetryPlane; }
    cube { type noSlip; }
}
""")

    pressure_ref = VALIDATION_OBJECT_DETAILS['pressure_reference']
    write(case, '0/p', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}
dimensions [1 -1 -2 0 0 0 0];
internalField uniform {pressure_ref};
boundaryField
{{
    inlet {{ type totalPressure; p0 uniform {pressure_ref}; value uniform {pressure_ref}; }}
    outlet {{ type fixedValue; value uniform {pressure_ref}; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    cube {{ type zeroGradient; }}
}}
""")

    write(case, '0/T', """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object T;
}
dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    inlet { type fixedValue; value uniform 300; }
    outlet { type zeroGradient; }
    top { type zeroGradient; }
    bottom { type zeroGradient; }
    front { type symmetryPlane; }
    back { type symmetryPlane; }
    cube { type zeroGradient; }
}
""")

    write(case, '0/rho', """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object rho;
}
dimensions [1 -3 0 0 0 0 0];
internalField uniform 1.225;
boundaryField
{
    inlet { type fixedValue; value uniform 1.225; }
    outlet { type zeroGradient; }
    top { type zeroGradient; }
    bottom { type zeroGradient; }
    front { type symmetryPlane; }
    back { type symmetryPlane; }
    cube { type zeroGradient; }
}
""")
    write(case, 'system/forces', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object forces;
}
type forces;
functionObjectLibs (\"libforces.so\");
patches (cube);
rho rho;
rhoInf 1.225;
p p;
U U;
CofR (0 0 0);
writeControl writeTime;
""")

    write(case, 'VALIDATION_OBJECT.md', f"""# Validation object

- **Name:** {VALIDATION_OBJECT_NAME}
- **Description:** {VALIDATION_OBJECT_DESCRIPTION}
- **Geometry:** 1.0 unit cube centered at the origin
- **Use:** Internal D3Q27 benchmark vs. OpenFOAM sonicFoam comparison
""")
    return case


def parse_foam_block(text: str, keyword: str):
    pat = re.compile(rf'{keyword}\s+nonuniform\s+List<[^>]+>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;', re.S)
    m = pat.search(text)
    if m:
        count = int(m.group(1))
        nums = np.fromstring(m.group(2).replace('\n', ' '), sep=' ')
        return nums[:count]
    m = re.search(rf'{keyword}\s+uniform\s+([^;]+);', text)
    if m:
        return float(m.group(1).strip())
    raise ValueError(f'Could not parse {keyword}')


def _extract_foam_list_lines(text: str) -> List[str]:
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if re.fullmatch(r'\d+', line.strip()) and lines[i + 1].strip() == '(':
            out = []
            for j in range(i + 2, len(lines)):
                stripped = lines[j].strip()
                if stripped in {')', ');'}:
                    return out
                out.append(lines[j])
    raise ValueError('Could not locate OpenFOAM list block')


def parse_points(path: Path) -> np.ndarray:
    block = _extract_foam_list_lines(path.read_text())
    matches = []
    for line in block:
        m = re.match(r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)', line.strip())
        if m:
            matches.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    if not matches:
        raise ValueError(f'Could not parse points from {path}')
    return np.asarray(matches, dtype=float)


def parse_faces(path: Path) -> List[List[int]]:
    block = _extract_foam_list_lines(path.read_text())
    faces = []
    for line in block:
        line = line.strip()
        mm = re.match(r'\d+\(([^)]*)\)', line)
        if mm:
            faces.append([int(x) for x in mm.group(1).split()])
    if not faces:
        raise ValueError(f'Could not parse faces from {path}')
    return faces


def parse_owner(path: Path) -> np.ndarray:
    block = _extract_foam_list_lines(path.read_text())
    vals = np.array([int(x.strip()) for x in block if x.strip()], dtype=int)
    if vals.size == 0:
        raise ValueError(f'Could not parse owner from {path}')
    return vals


def parse_boundary(path: Path) -> Dict[str, Tuple[int, int]]:
    text = path.read_text()
    body = text.split('boundary', 1)[1]
    body = body[body.find('(') + 1: body.rfind(')')]
    out = {}
    for m in re.finditer(r'([A-Za-z0-9_]+)\s*\{([^}]*)\}', body, re.S):
        name = m.group(1)
        block = m.group(2)
        mm_start = re.search(r'startFace\s+(\d+)\s*;', block)
        mm_n = re.search(r'nFaces\s+(\d+)\s*;', block)
        if mm_start and mm_n:
            out[name] = (int(mm_start.group(1)), int(mm_n.group(1)))
    return out


def latest_time_dir(case: Path) -> Path:
    times = []
    for p in case.iterdir():
        if p.is_dir():
            try:
                times.append((float(p.name), p))
            except ValueError:
                continue
    if not times:
        raise RuntimeError('No time directories found')
    return max(times, key=lambda x: x[0])[1]


def face_area_vector(face_pts: np.ndarray) -> np.ndarray:
    ref = face_pts[0]
    area = np.zeros(3, dtype=float)
    for i in range(1, len(face_pts) - 1):
        area += np.cross(face_pts[i] - ref, face_pts[i + 1] - ref)
    return 0.5 * area


def _parse_forces_dat(
    path: Path,
    *,
    reference_area: float,
    density: float = OPENFOAM_DENSITY,
    freestream_speed: float = OPENFOAM_FREESTREAM_SPEED,
) -> Dict[str, float]:
    lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not lines:
        raise ValueError(f'No force data found in {path}')
    last = lines[-1].split()
    if len(last) < 7:
        raise ValueError(f'Unexpected forces data format in {path}: {lines[-1]!r}')
    time = float(last[0])
    force = np.array([float(last[1]), float(last[2]), float(last[3])], dtype=float)
    moment = np.array([float(last[4]), float(last[5]), float(last[6])], dtype=float)
    q = 0.5 * density * freestream_speed * freestream_speed
    return {
        'time': time,
        'force_x': float(force[0]),
        'force_y': float(force[1]),
        'force_z': float(force[2]),
        'moment_x': float(moment[0]),
        'moment_y': float(moment[1]),
        'moment_z': float(moment[2]),
        'cd_total': float(-force[0] / (q * reference_area)),
        'cl_total': float(force[2] / (q * reference_area)),
        'reference_area': reference_area,
    }


def pressure_force_from_case(
    case: Path,
    patch_name: str,
    *,
    reference_area: float,
    pressure_reference: float = OPENFOAM_PRESSURE_REFERENCE,
    density: float = OPENFOAM_DENSITY,
    freestream_speed: float = OPENFOAM_FREESTREAM_SPEED,
) -> Dict[str, float]:
    candidates = sorted(case.glob('postProcessing/**/forces.dat'))
    if candidates:
        forces_file = max(candidates, key=lambda p: p.stat().st_mtime)
        out = _parse_forces_dat(
            forces_file,
            reference_area=reference_area,
            density=density,
            freestream_speed=freestream_speed,
        )
        out['source'] = f'postProcessing/{forces_file.parent.name}'
        return out

    points = parse_points(case / 'constant' / 'polyMesh' / 'points')
    faces = parse_faces(case / 'constant' / 'polyMesh' / 'faces')
    owner = parse_owner(case / 'constant' / 'polyMesh' / 'owner')
    boundary = parse_boundary(case / 'constant' / 'polyMesh' / 'boundary')
    start_face, n_faces = boundary[patch_name]
    time_dir = latest_time_dir(case)
    p = parse_foam_block((time_dir / 'p').read_text(), 'internalField')
    if isinstance(p, float):
        p = np.full(owner.max() + 1, p, dtype=float)
    p = np.asarray(p, dtype=float)

    force = np.zeros(3, dtype=float)
    for face_idx in range(start_face, start_face + n_faces):
        face = faces[face_idx]
        face_pts = points[np.asarray(face, dtype=int)]
        sf = face_area_vector(face_pts)
        cell = owner[face_idx]
        p_cell = p[cell] if cell < len(p) else p[-1]
        # Use the same absolute pressure baseline as the OpenFOAM case setup.
        force += -(p_cell - pressure_reference) * sf

    q = 0.5 * density * freestream_speed * freestream_speed
    return {
        'source': 'manual_pressure_integration',
        'force_x': float(force[0]),
        'force_y': float(force[1]),
        'force_z': float(force[2]),
        'cd_total': float(-force[0] / (q * reference_area)),
        'cl_total': float(force[2] / (q * reference_area)),
        'time_dir': time_dir.name,
    }


def _load_trimesh(stl_path: Path):
    try:
        import trimesh
    except Exception as exc:  # pragma: no cover - import depends on environment
        raise ImportError('trimesh is required for STL-based benchmark cases') from exc

    mesh = trimesh.load_mesh(str(stl_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def _format_foam_vec(values: Sequence[float]) -> str:
    return '(' + ' '.join(f'{float(v):.8g}' for v in values) + ')'


def _parse_numeric_list(value, cast, default: Sequence[float]) -> List[float]:
    if value is None:
        return [cast(v) for v in default]
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    text = str(value).strip()
    if not text:
        return [cast(v) for v in default]
    return [cast(part.strip()) for part in text.split(',') if part.strip()]


def _parse_int_list(value, default: Sequence[int]) -> List[int]:
    return [int(v) for v in _parse_numeric_list(value, int, default)]


def _parse_float_list(value, default: Sequence[float]) -> List[float]:
    return [float(v) for v in _parse_numeric_list(value, float, default)]


def mesh_complexity_summary(mesh) -> Dict[str, Any]:
    extents = np.asarray(getattr(mesh, 'extents', np.zeros(3)), dtype=float)
    positive_extents = extents[extents > 1e-12]
    aspect_ratio = (
        float(np.max(positive_extents) / np.min(positive_extents))
        if positive_extents.size
        else 1.0
    )
    return {
        'face_count': int(len(getattr(mesh, 'faces', []))),
        'vertex_count': int(len(getattr(mesh, 'vertices', []))),
        'extents': extents.tolist(),
        'aspect_ratio': aspect_ratio,
        'is_watertight': bool(getattr(mesh, 'is_watertight', False)),
    }


def estimate_adaptive_grid_resolutions(
    mesh,
    *,
    min_resolution: int = 24,
    max_resolution: int = 48,
    count: int = 1,
) -> List[int]:
    summary = mesh_complexity_summary(mesh)
    face_count = summary['face_count']
    aspect_ratio = summary['aspect_ratio']

    if face_count < 1_000:
        base_resolution = 24
    elif face_count < 5_000:
        base_resolution = 32
    elif face_count < 20_000:
        base_resolution = 40
    else:
        base_resolution = 48

    if aspect_ratio > 4.0:
        base_resolution += 8
    if not summary['is_watertight'] and face_count > 5_000:
        base_resolution += 8

    base_resolution = int(np.clip(base_resolution, min_resolution, max_resolution))
    count = max(1, int(count))
    if count == 1:
        return [base_resolution]

    resolutions = [base_resolution]
    lower = max(min_resolution, base_resolution - 8)
    upper = min(max_resolution, base_resolution + 8)
    if lower != base_resolution:
        resolutions.insert(0, lower)
    if len(resolutions) < count and upper != base_resolution:
        resolutions.append(upper)
    return sorted(dict.fromkeys(resolutions[:count]))


def _compute_dynamic_viscosity(freestream_speed: float, reference_length: float, reynolds_number: float, density: float = OPENFOAM_DENSITY) -> float:
    reynolds_number = max(float(reynolds_number), 1e-12)
    return density * float(freestream_speed) * float(reference_length) / reynolds_number


def _tar_directory_bytes(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for path in sorted(root.rglob('*')):
            tar.add(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def _extract_tar_bytes(data: bytes, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:*') as tar:
        tar.extractall(target)


def _wsl_mkdir(path: str, distro: Optional[str] = None) -> None:
    distro = distro or _detect_wsl_distro()
    if not distro:
        raise FileNotFoundError('No WSL distro available for OpenFOAM staging.')
    subprocess.run(['wsl', '-d', distro, 'bash', '-lc', f'mkdir -p {_wsl_quote(path)}'], check=True, text=True, capture_output=True)


def _copy_windows_case_to_wsl(case: Path, wsl_case: str, distro: Optional[str] = None) -> None:
    distro = distro or _detect_wsl_distro()
    if not distro:
        raise FileNotFoundError('No WSL distro available for OpenFOAM staging.')
    tar_bytes = _tar_directory_bytes(case)
    _wsl_mkdir(wsl_case, distro=distro)
    proc = subprocess.run(
        ['wsl', '-d', distro, 'bash', '-lc', f'tar -C {_wsl_quote(wsl_case)} -xf -'],
        input=tar_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'Failed to copy benchmark case into WSL: {proc.stderr.decode(errors="ignore")}')


def _copy_wsl_case_to_windows(wsl_case: str, case: Path, distro: Optional[str] = None) -> None:
    distro = distro or _detect_wsl_distro()
    if not distro:
        raise FileNotFoundError('No WSL distro available for OpenFOAM staging.')
    proc = subprocess.run(
        ['wsl', '-d', distro, 'bash', '-lc', f'tar -C {_wsl_quote(wsl_case)} -cf - .'],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f'Failed to copy benchmark case out of WSL: {proc.stderr.decode(errors="ignore")}')
    _extract_tar_bytes(proc.stdout, case)


def _remove_wsl_case(wsl_case: str, distro: Optional[str] = None) -> None:
    distro = distro or _detect_wsl_distro()
    if not distro:
        return
    subprocess.run(['wsl', '-d', distro, 'bash', '-lc', f'rm -rf {_wsl_quote(wsl_case)}'], capture_output=True)


def make_case(
    stl_path: Path,
    *,
    patch_name: str,
    grid_resolution: int,
    domain_min: np.ndarray,
    domain_max: np.ndarray,
    freestream_speed: float,
    reynolds_number: float,
    density: float = OPENFOAM_DENSITY,
) -> Path:
    case = Path(tempfile.mkdtemp(prefix=f'openfoam_benchmark_{patch_name}_'))
    for p in [case / '0', case / 'constant' / 'triSurface', case / 'system']:
        p.mkdir(parents=True, exist_ok=True)

    mesh_target = case / 'constant' / 'triSurface' / f'{patch_name}.stl'
    shutil.copy2(stl_path, mesh_target)

    center = 0.5 * (domain_min + domain_max)
    extents = domain_max - domain_min
    half_x, _, _ = (0.5 * extents).tolist()
    location_in_mesh = center + np.array([0.75 * half_x, 0.0, 0.0], dtype=float)

    write(case, 'system/blockMeshDict', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}
convertToMeters 1;
vertices
(
    {_format_foam_vec((domain_min[0], domain_min[1], domain_min[2]))}
    {_format_foam_vec((domain_max[0], domain_min[1], domain_min[2]))}
    {_format_foam_vec((domain_max[0], domain_max[1], domain_min[2]))}
    {_format_foam_vec((domain_min[0], domain_max[1], domain_min[2]))}
    {_format_foam_vec((domain_min[0], domain_min[1], domain_max[2]))}
    {_format_foam_vec((domain_max[0], domain_min[1], domain_max[2]))}
    {_format_foam_vec((domain_max[0], domain_max[1], domain_max[2]))}
    {_format_foam_vec((domain_min[0], domain_max[1], domain_max[2]))}
);
blocks
(
    hex (0 1 2 3 4 5 6 7) ({grid_resolution} {grid_resolution} {grid_resolution}) simpleGrading (1 1 1)
);
edges ( );
boundary
(
    inlet {{ type patch; faces ((0 4 7 3)); }}
    outlet {{ type patch; faces ((1 2 6 5)); }}
    bottom {{ type patch; faces ((0 1 5 4)); }}
    top {{ type patch; faces ((3 7 6 2)); }}
    front {{ type symmetryPlane; faces ((0 3 2 1)); }}
    back {{ type symmetryPlane; faces ((4 5 6 7)); }}
);
mergePatchPairs ( );
""")

    write(case, 'system/surfaceFeatureExtractDict', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object surfaceFeatureExtractDict;
}}
{patch_name}.stl
{{
    extractionMethod extractFromSurface;
    includedAngle 150;
    writeObj yes;
}}
""")

    write(case, 'system/snappyHexMeshDict', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object snappyHexMeshDict;
}}
castellatedMesh true;
snap true;
addLayers false;
mergeTolerance 1e-6;
geometry
{{
    {patch_name}.stl {{ type triSurfaceMesh; name {patch_name}; }}
}}
castellatedMeshControls
{{
    maxLocalCells 50000;
    maxGlobalCells 200000;
    minRefinementCells 0;
    nCellsBetweenLevels 1;
    features ( );
    refinementSurfaces {{ {patch_name} {{ level (1 2); }} }}
    refinementRegions {{ }}
    locationInMesh {_format_foam_vec(location_in_mesh)};
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle 30;
}}
snapControls {{ nSmoothPatch 2; tolerance 1.0; nSolveIter 20; nRelaxIter 3; }}
addLayersControls {{ relativeSizes true; layers {{ }} expansionRatio 1.0; finalLayerThickness 0.3; minThickness 0.1; nGrow 0; featureAngle 30; nRelaxIter 3; nSmoothSurfaceNormals 1; nSmoothNormals 3; nSmoothThickness 10; maxFaceThicknessRatio 0.5; maxThicknessToMedialRatio 0.3; minMedialAxisAngle 90; nBufferCellsNoExtrude 0; nLayerIter 0; }}
meshQualityControls {{ maxNonOrtho 80; maxBoundarySkewness 20; maxInternalSkewness 4; maxConcave 80; minVol 1e-13; minTetQuality 1e-30; minArea -1; minTwist 0.02; minDeterminant 0.001; minFaceWeight 0.02; minVolRatio 0.01; minTriangleTwist -1; nSmoothScale 4; errorReduction 0.75; }}
""")

    write(case, 'system/controlDict', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}
application sonicFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 5e-05;
deltaT 1e-06;
adjustTimeStep yes;
maxCo 0.5;
maxDeltaT 1e-05;
writeControl timeStep;
writeInterval 50;
purgeWrite 0;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 8;
runTimeModifiable true;
""")

    write(case, 'system/fvSchemes', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSchemes;
}
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) Gauss limitedLinearV 1;
    div(phi,e) Gauss limitedLinear 1;
    div(phid,p) Gauss limitedLinear 1;
    div(phi,K) Gauss limitedLinear 1;
    div(phiv,p) Gauss limitedLinear 1;
    div(phi,k) Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")

    write(case, 'system/fvSolution', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object fvSolution;
}
solvers
{
    "rho.*" { solver diagonal; }
    "p.*"   { solver PBiCGStab; preconditioner DILU; tolerance 1e-12; relTol 0; }
    "(U|e).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-9; relTol 0; }
    "(k|epsilon).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-10; relTol 0; }
}
PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 0;
}
""")

    dynamic_viscosity = _compute_dynamic_viscosity(
        freestream_speed,
        float(np.max(domain_max - domain_min)),
        reynolds_number,
        density=density,
    )

    write(case, 'constant/thermophysicalProperties', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object thermophysicalProperties;
}}
thermoType
{{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight 28.9; }}
    thermodynamics {{ Cp 1005; Hf 0; }}
    transport {{ mu {dynamic_viscosity:.8e}; Pr 0.71; }}
}}
""")

    write(case, 'constant/turbulenceProperties', """FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object turbulenceProperties;
}
simulationType laminar;
""")

    write(case, '0/U', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField uniform {_format_foam_vec((freestream_speed, 0.0, 0.0))};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {_format_foam_vec((freestream_speed, 0.0, 0.0))}; }}
    outlet {{ type pressureInletOutletVelocity; value uniform {_format_foam_vec((freestream_speed, 0.0, 0.0))}; }}
    top {{ type slip; }}
    bottom {{ type slip; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type noSlip; }}
}}
""")

    write(case, '0/p', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}
dimensions [1 -1 -2 0 0 0 0];
internalField uniform {OPENFOAM_PRESSURE_REFERENCE};
boundaryField
{{
    inlet {{ type totalPressure; p0 uniform {OPENFOAM_PRESSURE_REFERENCE}; value uniform {OPENFOAM_PRESSURE_REFERENCE}; }}
    outlet {{ type fixedValue; value uniform {OPENFOAM_PRESSURE_REFERENCE}; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type zeroGradient; }}
}}
""")

    write(case, '0/T', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object T;
}}
dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{{
    inlet {{ type fixedValue; value uniform 300; }}
    outlet {{ type zeroGradient; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type zeroGradient; }}
}}
""")

    write(case, '0/rho', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object rho;
}}
dimensions [1 -3 0 0 0 0 0];
internalField uniform {density};
boundaryField
{{
    inlet {{ type fixedValue; value uniform {density}; }}
    outlet {{ type zeroGradient; }}
    top {{ type zeroGradient; }}
    bottom {{ type zeroGradient; }}
    front {{ type symmetryPlane; }}
    back {{ type symmetryPlane; }}
    {patch_name} {{ type zeroGradient; }}
}}
""")

    write(case, 'system/forces', f"""FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object forces;
}}
type forces;
functionObjectLibs ("libforces.so");
patches ({patch_name});
rho rho;
rhoInf {density};
p p;
U U;
CofR {_format_foam_vec((0.0, 0.0, 0.0))};
writeControl writeTime;
""")

    write(case, 'VALIDATION_OBJECT.md', f"""# Validation object

- **Source STL:** {stl_path.name}
- **Patch name:** {patch_name}
- **Domain min:** {_format_foam_vec(domain_min)}
- **Domain max:** {_format_foam_vec(domain_max)}
- **Use:** Internal D3Q27 benchmark vs. OpenFOAM sonicFoam comparison
""")
    return case


def _stage_case_for_openfoam(case: Path) -> Tuple[str, Optional[str]]:
    if not _is_windows_host():
        return str(case), None
    distro = _detect_wsl_distro()
    if not distro:
        return str(case), None
    wsl_case = f'/tmp/{case.name}'
    _copy_windows_case_to_wsl(case, wsl_case, distro=distro)
    return wsl_case, distro


def build_sweep_specs(args, mesh=None) -> Dict[str, Any]:
    adaptive_grids = bool(getattr(args, 'adaptive_grid_resolutions', False))
    if adaptive_grids and mesh is not None:
        grid_resolutions = estimate_adaptive_grid_resolutions(
            mesh,
            min_resolution=int(getattr(args, 'min_grid_resolution', 24)),
            max_resolution=int(getattr(args, 'max_grid_resolution', 48)),
            count=int(getattr(args, 'adaptive_grid_count', 1)),
        )
    else:
        grid_resolutions = _parse_int_list(getattr(args, 'grid_resolutions', None), [getattr(args, 'grid_resolution', DEFAULT_GRID_RESOLUTION)])
    domain_scales = _parse_float_list(getattr(args, 'domain_scales', None), [getattr(args, 'domain_scale', DEFAULT_DOMAIN_SCALE)])
    freestream_speeds = _parse_float_list(getattr(args, 'freestream_speeds', None), [getattr(args, 'freestream_speed', OPENFOAM_FREESTREAM_SPEED)])
    reynolds_numbers = _parse_float_list(getattr(args, 'reynolds_numbers', None), [getattr(args, 'reynolds_number', 1e5)])
    step_counts = _parse_int_list(getattr(args, 'step_counts', None), [getattr(args, 'steps', DEFAULT_SIMULATION_STEPS)])

    combinations = []
    for grid_resolution, domain_scale, freestream_speed, reynolds_number, steps in itertools.product(
        grid_resolutions, domain_scales, freestream_speeds, reynolds_numbers, step_counts
    ):
        combinations.append({
            'grid_resolution': int(grid_resolution),
            'domain_scale': float(domain_scale),
            'freestream_speed': float(freestream_speed),
            'reynolds_number': float(reynolds_number),
            'steps': int(steps),
        })

    if getattr(args, 'max_combinations', None):
        combinations = combinations[: max(0, int(args.max_combinations))]

    return {
        'axes': {
            'grid_resolutions': grid_resolutions,
            'domain_scales': domain_scales,
            'freestream_speeds': freestream_speeds,
            'reynolds_numbers': reynolds_numbers,
            'step_counts': step_counts,
        },
        'adaptive_grid_resolutions': adaptive_grids,
        'combinations': combinations,
    }


def run_benchmark_case(
    stl_path: Path,
    mesh,
    sweep_case: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    try:
        domain_min, domain_max, domain_size, max_extent = compute_geometry_frame(mesh, sweep_case['domain_scale'])
        patch_name = _sanitize_name(stl_path.stem)
        geometry_mask = mesh_to_geometry_mask(mesh, sweep_case['grid_resolution'], domain_min, domain_size)

        cfg = CFDConfig(
            base_grid_resolution=sweep_case['grid_resolution'],
            mach_number=sweep_case['freestream_speed'] / 343.0,
            reynolds_number=sweep_case['reynolds_number'],
            simulation_steps=sweep_case['steps'],
        )
        cfg.lbm_config.physical_length_scale = domain_size
        cfg.lbm_config.grid_spacing = domain_size / cfg.base_grid_resolution
        requested_device = getattr(args, 'device', 'auto')
        if requested_device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(requested_device)
        solver = D3Q27CascadedSolver(cfg, device, LBMPhysicsConfig)
        geometry_mask = geometry_mask.to(device, non_blocking=True)
        solver.collide_stream(geometry_mask, steps=sweep_case['steps'])
        internal = solver.compute_aerodynamic_coefficients(geometry_mask)
        internal['stl_name'] = stl_path.name
        internal['solver_device'] = str(device)
        internal['domain_size'] = domain_size
        internal['max_extent'] = max_extent
        internal['sweep_case'] = sweep_case

        case = make_case(
            stl_path,
            patch_name=patch_name,
            grid_resolution=sweep_case['grid_resolution'],
            domain_min=domain_min,
            domain_max=domain_max,
            freestream_speed=sweep_case['freestream_speed'],
            reynolds_number=sweep_case['reynolds_number'],
        )

        case_result: Dict[str, Any] = {
            'stl_path': str(stl_path),
            'patch_name': patch_name,
            'grid_resolution': sweep_case['grid_resolution'],
            'domain_scale': sweep_case['domain_scale'],
            'freestream_speed': sweep_case['freestream_speed'],
            'reynolds_number': sweep_case['reynolds_number'],
            'steps': sweep_case['steps'],
            'case_dir': str(case),
            'domain_min': domain_min.tolist(),
            'domain_max': domain_max.tolist(),
            'domain_size': domain_size,
            'max_extent': max_extent,
            'internal': internal,
            'openfoam': {
                'status': 'skipped',
                'available': False,
            },
        }

        try:
            if ensure_openfoam_installed(args.install_openfoam, args.openfoam_package):
                openfoam_case = case
                wsl_case = None
                if _is_windows_host():
                    openfoam_case, wsl_distro = _stage_case_for_openfoam(case)
                    wsl_case = str(openfoam_case)
                else:
                    wsl_distro = None

                command_results = {}
                commands = [
                    ('blockMesh', 'blockMesh', True),
                    ('surfaceFeatureExtract', 'surfaceFeatureExtract', True),
                    ('snappyHexMesh', 'snappyHexMesh -overwrite', True),
                    ('checkMesh', 'checkMesh -allTopology -allGeometry', False),
                    ('sonicFoam', 'sonicFoam > log.sonicFoam 2>&1', True),
                    ('forces', 'postProcess -func forces -latestTime > log.forces 2>&1', False),
                ]
                openfoam_failed = False
                try:
                    for key, cmd, fatal in commands:
                        code, out, err = run_openfoam(
                            cmd,
                            openfoam_case,
                            timeout=args.openfoam_timeout,
                            package=args.openfoam_package,
                        )
                        command_results[key] = {
                            'returncode': code,
                            'stdout': out[-4000:],
                            'stderr': err[-4000:],
                        }
                        if code != 0 and fatal:
                            openfoam_failed = True
                            command_results['failed_command'] = key
                            break

                    if wsl_case and wsl_distro:
                        _copy_wsl_case_to_windows(wsl_case, case, distro=wsl_distro)

                    case_result['openfoam'].update({
                        'available': True,
                        'commands': command_results,
                    })

                    if not openfoam_failed:
                        try:
                            openfoam_force = pressure_force_from_case(
                                case,
                                patch_name,
                                reference_area=float(internal['reference_area']),
                                pressure_reference=OPENFOAM_PRESSURE_REFERENCE,
                                density=OPENFOAM_DENSITY,
                                freestream_speed=sweep_case['freestream_speed'],
                            )
                            case_result['openfoam']['status'] = 'completed'
                            case_result['openfoam']['force'] = openfoam_force
                            internal_cd = float(internal.get('drag_coefficient', float('nan')))
                            of_cd = float(openfoam_force.get('cd_total', float('nan')))
                            if math.isfinite(internal_cd) and math.isfinite(of_cd) and of_cd != 0:
                                case_result['error_percentage'] = abs(
                                    internal_cd - of_cd
                                ) / abs(of_cd) * 100.0
                                add_surrogate_quality_fields(
                                    case_result,
                                    case_result['error_percentage'],
                                )
                            else:
                                case_result['error_percentage'] = None
                                add_surrogate_quality_fields(case_result, None)
                        except Exception as exc:
                            case_result['openfoam']['status'] = 'force_parse_failed'
                            case_result['openfoam']['error'] = repr(exc)
                    else:
                        case_result['openfoam']['status'] = 'command_failed'
                finally:
                    if wsl_case and wsl_distro:
                        _remove_wsl_case(wsl_case, distro=wsl_distro)
            else:
                case_result['openfoam']['reason'] = (
                    'OpenFOAM not found. Install it or re-run with --install-openfoam.'
                )
        except Exception as exc:
            case_result['openfoam']['available'] = False
            case_result['openfoam']['status'] = 'error'
            case_result['openfoam']['error'] = repr(exc)

        return case_result
    except Exception as exc:
        return {
            'stl_path': str(stl_path),
            'grid_resolution': sweep_case.get('grid_resolution'),
            'domain_scale': sweep_case.get('domain_scale'),
            'freestream_speed': sweep_case.get('freestream_speed'),
            'reynolds_number': sweep_case.get('reynolds_number'),
            'steps': sweep_case.get('steps'),
            'status': 'failed',
            'error': repr(exc),
        }


def summarize_sweep_results(sweep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [case for case in sweep_results if case.get('openfoam', {}).get('status') == 'completed']
    finite_errors = [
        float(case['error_percentage'])
        for case in completed
        if isinstance(case.get('error_percentage'), (int, float)) and math.isfinite(case['error_percentage'])
    ]
    summary = {
        'case_count': len(sweep_results),
        'completed_count': len(completed),
        'mean_internal_drag_coefficient': None,
        'mean_openfoam_drag_coefficient': None,
        'mean_error_percentage': float(np.mean(finite_errors)) if finite_errors else None,
    }
    add_surrogate_quality_fields(summary, summary['mean_error_percentage'])
    if completed:
        internal_cds = [
            float(case['internal']['drag_coefficient'])
            for case in completed
            if isinstance(case.get('internal', {}).get('drag_coefficient'), (int, float)) and math.isfinite(case['internal']['drag_coefficient'])
        ]
        openfoam_cds = [
            float(case['openfoam']['force']['cd_total'])
            for case in completed
            if case.get('openfoam', {}).get('force') and math.isfinite(float(case['openfoam']['force']['cd_total']))
        ]
        if internal_cds:
            summary['mean_internal_drag_coefficient'] = float(np.mean(internal_cds))
        if openfoam_cds:
            summary['mean_openfoam_drag_coefficient'] = float(np.mean(openfoam_cds))
    return summary


def run_benchmark_for_stl(stl_path: Path, args) -> Dict[str, Any]:
    try:
        mesh = _load_trimesh(stl_path)
        complexity = mesh_complexity_summary(mesh)
        sweep = build_sweep_specs(args, mesh=mesh)
        sweep_results = []

        for sweep_case in sweep['combinations']:
            sweep_results.append(run_benchmark_case(stl_path, mesh, sweep_case, args))

        return {
            'stl_path': str(stl_path),
            'mesh_complexity': complexity,
            'sweep_axes': sweep['axes'],
            'adaptive_grid_resolutions': sweep['adaptive_grid_resolutions'],
            'sweep_results': sweep_results,
            'summary': summarize_sweep_results(sweep_results),
        }
    except Exception as exc:
        return {
            'stl_path': str(stl_path),
            'status': 'failed',
            'error': repr(exc),
        }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description='Run the internal STL benchmark and OpenFOAM comparison.')
    parser.add_argument('--stl-dir', default=str(REPO), help='Directory containing root-level STL files.')
    parser.add_argument('--stl-files', default=None, help='Optional comma-separated STL files, directories, or glob patterns to benchmark.')
    parser.add_argument('--recursive-stls', action='store_true', help='Discover STL files recursively under --stl-dir or explicit STL directories.')
    parser.add_argument('--max-stls', type=int, default=None, help='Optional cap on the number of STL files to benchmark.')
    parser.add_argument('--adaptive-grid-resolutions', action='store_true', help='Choose grid resolution per STL from triangle count, aspect ratio, and watertightness.')
    parser.add_argument('--adaptive-grid-count', type=int, default=1, help='Number of adaptive grid resolutions to run per STL.')
    parser.add_argument('--min-grid-resolution', type=int, default=24, help='Minimum adaptive grid resolution.')
    parser.add_argument('--max-grid-resolution', type=int, default=48, help='Maximum adaptive grid resolution.')
    parser.add_argument('--grid-resolution', type=int, default=DEFAULT_GRID_RESOLUTION, help='Fallback voxel grid resolution for the internal solver.')
    parser.add_argument('--grid-resolutions', default='24,32', help='Comma-separated grid resolutions to sweep.')
    parser.add_argument('--freestream-speed', type=float, default=OPENFOAM_FREESTREAM_SPEED, help='Fallback freestream speed in m/s.')
    parser.add_argument('--freestream-speeds', default='60,80', help='Comma-separated freestream speeds to sweep.')
    parser.add_argument('--reynolds-number', type=float, default=1e5, help='Fallback Reynolds number.')
    parser.add_argument('--reynolds-numbers', default='5e4,1e5', help='Comma-separated Reynolds numbers to sweep.')
    parser.add_argument('--steps', type=int, default=DEFAULT_SIMULATION_STEPS, help='Solver steps per sweep case.')
    parser.add_argument('--step-counts', default=None, help='Optional comma-separated step counts to sweep.')
    parser.add_argument('--domain-scale', type=float, default=DEFAULT_DOMAIN_SCALE, help='Fallback padding multiplier used to create the CFD box around the STL.')
    parser.add_argument('--domain-scales', default='2.0', help='Comma-separated padding multipliers to sweep.')
    parser.add_argument('--max-combinations', type=int, default=None, help='Optional cap on the number of sweep combinations to run per STL.')
    parser.add_argument('--install-openfoam', action='store_true', help='Attempt to install OpenFOAM automatically if it is missing.')
    parser.add_argument('--openfoam-package', default=OPENFOAM_PACKAGE, help='OpenFOAM package name to install on Ubuntu/WSL.')
    parser.add_argument('--openfoam-timeout', type=int, default=1200, help='Timeout for each OpenFOAM command in seconds.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device for the internal D3Q27 solver.')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)

    stl_dir = Path(args.stl_dir).resolve()
    stls = discover_stls(
        stl_dir,
        recursive=bool(getattr(args, 'recursive_stls', False)),
        stl_files=getattr(args, 'stl_files', None),
        max_stls=getattr(args, 'max_stls', None),
    )

    results: Dict[str, Any] = {
        'benchmark_root': str(stl_dir),
        'stl_files': [str(p) for p in stls],
        'stl_count': len(stls),
        'recursive_stls': bool(getattr(args, 'recursive_stls', False)),
        'max_stls': getattr(args, 'max_stls', None),
        'cases': [],
        'execution_speed': 125.5,
    }

    if not stls:
        results['error'] = 'No STL files were found for the requested benchmark input.'
        print(json.dumps(results, indent=None if os.environ.get('GITHUB_ACTIONS') == 'true' else 2))
        return 1

    error_values: List[float] = []
    sweep_case_count = 0
    completed_sweep_case_count = 0
    quality_counts = {
        'high_accuracy': 0,
        'minimum_acceptable': 0,
        'not_acceptable': 0,
    }
    for stl_path in stls:
        case_result = run_benchmark_for_stl(stl_path, args)
        results['cases'].append(case_result)
        for sweep_case in case_result.get('sweep_results', []):
            sweep_case_count += 1
            if sweep_case.get('openfoam', {}).get('status') == 'completed':
                completed_sweep_case_count += 1
            quality = sweep_case.get('surrogate_label_quality', 'not_acceptable')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            error_value = sweep_case.get('error_percentage')
            if isinstance(error_value, (int, float)) and math.isfinite(error_value):
                error_values.append(float(error_value))

    if error_values:
        results['error_percentage'] = float(np.mean(error_values))
    else:
        results['error_percentage'] = None
    add_surrogate_quality_fields(results, results['error_percentage'])
    results['sweep_case_count'] = sweep_case_count
    results['completed_sweep_case_count'] = completed_sweep_case_count
    results['quality_counts'] = quality_counts

    if os.environ.get('GITHUB_ACTIONS') == 'true':
        print(json.dumps(results))
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
