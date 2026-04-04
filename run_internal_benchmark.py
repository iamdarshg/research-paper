from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path('CLI').resolve()))
from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig
from advanced_lbm_solver import D3Q27CascadedSolver

REPO = Path(__file__).resolve().parent
OF_ROOT = Path(os.environ.get('OPENFOAM_ROOT', '/home/darsh/.openclaw/openfoam/usr/share/openfoam'))
OF_CMD = f'source "{OF_ROOT / "etc" / "bashrc"}" >/dev/null 2>&1 && '


def run(cmd: str, cwd: Path, timeout: int = 600):
    proc = subprocess.run(['bash', '-lc', OF_CMD + cmd], cwd=cwd, text=True, capture_output=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def write(case: Path, rel: str, content: str) -> None:
    path = case / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def make_case() -> Path:
    case = Path(tempfile.mkdtemp(prefix='openfoam_sonic_cube_'))
    for p in [case / '0', case / 'constant' / 'triSurface', case / 'system']:
        p.mkdir(parents=True, exist_ok=True)

    # Simple cube STL centered at the origin
    write(case, 'constant/triSurface/cube.stl', """solid cube
facet normal 0 0 -1
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex 0.5 -0.5 -0.5
  vertex 0.5 0.5 -0.5
 endloop
endfacet
facet normal 0 0 -1
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex 0.5 0.5 -0.5
  vertex -0.5 0.5 -0.5
 endloop
endfacet
facet normal 0 0 1
 outer loop
  vertex -0.5 -0.5 0.5
  vertex 0.5 0.5 0.5
  vertex 0.5 -0.5 0.5
 endloop
endfacet
facet normal 0 0 1
 outer loop
  vertex -0.5 -0.5 0.5
  vertex -0.5 0.5 0.5
  vertex 0.5 0.5 0.5
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex 0.5 -0.5 0.5
  vertex 0.5 -0.5 -0.5
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex -0.5 -0.5 0.5
  vertex 0.5 -0.5 0.5
 endloop
endfacet
facet normal 0 1 0
 outer loop
  vertex -0.5 0.5 -0.5
  vertex 0.5 0.5 -0.5
  vertex 0.5 0.5 0.5
 endloop
endfacet
facet normal 0 1 0
 outer loop
  vertex -0.5 0.5 -0.5
  vertex 0.5 0.5 0.5
  vertex -0.5 0.5 0.5
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex -0.5 0.5 -0.5
  vertex -0.5 0.5 0.5
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex -0.5 -0.5 -0.5
  vertex -0.5 0.5 0.5
  vertex -0.5 -0.5 0.5
 endloop
endfacet
facet normal 1 0 0
 outer loop
  vertex 0.5 -0.5 -0.5
  vertex 0.5 0.5 0.5
  vertex 0.5 0.5 -0.5
 endloop
endfacet
facet normal 1 0 0
 outer loop
  vertex 0.5 -0.5 -0.5
  vertex 0.5 -0.5 0.5
  vertex 0.5 0.5 0.5
 endloop
endfacet
endsolid cube
""")

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
    locationInMesh (0 0 0);
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

    write(case, '0/p', """FoamFile
{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}
dimensions [1 -1 -2 0 0 0 0];
internalField uniform 101325;
boundaryField
{
    inlet { type totalPressure; p0 uniform 101325; value uniform 101325; }
    outlet { type fixedValue; value uniform 101325; }
    top { type zeroGradient; }
    bottom { type zeroGradient; }
    front { type symmetryPlane; }
    back { type symmetryPlane; }
    cube { type zeroGradient; }
}
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


def pressure_force_from_case(case: Path, patch_name: str = 'cube', p_ref: float = 101325.0) -> Dict[str, float]:
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
        p_gauge = p_cell - p_ref
        force += -p_gauge * sf

    rho = 1.225
    u_inf = 80.0
    q = 0.5 * rho * u_inf * u_inf
    area_ref = 1.0
    return {
        'pressure_force_x': float(force[0]),
        'pressure_force_y': float(force[1]),
        'pressure_force_z': float(force[2]),
        'cd_pressure': float(force[0] / (q * area_ref)),
        'cl_pressure': float(force[2] / (q * area_ref)),
        'time_dir': time_dir.name,
    }


def main():
    n = 16
    mask = torch.zeros((n, n, n), dtype=torch.float32)
    mask[5:11, 5:11, 5:11] = 1.0
    solver = D3Q27CascadedSolver(CFDConfig(base_grid_resolution=n, mach_number=0.08, reynolds_number=1e5, simulation_steps=20), torch.device('cpu'), LBMPhysicsConfig)
    solver.collide_stream(mask, steps=5)
    internal = solver.compute_aerodynamic_coefficients(mask)

    case = make_case()
    results = {'case_dir': str(case), 'internal': internal}
    commands = [
        'blockMesh',
        'surfaceFeatureExtract',
        'snappyHexMesh -overwrite',
        'checkMesh -allTopology -allGeometry',
        'sonicFoam > log.sonicFoam 2>&1',
    ]
    for cmd in commands:
        code, out, err = run(cmd, case, timeout=1200)
        results[f'cmd_{cmd.split()[0]}'] = {'returncode': code, 'stdout': out[-4000:], 'stderr': err[-4000:]}
        if code != 0:
            print(json.dumps(results, indent=2))
            return 1

    try:
        results['openfoam_force'] = pressure_force_from_case(case)
    except Exception as exc:
        results['force_error'] = repr(exc)
        print(json.dumps(results, indent=2))
        return 1

    print(json.dumps(results, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
