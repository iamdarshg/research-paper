"""Dockerized OpenFOAM CFD runner for folded plane."""
import yaml
import numpy as np
import trimesh
import docker
import os
import tempfile
import shutil
from pathlib import Path
from src.folding.folder import fold_sheet
from src.surrogate.aero_model import surrogate_cfd
from src.folding.sheet import load_config

def load_config():
    config_path = Path(__file__).parent.parent.parent / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

def prepare_case(stl_path, case_dir, num_cells, v_inf, length):
    """
    Copy STL, setup blockMeshDict, snappyHexMeshDict, surfaceFeatureExtractDict,
    and all other OpenFOAM files for a simple case.
    """
    config = load_config()

    # Basic case template directories
    (case_dir / '0').mkdir(exist_ok=True)
    (case_dir / 'constant').mkdir(exist_ok=True)
    # Ensure constant/triSurface exists before copying STL
    (case_dir / 'constant/triSurface').mkdir(parents=True, exist_ok=True)
    (case_dir / 'system').mkdir(exist_ok=True)
    # The stl_path passed to prepare_case is expected to be `case_dir / 'constant/triSurface/airplane.stl'`,
    # so the copy should not be necessary here if `sheet.export` is already writing directly.
    # However, if stl_path is initially a source path, then this copy is correct.
    # Based on run_openfoam_cfd, `stl_path` is the *destination* path for `sheet.export`.
    # So, the `shutil.copy` is redundant/incorrect here.
    # The `stl_path` passed to this function `prepare_case` is `case_dir / 'constant/triSurface/airplane.stl'`.
    # So, it doesn't need to be copied. It will be created by `sheet.export` directly.
    # Let's remove the redundant shutil.copy and ensure `case_dir / 'constant/triSurface'` exists.
    # `stl_path` in `run_openfoam_cfd` is already set as `case_dir / 'constant/triSurface/airplane.stl'`.
    # So, `prepare_case` should just ensure the directory structure is there.
    # This `shutil.copy` line should be removed or commented out.
    # It also assumes `stl_path` is a *source* path, but it's a *destination*.
    # Let's verify `run_openfoam_cfd` again.
    
    # In `run_openfoam_cfd`:
    # stl_path = case_dir / 'constant/triSurface/airplane.stl'
    # sheet.export(stl_path)
    # This is correct. So the `shutil.copy` in `prepare_case` is wrong.
    
    # Remove this: `shutil.copy(stl_path, case_dir / 'constant/triSurface/airplane.stl')`
    # The parent directories are already ensured by `mkdir(parents=True)`.

    # blockMeshDict - dynamic domain based on v_inf for free stream
    bbox_file = stl_path.with_suffix('') / '_bbox.txt'
    # Fallback bbox values if file doesn't exist
    bbox_min_default = np.array([-0.15, -0.125, -0.1], dtype=np.float32)
    bbox_max_default = np.array([0.35, 0.25, 0.15], dtype=np.float32)

    if bbox_file.exists():
        bbox_data = np.loadtxt(bbox_file, dtype=np.float32)
        bbox_min, bbox_max = bbox_data[0], bbox_data[1]
    else:
        bbox_min, bbox_max = bbox_min_default, bbox_max_default
    
    span = bbox_max[1] - bbox_min[1]
    # length = bbox_max[0] - bbox_min[0] # length is passed as an argument
    height = bbox_max[2] - bbox_min[2]
    
    domain_l = 2 * length  # Length of domain
    domain_h = 10 * height  # Height
    domain_w = 5 * span  # Width

    nx = int(num_cells**(1/3)) * 2
    ny = int(num_cells**(1/3)) * 3  # More in width for span
    nz = int(num_cells**(1/3)) * 1.5

    block_str = f"""
/*--------------------------------*- C++ -*----------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
scale 1;

// vertices
vertices
(
    (-{domain_l} -{domain_w/2} -{domain_h/2})  // 0
    ({3*length} -{domain_w/2} -{domain_h/2})   // 1
    ({3*length} {domain_w/2} -{domain_h/2})    // 2
    (-{domain_l} {domain_w/2} -{domain_h/2})   // 3
    (-{domain_l} -{domain_w/2} {domain_h/2})   // 4
    ({3*length} -{domain_w/2} {domain_h/2})    // 5
    ({3*length} {domain_w/2} {domain_h/2})     // 6
    (-{domain_l} {domain_w/2} {domain_h/2})    // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (2 1 1)
);

edges ();

boundary
(
    inlet
    {{
        type patch;
        faces ((0 4 3 7));
    }}
    outlet
    {{
        type patch;
        faces ((1 5 6 2));
    }}
    topBottom
    {{
        type symmetryPlane;
        faces ((4 5 6 7) (0 1 2 3));
    }}
    side
    {{
        type symmetryPlane;
        faces ((0 1 5 4));
    }}
    airplane
    {{
        type wall;
        faces ();
    }}
);

mergePatchPairs ();
"""
    with open(case_dir / 'system/blockMeshDict', 'w') as f:
        f.write(block_str)

    # controlDict with forces
    control_dict = """
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   200;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;

functions
{{
    forces
    {{
        type forces;
        functionObjectLibs ("libforces.so"); // Changed to .so for Linux/Docker
        patches ("airplane");
        rho rhoInf;
        rhoInf 1.225;
        log true;
    }}
}}
"""
    with open(case_dir / 'system/controlDict', 'w') as f:
        f.write(control_dict)

    # fvSchemes
    fv_schemes = """
ddtSchemes
{{
    default steadyState;
}}

gradSchemes
{{
    default Gauss linear;
}}

divSchemes
{{
    default none;
    div(phi,U) Gauss linearUpwind grad(U);
    div(phi,k) Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div(phi,omega) Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default Gauss linear corrected;
}}
"""
    with open(case_dir / 'system/fvSchemes', 'w') as f:
        f.write(fv_schemes)

    # fvSolution
    fv_solution = """
solvers
{{
    p
    {{
        solver PCG;
        preconditioner DIC;
        tolerance 1e-6;
        relTol 0.01;
    }}
    U
    {{
        solver PBiCG;
        preconditioner DILU;
        tolerance 1e-6;
        relTol 0.01;
    }}
    k
    {{
        solver PBiCG;
        preconditioner DILU;
        tolerance 1e-7;
        relTol 0.01;
    }}
    epsilon
    {{
        solver PBiCG;
        preconditioner DILU;
        tolerance 1e-7;
        relTol 0.01;
    }}
    omega
    {{
        solver PBiCG;
        preconditioner DILU;
        tolerance 1e-7;
        relTol 0.01;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
}}
"""
    with open(case_dir / 'system/fvSolution', 'w') as f:
        f.write(fv_solution)

    # U initial
    u_str = f"""
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({v_inf} 0 0);
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value uniform ({v_inf} 0 0);
    }}
    outlet
    {{
        type zeroGradient;
    }}
    topBottom
    {{
        type symmetryPlane;
    }}
    side
    {{
        type symmetryPlane;
    }}
    airplane
    {{
        type noSlip;
    }}
}}
"""
    with open(case_dir / '0/U', 'w') as f:
        f.write(u_str)

    # p initial
    p_str = """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{{
    inlet
    {{
        type zeroGradient;
    }}
    outlet
    {{
        type fixedValue;
        value uniform 0;
    }}
    topBottom
    {{
        type symmetryPlane;
    }}
    side
    {{
        type symmetryPlane;
    }}
    airplane
    {{
        type zeroGradient;
    }}
}}
"""
    with open(case_dir / '0/p', 'w') as f:
        f.write(p_str)

    # transportProperties
    transport = """
transportModel Newtonian;
nu nu [ 0 2 -1 0 0 0 0 ] 1.5e-5;
"""
    with open(case_dir / 'constant/transportProperties', 'w') as f:
        f.write(transport)

    # write snappyHexMeshDict
    snappy_hex_mesh_dict = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 2000000;
    minRefinementCells 0;
    nCellsBetweenLevels 1;
    features
    (
        {{
            file "airplane.eMesh";
            level 3;
        }}
    );
    refinementSurfaces
    {{
        airplane
        {{
            level (3 3);
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}};
    locationInMesh ({length/2} 0 0); // Location needs to be inside the fluid domain, ensure it is.
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 4.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}
addLayersControls
{{
    relativeSizes true;
    layers
    {{
        airplane
        {{
            nSurfaceLayers 1;
        }}
    }}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 30;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessRatio 0.1;
    nSmoothNormalThickness 5;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minFlatness 0.5;
    minVol 1e-13;
    minArea -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}
writeFlags
(
    scalarLevels
    internalEdges
);
mergeTolerance 1e-6;
"""
    with open(case_dir / 'system/snappyHexMeshDict', 'w') as f:
        f.write(snappy_hex_mesh_dict)

    # surfaceFeatureExtractDict
    surface_feature_dict = """
features
(
    {{
        name "airplane_features";
        file "airplane.stl";
        extractFromSurface true;
        includedAngle 150;
    }}
);
"""
    with open(case_dir / 'system/surfaceFeatureExtractDict', 'w') as f:
        f.write(surface_feature_dict)

def run_openfoam_cfd(action, state, fidelity='low'):
    """
    Run CFD, return aero coeffs.
    
    Args:
        action: fold action
        state: env state
        fidelity: 'low'/'high' -> cells
    
    Returns:
        dict cl, cd, range_est
    """
    config = load_config()
    if fidelity == 'low':
        cells = config['cfd']['low_cells']
    else:
        cells = min(config['project']['max_cells_cfd'], config['cfd']['low_cells'] + config['cfd']['cell_increment'])
    
    with tempfile.TemporaryDirectory() as tmpdir:
        case_dir = Path(tmpdir)
        resolution = 50
        
        # Fold and export STL
        sheet = fold_sheet(action, resolution=resolution)
        stl_path = case_dir / 'constant/triSurface/airplane.stl' # Updated path
        stl_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist
        sheet.export(stl_path)
        
        # Get length from sheet bounds
        bbox = sheet.bounds
        length = bbox[1][0] - bbox[0][0] # Calculate length here
        
        v_inf = state['throw_speed_mps']
        prepare_case(stl_path, case_dir, cells, v_inf, length) # Pass length to prepare_case
        
        client = docker.from_env()
        
        # Debugging the OpenFOAM environment sourcing and command execution within the container
        cfd_command = [
            '/bin/bash', '-exc',
            (
                'echo "Attempting to source OpenFOAM bashrc..." && '
                'ls /opt/openfoam11/etc/bashrc || { echo "Error: bashrc not found at /opt/openfoam11/etc/bashrc"; exit 1; } && '
                'source /opt/openfoam11/etc/bashrc && '
                'echo "bashrc sourced. Checking OpenFOAM environment variables:" && '
                'env | grep WM_PROJECT || { echo "Error: OpenFOAM environment variables not set after sourcing"; exit 1; } && '
                'echo "WM_PROJECT_DIR: $WM_PROJECT_DIR" && '
                'echo "Running blockMesh -help to verify command availability..." && '
                'blockMesh -help || { echo "Error: blockMesh command failed to execute after sourcing"; exit 1; } && '
                
                'echo "OpenFOAM environment confirmed. Proceeding with full CFD pipeline..." && '
                'cd /case && '
                'blockMesh || { echo "blockMesh failed"; exit 1; } && '
                'surfaceFeatureExtract || { echo "surfaceFeatureExtract failed"; exit 1; } && '
                'snappyHexMesh -overwrite || { echo "snappyHexMesh failed"; exit 1; } && '
                'checkMesh || { echo "checkMesh failed"; exit 1; } && '
                'simpleFoam || { echo "simpleFoam failed"; exit 1; } && '
                'postProcess -func forces -latestTime || { echo "postProcess failed"; exit 1; } && '
                'ls -R postProcessing || { echo "ls postProcessing failed"; exit 1; } && '
                'tail -n 100 postProcessing/id/forces/0/forces.dat || true'
            )
        ]

        # Run the container and capture its logs
        result = client.containers.run(
            config['cfd']['docker_image'],
            command=cfd_command,
            volumes={str(case_dir): {'bind': '/case', 'mode': 'rw'}},
            working_dir='/case',
            remove=True,
            detach=False, # Run in foreground to get all logs directly
            mem_limit='4g' # Increased memory limit due to snappyHexMesh
        )
        
        container_logs = result.decode('utf-8') # result is the raw output (bytes)
        print("CFD Container Logs:\n", container_logs)

        # Parse forces.dat for CD/CL
        forces_path = case_dir / 'postProcessing/id/forces/0/forces.dat'
        if forces_path.exists():
            with open(forces_path) as f:
                lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
            if lines:
                values = lines[-1].split()
                # Assuming forces.dat format: time (Fx Fy Fz) (Mx My Mz)
                # We need to extract Fx and Fz for Cd and Cl
                total_force_line = lines[-1]
                # Split the line by parentheses and then by spaces to get individual values
                parts = total_force_line.split('(')
                forces_str = parts[1].split(')')[0].strip()
                moments_str = parts[2].split(')')[0].strip()

                fx, fy, fz = map(float, forces_str.split())
                # For a typical airfoil in x-z plane, Fx is drag, Fz is lift.
                # However, OpenFOAM's forces function might output in the global coordinate system.
                # Assuming lift is primarily in Z and drag in X against flow for simplicity.
                # Adjust if OpenFOAM's default force reporting convention is different.
                tx = -fx # Drag
                tz = fz # Lift

                rho = state['air_density_kgm3'] # Use state variable for density
                v_inf = state['throw_speed_mps']
                
                # Approximate projected area
                # Use actual sheet bounds for more accurate projected area
                Ap = (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1]) * 0.8 # Assume 0.8 fill factor
                
                dyn_press = 0.5 * rho * v_inf ** 2 * Ap
                
                # Avoid division by zero
                if dyn_press <= 0:
                    print("WARNING: Dynamic pressure is zero or negative. CFD result invalid.")
                    return {'cl': 0.0, 'cd': 0.0, 'range_est': 0.0}

                cd = tx / dyn_press  # Fx (drag)
                cl = tz / dyn_press  # Fz (lift)
            else:
                # If forces.dat exists but is empty/malformed, this indicates a CFD failure.
                # Do not fallback to surrogate, instead raise an error or return zero.
                print("WARNING: forces.dat found but empty or malformed. CFD calculation failed.")
                return {'cl': 0.0, 'cd': 0.0, 'range_est': 0.0}
        else:
            # If forces.dat does not exist, CFD run likely failed.
            print("ERROR: forces.dat not found. CFD simulation failed to produce results.")
            # Do not fallback to surrogate as per task. Return zero results.
            return {'cl': 0.0, 'cd': 0.0, 'range_est': 0.0}
        
        # Est range from CFD results
        # Re-using the simplified range formula with a placeholder for glide angle
        v_inf = state['throw_speed_mps']
        ld = cl / cd if cd > 0 else 0.0
        g = 9.81
        glide_angle_rad = np.deg2rad(10) # Assuming 10 degrees as a typical glide angle
        range_est = ld * v_inf**2 * np.sin(2 * glide_angle_rad) / g
        
        return {'cl': cl, 'cd': cd, 'range_est': range_est}
