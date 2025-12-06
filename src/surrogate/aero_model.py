"""Physics-inspired CFD surrogate for paper airplane aerodynamics with GPU optimization."""
import yaml
import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Any, Dict, Optional, Tuple # Import Any for type hinting
from ..folding.sheet import load_config
import subprocess
import json


CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FluidX3D integration flag
USE_FLUIDX3D = True
FLUIDX3D_EXE = None

def compute_aero_features(mesh):
    """
    Extract key aero features from 3D mesh, performing calculations on GPU where possible.
    The initial mesh properties (vertices, normals, area) are still CPU-derived from trimesh.
    """
    # Convert mesh attributes to PyTorch tensors on the device
    verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE)
    normals_t = torch.tensor(mesh.face_normals, dtype=torch.float32, device=DEVICE)
    area_t = torch.tensor(mesh.area, dtype=torch.float32, device=DEVICE)

    # Planform: project to XY, then calculate bbox and area on GPU
    proj_verts_t = verts_t[:, :2]
    min_xy = proj_verts_t.min(dim=0).values
    max_xy = proj_verts_t.max(dim=0).values
    bbox_t = max_xy - min_xy
    
    # Ensure values are positive for area calculation, prevent issues with flat meshes
    bbox_x = torch.max(bbox_t[0], torch.tensor(0.001, device=DEVICE)) 
    bbox_y = torch.max(bbox_t[1], torch.tensor(0.001, device=DEVICE))
    area_proj_t = bbox_x * bbox_y * 0.8  # Approx fill factor

    # Span (max Y), mean chord (area/span) on GPU
    span_t = bbox_y
    mean_chord_t = torch.where(span_t > 0, area_proj_t / span_t, torch.tensor(0.05, device=DEVICE))

    # Aspect ratio on GPU
    AR_t = torch.where(area_proj_t > 0, span_t ** 2 / area_proj_t, torch.tensor(10.0, device=DEVICE))

    # Camber: avg |Z| / chord on GPU
    camber_t = torch.where(mean_chord_t > 0, torch.mean(torch.abs(verts_t[:, 2])) / mean_chord_t, torch.tensor(0.0, device=DEVICE))

    # Dihedral proxy: avg normal tilt from Z on GPU
    z_tilt_t = torch.mean(torch.abs(normals_t[:, 2] - 1.0))

    # Wet area approx on GPU
    area_wet_t = area_t * 2  # Both sides

    return {
        'area_proj': area_proj_t,
        'span': span_t,
        'mean_chord': mean_chord_t,
        'AR': AR_t,
        'camber': camber_t,
        'dihedral': z_tilt_t,
        'area_wet': area_wet_t
    }

def compute_aero_features_batch(features_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """
    Converts a list of feature dictionaries (each representing aero features for a mesh)
    into a dictionary of batched PyTorch tensors on the configured device.

    Args:
        features_list (list[dict[str, Any]]): A list where each element is a dictionary
                                             containing aerodynamic features for a single mesh.

    Returns:
        dict[str, torch.Tensor]: A dictionary where keys are feature names and values
                                 are batched PyTorch tensors, ready for GPU processing.
    """
    keys = ['area_proj', 'span', 'mean_chord', 'AR', 'camber', 'dihedral', 'area_wet']
    batch = {key: torch.stack([torch.tensor(f[key], device=DEVICE, dtype=torch.float32) if isinstance(f[key], (int, float, np.ndarray)) else f[key] for f in features_list], dim=0)
             for key in keys}
    return batch

def compute_inviscid_cl_cd_batch(span_t, chord_t, ar_t, aoa_rad_t, camber_t):
    """
    Vectorized GPU computation of CL, CD_i using lifting line theory (batch).

    Args:
        All inputs are torch tensors on DEVICE
    Returns:
        cl, cd_i as torch tensors
    """
    # Effective angle of attack with camber effect
    effective_aoa = aoa_rad_t + camber_t * np.deg2rad(5)

    # Lifting line theory (Glauert correction for 3D)
    cl_2d = 2 * np.pi * effective_aoa
    cl = cl_2d / (1 + cl_2d / (np.pi * ar_t + 1e-8))
    cd_i = cl**2 / (np.pi * ar_t * 0.8 + 1e-8)

    return cl, cd_i

def surrogate_cfd(mesh, state, use_cfd: bool = True):
    """
    Surrogate prediction: CL, CD, est_range_m (single mesh, slow path).
    
    Args:
        mesh: trimesh.Trimesh folded
        state: dict from config
        use_cfd: If True, try FluidX3D first; otherwise use physics-based surrogate
    
    Returns:
        dict: cl, cd, range_est
    """
    config = load_config()
    aoa_deg = state.get('angle_of_attack_deg', config['goals']['angle_of_attack_deg'])
    aoa_rad = np.deg2rad(aoa_deg)
    rho = state.get('air_density_kgm3', config['environment']['air_density_kgm3'])
    mu = state.get('air_viscosity_pas', config['environment']['air_viscosity_pas'])
    v_inf = state.get('throw_speed_mps', config['goals']['throw_speed_mps'])
    
    # Try FluidX3D CFD first if enabled
    if use_cfd and USE_FLUIDX3D:
        try:
            reynolds = rho * v_inf * 0.1 / mu  # Approximate chord = 0.1m
            cfd_results = run_fluidx3d_cfd(
                mesh,
                v_inf=v_inf,
                aoa_deg=aoa_deg,
                reynolds=reynolds,
                iterations=5000
            )
            if 'source' in cfd_results and cfd_results['source'] == 'fluidx3d':
                return cfd_results
        except Exception as e:
            print(f"FluidX3D failed: {e}, using surrogate")
    
    # Fallback to physics-based surrogate
    features = compute_aero_features(mesh)
    chord = features['mean_chord']

    # Reynolds
    Re = rho * v_inf * chord / mu

    # Inviscid CL, CD_i from lifting line
    cl, cd_i = compute_inviscid_cl_cd_scalar(features['span'], chord, features['AR'], aoa_rad, features['camber'])

    # Add dihedral effect
    cl += 0.1 * features['dihedral']
    
    # Viscous drag: skin friction + pressure drag estimate
    c_f_laminar = 1.328 / ((Re + 1e-8)**0.5)
    c_f_turbulent = 0.0744 / (Re**0.2 + 1e-8)
    c_f = c_f_laminar if Re < 5e5 else c_f_turbulent
    cd_viscous = 1.1 * c_f * 2 * features['area_wet'] / (features['area_proj'] + 1e-8)

    stall_factor = 1.0 + 2.0 * np.maximum(0, aoa_rad - np.deg2rad(15)) / np.pi
    cd = cd_viscous + cd_i * stall_factor
    
    # Glide ratio L/D
    ld = cl / (cd + 1e-8) * 0.8
    
    # Est range: simplified glide
    g = 9.81
    range_est = ld * v_inf**2 * np.sin(2 * np.deg2rad(10)) / g
    
    # Convert tensors to scalar floats for single evaluation
    return {
        'cl': cl.item(),
        'cd': cd.item(),
        'ld': ld.item(),
        'range_est': range_est.item(),
        'features': features
    }

def compute_inviscid_cl_cd_scalar(span, chord, ar, aoa_rad, camber):
    """Scalar version for single evaluation."""
    effective_aoa = aoa_rad + camber * np.deg2rad(5)
    cl_2d = 2 * np.pi * effective_aoa
    cl = cl_2d / (1 + cl_2d / (np.pi * ar + 1e-8))
    cd_i = cl**2 / (np.pi * ar * 0.8 + 1e-8)
    # Convert tensors to scalars
    return cl.item(), cd_i.item()

def surrogate_cfd_batch(features_list, states_list):
    """
    Batch GPU-optimized surrogate evaluation for multiple configurations.

    Args:
        features_list: List of feature dicts from compute_aero_features
        states_list: List of state dicts

    Returns:
        dict of torch tensors: cl, cd, ld, range_est (batch)
    """
    config = load_config()

    # Extract and prepare state tensors
    aoa_list = [np.deg2rad(s.get('angle_of_attack_deg', config['goals']['angle_of_attack_deg']))
                for s in states_list]
    rho_list = [s.get('air_density_kgm3', config['environment']['air_density_kgm3'])
                for s in states_list]
    mu_list = [s.get('air_viscosity_pas', config['environment']['air_viscosity_pas'])
               for s in states_list]
    v_inf_list = [s.get('throw_speed_mps', config['goals']['throw_speed_mps'])
                  for s in states_list]

    # Convert to GPU tensors
    aoa_t = torch.tensor(aoa_list, dtype=torch.float32, device=DEVICE)
    rho_t = torch.tensor(rho_list, dtype=torch.float32, device=DEVICE)
    mu_t = torch.tensor(mu_list, dtype=torch.float32, device=DEVICE)
    v_inf_t = torch.tensor(v_inf_list, dtype=torch.float32, device=DEVICE)

    # Feature tensors
    features_batch = compute_aero_features_batch(features_list)

    # Reynolds number (vectorized)
    chord_t = features_batch['mean_chord']
    Re = rho_t * v_inf_t * chord_t / (mu_t + 1e-8)

    # Inviscid aerodynamics (batch)
    cl, cd_i = compute_inviscid_cl_cd_batch(
        features_batch['span'], chord_t, features_batch['AR'], aoa_t, features_batch['camber']
    )

    # Add dihedral effect
    cl = cl + 0.1 * features_batch['dihedral']

    # Viscous drag (vectorized)
    c_f_laminar = 1.328 / torch.sqrt(Re + 1e-8)
    c_f_turbulent = 0.0744 / (Re**0.2 + 1e-8)
    c_f = torch.where(Re < 5e5, c_f_laminar, c_f_turbulent)
    cd_viscous = 1.1 * c_f * 2 * features_batch['area_wet'] / (features_batch['area_proj'] + 1e-8)

    # Stall modeling (vectorized)
    stall_factor = 1.0 + 2.0 * torch.maximum(torch.zeros_like(aoa_t), aoa_t - np.deg2rad(15)) / np.pi
    cd = cd_viscous + cd_i * stall_factor

    # Glide ratio
    ld = cl / (cd + 1e-8) * 0.8

    # Range estimation (vectorized)
    g = 9.81
    range_est = ld * v_inf_t**2 * np.sin(2 * np.deg2rad(10)) / g

    return {
        'cl': cl,
        'cd': cd,
        'ld': ld,
        'range_est': range_est,
        'Re': Re
    }


def find_fluidx3d_executable() -> Optional[Path]:
    """Locate FluidX3D executable on system."""
    import platform
    import os
    
    candidates = []
    
    if platform.system() == 'Windows':
        candidates.extend([
            Path(os.environ.get('PROGRAMFILES', 'C:\\Program Files')) / 'FluidX3D' / 'FluidX3D.exe',
            Path(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)')) / 'FluidX3D' / 'FluidX3D.exe',
            Path.home() / 'FluidX3D' / 'FluidX3D.exe',
            Path.cwd() / 'FluidX3D' / 'FluidX3D.exe',
        ])
    else:
        candidates.extend([
            Path('/usr/local/bin/FluidX3D'),
            Path('/opt/FluidX3D/FluidX3D'),
            Path.home() / 'FluidX3D' / 'FluidX3D',
        ])

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    try:
        result = subprocess.run(['which' if platform.system() != 'Windows' else 'where', 'FluidX3D'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return Path(result.stdout.strip().split('\n')[0])
    except Exception:
        pass

    return None


def run_fluidx3d_cfd(
    mesh: trimesh.Trimesh,
    v_inf: float = 10.0,
    aoa_deg: float = 5.0,
    reynolds: float = 1e5,
    iterations: int = 5000,
    temp_dir: Optional[Path] = None
) -> Dict[str, float]:
    """
    Run FluidX3D CFD simulation for paper airplane.
    
    Args:
        mesh: Airplane mesh (trimesh object)
        v_inf: Free stream velocity (m/s)
        aoa_deg: Angle of attack (degrees)
        reynolds: Reynolds number
        iterations: LBM iterations
        temp_dir: Temporary directory for simulation
    
    Returns:
        Dictionary with 'cl', 'cd', 'ld', 'range_est' keys
    """
    global FLUIDX3D_EXE
    
    if FLUIDX3D_EXE is None:
        FLUIDX3D_EXE = find_fluidx3d_executable()
    
    if FLUIDX3D_EXE is None:
        # Fallback to surrogate if FluidX3D not available
        return surrogate_cfd(mesh, {
            'v_inf': v_inf,
            'aoa_deg': aoa_deg,
            'reynolds': reynolds
        }, use_cfd=False)
    
    import tempfile
    import shutil
    
    work_dir = temp_dir or Path(tempfile.mkdtemp(prefix='fluidx3d_'))
    
    try:
        # Export STL
        stl_path = work_dir / 'airplane.stl'
        mesh.export(str(stl_path))
        
        # Create config file (JSON format for FluidX3D)
        config = {
            'stl_file': str(stl_path),
            'output_dir': str(work_dir),
            'velocity': v_inf,
            'angle_of_attack': aoa_deg,
            'reynolds': reynolds,
            'iterations': iterations,
            'lattice': 'D3Q27',
            'convergence_criterion': 1e-8
        }
        
        config_path = work_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run FluidX3D
        cmd = [
            str(FLUIDX3D_EXE),
            '--stl', str(stl_path),
            '--velocity', str(v_inf),
            '--aoa', str(aoa_deg),
            '--reynolds', str(reynolds),
            '--iterations', str(iterations),
            '--output', str(work_dir)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"FluidX3D warning: {result.stderr}")
            # Fallback to surrogate
            return surrogate_cfd(mesh, {
                'v_inf': v_inf,
                'aoa_deg': aoa_deg,
                'reynolds': reynolds
            }, use_cfd=False)
        
        # Parse results from output
        results_file = work_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
                return {
                    'cl': float(results.get('cl', 0.5)),
                    'cd': float(results.get('cd', 0.05)),
                    'ld': float(results.get('ld', 10.0)),
                    'range_est': float(results.get('range', 20.0)),
                    'source': 'fluidx3d'
                }
        else:
            # Parse from stdout
            cl = cd = 0.0
            for line in result.stdout.split('\n'):
                if 'CL' in line.upper():
                    try:
                        cl = float(line.split()[-1])
                    except:
                        pass
                elif 'CD' in line.upper():
                    try:
                        cd = float(line.split()[-1])
                    except:
                        pass
            
            ld = cl / (cd + 1e-8) if cd > 0 else 10.0
            g = 9.81
            range_est = ld * v_inf**2 * np.sin(2 * np.deg2rad(10)) / g
            
            return {
                'cl': cl,
                'cd': cd,
                'ld': ld,
                'range_est': range_est,
                'source': 'fluidx3d'
            }
    
    except subprocess.TimeoutExpired:
        print("FluidX3D timeout - using surrogate")
        return surrogate_cfd(mesh, {
            'v_inf': v_inf,
            'aoa_deg': aoa_deg,
            'reynolds': reynolds
        }, use_cfd=False)
    
    except Exception as e:
        print(f"FluidX3D error: {e} - falling back to surrogate")
        return surrogate_cfd(mesh, {
            'v_inf': v_inf,
            'aoa_deg': aoa_deg,
            'reynolds': reynolds
        }, use_cfd=False)
    
    finally:
        # Cleanup
        if temp_dir is None and work_dir.exists():
            try:
                shutil.rmtree(work_dir)
            except:
                pass
