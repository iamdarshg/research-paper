"""Physics-inspired CFD surrogate for paper airplane aerodynamics with GPU optimization."""
import yaml
import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Any # Import Any for type hinting
from ..folding.sheet import load_config


CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def surrogate_cfd(mesh, state):
    """
    Surrogate prediction: CL, CD, est_range_m (single mesh, slow path).
    
    Args:
        mesh: trimesh.Trimesh folded
        state: dict from config
    
    Returns:
        dict: cl, cd, range_est
    """
    config = load_config()
    aoa_rad = np.deg2rad(state.get('angle_of_attack_deg', config['goals']['angle_of_attack_deg']))
    rho = state.get('air_density_kgm3', config['environment']['air_density_kgm3'])
    mu = state.get('air_viscosity_pas', config['environment']['air_viscosity_pas'])
    v_inf = state.get('throw_speed_mps', config['goals']['throw_speed_mps'])
    
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
