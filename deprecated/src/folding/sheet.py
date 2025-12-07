"""Parametric A4 sheet mesh generator."""
import yaml
import numpy as np
import trimesh
from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / 'config.yaml'

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def create_sheet(width_mm=210.0, height_mm=297.0, resolution=50):
    """
    Create triangulated mesh of flat A4 sheet.
    
    Args:
        width_mm, height_mm: Dimensions.
        resolution: Grid cells per side.
    
    Returns:
        trimesh.Trimesh
    """
    config = load_config()
    width = config['project']['sheet_width_mm'] / 1000.0  # to meters
    height = config['project']['sheet_height_mm'] / 1000.0
    
    # Create grid vertices
    x = np.linspace(0, width, resolution + 1)
    y = np.linspace(0, height, resolution + 1)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
    
    # Triangles: two per quad
    tris = []
    for i in range(resolution):
        for j in range(resolution):
            v0 = i * (resolution + 1) + j
            v1 = v0 + 1
            v2 = v0 + resolution + 1
            v3 = v2 + 1
            tris.extend([[v0, v1, v2], [v1, v3, v2]])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=tris)
    mesh.fix_normals()  # Upward facing
    return mesh
