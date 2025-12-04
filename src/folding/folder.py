"""Paper folding simulator: creases -> 3D mesh."""
import numpy as np
import trimesh
from shapely.geometry import LineString, Point
from .sheet import create_sheet, load_config

# load_config imported from sheet

def fold_sheet(action, resolution=50):
    """
    Fold flat sheet based on RL action.

    Args:
        action: np.array (n_folds * 5): [x1,y1,x2,y2,angle] normalized [0,1]
        resolution: Sheet grid res

    Returns:
        trimesh.Trimesh folded
    """
    config = load_config()
    n_folds = config['project']['n_folds']
    assert len(action) == n_folds * 5  # x1,y1,x2,y2,angle
    
    sheet = create_sheet(resolution=resolution)
    vertices = sheet.vertices.copy()
    
    width, height = config['project']['sheet_width_mm'] / 1000.0, config['project']['sheet_height_mm'] / 1000.0
    h, w = height, width  # Normalize inverse
    
    for i in range(n_folds):
        index = i * 5
        x1, y1, x2, y2, angle_val = action[index:index+5]
        p1 = np.array([x1 * w, y1 * h])
        p2 = np.array([x2 * w, y2 * h])

        # Invalid crease check
        if np.allclose(p1, p2) or np.linalg.norm(p2 - p1) < 0.01:
            continue

        crease_len = np.linalg.norm(p2 - p1)
        dir_crease = (p2 - p1) / crease_len
        angle_rad = (angle_val - 0.5) * np.pi  # -pi/2 to pi/2

        # For each vertex
        for j, v in enumerate(vertices):
            proj = v[:2]

            vec_to_proj = proj - p1
            proj_along = np.dot(vec_to_proj, dir_crease)
            if proj_along < 0 or proj_along > crease_len:
                continue

            perp_dist_vec = vec_to_proj - proj_along * dir_crease
            dist = np.linalg.norm(perp_dist_vec)
            if dist < 0.001:
                continue

            side = np.cross(dir_crease, perp_dist_vec / dist)

            # Simple displacement, can be refined to rotation
            z_disp = side * np.sin(angle_rad) * np.exp(-dist / 0.02)  # Smooth decay
            vertices[j, 2] += z_disp
    
    # Recreate mesh (approx, faces same)
    sheet.vertices = vertices
    sheet.fix_normals()
    return sheet

# Note: Simplified displacement fold. Refine with panel rotation later.
