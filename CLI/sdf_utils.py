import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple

def compute_sdf(voxel_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Field from a binary voxel grid.
    Positive values are outside (fluid), negative values are inside (solid).
    Uses Euclidean Distance Transform (EDT).
    """
    # Ensure binary mask
    mask = (voxel_grid > 0.5).cpu().numpy()

    # Distance to nearest solid (for fluid cells)
    # distance_transform_edt(input) calculates the distance to the closest zero-value
    # so we pass ~mask to find distance to solid (1s in mask)
    dist_outside = distance_transform_edt(~mask)

    # Distance to nearest fluid (for solid cells)
    dist_inside = distance_transform_edt(mask)

    # Combined SDF: fluid is positive, solid is negative
    # Subtracting 0.5 to place the zero-crossing at the voxel boundary interface
    sdf = dist_outside - dist_inside

    return torch.from_numpy(sdf).to(voxel_grid.device, dtype=torch.float32)

def compute_all_link_distances(voxel_grid: torch.Tensor, ex: torch.Tensor, ey: torch.Tensor, ez: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized wall distance 'q' for all 27 D3Q27 lattice directions.
    Returns tensor of shape [27, D, H, W].
    q = distance_to_wall / link_length (0 < q <= 1)
    """
    sdf = compute_sdf(voxel_grid)
    res = voxel_grid.shape[0]
    num_dirs = ex.shape[0]

    q_all = torch.ones((num_dirs, res, res, res), device=voxel_grid.device, dtype=torch.float32)

    for i in range(num_dirs):
        dx, dy, dz = int(ex[i].item()), int(ey[i].item()), int(ez[i].item())
        if dx == 0 and dy == 0 and dz == 0:
            continue

        # Neighbor SDF values
        sdf_neighbor = torch.roll(sdf, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))

        # Links that cross the boundary: current is fluid (>0), neighbor is solid (<=0)
        crossing = (sdf > 0) & (sdf_neighbor <= 0)

        # Linear interpolation for q: sdf(x) / (sdf(x) - sdf(x+e))
        # This assumes the wall is at sdf=0.
        denom = sdf[crossing] - sdf_neighbor[crossing]
        q_all[i][crossing] = torch.clamp(sdf[crossing] / (denom + 1e-12), 0.01, 1.0)

    return q_all
