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
    # Fix B: Support non-cubic tensors
    shape = voxel_grid.shape
    sdf = compute_sdf(voxel_grid)
    num_dirs = ex.shape[0]

    # Fix C: Avoid boundary wraparound using padding
    sdf_padded = torch.nn.functional.pad(sdf, (1, 1, 1, 1, 1, 1), mode='constant', value=10.0)

    q_all = torch.ones((num_dirs, *shape), device=voxel_grid.device, dtype=torch.float32)

    for i in range(num_dirs):
        dx, dy, dz = int(ex[i].item()), int(ey[i].item()), int(ez[i].item())
        if dx == 0 and dy == 0 and dz == 0:
            continue

        # Neighbor SDF values (using padded version to avoid wraparound)
        # shifts in roll move elements from end to start.
        # For neighbor at x+e, we need to look at shifted sdf.
        # But padding is safer than roll for non-periodic.

        # Extract the same shape as sdf but shifted
        # Padded is [D+2, H+2, W+2]. If dx=1, we want [2:D+2]
        d_slice = slice(1+dx, 1+dx+shape[0])
        h_slice = slice(1+dy, 1+dy+shape[1])
        w_slice = slice(1+dz, 1+dz+shape[2])
        sdf_neighbor = sdf_padded[d_slice, h_slice, w_slice]

        # Links that cross the boundary: current is fluid (>0), neighbor is solid (<=0)
        crossing = (sdf > 0) & (sdf_neighbor <= 0)

        # Linear interpolation for q: sdf(x) / (sdf(x) - sdf(x+e))
        # This assumes the wall is at sdf=0.
        denom = sdf[crossing] - sdf_neighbor[crossing]
        q_all[i][crossing] = torch.clamp(sdf[crossing] / (denom + 1e-12), 0.01, 1.0)

    return q_all
