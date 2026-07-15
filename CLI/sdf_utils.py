import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple
from threading import Lock


_EDT_WORKSPACES = {}
_EDT_WORKSPACE_LOCK = Lock()


def _edt_workspace(shape):
    workspace = _EDT_WORKSPACES.get(tuple(shape))
    if workspace is None:
        workspace = (
            np.empty(shape, dtype=np.float64),
            np.empty(shape, dtype=np.float64),
            np.empty((len(shape), *shape), dtype=np.int32),
        )
        _EDT_WORKSPACES[tuple(shape)] = workspace
    return workspace


def prepare_edt_workspace(shape) -> None:
    """Reserve exact EDT output storage before high-memory model construction."""
    normalized_shape = tuple(int(dimension) for dimension in shape)
    if len(normalized_shape) != 3 or any(dimension <= 0 for dimension in normalized_shape):
        raise ValueError("EDT workspace shape must contain three positive dimensions")
    with _EDT_WORKSPACE_LOCK:
        _edt_workspace(normalized_shape)

def compute_sdf(voxel_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Field from a binary voxel grid.
    Positive values are outside (fluid), negative values are inside (solid).
    Uses Euclidean Distance Transform (EDT).
    """
    # Ensure binary mask
    mask = (voxel_grid > 0.5).cpu().numpy()

    # SciPy otherwise allocates a feature-index field internally for every EDT.
    # Reusing caller-owned arrays avoids allocator spikes across the 33 sequential
    # direct-solver evaluations in one optimizer update.
    with _EDT_WORKSPACE_LOCK:
        dist_outside, dist_inside, feature_indices = _edt_workspace(mask.shape)
        distance_transform_edt(
            ~mask,
            return_distances=True,
            return_indices=True,
            distances=dist_outside,
            indices=feature_indices,
        )
        distance_transform_edt(
            mask,
            return_distances=True,
            return_indices=True,
            distances=dist_inside,
            indices=feature_indices,
        )
        np.subtract(dist_outside, dist_inside, out=dist_outside)
        # The dtype conversion makes an owning copy before the workspace is reused.
        sdf = torch.from_numpy(dist_outside).to(voxel_grid.device, dtype=torch.float32)

    return sdf

def compute_all_link_distances(voxel_grid: torch.Tensor, ex: torch.Tensor, ey: torch.Tensor, ez: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized wall distance 'q' for all 27 D3Q27 lattice directions (Issue #15).
    Returns tensor of shape [27, D, H, W].
    q = distance_to_wall / link_length (0 < q <= 1)
    """
    # Support non-cubic tensors
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected 3D voxel grid, got {voxel_grid.ndim}D")
    D, H, W = voxel_grid.shape
    sdf = compute_sdf(voxel_grid)
    num_dirs = ex.shape[0]

    # Avoid boundary wraparound using padding
    # We pad the SDF so that 'neighbors' outside the domain appear far away (fluid)
    # Using 10.0 ensures we don't accidentally detect a boundary link to the opposite face
    sdf_padded = torch.nn.functional.pad(sdf, (1, 1, 1, 1, 1, 1), mode='constant', value=10.0)

    q_all = torch.ones((num_dirs, D, H, W), device=voxel_grid.device, dtype=torch.float32)

    for i in range(num_dirs):
        dx, dy, dz = int(ex[i].item()), int(ey[i].item()), int(ez[i].item())
        if dx == 0 and dy == 0 and dz == 0:
            continue

        # Neighbor SDF values (using padded version to avoid wraparound)
        # For neighbor at x+e, we look at the padded slice shifted by dx, dy, dz.
        # Padded is [D+2, H+2, W+2]. Interior is at [1:D+1, 1:H+1, 1:W+1]
        # Neighbor of (x,y,z) in direction (dx,dy,dz) is at (1+x+dx, 1+y+dy, 1+z+dz) in padded
        d_start, d_end = 1 + dx, 1 + dx + D
        h_start, h_end = 1 + dy, 1 + dy + H
        w_start, w_end = 1 + dz, 1 + dz + W

        sdf_neighbor = sdf_padded[d_start:d_end, h_start:h_end, w_start:w_end]

        # Links that cross the boundary: current is fluid (>0), neighbor is solid (<=0)
        crossing = (sdf > 0) & (sdf_neighbor <= 0)

        # Linear interpolation for q: sdf(x) / (sdf(x) - sdf(x+e))
        # This assumes the wall is at sdf=0.
        denom = sdf[crossing] - sdf_neighbor[crossing]
        q_all[i][crossing] = torch.clamp(sdf[crossing] / (denom + 1e-12), 0.01, 1.0)

    return q_all
