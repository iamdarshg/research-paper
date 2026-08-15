import threading
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple


# Per-thread EDT workspaces. scipy's distance_transform_edt releases the GIL
# during the C computation (measured ~3.6x on 8 threads for 96^3), so the 33
# per-update direct-solver SDF evaluations can run concurrently on separate
# workspaces instead of serializing on a single shared buffer. Each workspace
# is 2 x float64[96^3] + int32[3,96^3] ~= 25 MiB, so N threads cost ~25N MiB.
# prepare_edt_workspace() still primes the calling (main) thread's workspace
# before high-memory model construction.
_THREAD_EDT_WORKSPACES = threading.local()


def _edt_workspace(shape):
    normalized_shape = tuple(int(d) for d in shape)
    workspaces = getattr(_THREAD_EDT_WORKSPACES, "_workspaces", None)
    if workspaces is None:
        workspaces = {}
        _THREAD_EDT_WORKSPACES._workspaces = workspaces
    workspace = workspaces.get(normalized_shape)
    if workspace is None:
        workspace = (
            np.empty(normalized_shape, dtype=np.float64),
            np.empty(normalized_shape, dtype=np.float64),
            np.empty((len(normalized_shape), *normalized_shape), dtype=np.int32),
        )
        workspaces[normalized_shape] = workspace
    return workspace


def prepare_edt_workspace(shape) -> None:
    """Reserve exact EDT output storage before high-memory model construction."""
    normalized_shape = tuple(int(dimension) for dimension in shape)
    if len(normalized_shape) != 3 or any(dimension <= 0 for dimension in normalized_shape):
        raise ValueError("EDT workspace shape must contain three positive dimensions")
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
    # direct-solver evaluations in one optimizer update. Each thread owns its own
    # workspace (_edt_workspace is thread-local), so concurrent EDTs from a
    # thread pool never share buffers and need no lock.
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
    num_dirs = ex.shape[0]

    # Read the 27 lattice directions once. Direction index 0 is the zero link,
    # which the original loop skipped; every other index was written into q_all.
    directions = []
    for i in range(num_dirs):
        dx, dy, dz = int(ex[i].item()), int(ey[i].item()), int(ez[i].item())
        if not (dx == 0 and dy == 0 and dz == 0):
            directions.append((i, dx, dy, dz))

    q_all = torch.ones((num_dirs, D, H, W), device=voxel_grid.device, dtype=torch.float32)
    if not directions:
        return q_all

    # The aircraft occupies only ~1-4% of the box, so the full-volume EDT is
    # mostly wasted work. Crop the SDF + link algebra to the solid bounding box
    # expanded by a margin of 2 cells. Every crossing cell is fluid (sdf > 0,
    # positive sdf = outside/fluid in this codebase's EDT convention) and
    # therefore lies inside the bbox; its neighbor in the link direction is
    # at most 1 cell outside the bbox. Margin 2 keeps the entire crossing set
    # and its neighbors inside the crop, and because the crop still contains
    # every solid cell, the EDT values inside it are unchanged from the
    # full-volume EDT. All cells outside the crop keep the initialized 1.0.
    solid = voxel_grid > 0.5
    occupied = torch.nonzero(solid)
    if occupied.numel() == 0:
        return q_all

    mins = occupied.min(dim=0).values
    maxs = occupied.max(dim=0).values
    lo = torch.clamp(mins - 2, min=0)
    hi = torch.clamp(maxs + 3, max=torch.tensor([D, H, W], device=voxel_grid.device))
    lo_z, lo_y, lo_x = (int(v) for v in lo.tolist())
    hi_z, hi_y, hi_x = (int(v) for v in hi.tolist())

    crop = voxel_grid[lo_z:hi_z, lo_y:hi_y, lo_x:hi_x]
    sdf_crop = compute_sdf(crop)
    cD, cH, cW = sdf_crop.shape

    # Avoid boundary wraparound using padding
    # We pad the SDF so that 'neighbors' outside the domain appear far away (fluid)
    # Using 10.0 ensures we don't accidentally detect a boundary link to the opposite face
    sdf_crop_padded = torch.nn.functional.pad(sdf_crop, (1, 1, 1, 1, 1, 1), mode='constant', value=10.0)

    # Stack the 26 shifted neighbor slices into one tensor, in the exact index
    # order the original loop wrote (ascending direction index, zero link absent).
    neighbor_slices = []
    for _i, dx, dy, dz in directions:
        neighbor_slices.append(
            sdf_crop_padded[
                1 + dx:1 + dx + cD,
                1 + dy:1 + dy + cH,
                1 + dz:1 + dz + cW,
            ]
        )
    sdf_neighbors = torch.stack(neighbor_slices, dim=0)  # [26, cD, cH, cW]
    sdf_view = sdf_crop.unsqueeze(0)

    # Links that cross the boundary: current is fluid (>0, positive sdf =
    # outside/fluid), neighbor is solid (<=0, crossing into solid).
    crossing = (sdf_view > 0) & (sdf_neighbors <= 0)

    # Linear interpolation for q: sdf(x) / (sdf(x) - sdf(x+e))
    # This assumes the wall is at sdf=0. Non-crossing cells stay at 1.0.
    denom = sdf_view - sdf_neighbors
    q = torch.clamp(sdf_view / (denom + 1e-12), 0.01, 1.0)
    q = torch.where(crossing, q, torch.ones_like(q))

    for idx, (i, _dx, _dy, _dz) in enumerate(directions):
        q_all[i, lo_z:hi_z, lo_y:hi_y, lo_x:hi_x] = q[idx]

    return q_all
