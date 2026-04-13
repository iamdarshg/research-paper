# Copyright (C) 2025 Darsh Gupta
# Licensed under GPL-v2

import torch
import numpy as np

def compute_sdf(voxel_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute Signed Distance Field (SDF) from binary voxel grid.
    Positive values are outside (fluid), negative values are inside (solid).
    Uses distance transform from scipy.
    """
    from scipy.ndimage import distance_transform_edt

    grid_np = voxel_grid.cpu().numpy()

    # Distance to the nearest solid voxel (outside)
    dist_outside = distance_transform_edt(grid_np == 0)

    # Distance to the nearest fluid voxel (inside)
    dist_inside = distance_transform_edt(grid_np == 1)

    # Sub-voxel correction: if we assume the interface is at 0.5 voxel units
    sdf = dist_outside - dist_inside
    return torch.from_numpy(sdf).to(voxel_grid.device).float()

def compute_bfl_q(sdf: torch.Tensor, direction: tuple) -> torch.Tensor:
    """
    Compute the distance q from fluid node to wall along a lattice direction.
    sdf: [D, H, W] SDF field (positive outside)
    direction: (dx, dy, dz)
    Returns q: [D, H, W] where 0 < q <= 1 for boundary links.
    """
    dx, dy, dz = direction
    sdf_neighbor = torch.roll(sdf, shifts=(-dx, -dy, -dz), dims=(0, 1, 2))

    # Boundary link: fluid node (sdf > 0) with solid neighbor (sdf <= 0)
    # q is the fraction of the lattice link in the fluid.
    # q = sdf_fluid / (sdf_fluid - sdf_solid)

    # We only care about links that cross the boundary (sdf > 0 and sdf_neighbor <= 0)
    q = torch.abs(sdf) / (torch.abs(sdf) + torch.abs(sdf_neighbor) + 1e-12)

    # Clip q to (0, 1] to avoid numerical issues
    q = torch.clamp(q, 1e-3, 1.0)

    return q

def add_box_to_sdf(sdf: torch.Tensor, padding: int = 0) -> torch.Tensor:
    """
    Treat domain boundaries as solid walls in the SDF.
    Negative SDF means solid.
    """
    res = sdf.shape[0]
    # Create a box SDF: distance to nearest domain boundary
    # For a cube [0, res-1], the distance to boundary is min(x, res-1-x, y, ...)
    coords = torch.stack(torch.meshgrid(
        torch.arange(res, device=sdf.device),
        torch.arange(res, device=sdf.device),
        torch.arange(res, device=sdf.device),
        indexing='ij'
    ), dim=-1)

    dist_to_min = coords
    dist_to_max = (res - 1) - coords
    dist_to_boundary = torch.min(torch.min(dist_to_min, dist_to_max), dim=-1).values

    # Domain walls are at dist_to_boundary == 0
    # Inside domain: dist_to_boundary > 0 (fluid-ish)
    # Outside: negative
    box_sdf = dist_to_boundary.float() - 0.5

    # Combine with existing SDF (minimum means union of solids)
    return torch.min(sdf, box_sdf)

def get_full_mask(geometry_mask: torch.Tensor) -> torch.Tensor:
    """Combine geometry mask with domain boundary mask"""
    res = geometry_mask.shape[0]
    full_mask = geometry_mask.clone()
    # Set boundaries to solid (except maybe inlet/outlet?)
    # For now, let's just make everything solid if it's on the boundary
    full_mask[0, :, :] = 1.0
    full_mask[-1, :, :] = 1.0
    full_mask[:, 0, :] = 1.0
    full_mask[:, -1, :] = 1.0
    full_mask[:, :, 0] = 1.0
    full_mask[:, :, -1] = 1.0
    return full_mask
