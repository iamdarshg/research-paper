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
    """Compute SDF. Uses GPU dilation when input is on CUDA, CPU scipy otherwise."""
    if voxel_grid.is_cuda:
        from gpu_sdf import gpu_sdf
        return gpu_sdf(voxel_grid).float()
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

# Fixed D3Q27 stencil: the 26 non-zero link directions are read from the
# stencil tensors once per stencil and cached, so the GPU q-algebra does not
# re-read (and re-sync) ex/ey/ez on every solve. The cached value is a LIST of
# ``(i, dx, dy, dz)`` integer lattice-link offset tuples; the ``[num_dirs, D, H,
# W]`` stacks are built PER-CALL from the current sdf shape, never from the
# cache (see compute_link_q). The lattice offsets themselves are
# resolution-INDEPENDENT, so the cache is safe across resolutions. The PRIMARY
# key is a stencil fingerprint (the nonzero ``(i, dx, dy, dz)`` tuples), not the
# bare direction count, so a FUTURE stencil with the same number of non-zero
# links but different offsets/ordering cannot silently collide with D3Q27 (the
# review's finding). Computing the fingerprint reads the small offset tensors,
# so a second, identity-keyed fast path serves repeat calls that pass the SAME
# tensor objects (the steady-state case: solver ex/ey/ez are long-lived) without
# re-reading them; the stored strong reference keeps each id from being reused,
# so a different stencil's tensors can never be mistaken for a cached one. Every
# production caller uses the D3Q27 lattice; the only non-27-direction caller is
# the non-cubic unit test with a single link.
_LINK_DIRECTIONS_CACHE: dict = {}       # fingerprint -> directions list
_LINK_ID_FAST_CACHE: dict = {}          # (id(ex), id(ey), id(ez)) -> (directions, (ex, ey, ez))


def _link_directions(ex: torch.Tensor, ey: torch.Tensor, ez: torch.Tensor):
    id_key = (id(ex), id(ey), id(ez))
    fast = _LINK_ID_FAST_CACHE.get(id_key)
    if fast is not None:
        return fast[0]
    num_dirs = int(ex.shape[0])
    ex_l = ex.detach().cpu().tolist()
    ey_l = ey.detach().cpu().tolist()
    ez_l = ez.detach().cpu().tolist()
    directions = [
        (i, int(ex_l[i]), int(ey_l[i]), int(ez_l[i]))
        for i in range(num_dirs)
        if not (int(ex_l[i]) == 0 and int(ey_l[i]) == 0 and int(ez_l[i]) == 0)
    ]
    # Stencil fingerprint: the nonzero offset triples (with their direction
    # indices, in ascending order). Identical for the same stencil at any
    # resolution; distinct for any stencil whose nonzero link set or index
    # assignment differs, which is exactly the collision the review flagged.
    fingerprint = tuple(directions)
    cached = _LINK_DIRECTIONS_CACHE.get(fingerprint)
    if cached is None:
        cached = directions
        _LINK_DIRECTIONS_CACHE[fingerprint] = cached
    _LINK_ID_FAST_CACHE[id_key] = (cached, (ex, ey, ez))
    return cached


def compute_link_q(sdf: torch.Tensor, ex: torch.Tensor, ey: torch.Tensor, ez: torch.Tensor) -> torch.Tensor:
    """Compute the normalized wall-distance 'q' field from a full-volume SDF.

    OFFLOAD-3: the pure Issue-#15 q-algebra over an already-computed SDF on the
    target device. The scipy EDT that produced ``sdf`` stays on the CPU thread
    pool; every op here runs on ``sdf.device``, so the per-solve H2D transfer is
    the [D, H, W] SDF (3.5 MB at 96^3) instead of the [27, D, H, W] q field
    (95.5 MB). No bbox crop is applied: all crossing cells lie within one cell
    of the solid bounding box and their link neighbors within one more cell, so
    the full-volume EDT is bit-identical to the cropped EDT at every crossing
    cell, and cells far from the solid keep q = 1.0 through the ``where``.

    R12: the 26-link algebra is evaluated one direction at a time, writing
    directly into the pre-filled 1.0 ``q_all``. The stacked [num_dirs, D, H, W]
    temporaries the batched form materialized (``sdf_neighbors``, ``crossing``,
    ``denom``, ``q``, and the ``where`` result — ~5 x 88 MB fp32 at 96^3) are
    replaced by per-direction [D, H, W] temporaries (~3.5 MB each). Every
    element's arithmetic is unchanged, so the result is bit-identical.
    """
    if sdf.ndim != 3:
        raise ValueError(f"Expected 3D SDF, got {sdf.ndim}D")
    D, H, W = sdf.shape
    num_dirs = int(ex.shape[0])

    directions = _link_directions(ex, ey, ez)
    q_all = torch.ones((num_dirs, D, H, W), device=sdf.device, dtype=torch.float32)
    if not directions:
        return q_all

    # Avoid boundary wraparound using padding. We pad the SDF so that
    # 'neighbors' outside the domain appear far away (fluid). Using 10.0
    # ensures we don't accidentally detect a boundary link to the opposite face.
    sdf_padded = torch.nn.functional.pad(sdf, (1, 1, 1, 1, 1, 1), mode='constant', value=10.0)

    # One direction at a time (R12): the per-element formula below is identical
    # to the batched form, so only the working-set size changes. Non-crossing
    # cells keep the pre-filled 1.0, matching the original ``where(..., ones)``.
    for _idx, (i, dx, dy, dz) in enumerate(directions):
        neighbor_slice = sdf_padded[1 + dx:1 + dx + D, 1 + dy:1 + dy + H, 1 + dz:1 + dz + W]

        # Links that cross the boundary: current is fluid (>0, positive sdf =
        # outside/fluid), neighbor is solid (<=0, crossing into solid).
        crossing = (sdf > 0) & (neighbor_slice <= 0)

        # Linear interpolation for q: sdf(x) / (sdf(x) - sdf(x+e))
        # This assumes the wall is at sdf=0. Non-crossing cells stay at 1.0.
        denom = sdf - neighbor_slice
        q_dir = torch.clamp(sdf / (denom + 1e-12), 0.01, 1.0)
        q_all[i] = torch.where(crossing, q_dir, q_all[i])

    return q_all


def compute_all_link_distances(voxel_grid: torch.Tensor, ex: torch.Tensor, ey: torch.Tensor, ez: torch.Tensor, return_sdf: bool = False) -> torch.Tensor:
    """
    Compute normalized wall distance 'q' for all 27 D3Q27 lattice directions (Issue #15).
    Returns tensor of shape [27, D, H, W].
    q = distance_to_wall / link_length (0 < q <= 1)

    OFFLOAD-3: the SDF (scipy EDT) is computed here; the q-algebra is deferred to
    :func:`compute_link_q`, which runs on the SDF's device. When ``return_sdf`` is
    True, only the [D, H, W] SDF is returned (the thread-pool pre-warm path); the
    caller runs the q-algebra on the solve device.
    """
    # Support non-cubic tensors
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected 3D voxel grid, got {voxel_grid.ndim}D")

    # OFFLOAD-3: the pre-warm pool only needs the SDF; the q-algebra is deferred
    # to the solve device in D3Q27Solver._get_q. This is the only warm path, so
    # it avoids both the crop bbox scan (33 torch.nonzero host syncs per update)
    # and the 95.5 MB [27, D, H, W] H2D transfer per solve (3.5 MB SDF instead).
    if return_sdf:
        return compute_sdf(voxel_grid)

    # Preserve the original short-circuit for an empty grid (all-1.0 q). This
    # guard only runs on the rare cold path, so its single host sync is not on
    # the 33-per-update hot path.
    num_dirs = int(ex.shape[0])
    D, H, W = voxel_grid.shape
    solid = voxel_grid > 0.5
    if not torch.any(solid):
        return torch.ones((num_dirs, D, H, W), device=voxel_grid.device, dtype=torch.float32)

    sdf = compute_sdf(voxel_grid)
    return compute_link_q(sdf, ex, ey, ez)
