"""Task 34: parity gates for the compressed batched workspace.

The compressed path replaces the fused ``[C, 27, D, H, W]`` q-field with a
compact active-voxel table (sparse q) and collapses the 3 live population
buffers to 2 ping-pong buffers. These tests pin the two exactness claims:

1. The split stream (plain full-lattice pull stream + sparse BFL correction)
   reproduces the fused ``stream_bfl_d3q27_batch`` output BITWISE for every
   fixture, and the compact ``q_flat``/``active_flat`` tables are consistent
   with the full q-field at active voxels.
2. The full batched solver path that builds q INTERNALLY (``q_stack=None``,
   the production SPSA shape) stays within the Task 10 envelope of the
   sequential solver.

Nothing here weakens the existing Task 10 / kernel-parity gates.
"""

import math

import pytest
import torch

from advanced_lbm_solver import D3Q27Solver
from sdf_utils import compute_all_link_distances
from utils import compute_tensor_content_hash

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    from d3q27_kernels import stream_bfl_d3q27_batch, stream_bfl_d3q27_batch_compressed, triton
except Exception:  # pragma: no cover - optional dependency
    stream_bfl_d3q27_batch = None
    stream_bfl_d3q27_batch_compressed = None
    triton = None

requires_fused = pytest.mark.skipif(
    not (CUDA_AVAILABLE and triton is not None and stream_bfl_d3q27_batch is not None),
    reason="batched stream/BFL kernels require CUDA and Triton",
)

FIELD_ATOL_5STEP = 4e-6
FORCE_RTOL = 5e-4
FORCE_ATOL = 2.5e-5

GRID = 32
STEPS = 5
OMEGA = 1.6


def _sphere_mask(n: int, radius_frac: float = 0.27, center=None) -> torch.Tensor:
    coords = torch.arange(n, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    if center is None:
        center = ((n - 1) / 2.0,) * 3
    r2 = (z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2
    return (r2 <= (radius_frac * n) ** 2).float()


def _cube_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    lo, hi = n // 3, 2 * n // 3
    mask[lo:hi, lo:hi, lo:hi] = 1.0
    return mask


def _face_touching_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    mask[0 : n // 3, 0 : n // 3, 2 * n // 3 : n] = 1.0
    return mask


def _disconnected_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    q = n // 4
    mask[q : q + 3, q : q + 3, q : q + 3] = 1.0
    mask[3 * q : 3 * q + 3, 3 * q : 3 * q + 3, 3 * q : 3 * q + 3] = 1.0
    return mask


FIXTURES = {
    "sphere": _sphere_mask,
    "cube": _cube_mask,
    "face_touching": _face_touching_mask,
    "disconnected": _disconnected_mask,
}


def _make_solver(n: int, device: torch.device) -> D3Q27Solver:
    solver = D3Q27Solver(
        n,
        device,
        inlet_velocity_lu=0.05,
        use_triton_streaming=False,
        use_fused_stream_bfl=True,
    )
    rho = torch.ones(n, n, n, device=device)
    ux = torch.full((n, n, n), 0.05, device=device)
    uy = torch.zeros(n, n, n, device=device)
    uz = torch.zeros(n, n, n, device=device)
    solver.f.copy_(solver.compute_equilibrium(rho, ux, uy, uz))
    solver.reset_force_accounting()
    return solver


@requires_fused
@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_compressed_kernel_bitwise_matches_fused(fixture_name):
    """The split stream (plain stream + sparse BFL correction) must reproduce
    the fused batched kernel bitwise for every fixture, including empty/full."""
    n = GRID
    device = torch.device("cuda")
    solver = _make_solver(n, device)
    mask = FIXTURES[fixture_name](n).to(device).unsqueeze(0)
    C = 1
    solver._init_batch_equilibrium(C)
    q_stack = torch.stack([compute_all_link_distances(mask[c], solver.ex, solver.ey, solver.ez) for c in range(C)], dim=0).contiguous()
    sparse = solver._build_bfl_sparse_tables(mask, [f"h{fixture_name}"], q_stack)

    f_pre = solver._f_batch.clone()
    f_fused = torch.empty_like(f_pre)
    f_comp = torch.empty_like(f_pre)
    solid_u8 = (mask > 0.5).to(torch.uint8).contiguous()
    assert stream_bfl_d3q27_batch(f_pre, f_fused, solid_u8, q_stack, solver.ex, solver.ey, solver.ez, solver.opposite)
    assert stream_bfl_d3q27_batch_compressed(f_pre, f_comp, sparse, solver.ex, solver.ey, solver.ez, solver.opposite)
    torch.testing.assert_close(f_comp, f_fused, rtol=0.0, atol=0.0)


@requires_fused
def test_sparse_tables_consistent_with_full_q():
    """The compact q_flat values must equal the full q-field at the active voxels
    (guards against per-(c, i) table misalignment), and the active offsets must
    exactly match the fused kernel's boundary-link set."""
    n = GRID
    device = torch.device("cuda")
    solver = _make_solver(n, device)
    mask = torch.stack([_sphere_mask(n), _cube_mask(n)], dim=0).to(device)
    C = 2
    q_stack = torch.stack([compute_all_link_distances(mask[c], solver.ex, solver.ey, solver.ez) for c in range(C)], dim=0).contiguous()
    sparse = solver._build_bfl_sparse_tables(mask, [f"h{c}" for c in range(C)], q_stack)
    bl = solver._boundary_links_batch(mask)
    N = n * n * n
    for c in range(C):
        for i in range(1, 27):
            pair = c * 26 + (i - 1)
            start = int(sparse["pair_start"][pair].item())
            cnt = int(sparse["pair_count"][pair].item())
            idx = bl[c, i - 1].reshape(-1).nonzero(as_tuple=False).reshape(-1)
            assert int(idx.numel()) == cnt, f"c={c} i={i}: active count mismatch"
            assert torch.equal(sparse["active_flat"][start:start + cnt], idx.to(torch.int32)), (
                f"c={c} i={i}: active offsets mismatch"
            )
            assert torch.equal(
                sparse["q_flat"][start:start + cnt],
                q_stack[c, i].reshape(N)[idx],
            ), f"c={c} i={i}: compact q mismatch at active voxels"


@requires_fused
@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_batched_two_buffer_internal_q_vs_sequential(fixture_name):
    """The 2-buffer batched path with internally-built sparse q (production
    ``q_stack=None`` shape) stays within the Task 10 envelope of the sequential
    solver."""
    n = GRID
    device = torch.device("cuda")
    mask = FIXTURES[fixture_name](n)
    # sequential reference (fused sequential kernel untouched by Task 34)
    seq = _make_solver(n, device)
    mask_c = mask.to(device)
    seq_hash = compute_tensor_content_hash(mask_c)
    for _ in range(STEPS):
        seq.collide_and_stream(OMEGA, mask_c, geom_hash=seq_hash)

    # batched path, q built internally (no q_stack)
    batched = _make_solver(n, device)
    mask_stack = mask_c.unsqueeze(0)
    geom_hashes = [compute_tensor_content_hash(mask_stack[0])]
    batched._init_batch_equilibrium(1)
    batched.reset_force_accounting_batch(1)
    for _ in range(STEPS):
        batched.collide_and_stream_batch(OMEGA, mask_stack, geom_hashes=geom_hashes)

    field_diff = float((seq.f - batched._f_batch[0]).abs().max().item())
    assert field_diff <= FIELD_ATOL_5STEP, f"{fixture_name}: field diff {field_diff}"
    for seq_attr, bat_attr in (
        ("force_x_accum", "_force_x_accum_batch"),
        ("force_z_accum", "_force_z_accum_batch"),
        ("projected_drag_accum", "_projected_drag_accum_batch"),
    ):
        ref_val = float(getattr(seq, seq_attr).item())
        bat_val = float(getattr(batched, bat_attr)[0].item())
        assert math.isclose(ref_val, bat_val, rel_tol=FORCE_RTOL, abs_tol=FORCE_ATOL), (
            f"{fixture_name}: {seq_attr} reference={ref_val} batched={bat_val}"
        )


def test_batched_fallback_compact_q_cpu_vs_sequential():
    """The Triton-free fallback path (_stream_batch_fallback +
    _apply_bfl_boundary_batch_item consuming the COMPACT sparse-q tables) must
    match the sequential solver. On CPU the batched path always takes the
    fallback (the Triton kernels require CUDA), so this gates the compact-q
    fallback the brief explicitly named (task-34-brief.md, Lever 1).
    """
    n = 16
    device = torch.device("cpu")
    seq = _make_solver(n, device)
    mask_c = _sphere_mask(n).to(device)
    seq_hash = compute_tensor_content_hash(mask_c)
    for _ in range(STEPS):
        seq.collide_and_stream(OMEGA, mask_c, geom_hash=seq_hash)

    batched = _make_solver(n, device)
    mask_stack = mask_c.unsqueeze(0)
    geom_hashes = [compute_tensor_content_hash(mask_stack[0])]
    batched._init_batch_equilibrium(1)
    batched.reset_force_accounting_batch(1)
    for _ in range(STEPS):
        batched.collide_and_stream_batch(OMEGA, mask_stack, geom_hashes=geom_hashes)

    field_diff = float((seq.f - batched._f_batch[0]).abs().max().item())
    assert field_diff <= FIELD_ATOL_5STEP, f"fallback: field diff {field_diff}"
    for seq_attr, bat_attr in (
        ("force_x_accum", "_force_x_accum_batch"),
        ("force_z_accum", "_force_z_accum_batch"),
        ("projected_drag_accum", "_projected_drag_accum_batch"),
    ):
        ref_val = float(getattr(seq, seq_attr).item())
        bat_val = float(getattr(batched, bat_attr)[0].item())
        assert math.isclose(ref_val, bat_val, rel_tol=FORCE_RTOL, abs_tol=FORCE_ATOL), (
            f"fallback: {seq_attr} reference={ref_val} batched={bat_val}"
        )
