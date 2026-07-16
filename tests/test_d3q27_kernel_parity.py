"""Parity gates for the fused D3Q27 pull-stream + BFL kernel.

The PyTorch implementation in ``advanced_lbm_solver.D3Q27Solver`` remains the
authoritative reference. These tests compare the fused Triton path against it
on fixtures covering axial and diagonal boundaries, q values on both sides of
0.5, domain faces, empty/full/disconnected masks, and (when present) a real
96^3 corpus aircraft. The fused path may only be enabled for training while
every test here passes.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from advanced_lbm_solver import D3Q27Solver
from sdf_utils import compute_all_link_distances

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    from d3q27_kernels import stream_bfl_d3q27, triton
except Exception:  # pragma: no cover - optional dependency
    stream_bfl_d3q27 = None
    triton = None

requires_fused = pytest.mark.skipif(
    not (CUDA_AVAILABLE and triton is not None and stream_bfl_d3q27 is not None),
    reason="fused stream/BFL kernel requires CUDA and Triton",
)

# Tolerances: the reference path is deterministic, but Triton contracts
# multiply-add chains into FMAs, so bitwise equality is not attainable.
# Measured on 2026-07-16 (RTX 4060 Laptop, torch 2.x, triton 3.5.1):
# - one-step max population diff: 1.9e-9 (a single ULP of the ~0.04 scale);
# - five-step max population diff: < 1e-6 across all fixtures;
# - force summation: computed on identical solver state, the fused-path
#   accumulation is bitwise equal to the reference loop (diff 0.0; verified
#   directly, 2026-07-16). The only force divergence is the FMA field drift
#   projected through the momentum-exchange sum, whose gross term magnitude
#   is ~380 while the net force is ~1e-2 (a ~4e4x sign cancellation). The
#   measured 5-step accumulated force diff is <= 6.5e-6, consistent with
#   gross_magnitude x fp32 drift, so the meaningful envelope is ABSOLUTE
#   (scaled to gross magnitude), not relative to the small net value.
# Gates are set at roughly 4x the measured worst case; they are not tuned
# to pass any later regression.
FIELD_ATOL_1STEP = 1e-8
FIELD_ATOL_5STEP = 4e-6
FORCE_RTOL = 5e-4
FORCE_ATOL = 2.5e-5


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
    """Solid block touching the x=0, y=0, and z=n-1 domain faces."""
    mask = torch.zeros(n, n, n)
    mask[0 : n // 3, 0 : n // 3, 2 * n // 3 : n] = 1.0
    return mask


def _disconnected_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    q = n // 4
    mask[q : q + 3, q : q + 3, q : q + 3] = 1.0
    mask[3 * q : 3 * q + 3, 3 * q : 3 * q + 3, 3 * q : 3 * q + 3] = 1.0
    return mask


def _real_aircraft_mask(n: int):
    voxel_dir = Path(__file__).resolve().parents[1] / "build" / "aircraftverse_geometry_only_20260715" / "voxels"
    if not voxel_dir.exists():
        return None
    for candidate in sorted(voxel_dir.glob("*.npy")):
        arr = np.load(candidate)
        if arr.shape == (n, n, n):
            return torch.from_numpy(arr.astype(np.float32))
    return None


def _make_solver(n: int, device: torch.device, fused: bool) -> D3Q27Solver:
    solver = D3Q27Solver(
        n,
        device,
        inlet_velocity_lu=0.05,
        use_triton_streaming=False,
        use_fused_stream_bfl=fused,
    )
    # Deterministic non-trivial starting populations shared by both backends.
    rho = torch.ones(n, n, n, device=device)
    ux = torch.full((n, n, n), 0.05, device=device)
    uy = torch.zeros(n, n, n, device=device)
    uz = torch.zeros(n, n, n, device=device)
    feq = solver.compute_equilibrium(rho, ux, uy, uz)
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    noise = torch.randn(feq.shape, generator=generator) * 1e-4
    solver.f.copy_(feq + noise.to(device))
    solver.reset_force_accounting()
    return solver


def _run_pair(mask: torch.Tensor, n: int, steps: int, omega: float = 1.6):
    device = torch.device("cuda")
    mask = mask.to(device)
    reference = _make_solver(n, device, fused=False)
    fused = _make_solver(n, device, fused=True)
    assert fused.use_fused_stream_bfl, "fused backend did not activate"
    torch.testing.assert_close(reference.f, fused.f, rtol=0.0, atol=0.0)

    diffs = []
    for _ in range(steps):
        reference.collide_and_stream(omega, mask)
        fused.collide_and_stream(omega, mask)
        diffs.append(float((reference.f - fused.f).abs().max().item()))
    return reference, fused, diffs


FIXTURES = {
    "sphere": _sphere_mask,
    "cube": _cube_mask,
    "face_touching": _face_touching_mask,
    "disconnected": _disconnected_mask,
}


@requires_fused
@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_one_step_field_parity(fixture_name):
    n = 32
    mask = FIXTURES[fixture_name](n)
    _, _, diffs = _run_pair(mask, n, steps=1)
    assert diffs[0] <= FIELD_ATOL_1STEP, f"{fixture_name}: 1-step max field diff {diffs[0]}"


@requires_fused
@pytest.mark.parametrize("fixture_name", sorted(FIXTURES))
def test_five_step_parity_fields_and_forces(fixture_name):
    n = 32
    mask = FIXTURES[fixture_name](n)
    reference, fused, diffs = _run_pair(mask, n, steps=5)
    assert max(diffs) <= FIELD_ATOL_5STEP, f"{fixture_name}: 5-step max field diff {max(diffs)}"

    for attr in ("force_x_accum", "force_z_accum", "projected_drag_accum"):
        ref_val = float(getattr(reference, attr).item())
        fused_val = float(getattr(fused, attr).item())
        assert math.isclose(ref_val, fused_val, rel_tol=FORCE_RTOL, abs_tol=FORCE_ATOL), (
            f"{fixture_name}: {attr} reference={ref_val} fused={fused_val}"
        )

    # Macroscopic fields after the run.
    for attr in ("rho", "velocity_x", "velocity_y", "velocity_z", "pressure"):
        ref_field = getattr(reference, attr)
        fused_field = getattr(fused, attr)
        max_diff = float((ref_field - fused_field).abs().max().item())
        assert max_diff <= FIELD_ATOL_5STEP, f"{fixture_name}: {attr} max diff {max_diff}"


@requires_fused
def test_empty_and_full_masks():
    n = 32
    for name, mask in (("empty", torch.zeros(n, n, n)), ("full", torch.ones(n, n, n))):
        _, _, diffs = _run_pair(mask, n, steps=3)
        assert max(diffs) <= FIELD_ATOL_5STEP, f"{name}: max field diff {max(diffs)}"


@requires_fused
def test_q_branches_are_exercised():
    """The sphere fixture must contain crossing links with q on both sides of 0.5."""
    n = 32
    mask = _sphere_mask(n).cuda()
    solver = _make_solver(n, torch.device("cuda"), fused=False)
    q = solver._get_q(mask)
    links = solver._boundary_links(mask)
    crossing_q = torch.cat([q[i][links[i - 1]] for i in range(1, 27)])
    assert crossing_q.numel() > 0
    assert bool((crossing_q < 0.5).any()), "no q < 0.5 links; low branch untested"
    assert bool((crossing_q >= 0.5).any()), "no q >= 0.5 links; high branch untested"
    assert bool((crossing_q < 0.45).any()) and bool((crossing_q > 0.55).any()), (
        "q distribution too narrow to exercise both BFL formulas meaningfully"
    )


@requires_fused
def test_real_aircraft_geometry_parity_96():
    n = 96
    mask = _real_aircraft_mask(n)
    if mask is None:
        pytest.skip("no 96^3 corpus voxel file available")
    reference, fused, diffs = _run_pair(mask, n, steps=5)
    assert max(diffs) <= FIELD_ATOL_5STEP, f"real aircraft: 5-step max field diff {max(diffs)}"
    for attr in ("force_x_accum", "force_z_accum", "projected_drag_accum"):
        ref_val = float(getattr(reference, attr).item())
        fused_val = float(getattr(fused, attr).item())
        assert math.isclose(ref_val, fused_val, rel_tol=FORCE_RTOL, abs_tol=FORCE_ATOL), (
            f"real aircraft: {attr} reference={ref_val} fused={fused_val}"
        )


@requires_fused
def test_fused_repeatability():
    """Repeated fused runs from identical state must agree bitwise with themselves."""
    n = 32
    mask = _sphere_mask(n)
    _, fused_a, _ = _run_pair(mask, n, steps=3)
    _, fused_b, _ = _run_pair(mask, n, steps=3)
    torch.testing.assert_close(fused_a.f, fused_b.f, rtol=0.0, atol=0.0)


def test_reference_path_unchanged_without_flag():
    """Without the flag the solver must use the reference path even on CUDA."""
    device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
    solver = D3Q27Solver(16, device, inlet_velocity_lu=0.05)
    assert solver.use_fused_stream_bfl is False
    mask = _cube_mask(16).to(device)
    solver.collide_and_stream(1.6, mask)  # must not raise
    assert torch.isfinite(solver.f).all()
