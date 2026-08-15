"""Task 8: vectorized-vs-loop parity for the momentum-exchange force kernel.

The fused-parity gates (test_direct_solver_fused_parity.py,
test_d3q27_kernel_parity.py) compare two full pipelines that genuinely differ in
the streaming/BFL backend (PyTorch ``_apply_bfl_boundary`` vs Triton
``stream_bfl_d3q27``) but that both share the vectorized 26-dir force kernel.
They therefore cannot detect a regression inside the force accumulation itself.

This test pins fusion #1 (vectorized 26-dir momentum-exchange sum) directly:
on identical live solver state, the vectorized
``D3Q27Solver._accumulate_momentum_exchange_force_nosync`` is compared against
a small independent inline 26-direction loop reference, and the absolute diff
must stay below FORCE_ATOL (2.5e-5) for BOTH the x and z components. Parity is
LOW (reduction order, ~1e-13 relative); the envelope is the same absolute
FORCE_ATOL used by the fused-vs-reference gates.

CPU-runnable: no CUDA/Triton dependency. Uses the non-fused reference path to
advance live state (``_apply_bfl_boundary`` computes q via the SciPy EDT).
"""

import pytest
import torch

from advanced_lbm_solver import D3Q27Solver

FORCE_ATOL = 2.5e-5
GRID = 32


def _make_solver() -> D3Q27Solver:
    device = torch.device("cpu")
    solver = D3Q27Solver(GRID, device, inlet_velocity_lu=0.05)
    rho = torch.ones(GRID, GRID, GRID, device=device)
    ux = torch.full((GRID, GRID, GRID), 0.05, device=device)
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    solver.f.copy_(solver.compute_equilibrium(rho, ux, uy, uz))
    generator = torch.Generator().manual_seed(20260716)
    solver.f.add_(torch.randn(solver.f.shape, generator=generator) * 1e-4)
    solver.reset_force_accounting()
    return solver


def _sphere_mask(n: int) -> torch.Tensor:
    coords = torch.arange(n, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (n - 1) / 2.0
    r2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return (r2 <= (0.27 * n) ** 2).float()


def _cube_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    lo, hi = n // 3, 2 * n // 3
    mask[lo:hi, lo:hi, lo:hi] = 1.0
    return mask


def _loop_reference(solver: D3Q27Solver, boundary_links: torch.Tensor):
    """Independent 26-direction inline loop reference (the pre-Task-8 form)."""
    fx = torch.tensor(0.0, device=solver.device)
    fz = torch.tensor(0.0, device=solver.device)
    for i in range(1, 27):
        link_idx = i - 1
        active = boundary_links[link_idx]
        f_in = solver.f_pre_stream[i][active]
        opp_i = solver._opposite_list[i]
        f_out = solver.f_temp[opp_i][active]
        fx += torch.sum(solver.ex[i] * (f_in + f_out))
        fz += torch.sum(solver.ez[i] * (f_in + f_out))
    return fx, fz


@pytest.mark.parametrize("mask", [_sphere_mask(GRID), _cube_mask(GRID)])
def test_force_vectorize_vs_loop_parity(mask):
    solver = _make_solver()
    # Advance two steps so f_pre_stream / f_temp hold live post-stream state
    # (the same state the momentum-exchange sum sees during a real solve).
    for _ in range(2):
        solver.collide_and_stream(1.6, mask)

    boundary_links = solver._boundary_links(mask)
    fx_vec, fz_vec = solver._accumulate_momentum_exchange_force_nosync(mask)
    fx_ref, fz_ref = _loop_reference(solver, boundary_links)

    fx_diff = float((fx_vec - fx_ref).abs().item())
    fz_diff = float((fz_vec - fz_ref).abs().item())
    assert fx_diff < FORCE_ATOL, (
        f"fx vectorized-vs-loop diff {fx_diff:.3e} >= FORCE_ATOL {FORCE_ATOL}"
    )
    assert fz_diff < FORCE_ATOL, (
        f"fz vectorized-vs-loop diff {fz_diff:.3e} >= FORCE_ATOL {FORCE_ATOL}"
    )
