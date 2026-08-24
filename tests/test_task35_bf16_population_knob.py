"""Task 35: bf16 population-storage knob for the batched workspace.

The knob ``D3Q27Solver.batch_population_dtype`` defaults to fp32 (the
production precision contract). Setting it to ``torch.bfloat16`` stores the two
resident batch population buffers (``_f_batch`` / ``_f_swap_batch``) in bf16
(halving that workspace); all arithmetic stays fp32 (bf16 STORAGE, fp32
COMPUTE). These are behavior assertions (dtype toggling + end-to-end finite
solve), not noise/precision tests: the fp32 default path is pinned bit-exact by
the Task 10 / Task 34 / kernel-parity gates, and the bf16-vs-fp32 precision
delta is measured by the experiment probe, not asserted here.
"""

import torch

from advanced_lbm_solver import D3Q27Solver
from utils import compute_tensor_content_hash

GRID = 16
STEPS = 5
OMEGA = 1.6


def _sphere_mask(n: int, radius_frac: float = 0.27) -> torch.Tensor:
    coords = torch.arange(n, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (n - 1) / 2.0
    r2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return (r2 <= (radius_frac * n) ** 2).float()


def _make_solver(n: int, device: torch.device) -> D3Q27Solver:
    return D3Q27Solver(
        n,
        device,
        inlet_velocity_lu=0.05,
        use_triton_streaming=False,
        use_fused_stream_bfl=True,
    )


def test_knob_defaults_to_fp32_and_allocates_fp32_buffers():
    device = torch.device("cpu")
    solver = _make_solver(GRID, device)
    assert solver.batch_population_dtype == torch.float32
    solver._init_batch_equilibrium(2)
    assert solver._f_batch.dtype == torch.float32
    assert solver._f_swap_batch.dtype == torch.float32


def test_knob_toggles_buffer_dtype_and_reallocates():
    device = torch.device("cpu")
    solver = _make_solver(GRID, device)
    solver._init_batch_equilibrium(2)
    fp32_f = solver._f_batch

    solver.batch_population_dtype = torch.bfloat16
    # Same C but different dtype -> must reallocate (dtype change is a realloc
    # trigger, not just a shape check).
    solver._init_batch_equilibrium(2)
    assert solver._f_batch.dtype == torch.bfloat16
    assert solver._f_swap_batch.dtype == torch.bfloat16
    assert solver._f_batch is not fp32_f

    # Macroscopic fields stay fp32 (compute is fp32 regardless of storage).
    assert solver._rho_batch.dtype == torch.float32
    assert solver._velocity_x_batch.dtype == torch.float32

    solver.batch_population_dtype = torch.float32
    solver._init_batch_equilibrium(2)
    assert solver._f_batch.dtype == torch.float32
    assert solver._f_swap_batch.dtype == torch.float32


def test_bf16_storage_solve_runs_and_is_finite():
    """The bf16-storage batched path completes a 5-step solve and returns
    finite macroscopic fields / populations (behavior assertion, not a
    precision gate). Uses the Triton-free CPU fallback so it runs everywhere."""
    device = torch.device("cpu")
    solver = _make_solver(GRID, device)
    solver.batch_population_dtype = torch.bfloat16
    mask = _sphere_mask(GRID).to(device).unsqueeze(0)
    geom_hashes = [compute_tensor_content_hash(mask[0])]
    solver._init_batch_equilibrium(1)
    solver.reset_force_accounting_batch(1)
    for _ in range(STEPS):
        ux, uy, uz, rho = solver.collide_and_stream_batch(
            OMEGA, mask, geom_hashes=geom_hashes
        )
        assert torch.isfinite(ux).all()
        assert torch.isfinite(uy).all()
        assert torch.isfinite(uz).all()
        assert torch.isfinite(rho).all()
    assert solver._f_batch.dtype == torch.bfloat16
    assert torch.isfinite(solver._f_batch).all()
    assert torch.isfinite(solver._force_x_accum_batch).all()
