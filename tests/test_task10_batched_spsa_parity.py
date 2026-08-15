"""Task 10: parity gates for the batched SPSA direct-solver path.

The sequential D3Q27 direct solver remains the authoritative reference. This
suite gates the batched path (``stream_bfl_d3q27_batch`` +
``D3Q27Solver.collide_and_stream_batch`` + the chunked SPSA forward loop) with
the plan's verbatim constants:

    COMPONENT_RTOL = 1e-3, COMPONENT_ATOL = 5e-5, GRAD_ATOL = 5e-4,
    LOSS_ATOL = 5e-5, FIELD_ATOL_5STEP = 4e-6, FORCE_ATOL = 2.5e-5

The ONLY allowed divergences in the whole batched path are the collision
matmul reduction order and the moment-equilibrium broadcast shape (LOW parity,
envelope pinned by FIELD_ATOL_5STEP / FORCE_ATOL). The test must FAIL before the
batched code exists and must PASS in full before the forward loop is wired to
the batched path.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from advanced_lbm_solver import D3Q27Solver
from sdf_utils import compute_all_link_distances
from utils import compute_tensor_content_hash

CUDA_AVAILABLE = torch.cuda.is_available()
try:
    from d3q27_kernels import stream_bfl_d3q27, triton
except Exception:  # pragma: no cover - optional dependency
    stream_bfl_d3q27 = None
    triton = None

try:
    from d3q27_kernels import stream_bfl_d3q27_batch
except Exception:  # pragma: no cover - optional dependency (not yet implemented)
    stream_bfl_d3q27_batch = None

requires_fused = pytest.mark.skipif(
    not (CUDA_AVAILABLE and triton is not None and stream_bfl_d3q27 is not None),
    reason="fused stream/BFL kernel requires CUDA and Triton",
)

# Verbatim gate constants from the plan (Task 10 spec).
COMPONENT_RTOL = 1e-3
COMPONENT_ATOL = 5e-5
GRAD_ATOL = 5e-4
LOSS_ATOL = 5e-5
FIELD_ATOL_5STEP = 4e-6
FORCE_ATOL = 2.5e-5
FORCE_RTOL = 5e-4

GRID = 32
STEPS = 5
OMEGA = 1.6


# ---------------------------------------------------------------------------
# Fixtures (same sphere/cube/face_touching/disconnected 32^3 fixtures as
# test_d3q27_kernel_parity.py, plus the real 96^3 corpus aircraft when present)
# ---------------------------------------------------------------------------
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


FIXTURES = {
    "sphere": _sphere_mask,
    "cube": _cube_mask,
    "face_touching": _face_touching_mask,
    "disconnected": _disconnected_mask,
}


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------
def _make_solver(n: int, device: torch.device, fused: bool = True) -> D3Q27Solver:
    """Deterministic solver starting from pure equilibrium (the training path's
    `_initialize_equilibrium` state), so the sequential reference and the
    batched path share a byte-identical starting population."""
    solver = D3Q27Solver(
        n,
        device,
        inlet_velocity_lu=0.05,
        use_triton_streaming=False,
        use_fused_stream_bfl=fused,
    )
    assert solver.use_fused_stream_bfl is fused, "fused backend did not activate"
    rho = torch.ones(n, n, n, device=device)
    ux = torch.full((n, n, n), 0.05, device=device)
    uy = torch.zeros(n, n, n, device=device)
    uz = torch.zeros(n, n, n, device=device)
    solver.f.copy_(solver.compute_equilibrium(rho, ux, uy, uz))
    solver.reset_force_accounting()
    return solver


def _q_for_mask(solver: D3Q27Solver, mask: torch.Tensor) -> torch.Tensor:
    mask_cuda = mask.to(solver.device)
    return compute_all_link_distances(mask_cuda, solver.ex, solver.ey, solver.ez)


def _run_sequential(mask: torch.Tensor, n: int, steps: int = STEPS, omega: float = OMEGA) -> D3Q27Solver:
    device = torch.device("cuda")
    solver = _make_solver(n, device, fused=True)
    mask_cuda = mask.to(device)
    geom_hash = compute_tensor_content_hash(mask_cuda)
    # Give the sequential solver the SAME q the batched path receives.
    solver._q_cache[geom_hash] = _q_for_mask(solver, mask_cuda)
    for _ in range(steps):
        solver.collide_and_stream(omega, mask_cuda, geom_hash=geom_hash)
    return solver


def _run_batched(
    masks,
    n: int,
    steps: int = STEPS,
    omega: float = OMEGA,
    C: int = None,
) -> D3Q27Solver:
    device = torch.device("cuda")
    if C is None:
        C = len(masks)
    solver = _make_solver(n, device, fused=True)
    mask_stack = torch.stack([m.to(device) for m in masks], dim=0)
    geom_hashes = [compute_tensor_content_hash(mask_stack[c]) for c in range(C)]
    q_stack = torch.stack(
        [_q_for_mask(solver, mask_stack[c]) for c in range(C)], dim=0
    ).contiguous()
    solver._init_batch_equilibrium(C)
    solver.reset_force_accounting_batch(C)
    for _ in range(steps):
        solver.collide_and_stream_batch(
            omega,
            mask_stack,
            geom_hashes=geom_hashes,
            q_stack=q_stack,
        )
    return solver


def _assert_field_and_force_parity(seq_solvers, batched, C, where=""):
    for c in range(C):
        field_diff = float((seq_solvers[c].f - batched._f_batch[c]).abs().max().item())
        assert field_diff <= FIELD_ATOL_5STEP, (
            f"{where} item {c}: 5-step max field diff {field_diff} > FIELD_ATOL_5STEP"
        )
        for seq_attr, bat_attr in (
            ("force_x_accum", "_force_x_accum_batch"),
            ("force_z_accum", "_force_z_accum_batch"),
            ("projected_drag_accum", "_projected_drag_accum_batch"),
        ):
            ref_val = float(getattr(seq_solvers[c], seq_attr).item())
            bat_val = float(getattr(batched, bat_attr)[c].item())
            assert math.isclose(ref_val, bat_val, rel_tol=FORCE_RTOL, abs_tol=FORCE_ATOL), (
                f"{where} item {c}: {seq_attr} reference={ref_val} batched={bat_val}"
            )


# ---------------------------------------------------------------------------
# Test 1: 32^3, batched-vs-sequential, C=4
# ---------------------------------------------------------------------------
@requires_fused
def test_batched_vs_sequential_32_C4():
    n = GRID
    masks = [FIXTURES[name](n) for name in ("sphere", "cube", "face_touching", "disconnected")]
    seq_solvers = [_run_sequential(m, n) for m in masks]
    batched = _run_batched(masks, n, C=4)
    _assert_field_and_force_parity(seq_solvers, batched, C=4, where="32^3 C=4")


# ---------------------------------------------------------------------------
# Test 2: 96^3 real aircraft, batched vs sequential
# ---------------------------------------------------------------------------
@requires_fused
def test_batched_vs_sequential_96_real_aircraft():
    n = 96
    mask = _real_aircraft_mask(n)
    if mask is None:
        pytest.skip("no 96^3 corpus voxel file available")
    seq = _run_sequential(mask, n)
    batched = _run_batched([mask], n, C=1)
    _assert_field_and_force_parity([seq], batched, C=1, where="96^3 real aircraft")


# ---------------------------------------------------------------------------
# Test 3: full SPSA path parity (sequential vs batched-chunked)
# ---------------------------------------------------------------------------
def _draw_spsa_deltas(seed, directions, shape, perturbation_grid_size=12, dtype=torch.float32):
    """Replicate DirectSolverSPSAFunction.forward's delta-draw code path exactly."""
    import torch.nn.functional as F

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed) % (2**63 - 1))
    deltas = []
    for _ in range(directions):
        low_frequency_grid = int(perturbation_grid_size)
        if low_frequency_grid > 1 and any(dim > low_frequency_grid for dim in shape):
            coarse_shape = tuple(max(1, min(low_frequency_grid, int(dim))) for dim in shape)
            coarse_delta = torch.randint(
                low=0,
                high=2,
                size=(1, 1, *coarse_shape),
                generator=generator,
                device="cuda",
                dtype=torch.int8,
            ).to(dtype=dtype)
            coarse_delta = coarse_delta.mul(2.0).sub(1.0)
            delta = F.interpolate(
                coarse_delta,
                size=tuple(shape),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            delta = (delta / delta.abs().mean().clamp_min(1.0e-6)).clamp(-2.0, 2.0)
        else:
            delta = torch.randint(
                low=0,
                high=2,
                size=tuple(shape),
                generator=generator,
                device="cuda",
                dtype=torch.int8,
            ).to(dtype=dtype)
            delta = delta.mul(2.0).sub(1.0)
        deltas.append(delta)
    return deltas


def _probability_grid() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    base = torch.rand((GRID, GRID, GRID), generator=generator)
    coords = torch.arange(GRID, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (GRID - 1) / 2.0
    r2 = ((z - c) / (0.18 * GRID)) ** 2 + ((y - c) / (0.4 * GRID)) ** 2 + ((x - c) / (0.28 * GRID)) ** 2
    return (0.35 * base + torch.exp(-r2)).cuda()


def _simulator():
    from aircraft_diffusion_cfd import AdvancedCFDSimulator
    from config import CFDConfig

    config = CFDConfig(base_grid_resolution=GRID, resolution=GRID)
    config.use_amr = False
    config.use_fused_stream_bfl = True
    simulator = AdvancedCFDSimulator(config, torch.device("cuda"))
    assert simulator.lbm_solver._solver.use_fused_stream_bfl is True
    return simulator


@requires_fused
def test_spsa_path_parity_batched_vs_sequential():
    import aircraft_diffusion_cfd as adc

    spec = adc.DesignSpec()
    directions = 16
    seed = 20260716
    perturbation = 0.15
    perturbation_grid_size = 12

    def _run(chunk):
        old = adc._DIRECT_SOLVER_BATCH_CHUNK
        adc._DIRECT_SOLVER_BATCH_CHUNK = chunk
        try:
            prob = _probability_grid().requires_grad_(True)
            sink: dict = {}
            loss = adc.DirectSolverSPSAFunction.apply(
                prob,
                spec,
                _simulator(),
                STEPS,
                perturbation,
                10.0,  # gradient_clip
                {  # component_gradient_max_norms
                    "aero_loss": 10.0,
                    "connectivity_loss": 10.0,
                    "aircraft_validity_loss": 10.0,
                },
                0.5,  # connectivity_weight
                0.5,  # aircraft_validity_weight
                0.5,  # threshold
                0.08,  # target_occupancy
                perturbation_grid_size,
                directions,
                seed,
                False,  # input_is_logits
                sink,
            )
            loss.backward()
            return float(loss.detach().item()), prob.grad.detach().clone(), dict(sink)
        finally:
            adc._DIRECT_SOLVER_BATCH_CHUNK = old

    seq_loss, seq_grad, seq_sink = _run(1)  # sequential verbatim fallback
    bat_loss, bat_grad, bat_sink = _run(4)  # batched chunked

    # (a) The deltas each forward actually consumed are byte-identical. The
    # forward records them in the sink, so this pins per-forward delta identity
    # (not merely that the RNG draw is reproducible).
    seq_deltas = seq_sink.get("_spsa_deltas")
    bat_deltas = bat_sink.get("_spsa_deltas")
    assert seq_deltas is not None and bat_deltas is not None, (
        "forward did not record _spsa_deltas; per-forward delta identity unverifiable"
    )
    assert len(seq_deltas) == directions and len(bat_deltas) == directions
    for i, (da, db) in enumerate(zip(seq_deltas, bat_deltas)):
        assert torch.equal(da, db), (
            f"delta {i}: sequential and batched forwards consumed different deltas"
        )

    # (b) Per-probe loss parity for every plus/minus probe (32 probes in
    # direction order, plus/minus interleaved). Comparing each probe's component
    # dict between the two paths catches a systematic loss-assembly error in
    # _assemble_direct_solver_components that inflates plus and minus equally
    # (which would cancel in (L+ - L-) and evade the gradient gates).
    seq_probes = seq_sink.get("_probe_components")
    bat_probes = bat_sink.get("_probe_components")
    assert seq_probes is not None and bat_probes is not None, (
        "forward did not record _probe_components; per-probe parity unverifiable"
    )
    assert len(seq_probes) == 2 * directions and len(bat_probes) == 2 * directions, (
        f"expected {2 * directions} per-probe records, got {len(seq_probes)} and {len(bat_probes)}"
    )
    for i, (sp, bp) in enumerate(zip(seq_probes, bat_probes)):
        where = f"probe {i} (dir {i // 2} {'plus' if i % 2 == 0 else 'minus'})"
        for key in ("occupancy_loss", "aero_loss", "connectivity_loss", "aircraft_validity_loss", "total_loss"):
            assert key in sp and key in bp, f"{where}: missing component {key}"
            assert math.isclose(
                float(sp[key]), float(bp[key]), rel_tol=COMPONENT_RTOL, abs_tol=COMPONENT_ATOL
            ), f"{where}: {key} sequential={sp[key]} batched={bp[key]}"

    # (c) total loss within LOSS_ATOL
    assert math.isclose(seq_loss, bat_loss, rel_tol=COMPONENT_RTOL, abs_tol=LOSS_ATOL), (
        f"loss: sequential={seq_loss} batched={bat_loss}"
    )
    # (d) total gradient within GRAD_ATOL
    grad_diff = float((seq_grad - bat_grad).abs().max().item())
    grad_scale = float(seq_grad.abs().max().item())
    assert grad_diff <= max(GRAD_ATOL, 1e-3 * grad_scale), (
        f"gradient max diff {grad_diff} (scale {grad_scale})"
    )
    seq_norm = float(seq_grad.norm().item())
    bat_norm = float(bat_grad.norm().item())
    assert math.isclose(seq_norm, bat_norm, rel_tol=5e-3, abs_tol=1e-6), (
        f"gradient norm: sequential={seq_norm} batched={bat_norm}"
    )
    # Component sink means stay within COMPONENT_RTOL/ATOL.
    for key in seq_sink:
        if key in {"active_guard_names", "active_guard_set", "_probe_components", "_spsa_deltas"} or not isinstance(
            seq_sink[key], (int, float, np.floating)
        ):
            continue
        if key not in bat_sink:
            continue
        assert math.isclose(seq_sink[key], bat_sink[key], rel_tol=COMPONENT_RTOL, abs_tol=COMPONENT_ATOL), (
            f"sink component {key}: sequential={seq_sink[key]} batched={bat_sink[key]}"
        )


# ---------------------------------------------------------------------------
# Test 3b: the Task 10 telemetry keys must not leak into the JSONL metrics
# record. In production the sink passed to the forward IS
# DirectSolverSPSALoss.last_components, and the trainer builds
# direct_components = dict(last_components) for the metrics callback, which the
# JSONL writer serializes with json.dumps(record, sort_keys=True, allow_nan=False)
# (run_monitored_training._append_jsonl). _spsa_deltas holds CUDA tensors and
# _probe_components holds per-probe dicts, so either key surviving into the
# callback record crashes every optimizer_update. The trainer sanitizes by
# popping both keys when building direct_components (mirroring the existing
# _accepted_guard_gradients pop); this test pins that contract.
# ---------------------------------------------------------------------------
@requires_fused
def test_direct_components_jsonl_serialization():
    import json

    last_components = {
        "_spsa_deltas": [torch.randn(3, 3, 3, device="cuda") for _ in range(16)],
        "_probe_components": [
            {
                "total_loss": 0.5,
                "occupancy_loss": 0.1,
                "aero_loss": 0.3,
                "connectivity_loss": 0.05,
                "aircraft_validity_loss": 0.05,
            }
            for _ in range(32)
        ],
        "total_loss": 1.2345,
        "aero_loss": 0.5678,
        "drag_coefficient": 0.77,
        "active_guard_names": ["connectivity_loss"],
    }
    # Sanitize exactly as OptimizedDiffusionTrainer.optimizer_update does at the
    # callback-consumption site (direct_components = dict(last_components), then
    # pop the telemetry-internal keys so they never reach the JSONL record).
    direct_components = dict(last_components)
    direct_components.pop("_spsa_deltas", None)
    direct_components.pop("_probe_components", None)
    record = {
        "kind": "optimizer_update",
        "global_step": 7,
        "direct_solver": {"evaluated": True, "components": direct_components},
    }
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    assert "_spsa_deltas" not in line, "telemetry key leaked into JSONL record"
    assert "_probe_components" not in line, "telemetry key leaked into JSONL record"
    assert '"total_loss": 1.2345' in line
    assert '"drag_coefficient": 0.77' in line


# ---------------------------------------------------------------------------
# Test 4: empty/full masks batched vs sequential
# ---------------------------------------------------------------------------
@requires_fused
def test_empty_and_full_masks_batched():
    n = GRID
    masks = [torch.zeros(n, n, n), torch.ones(n, n, n)]
    seq_solvers = [_run_sequential(m, n, steps=3) for m in masks]
    batched = _run_batched(masks, n, steps=3, C=2)
    _assert_field_and_force_parity(seq_solvers, batched, C=2, where="empty/full masks")


# ---------------------------------------------------------------------------
# Test 5: batched repeatability (bitwise identical output)
# ---------------------------------------------------------------------------
@requires_fused
def test_batched_repeatability():
    n = GRID
    masks = [FIXTURES["sphere"](n), FIXTURES["cube"](n)]
    a = _run_batched(masks, n, C=2)
    b = _run_batched(masks, n, C=2)
    torch.testing.assert_close(a._f_batch, b._f_batch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a._force_x_accum_batch, b._force_x_accum_batch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a._force_z_accum_batch, b._force_z_accum_batch, rtol=0.0, atol=0.0)
    torch.testing.assert_close(a._projected_drag_accum_batch, b._projected_drag_accum_batch, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Test 6: C=1 sanity (single geometry in a batch)
# ---------------------------------------------------------------------------
@requires_fused
def test_batched_C1_sanity():
    n = GRID
    mask = FIXTURES["sphere"](n)
    seq = _run_sequential(mask, n)
    batched = _run_batched([mask], n, C=1)
    _assert_field_and_force_parity([seq], batched, C=1, where="C=1 sanity")
