"""Direct-objective and SPSA-gradient parity for the fused stream/BFL backend.

Plan gates 5 and 6 (docs/benchmarks/grounded_500_28m_profile_20260715/
mrt_fusion_parallelization_plan.md): with identical geometry, seeds, and SPSA
deltas, every loss component and the resulting gradient tensor must match the
pytorch_reference backend within the measured kernel-parity envelope.

Run at 32^3 for test-suite speed; the underlying kernels are grid-size
agnostic and tests/test_d3q27_kernel_parity.py covers 96^3 fields directly.
"""

import math

import pytest
import torch

from aircraft_diffusion_cfd import (
    DesignSpec,
    DirectSolverSPSAFunction,
    _direct_measured_objective_for_single,
)
from config import CFDConfig

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

# Component/gradient tolerances derive from the kernel-parity envelope in
# tests/test_d3q27_kernel_parity.py (fields <= 4e-6 over five steps). The
# coefficient extraction adds no stochastic terms, so components track the
# field envelope; SPSA gradients difference two nearly-equal losses, which
# amplifies the relative (not absolute) error.
COMPONENT_RTOL = 1e-3
COMPONENT_ATOL = 5e-5
GRAD_ATOL = 5e-4
LOSS_ATOL = 5e-5

GRID = 32
STEPS = 5


def _simulator(fused: bool):
    from cfd_simulator import AdvancedCFDSimulator

    config = CFDConfig(base_grid_resolution=GRID, resolution=GRID)
    config.use_amr = False
    device = torch.device("cuda")
    simulator = AdvancedCFDSimulator(config, device)
    inner = simulator.lbm_solver._solver
    if fused:
        inner.use_fused_stream_bfl = True
        assert stream_bfl_d3q27 is not None
    else:
        assert inner.use_fused_stream_bfl is False
    return simulator


def _probability_grid() -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    base = torch.rand((GRID, GRID, GRID), generator=generator)
    # A soft ellipsoidal bias produces an aircraft-scale connected blob after
    # thresholding rather than salt-and-pepper noise.
    coords = torch.arange(GRID, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (GRID - 1) / 2.0
    r2 = ((z - c) / (0.18 * GRID)) ** 2 + ((y - c) / (0.4 * GRID)) ** 2 + ((x - c) / (0.28 * GRID)) ** 2
    return (0.35 * base + torch.exp(-r2)).cuda()


@requires_fused
def test_direct_objective_component_parity():
    prob = _probability_grid()
    spec = DesignSpec()
    ref = _direct_measured_objective_for_single(
        prob, spec, _simulator(False), STEPS,
        connectivity_weight=0.5, aircraft_validity_weight=0.5,
        threshold=0.5, target_occupancy=0.08, return_components=True,
    )
    fused = _direct_measured_objective_for_single(
        prob, spec, _simulator(True), STEPS,
        connectivity_weight=0.5, aircraft_validity_weight=0.5,
        threshold=0.5, target_occupancy=0.08, return_components=True,
    )
    assert set(ref) == set(fused)
    for key in sorted(ref):
        assert math.isclose(ref[key], fused[key], rel_tol=COMPONENT_RTOL, abs_tol=COMPONENT_ATOL), (
            f"component {key}: reference={ref[key]} fused={fused[key]}"
        )


@requires_fused
def test_spsa_loss_and_gradient_parity():
    spec = DesignSpec()
    directions = 4
    seed = 20260716

    results = {}
    for name, fused in (("reference", False), ("fused", True)):
        prob = _probability_grid().requires_grad_(True)
        sink: dict = {}
        loss = DirectSolverSPSAFunction.apply(
            prob, spec, _simulator(fused), STEPS,
            0.05,      # perturbation
            10.0,      # gradient_clip
            {           # component_gradient_max_norms
                "aero_loss": 10.0,
                "connectivity_loss": 10.0,
                "aircraft_validity_loss": 10.0,
            },
            0.5,       # connectivity_weight
            0.5,       # aircraft_validity_weight
            0.5,       # threshold
            0.08,      # target_occupancy
            0,         # perturbation_grid_size (0 = full-resolution deltas)
            directions,
            seed,
            False,     # input_is_logits
            sink,
        )
        loss.backward()
        results[name] = {
            "loss": float(loss.detach().item()),
            "grad": prob.grad.detach().clone(),
            "sink": dict(sink),
        }

    ref, fus = results["reference"], results["fused"]
    assert math.isclose(ref["loss"], fus["loss"], rel_tol=COMPONENT_RTOL, abs_tol=LOSS_ATOL), (
        f"loss: reference={ref['loss']} fused={fus['loss']}"
    )
    grad_diff = float((ref["grad"] - fus["grad"]).abs().max().item())
    grad_scale = float(ref["grad"].abs().max().item())
    assert grad_diff <= max(GRAD_ATOL, 1e-3 * grad_scale), (
        f"gradient max diff {grad_diff} (scale {grad_scale})"
    )
    ref_norm = float(ref["grad"].norm().item())
    fus_norm = float(fus["grad"].norm().item())
    assert math.isclose(ref_norm, fus_norm, rel_tol=5e-3, abs_tol=1e-6), (
        f"gradient norm: reference={ref_norm} fused={fus_norm}"
    )
