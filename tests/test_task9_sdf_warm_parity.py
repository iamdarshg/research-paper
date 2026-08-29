"""Task 9 regression tests: SDF warm-cache bit-parity, delta RNG-order parity, hash guard.

Covers the EXACT-SAFE crux of Task 9 (perf: thread the 33 per-solve SDF
computations, commit a481291):

1. warm-vs-cold bit-parity: the SPSA objective run with the thread-pool warm
   SDF cache must be BIT-IDENTICAL to the same objective run with the warm
   cache disabled (every EDT serial in _get_q). It must also NEVER fall back to
   a serial EDT in the warm run (a wrong warm-cache key would silently do so),
   and must leave zero warm entries behind (memory bound).
2. delta RNG-order determinism: the hoisted delta draws captured from the real
   forward must be byte-identical to the pre-hoist sequential draw order for the
   same seed (the brief's "delicate part").
3. device-folded-hash guard: a warm entry keyed on the CPU tensor hash must
   MISS a _get_q lookup that uses the CUDA geometry hash, never silently
   returning the wrong q.

The warm-cache path is backend-agnostic, so the full-forward tests run on the
non-fused pytorch reference backend and are gated only on CUDA (no Triton
requirement). The wrong-key guard exercises D3Q27Solver._get_q directly.
"""

import torch
import torch.nn.functional as F

import pytest

import aircraft_diffusion_cfd as adc
import advanced_lbm_solver as lbs
from aircraft_diffusion_cfd import (
    DesignSpec,
    DirectSolverSPSAFunction,
    _find_q_solver,
)
from advanced_lbm_solver import D3Q27Solver, compute_all_link_distances
from config import CFDConfig
from sdf_utils import compute_sdf, gpu_exact_available
from utils import compute_tensor_content_hash

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Task 9 SDF warm-cache tests require CUDA",
)

GRID = 32
STEPS = 5
SEED = 20260716


def _simulator():
    from cfd_simulator import AdvancedCFDSimulator

    config = CFDConfig(base_grid_resolution=GRID, resolution=GRID)
    config.use_amr = False
    simulator = AdvancedCFDSimulator(config, torch.device("cuda"))
    inner = simulator.lbm_solver._solver
    # Warm-cache logic is backend-agnostic; keep these tests runnable on any
    # CUDA box without a Triton dependency.
    inner.use_fused_stream_bfl = False
    return simulator


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Task 9 GPU exact SDF warm-cache test requires CUDA",
)
def test_gpu_exact_prewarm_keeps_sdf_on_solver_device(monkeypatch):
    """When available, SPSA prewarm must use exact GPU EDT without CPU SDF hops."""

    if not gpu_exact_available(torch.device("cuda")):
        pytest.skip("CuPy exact EDT is not installed or parity-approved")

    class ImmediateFuture:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class InlinePool:
        def submit(self, function, *args, **kwargs):
            return ImmediateFuture(function(*args, **kwargs))

    monkeypatch.setattr(adc, "_SDF_POOL", InlinePool())
    solver = D3Q27Solver(
        resolution=8,
        device=torch.device("cuda"),
        use_fused_stream_bfl=False,
    )
    simulator = type(
        "SimulatorStub",
        (),
        {"device": torch.device("cuda"), "lbm_solver": type("Root", (), {"_solver": solver})()},
    )()
    sample_field = torch.full((8, 8, 8), 0.2, device="cuda")
    sample_field[2:6, 2:6, 2:6] = 0.8
    delta = torch.ones_like(sample_field)

    adc._clear_direct_solver_sdf_warm_cache(simulator)
    adc._warm_direct_solver_sdfs(
        sample_field,
        sample_field,
        [delta],
        eps=0.1,
        input_is_logits=False,
        threshold=0.5,
        cfd_simulator=simulator,
    )

    warm_entry = next(iter(solver._warm_sdf_cache.values())).result()
    assert warm_entry.is_cuda
    expected = torch.zeros((8, 8, 8), device="cuda")
    expected[2:6, 2:6, 2:6] = 1.0
    torch.testing.assert_close(
        warm_entry,
        compute_sdf(expected, backend="gpu_exact"),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Task 9 GPU exact objective parity test requires CUDA",
)
def test_gpu_exact_matches_scipy_for_complete_direct_objective(monkeypatch):
    """The exact GPU SDF must preserve the full SPSA loss and gradient."""
    if not gpu_exact_available(torch.device("cuda")):
        pytest.skip("CuPy exact EDT is not installed or parity-approved")

    original_config_value = adc.config_value

    def run(backend):
        monkeypatch.setattr(
            adc,
            "config_value",
            lambda section, key, default=None: (
                backend
                if section == "training" and key == "sdf_backend"
                else original_config_value(section, key, default)
            ),
        )
        sim = _simulator()
        adc._clear_direct_solver_geometry_caches(sim)
        adc._clear_direct_solver_sdf_warm_cache(sim)
        prob = _probability_grid().requires_grad_(True)
        sink = {}
        loss = DirectSolverSPSAFunction.apply(
            prob,
            DesignSpec(),
            sim,
            1,
            0.05,
            10.0,
            {"aero_loss": 10.0, "connectivity_loss": 10.0, "aircraft_validity_loss": 10.0},
            0.5,
            0.5,
            0.5,
            0.08,
            0,
            1,
            SEED,
            False,
            sink,
        )
        loss.backward()
        torch.cuda.synchronize()
        return float(loss.detach()), prob.grad.detach().clone(), sink

    reference_loss, reference_grad, reference_sink = run("scipy_reference")
    gpu_loss, gpu_grad, gpu_sink = run("gpu_exact")
    assert gpu_loss == pytest.approx(reference_loss, rel=1e-5, abs=1e-5)
    torch.testing.assert_close(gpu_grad, reference_grad, rtol=1e-5, atol=1e-5)
    for key in ("aero_loss", "connectivity_loss", "aircraft_validity_loss"):
        assert gpu_sink[key] == pytest.approx(reference_sink[key], rel=1e-5, abs=1e-5)


def _probability_grid():
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    base = torch.rand((GRID, GRID, GRID), generator=generator)
    coords = torch.arange(GRID, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (GRID - 1) / 2.0
    r2 = ((z - c) / (0.18 * GRID)) ** 2 + ((y - c) / (0.4 * GRID)) ** 2 + ((x - c) / (0.28 * GRID)) ** 2
    return (0.35 * base + torch.exp(-r2)).cuda()


def _deep_eq(a, b):
    """Recursive exact equality for scalars, tensors, lists, dicts, tuples."""
    if isinstance(a, (float, int)) and isinstance(b, (float, int)):
        return a == b
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    return a == b


def _reference_interleaved_deltas(sample_field, direction_count, generator, perturbation_grid_size):
    """Frozen pre-hoist (draw-one-use-one) delta construction.

    Replicates the exact RNG call sequence DirectSolverSPSAFunction.forward used
    before Task 9 hoisted the draws. Any change to the forward's draw order,
    seed handling, or low-frequency branch diverges the real deltas from this
    reference, so the test fails.
    """
    fields_dtype = sample_field.dtype
    deltas = []
    for _ in range(direction_count):
        low_frequency_grid = int(perturbation_grid_size)
        if low_frequency_grid > 1 and any(dim > low_frequency_grid for dim in sample_field.shape):
            coarse_shape = tuple(
                max(1, min(low_frequency_grid, int(dim))) for dim in sample_field.shape
            )
            coarse_delta = torch.randint(
                low=0,
                high=2,
                size=(1, 1, *coarse_shape),
                generator=generator,
                device=sample_field.device,
                dtype=torch.int8,
            ).to(dtype=fields_dtype)
            coarse_delta = coarse_delta.mul(2.0).sub(1.0)
            delta = F.interpolate(
                coarse_delta,
                size=tuple(sample_field.shape),
                mode="trilinear",
                align_corners=False,
            )[0, 0]
            delta = (delta / delta.abs().mean().clamp_min(1.0e-6)).clamp(-2.0, 2.0)
        else:
            delta = torch.randint(
                low=0,
                high=2,
                size=tuple(sample_field.shape),
                generator=generator,
                device=sample_field.device,
                dtype=torch.int8,
            ).to(dtype=fields_dtype)
            delta = delta.mul(2.0).sub(1.0)
        deltas.append(delta)
    return deltas


@requires_cuda
def test_warm_vs_cold_spsa_bit_parity(monkeypatch):
    """The warm and cold SPSA objectives must be bit-identical.

    Warm = thread-pool pre-computed CPU EDTs (Task 9 path). Cold = every EDT
    computed serially inside _get_q (the original path). The warm run must make
    ZERO cold fallback EDTs (a wrong warm-cache key would silently fall back),
    must use the pool at least once, and must leave zero warm entries behind
    (the bounded-residency memory bound).
    """
    # This regression specifically compares the established CPU warm path to
    # the serial CPU fallback. The GPU exact backend has its own gate below.
    monkeypatch.setattr(adc, "gpu_exact_available", lambda *_args, **_kwargs: False)
    directions = 4

    counts = {"warm": 0, "cold": 0}
    orig_pool = adc.compute_all_link_distances
    orig_cold = lbs.compute_all_link_distances

    def warm_fn(*a, **k):
        counts["warm"] += 1
        return orig_pool(*a, **k)

    def cold_fn(*a, **k):
        counts["cold"] += 1
        return orig_cold(*a, **k)

    # warm_fn intercepts the pool submissions (_refill_sdf_pool);
    # cold_fn intercepts the _get_q serial fallback.
    monkeypatch.setattr(adc, "compute_all_link_distances", warm_fn)
    monkeypatch.setattr(lbs, "compute_all_link_distances", cold_fn)

    def run(warm: bool):
        sim = _simulator()
        adc._clear_direct_solver_geometry_caches(sim)
        adc._clear_direct_solver_sdf_warm_cache(sim)
        counts["warm"] = 0
        counts["cold"] = 0
        if not warm:
            # Force the cold path: no-op pre-warm -> every solve EDTs in _get_q.
            monkeypatch.setattr(adc, "_warm_direct_solver_sdfs", lambda *a, **k: None)

        prob = _probability_grid().requires_grad_(True)
        sink = {}
        loss = DirectSolverSPSAFunction.apply(
            prob, DesignSpec(), sim, STEPS, 0.05, 10.0,
            {"aero_loss": 10.0, "connectivity_loss": 10.0, "aircraft_validity_loss": 10.0},
            0.5, 0.5, 0.5, 0.08, 0, directions, SEED, False, sink,
        )
        loss.backward()
        torch.cuda.synchronize()
        return {
            "loss": float(loss.detach().item()),
            "grad": prob.grad.detach().clone(),
            "sink": dict(sink),
            "warm_submissions": counts["warm"],
            "cold_fallback_edts": counts["cold"],
            "warm_leftover": len(_find_q_solver(sim)._warm_sdf_cache),
        }

    warm = run(warm=True)
    cold = run(warm=False)

    assert warm["cold_fallback_edts"] == 0, (
        f"warm run fell back to {warm['cold_fallback_edts']} serial EDTs "
        "(wrong warm-cache key?)"
    )
    assert warm["warm_submissions"] >= 1, "warm pool was never used"
    assert cold["cold_fallback_edts"] >= 1, (
        "cold mode did not exercise the serial EDT path"
    )
    assert warm["warm_leftover"] == 0, (
        f"{warm['warm_leftover']} warm q tensors leaked past the batch item"
    )

    assert warm["loss"] == cold["loss"], (
        f"warm loss {warm['loss']!r} != cold loss {cold['loss']!r}"
    )
    assert torch.equal(warm["grad"], cold["grad"]), (
        "SPSA gradient differs between warm and cold paths"
    )
    assert set(warm["sink"]) == set(cold["sink"])
    mismatched = [
        k for k in warm["sink"] if not _deep_eq(warm["sink"][k], cold["sink"][k])
    ]
    assert not mismatched, f"component sink differs between warm and cold: {mismatched}"


@requires_cuda
@pytest.mark.parametrize("perturbation_grid_size", [0, 12])
def test_delta_hoist_rng_order_parity(monkeypatch, perturbation_grid_size):
    """The forward's hoisted delta draws must match the pre-hoist sequential order.

    Captures the actual deltas from the real forward (via the _warm_direct_solver_sdfs
    hook, which receives the hoisted list) and compares them byte-for-byte against an
    independent frozen reference that reproduces the original draw-one-use-one RNG
    sequence for the same seed. 0 exercises the full-resolution branch; 12 exercises
    the low-frequency coarse branch (the trainer's realistic path at 96^3).
    """
    directions = 4
    captured = {}

    def capture(sample_field, sample_probs, deltas, eps, input_is_logits, threshold, cfd_simulator):
        captured["deltas"] = [d.detach().clone() for d in deltas]

    monkeypatch.setattr(adc, "_warm_direct_solver_sdfs", capture)

    prob = _probability_grid().requires_grad_(True)
    sink = {}
    DirectSolverSPSAFunction.apply(
        prob, DesignSpec(), _simulator(), STEPS, 0.05, 10.0,
        {"aero_loss": 10.0, "connectivity_loss": 10.0, "aircraft_validity_loss": 10.0},
        0.5, 0.5, 0.5, 0.08, perturbation_grid_size, directions, SEED, False, sink,
    )
    torch.cuda.synchronize()

    assert "deltas" in captured, "forward never invoked the pre-warm hook"
    got = captured["deltas"]
    assert len(got) == directions

    sample_field = prob.detach().float()
    generator = torch.Generator(device=sample_field.device)
    generator.manual_seed(int(SEED) % (2**63 - 1))
    reference = _reference_interleaved_deltas(
        sample_field, directions, generator, perturbation_grid_size
    )

    for i, (g, r) in enumerate(zip(got, reference)):
        assert torch.equal(g, r), (
            f"delta {i} diverges from the pre-hoist sequential order "
            f"(perturbation_grid_size={perturbation_grid_size})"
        )


@requires_cuda
def test_warm_lookup_cpu_key_misses_device_folded_hash():
    """A CPU-keyed warm entry must MISS a GPU-keyed _get_q lookup.

    compute_tensor_content_hash folds tensor.device.type into the key, so a
    warm fill keyed on the CPU tensor can never match the hash
    simulate_aerodynamics computes on the CUDA solver-frame geometry. This pins
    that caveat: the wrong-keyed entry must stay in the warm store (miss), and
    _get_q must fall back to the true q rather than silently return a wrong one.
    """
    device = torch.device("cuda")
    solver = D3Q27Solver(resolution=16, device=device, use_fused_stream_bfl=False)

    geom = torch.zeros(16, 16, 16, dtype=torch.float32)
    geom[4:12, 5:11, 6:10] = 1.0
    geom_cpu = geom.contiguous()
    geom_gpu = geom_cpu.to(device)

    key_cpu = compute_tensor_content_hash(geom_cpu)
    key_gpu = compute_tensor_content_hash(geom_gpu)
    assert key_cpu != key_gpu, "device type must be folded into the content hash"

    sentinel = torch.full((27, 16, 16, 16), -12345.0, dtype=torch.float32)

    # A buggy CPU-keyed warm entry must MISS a GPU-keyed _get_q lookup.
    solver._warm_sdf_cache = {key_cpu: sentinel.clone()}
    q = solver._get_q(geom_gpu, geom_hash=key_gpu)

    assert key_cpu in solver._warm_sdf_cache, (
        "CPU-keyed warm entry was popped by a GPU-key lookup (must miss)"
    )
    assert not torch.equal(q, sentinel.to(device)), "lookup returned a wrong-keyed q"
    reference = compute_all_link_distances(geom_gpu, solver.ex, solver.ey, solver.ez)
    assert torch.equal(q, reference), "cold fallback did not compute the true q"

    # A correctly GPU-keyed warm entry IS popped and returned.
    solver._q_cache = {}  # forget the computed q so the warm pop is exercised
    solver._warm_sdf_cache = {key_gpu: sentinel.clone()}
    q2 = solver._get_q(geom_gpu, geom_hash=key_gpu)
    assert key_gpu not in solver._warm_sdf_cache, "GPU-keyed warm entry was not popped"
    assert torch.equal(q2, sentinel.to(device)), "GPU-keyed warm entry returned wrong q"
