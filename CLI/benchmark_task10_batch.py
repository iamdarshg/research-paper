"""Task 10 C-benchmark: collapse per-step kernel-launch overhead by batching
the 32 SPSA probe D3Q27 solves into chunks of C simultaneous solves.

Measures the probe phase (the only phase Task 10 changes) at 96^3:
  - pre-warm all 32 probe q (SDF) tensors BEFORE timing so the timed region is
    the GPU solve phase (the EDT/SDF CPU work is overlapped/identical across C),
  - run the 32 probes sequentially (C=1) or in chunks of C probes,
  - record wall time + peak VRAM (torch.cuda.max_memory_allocated).

A real 96^3 corpus aircraft voxel seeds the base probability grid; the 32
probes are the usual plus/minus SPSA perturbations.

Usage:
    python CLI/benchmark_task10_batch.py [--batch 1,2,4,8] [--grid 96]
"""
import argparse
import glob
import sys
import time
import contextlib
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CLI"))
sys.path.insert(0, str(ROOT))

from aircraft_diffusion_cfd import (  # noqa: E402
    AdvancedCFDSimulator,
    DesignSpec,
    _binarize_probability_grid_for_solver,
    _canonical_training_geometry_to_solver_xyz,
    _clear_direct_solver_batch_workspace,
    _clear_direct_solver_geometry_caches,
    _direct_measured_objective_for_single,
    _direct_measured_objectives_batch,
)
from config import CFDConfig  # noqa: E402
from sdf_utils import compute_all_link_distances  # noqa: E402
from utils import compute_tensor_content_hash  # noqa: E402

GRID = 96
STEPS = 5
SEED = 20260716
PERTURBATION = 0.15
PERT_GRID = 12


def _draw_spsa_deltas(seed, directions, shape):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed) % (2**63 - 1))
    deltas = []
    for _ in range(directions):
        if PERT_GRID > 1 and any(dim > PERT_GRID for dim in shape):
            coarse_shape = tuple(max(1, min(PERT_GRID, int(d))) for d in shape)
            coarse = torch.randint(0, 2, size=(1, 1, *coarse_shape), generator=generator,
                                   device="cuda", dtype=torch.int8).to(dtype=torch.float32)
            coarse = coarse.mul(2.0).sub(1.0)
            delta = F.interpolate(coarse, size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]
            delta = (delta / delta.abs().mean().clamp_min(1.0e-6)).clamp(-2.0, 2.0)
        else:
            delta = torch.randint(0, 2, size=tuple(shape), generator=generator,
                                  device="cuda", dtype=torch.int8).to(dtype=torch.float32)
            delta = delta.mul(2.0).sub(1.0)
        deltas.append(delta)
    return deltas


def _make_sim():
    config = CFDConfig(base_grid_resolution=GRID, resolution=GRID)
    config.use_amr = False
    config.use_fused_stream_bfl = True
    sim = AdvancedCFDSimulator(config, torch.device("cuda"))
    assert sim.lbm_solver._solver.use_fused_stream_bfl is True
    return sim


def _base_probability_grid():
    voxels = sorted(glob.glob(str(ROOT / "build" / "aircraftverse_geometry_only_20260715" / "voxels" / "*.npy")))
    if not voxels:
        raise RuntimeError("no corpus voxel files; set up the corpus or use --grid with a synthetic mask")
    # Pick the densest 96^3 corpus voxel so the geometry does real boundary work.
    best = None
    best_occ = -1.0
    for v in voxels:
        arr = np.load(v)
        if arr.shape != (GRID, GRID, GRID):
            continue
        occ = float(arr.mean())
        if occ > best_occ:
            best_occ = occ
            best = arr
    if best is None:
        raise RuntimeError(f"no corpus voxel matches {(GRID, GRID, GRID)}")
    return torch.from_numpy(best.astype(np.float32)).cuda()


def _build_probes(base_prob, deltas):
    probes = []
    for d in deltas:
        plus = (base_prob + PERTURBATION * d).clamp(0.0, 1.0)
        minus = (base_prob - PERTURBATION * d).clamp(0.0, 1.0)
        probes.append(plus)
        probes.append(minus)
    return probes


def _prewarm_q(sim, probes):
    """Compute each probe's q on GPU and stash in the warm cache (CPU tensors,
    matching the real path's CPU->GPU pop). Returns nothing."""
    solver = sim.lbm_solver._solver
    ex, ey, ez = solver.ex.cpu(), solver.ey.cpu(), solver.ez.cpu()
    warm = solver._warm_sdf_cache
    for g in probes:
        geom_cpu = _binarize_probability_grid_for_solver(g.detach().to("cpu"), threshold=0.5, target_occupancy=None)
        solver_geom_cpu = _canonical_training_geometry_to_solver_xyz(geom_cpu)
        solver_geom_gpu = solver_geom_cpu.to(sim.device)
        key = compute_tensor_content_hash(solver_geom_gpu)
        if key in warm:
            continue
        q = compute_all_link_distances(solver_geom_cpu, ex, ey, ez)
        warm[key] = q
    # Pop the base-solve key too (not used by the probe phase) is unnecessary.


def run_probes(sim, probes, batch_size, spec):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        if batch_size == 1:
            for g in probes:
                _direct_measured_objective_for_single(
                    g.detach(), spec, sim, STEPS, 0.0, 0.0, 0.5, 0.08,
                    return_components=False,
                )
                _clear_direct_solver_geometry_caches(sim)
        else:
            for start in range(0, len(probes), batch_size):
                _direct_measured_objectives_batch(
                    probes[start:start + batch_size], spec, sim, STEPS, 0.0, 0.0, 0.5, 0.08
                )
                _clear_direct_solver_geometry_caches(sim)
                _clear_direct_solver_batch_workspace(sim)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated() / 1.0e6
    return wall_ms, peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,2,4,8")
    ap.add_argument("--grid", type=int, default=GRID)
    ap.add_argument("--directions", type=int, default=16)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()
    globals()["GRID"] = args.grid
    batch_sizes = [int(x) for x in args.batch.split(",")]

    print(f"grid={GRID} steps={STEPS} directions={args.directions} batch_sizes={batch_sizes}")
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    spec = DesignSpec()
    sim = _make_sim()
    base_prob = _base_probability_grid()
    deltas = _draw_spsa_deltas(SEED, args.directions, (GRID, GRID, GRID))
    probes = _build_probes(base_prob, deltas)
    print(f"probes={len(probes)} base_occupancy={float(base_prob.mean().item()):.4f}")

    _prewarm_q(sim, probes)

    rows = []
    for bs in batch_sizes:
        best_ms = float("inf")
        peak_mb = 0.0
        for rep in range(max(1, args.repeat)):
            wall_ms, peak = run_probes(sim, probes, bs, spec)
            best_ms = min(best_ms, wall_ms)
            peak_mb = max(peak_mb, peak)
        rows.append((bs, best_ms, peak_mb))
        print(f"C={bs:2d}: wall={best_ms:9.2f} ms   peak_vram={peak_mb:9.1f} MB")

    base_ms = next((r[1] for r in rows if r[0] == 1), rows[0][1])
    print("\nsummary (C -> wall ms, peak MB, speedup):")
    for bs, ms, peak in rows:
        print(f"  C={bs:2d}: {ms:9.2f} ms  {peak:9.1f} MB  speedup={base_ms / max(ms, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
