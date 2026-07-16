#!/usr/bin/env python3
"""Reproducible D3Q27 kernel benchmark: pytorch_reference vs fused stream/BFL.

Times the solver segments the fusion plan targets (SDF/q preparation, the
combined collide+stream step, the isolated stream+BFL segment, and
momentum-exchange force accumulation) with CUDA events plus wall timers, and
verifies field parity between backends while it runs. Writes machine-readable
JSON and a Markdown summary.

This benchmark must not run while a training process is using the GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from advanced_lbm_solver import D3Q27Solver


def _sphere_mask(n: int, radius_frac: float = 0.27) -> torch.Tensor:
    coords = torch.arange(n, dtype=torch.float32)
    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
    c = (n - 1) / 2.0
    r2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return (r2 <= (radius_frac * n) ** 2).float()


def _cube_mask(n: int) -> torch.Tensor:
    mask = torch.zeros(n, n, n)
    lo, hi = n // 3, 2 * n // 3
    mask[lo:hi, lo:hi, lo:hi] = 1.0
    return mask


def _real_mask(n: int):
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "build/aircraftverse_geometry_only_20260716/voxels",
        "build/aircraftverse_geometry_only_20260715/voxels",
    ):
        voxel_dir = root / rel
        if not voxel_dir.exists():
            continue
        for candidate in sorted(voxel_dir.glob("*.npy")):
            arr = np.load(candidate)
            if arr.shape == (n, n, n):
                return torch.from_numpy(arr.astype(np.float32)), candidate.name
    return None, None


def _make_solver(n: int, device: torch.device, fused: bool) -> D3Q27Solver:
    solver = D3Q27Solver(
        n,
        device,
        inlet_velocity_lu=0.05,
        use_fused_stream_bfl=fused,
    )
    rho = torch.ones(n, n, n, device=device)
    ux = torch.full((n, n, n), 0.05, device=device)
    feq = solver.compute_equilibrium(rho, ux, torch.zeros_like(rho), torch.zeros_like(rho))
    generator = torch.Generator(device="cpu").manual_seed(20260716)
    solver.f.copy_(feq + (torch.randn(feq.shape, generator=generator) * 1e-4).to(device))
    solver.reset_force_accounting()
    return solver


def _cuda_time(fn, warmup: int = 1, iters: int = 5) -> float:
    """Median CUDA-event time of fn() in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))


def benchmark_geometry(name: str, mask_cpu: torch.Tensor, n: int, steps: int) -> dict:
    device = torch.device("cuda")
    mask = mask_cpu.to(device)
    omega = 1.6
    result: dict = {"geometry": name, "grid": n, "steps": steps}

    # SDF + q preparation (CPU SciPy EDT path), measured once uncached.
    solver_prep = _make_solver(n, device, fused=False)
    import time as _time

    t0 = _time.perf_counter()
    solver_prep._get_q(mask)
    result["sdf_q_prep_wall_ms"] = (_time.perf_counter() - t0) * 1000.0

    per_backend = {}
    fields = {}
    for backend, fused in (("pytorch_reference", False), ("fused_stream_bfl", True)):
        solver = _make_solver(n, device, fused=fused)
        if fused and not solver.use_fused_stream_bfl:
            per_backend[backend] = {"available": False}
            continue
        solver._get_q(mask)  # warm the q cache; prep cost reported separately

        torch.cuda.nvtx.range_push(f"{name}/{backend}/full_step")
        step_ms = _cuda_time(lambda: solver.collide_and_stream(omega, mask))
        torch.cuda.nvtx.range_pop()

        # Isolated stream+BFL segment on frozen pre-stream state.
        solver.f_pre_stream.copy_(solver.f)

        if fused:
            from d3q27_kernels import stream_bfl_d3q27

            q = solver._get_q(mask).contiguous()
            solid_u8 = (mask > 0.5).to(torch.uint8).contiguous()

            def _segment():
                stream_bfl_d3q27(
                    solver.f_pre_stream, solver.f_temp, solid_u8, q,
                    solver.ex, solver.ey, solver.ez, solver.opposite,
                )
        else:

            def _segment():
                for i in range(27):
                    solver.f_temp[i] = torch.roll(
                        solver.f[i], shifts=solver._stream_shifts[i], dims=(0, 1, 2)
                    )
                solver._apply_bfl_boundary(mask)

        torch.cuda.nvtx.range_push(f"{name}/{backend}/stream_bfl")
        segment_ms = _cuda_time(_segment)
        torch.cuda.nvtx.range_pop()

        if fused:
            force_fn = lambda: solver._accumulate_momentum_exchange_force_nosync(mask)
        else:
            force_fn = lambda: solver._accumulate_momentum_exchange_force(mask)
        force_ms = _cuda_time(force_fn)

        # Fresh solver for the multi-step run used in parity + throughput.
        runner = _make_solver(n, device, fused=fused)
        runner._get_q(mask)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(steps):
            runner.collide_and_stream(omega, mask)
        torch.cuda.synchronize()
        run_wall_ms = (_time.perf_counter() - t0) * 1000.0

        per_backend[backend] = {
            "available": True,
            "full_step_ms": step_ms,
            "stream_bfl_segment_ms": segment_ms,
            "force_accumulation_ms": force_ms,
            f"{steps}_step_wall_ms": run_wall_ms,
            "force_x_accum": float(runner.force_x_accum.item()),
            "force_z_accum": float(runner.force_z_accum.item()),
            "projected_drag_accum": float(runner.projected_drag_accum.item()),
        }
        fields[backend] = runner.f.clone()

    result["backends"] = per_backend
    if all(per_backend.get(b, {}).get("available") for b in ("pytorch_reference", "fused_stream_bfl")):
        diff = float((fields["pytorch_reference"] - fields["fused_stream_bfl"]).abs().max().item())
        result["max_population_diff_after_run"] = diff
        ref = per_backend["pytorch_reference"]
        fus = per_backend["fused_stream_bfl"]
        result["speedup_full_step"] = ref["full_step_ms"] / max(fus["full_step_ms"], 1e-9)
        result["speedup_stream_bfl_segment"] = (
            ref["stream_bfl_segment_ms"] / max(fus["stream_bfl_segment_ms"], 1e-9)
        )
        result["speedup_multi_step_wall"] = (
            ref[f"{steps}_step_wall_ms"] / max(fus[f"{steps}_step_wall_ms"], 1e-9)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", default="32,96")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device required for this benchmark")

    date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
    output_dir = Path(args.output_dir or f"build/d3q27_kernel_profile_{date_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_sizes = [int(v) for v in str(args.grid_sizes).split(",") if v.strip()]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "steps": int(args.steps),
        "results": [],
    }
    try:
        import triton

        report["triton_version"] = triton.__version__
    except Exception:
        report["triton_version"] = None

    for n in grid_sizes:
        geometries = [("sphere", _sphere_mask(n)), ("cube", _cube_mask(n))]
        real, real_name = _real_mask(n)
        if real is not None:
            geometries.append((f"real:{real_name}", real))
        for geom_name, mask in geometries:
            print(f"benchmarking {geom_name} at {n}^3 ...", flush=True)
            report["results"].append(benchmark_geometry(geom_name, mask, n, int(args.steps)))

    json_path = output_dir / "kernel_profile.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# D3Q27 Kernel Benchmark",
        "",
        f"Generated: `{report['created_at']}`  ",
        f"Device: `{report['device']}`, torch `{report['torch_version']}`, triton `{report['triton_version']}`",
        "",
        "| Geometry | Grid | Ref step (ms) | Fused step (ms) | Step speedup | Ref stream+BFL (ms) | Fused stream+BFL (ms) | Segment speedup | Max pop diff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        ref = row["backends"].get("pytorch_reference", {})
        fus = row["backends"].get("fused_stream_bfl", {})
        if not (ref.get("available") and fus.get("available")):
            continue
        lines.append(
            f"| {row['geometry']} | {row['grid']} | {ref['full_step_ms']:.2f} | {fus['full_step_ms']:.2f} "
            f"| {row['speedup_full_step']:.2f}x | {ref['stream_bfl_segment_ms']:.2f} | {fus['stream_bfl_segment_ms']:.2f} "
            f"| {row['speedup_stream_bfl_segment']:.2f}x | {row['max_population_diff_after_run']:.2e} |"
        )
    (output_dir / "kernel_profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
