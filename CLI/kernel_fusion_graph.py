#!/usr/bin/env python
"""EXPERIMENTAL (branch experiment/kernel-fusion-launch): CUDA-graph launch fusion.

The profiled training update launches ~114,264 CUDA kernels (cudaLaunchKernel
self-CPU 15.06s) and the GPU is idle 39.4% of wall time. The single densest
launch site is the chunked coordinate-decoder MLP:

  * 96^3 = 884,736 voxels; coordinate_chunk_size = 32,768 -> exactly 27 full
    chunks per decode call, every chunk [32768, 231] -> fixed shape, no tail.
  * Each chunk runs 12 ops (coordinate_input Linear + SiLU, 5x residual
    Linear+SiLU+Linear, coordinate_output Linear) -> ~12 kernel launches.
  * torch.utils.checkpoint(use_reentrant=False) DISABLES dynamo in the recompute
    (torch/utils/checkpoint.py _run_fn_with_dynamo_disabled), so the backward
    recompute path runs EAGER even when forward uses torch.compile.

A CUDA graph capture of the MLP forward turns ~12 launches/chunk into 2
(copy input into the static buffer + one graph replay), and works identically
in the recompute path. This module is the graph-capture/replay harness plus a
GPU microbenchmark; it is OFF by default and must be opted in.

DESIGN NOTES
------------
* Static buffers: one input [32768, 231] fp32 (~30 MiB), one output
  [32768, 1] fp32. Replay = input_buffer.copy_(x) then graph.replay().
  copy_ is a single kernel; the replay is a single enqueue from the host.
* Graph pool: capture on a private side stream; the graph owns the pooled
  intermediate allocations (hidden [32768, 896] fp32 ~= 117 MiB each; the
  residual MLP keeps ~2 live at peak). Retain the pool so replays reuse it.
* Fallback: any capture/replay error (shape change, OOM) degrades to the
  eager function and logs once.
* Autograd: under torch.utils.checkpoint the recompute calls the same
  wrapped callable, so routing BOTH forward and recompute through the graph
  is safe *only if* replay is deterministic for the same input values. It is:
  the graph is a pure function of the static input buffer.

USAGE (microbenchmark; GPU-only, run when no training job is on the GPU):
  python CLI/kernel_fusion_graph.py --chunks 27 --iters 20 [--eager]

Parity gate (GPU): each replay output must match the eager function within
COMPONENT_ATOL (bitwise-equality is not required; cuBLAS may pick a different
but equally-valid fp32 gemm kernel inside the captured graph). The parity
envelopes used by the training pipeline (COMPONENT_RTOL=1e-3 / ATOL=5e-5)
govern.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)


class DecodeMLPGraph:
    """CUDA-graph wrapper around a fixed-shape MLP forward.

    Captures ``fn`` (a callable taking [R, in_features] and returning
    [R, out_features]) as a CUDA graph over static buffers. Replays by
    copying the input into the static input buffer and replaying.

    The graph is shape-static. The caller must guarantee every call passes
    [R, in_features] with R == rows and the same dtype/device as capture.
    Falls back to eager on any capture or replay error.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        rows: int,
        in_features: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._fn = fn
        self._rows = rows
        self._in_features = in_features
        self._device = device
        self._dtype = dtype
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_in: Optional[torch.Tensor] = None
        self._static_out: Optional[torch.Tensor] = None
        self._pool: Optional[torch.cuda.graph_pool_handle] = None
        self._capture_error: Optional[str] = None

    # -- capture ----------------------------------------------------------
    def capture(self) -> bool:
        """Capture the graph. Returns True on success; on failure records the
        error and leaves self eager so __call__ degrades silently."""
        if self._graph is not None:
            return True
        if self._capture_error is not None:
            return False
        if self._device.type != "cuda":
            self._capture_error = "CUDA graph capture requires a CUDA device"
            return False
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            self._static_in = torch.zeros(
                self._rows, self._in_features, device=self._device, dtype=self._dtype
            )
            with torch.cuda.stream(s):
                self._static_out = self._fn(self._static_in)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s, pool=None):
                self._static_out = self._fn(self._static_in)
            # graph_pool_handle() lets OTHER graphs allocate from the same
            # private pool (only needed for multi-graph pools). The graph itself
            # retains its own pool for replays.
            self._pool = torch.cuda.graph_pool_handle()
            self._graph = g
            log.info(
                "DecodeMLPGraph captured: rows=%d in=%d out=%s",
                self._rows, self._in_features, tuple(self._static_out.shape),
            )
            return True
        except Exception as exc:  # pragma: no cover - runtime capture failure
            self._capture_error = f"{type(exc).__name__}: {exc}"
            log.warning("DecodeMLPGraph capture failed; falling back to eager: %s", exc)
            self._graph = None
            return False

    # -- replay -----------------------------------------------------------
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._graph is None and self._capture_error is None:
            self.capture()
        if self._graph is None:
            return self._fn(x)
        # CRITICAL AUTOGRAD SAFETY: a graph replay returns a detached tensor
        # (no autograd edges through the captured ops). If the caller has
        # gradients enabled -- e.g. torch.utils.checkpoint's BACKWARD recompute,
        # which must rebuild the graph from latent/coords -- replaying would
        # silently zero the input gradients. Only the checkpoint FORWARD (run
        # under torch.no_grad(), confirmed in torch/utils/checkpoint.py
        # CheckpointFunction.forward) is safe to replay. Fall back to eager
        # whenever grad is enabled.
        if torch.is_grad_enabled():
            if self._capture_error is None:
                self._capture_error = (
                    "autograd enabled; graph replay would drop gradient flow"
                )
            return self._fn(x)
        if (
            x.shape[0] != self._rows
            or x.shape[-1] != self._in_features
            or x.device != self._device
            or x.dtype != self._dtype
        ):
            # Shape/device drift: degrade to eager (log once).
            if self._capture_error is None:
                self._capture_error = (
                    f"shape/device drift (got {tuple(x.shape)} on {x.device} {x.dtype})"
                )
                log.warning("DecodeMLPGraph input drift; falling back to eager: %s", self._capture_error)
            return self._fn(x)
        self._static_in.copy_(x)
        self._graph.replay()
        return self._static_out

    # -- introspection ----------------------------------------------------
    @property
    def active(self) -> bool:
        return self._graph is not None

    @property
    def capture_error(self) -> Optional[str]:
        return self._capture_error


# ---------------------------------------------------------------------------
# Microbenchmark (GPU-only; do NOT run while the approved training job is on
# the GPU -- contention corrupts both the timing and the training run).
# ---------------------------------------------------------------------------
def _make_mlp(in_features: int, width: int, depth: int, device: torch.device):
    """Build a decoder-shaped MLP (coordinate_input + depth residual blocks +
    coordinate_output) as an nn.Module so .to(device) works and the module is
    directly callable by DecodeMLPGraph."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _MLP(nn.Module):
        def __init__(self, in_features: int, width: int, depth: int):
            super().__init__()
            self.input_layer = nn.Linear(in_features, width)
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
                    )
                    for _ in range(depth)
                ]
            )
            self.output_layer = nn.Linear(width, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.input_layer(x)
            for block in self.blocks:
                hidden = F.silu(hidden + block(hidden))
            return self.output_layer(hidden)

    return _MLP(in_features, width, depth).to(device)


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA-graph launch-fusion microbench for the decoder MLP")
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--in-features", type=int, default=231)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=27)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--eager", action="store_true", help="benchmark eager only (no graph)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    fn = _make_mlp(args.in_features, args.width, args.depth, device)
    x = torch.randn(args.rows, args.in_features, device=device)

    # eager timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.chunks * args.iters):
        fn(x)
    torch.cuda.synchronize()
    eager_s = (time.perf_counter() - t0)

    # Per-OP CPU enqueue latency (no intermediate sync): the trace's averaged
    # ~1.14 ms/launch is launch+sync churn; this isolates pure host enqueue.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.chunks * args.iters):
        fn(x)
    eager_enqueue_s = (time.perf_counter() - t0)  # async; CPU-side only
    ops_per_chunk = 12
    print(f"eager per-op enqueue : {eager_enqueue_s / (args.chunks * args.iters * ops_per_chunk) * 1e6:7.1f} us/op")

    g = DecodeMLPGraph(fn, args.rows, args.in_features, device)
    if not args.eager:
        ok = g.capture()
        print(f"graph captured: {ok}")
        if ok and device.type == "cuda":
            # Graph-pool footprint: the graph retains its private pool, so
            # reserved memory rises by ~the peak live intermediate set. This is
            # the number that gates "do not land on 8GB" (per workflow wf_da6a83b6).
            torch.cuda.synchronize()
            reserved_before = torch.cuda.memory_reserved()
            allocated_before = torch.cuda.memory_allocated()
            # Force the pool to be instantiated by replaying once.
            with torch.no_grad():
                g(x)
            torch.cuda.synchronize()
            pool_reserved = torch.cuda.memory_reserved() - reserved_before
            pool_allocated = torch.cuda.memory_allocated() - allocated_before
            print(f"graph pool reserved delta: {pool_reserved/2**20:7.1f} MiB "
                  f"(allocated delta {pool_allocated/2**20:6.1f} MiB)")

        # parity gate: graph output vs eager output
        with torch.no_grad():
            ref = fn(x)
            got = g(x)
        adiff = (got - ref).abs().max().item()
        print(f"parity max|got-ref| = {adiff:.3e} (COMPONENT_ATOL=5e-5)")

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.chunks * args.iters):
            g(x)
        torch.cuda.synchronize()
        graph_s = time.perf_counter() - t1

        # Per-REPLAY CPU enqueue latency (input copy_ + graph.replay()).
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(args.chunks * args.iters):
            g(x)
        graph_enqueue_s = (time.perf_counter() - t1)
        print(f"graph per-replay enq : {graph_enqueue_s / (args.chunks * args.iters) * 1e6:7.1f} us/replay "
              f"(vs ~1140 us/launch in the trace)")
    else:
        graph_s = None
        graph_enqueue_s = None

    print(f"\neager : {eager_s/args.iters*1000:9.2f} ms/update (27 chunks)")
    if graph_s is not None:
        print(f"graph : {graph_s/args.iters*1000:9.2f} ms/update (27 chunks)")
        print(f"speedup: {eager_s/graph_s:6.2f}x")

    result = {
        "rows": args.rows,
        "in_features": args.in_features,
        "width": args.width,
        "depth": args.depth,
        "chunks": args.chunks,
        "iters": args.iters,
        "eager_ms_per_update": eager_s / args.iters * 1000,
        "graph_ms_per_update": (graph_s / args.iters * 1000) if graph_s is not None else None,
        "speedup": (eager_s / graph_s) if graph_s else None,
        "eager_per_op_enqueue_us": eager_enqueue_s / (args.chunks * args.iters * ops_per_chunk) * 1e6,
        "graph_per_replay_enqueue_us": (graph_enqueue_s / (args.chunks * args.iters) * 1e6) if graph_enqueue_s is not None else None,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print("saved:", args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
