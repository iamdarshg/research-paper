from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional GPU acceleration path
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _stream_bounce_kernel(
        f,
        f_pre,
        f_temp,
        solid,
        ex,
        ey,
        ez,
        opposite,
        n: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        q = tl.program_id(0)
        pid = tl.program_id(1)
        offsets = pid * block + tl.arange(0, block)
        valid = offsets < total

        n2 = n * n
        x = offsets // n2
        rem = offsets - x * n2
        y = rem // n
        z = rem - y * n

        dx = tl.load(ex + q)
        dy = tl.load(ey + q)
        dz = tl.load(ez + q)

        src_x = (x - dx + n) % n
        src_y = (y - dy + n) % n
        src_z = (z - dz + n) % n
        src_offsets = src_x * n2 + src_y * n + src_z

        opp_q = tl.load(opposite + q)
        streamed = tl.load(f + q * total + src_offsets, mask=valid, other=0.0)
        reflected = tl.load(f_pre + opp_q * total + offsets, mask=valid, other=0.0)
        is_fluid = tl.load(solid + offsets, mask=valid, other=1.0) <= 0.5
        source_is_solid = tl.load(solid + src_offsets, mask=valid, other=0.0) > 0.5
        out = tl.where(is_fluid & source_is_solid & (q != 0), reflected, streamed)
        tl.store(f_temp + q * total + offsets, out, mask=valid)


def stream_bounce_d3q27(
    f: torch.Tensor,
    f_pre: torch.Tensor,
    f_temp: torch.Tensor,
    solid_mask: torch.Tensor,
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
    opposite: torch.Tensor,
    block_size: int = 256,
) -> bool:
    """Run fused D3Q27 streaming + fluid-node link bounce-back on CUDA via Triton.

    Returns True when the Triton path was used, False when the caller should
    fall back to the ordinary PyTorch implementation.
    """
    if triton is None or not f.is_cuda:
        return False
    if f.dim() != 4 or f.shape[0] != 27 or f.shape[1] != f.shape[2] or f.shape[1] != f.shape[3]:
        return False
    if not (f.is_contiguous() and f_pre.is_contiguous() and f_temp.is_contiguous()):
        return False

    n = int(f.shape[1])
    total = n * n * n
    grid = (27, triton.cdiv(total, block_size))
    _stream_bounce_kernel[grid](
        f,
        f_pre,
        f_temp,
        solid_mask.contiguous(),
        ex,
        ey,
        ez,
        opposite,
        n,
        total,
        block=block_size,
    )
    return True
