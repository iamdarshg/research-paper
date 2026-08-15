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


if triton is not None:
    @triton.jit
    def _stream_bfl_kernel(
        f_pre,
        f_out,
        solid,
        q_field,
        ex,
        ey,
        ez,
        opposite,
        n: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        """Fused pull-streaming + q-dependent BFL interpolation for D3Q27.

        Reproduces the reference PyTorch path exactly:
        - plain streaming is a periodic pull (`torch.roll` semantics);
        - the BFL override applies at fluid cells whose in-domain neighbor in
          direction i = opposite(k) is solid (zero-padded neighbor semantics);
        - q < 0.5:  f_out[k](x) = (1-2q) f_pre[i](x - e_i) + 2q f_pre[i](x)
          with the upstream neighbor read as zero outside the domain;
        - q >= 0.5: f_out[k](x) = (1/(2q)) f_pre[i](x) + (1 - 1/(2q)) s
          where s is the plain periodically streamed value for k at x.
        Each (k, x) output is written exactly once, so the launch is
        order-independent and requires no host synchronization.
        """
        k = tl.program_id(0)
        pid = tl.program_id(1)
        offsets = pid * block + tl.arange(0, block)
        valid = offsets < total

        n2 = n * n
        x = offsets // n2
        rem = offsets - x * n2
        y = rem // n
        z = rem - y * n

        dxk = tl.load(ex + k)
        dyk = tl.load(ey + k)
        dzk = tl.load(ez + k)

        # Plain streaming: periodic pull from x - e_k (torch.roll parity).
        sx = (x - dxk + n) % n
        sy = (y - dyk + n) % n
        sz = (z - dzk + n) % n
        streamed = tl.load(f_pre + k * total + (sx * n2 + sy * n + sz), mask=valid, other=0.0)

        # BFL incoming direction i = opposite(k). The lattice's opposite table
        # is NOT the geometric negation for edge directions, so e_i must be
        # loaded from the direction table rather than derived as -e_k; the
        # reference implementation indexes shifts by i and is authoritative.
        i = tl.load(opposite + k)
        dxi = tl.load(ex + i)
        dyi = tl.load(ey + i)
        dzi = tl.load(ez + i)

        # Boundary link: fluid at x, solid at the in-domain neighbor x + e_i.
        nb_x = x + dxi
        nb_y = y + dyi
        nb_z = z + dzi
        nb_in = (nb_x >= 0) & (nb_x < n) & (nb_y >= 0) & (nb_y < n) & (nb_z >= 0) & (nb_z < n)
        cell_solid = tl.load(solid + offsets, mask=valid, other=1) > 0
        nb_solid = tl.load(
            solid + (nb_x * n2 + nb_y * n + nb_z),
            mask=valid & nb_in,
            other=0,
        ) > 0
        active = valid & (~cell_solid) & nb_in & nb_solid & (i != 0)

        qi = tl.load(q_field + i * total + offsets, mask=active, other=1.0)
        f_i_here = tl.load(f_pre + i * total + offsets, mask=active, other=0.0)

        # Upstream fluid neighbor for direction i sits at x - e_i and reads as
        # zero outside the domain (reference zero-padding).
        up_x = x - dxi
        up_y = y - dyi
        up_z = z - dzi
        up_in = (up_x >= 0) & (up_x < n) & (up_y >= 0) & (up_y < n) & (up_z >= 0) & (up_z < n)
        f_i_up = tl.load(
            f_pre + i * total + (up_x * n2 + up_y * n + up_z),
            mask=active & up_in,
            other=0.0,
        )

        res_low = (1.0 - 2.0 * qi) * f_i_up + 2.0 * qi * f_i_here
        inv_2q = 1.0 / (2.0 * qi)
        res_high = inv_2q * f_i_here + (1.0 - inv_2q) * streamed
        res = tl.where(qi < 0.5, res_low, res_high)

        out = tl.where(active, res, streamed)
        tl.store(f_out + k * total + offsets, out, mask=valid)


def stream_bfl_d3q27(
    f_pre: torch.Tensor,
    f_out: torch.Tensor,
    solid_mask_u8: torch.Tensor,
    q_field: torch.Tensor,
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
    opposite: torch.Tensor,
    block_size: int = 256,
) -> bool:
    """Run the fused D3Q27 streaming + BFL boundary kernel on CUDA via Triton.

    Returns True when the fused path executed, False when the caller must use
    the reference PyTorch implementation. This kernel never substitutes the
    simplified bounce-back physics of `stream_bounce_d3q27`.
    """
    if triton is None or not f_pre.is_cuda:
        return False
    if f_pre.dim() != 4 or f_pre.shape[0] != 27 or f_pre.shape[1] != f_pre.shape[2] or f_pre.shape[1] != f_pre.shape[3]:
        return False
    if q_field.shape != f_pre.shape:
        return False
    if not (f_pre.is_contiguous() and f_out.is_contiguous() and q_field.is_contiguous()):
        return False
    if solid_mask_u8.dtype != torch.uint8 or not solid_mask_u8.is_contiguous():
        return False

    n = int(f_pre.shape[1])
    total = n * n * n
    grid = (27, triton.cdiv(total, block_size))
    _stream_bfl_kernel[grid](
        f_pre,
        f_out,
        solid_mask_u8,
        q_field,
        ex,
        ey,
        ez,
        opposite,
        n,
        total,
        block=block_size,
    )
    return True


if triton is not None:
    @triton.jit
    def _stream_bfl_kernel_batch(
        f_pre,
        f_out,
        solid,
        q_field,
        ex,
        ey,
        ez,
        opposite,
        n: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        """Batched fused pull-streaming + q-dependent BFL interpolation.

        Identical per-item formulas and write pattern as ``_stream_bfl_kernel``
        with a leading batch program id ``c``. Every load is offset by
        ``c * total`` (populations and solid) or ``c * 27 * total`` (q_field),
        so each (c, k, x) output is written exactly once and the launch is
        order-independent across the batch dim.
        """
        c = tl.program_id(0)
        k = tl.program_id(1)
        pid = tl.program_id(2)
        offsets = pid * block + tl.arange(0, block)
        valid = offsets < total

        n2 = n * n
        x = offsets // n2
        rem = offsets - x * n2
        y = rem // n
        z = rem - y * n

        f_pre_c = f_pre + c * 27 * total
        f_out_c = f_out + c * 27 * total
        solid_c = solid + c * total
        q_c = q_field + c * 27 * total

        dxk = tl.load(ex + k)
        dyk = tl.load(ey + k)
        dzk = tl.load(ez + k)

        # Plain streaming: periodic pull from x - e_k (torch.roll parity).
        sx = (x - dxk + n) % n
        sy = (y - dyk + n) % n
        sz = (z - dzk + n) % n
        streamed = tl.load(f_pre_c + k * total + (sx * n2 + sy * n + sz), mask=valid, other=0.0)

        # BFL incoming direction i = opposite(k).
        i = tl.load(opposite + k)
        dxi = tl.load(ex + i)
        dyi = tl.load(ey + i)
        dzi = tl.load(ez + i)

        # Boundary link: fluid at x, solid at the in-domain neighbor x + e_i.
        nb_x = x + dxi
        nb_y = y + dyi
        nb_z = z + dzi
        nb_in = (nb_x >= 0) & (nb_x < n) & (nb_y >= 0) & (nb_y < n) & (nb_z >= 0) & (nb_z < n)
        cell_solid = tl.load(solid_c + offsets, mask=valid, other=1) > 0
        nb_solid = tl.load(
            solid_c + (nb_x * n2 + nb_y * n + nb_z),
            mask=valid & nb_in,
            other=0,
        ) > 0
        active = valid & (~cell_solid) & nb_in & nb_solid & (i != 0)

        qi = tl.load(q_c + i * total + offsets, mask=active, other=1.0)
        f_i_here = tl.load(f_pre_c + i * total + offsets, mask=active, other=0.0)

        # Upstream fluid neighbor for direction i sits at x - e_i.
        up_x = x - dxi
        up_y = y - dyi
        up_z = z - dzi
        up_in = (up_x >= 0) & (up_x < n) & (up_y >= 0) & (up_y < n) & (up_z >= 0) & (up_z < n)
        f_i_up = tl.load(
            f_pre_c + i * total + (up_x * n2 + up_y * n + up_z),
            mask=active & up_in,
            other=0.0,
        )

        res_low = (1.0 - 2.0 * qi) * f_i_up + 2.0 * qi * f_i_here
        inv_2q = 1.0 / (2.0 * qi)
        res_high = inv_2q * f_i_here + (1.0 - inv_2q) * streamed
        res = tl.where(qi < 0.5, res_low, res_high)

        out = tl.where(active, res, streamed)
        tl.store(f_out_c + k * total + offsets, out, mask=valid)


def stream_bfl_d3q27_batch(
    f_pre: torch.Tensor,
    f_out: torch.Tensor,
    solid_mask_u8: torch.Tensor,
    q_field: torch.Tensor,
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
    opposite: torch.Tensor,
    block_size: int = 256,
) -> bool:
    """Run the batched fused D3Q27 streaming + BFL kernel on CUDA via Triton.

    Operates on ``[C, 27, D, H, W]`` populations and q and ``[C, D, H, W]``
    uint8 solid masks. Per-item results are bitwise-identical to the
    sequential ``stream_bfl_d3q27`` (same formulas, same reduction-free
    streaming). Returns True when the fused path executed, False when the
    caller must fall back to a batched PyTorch reference implementation.
    """
    if triton is None or not f_pre.is_cuda:
        return False
    if f_pre.dim() != 5 or f_pre.shape[1] != 27:
        return False
    n = int(f_pre.shape[2])
    if f_pre.shape[2] != f_pre.shape[3] or f_pre.shape[2] != f_pre.shape[4]:
        return False
    if q_field.shape != f_pre.shape:
        return False
    if not (f_pre.is_contiguous() and f_out.is_contiguous() and q_field.is_contiguous()):
        return False
    if solid_mask_u8.dtype != torch.uint8 or not solid_mask_u8.is_contiguous():
        return False
    C = int(f_pre.shape[0])
    if solid_mask_u8.shape != (C, n, n, n):
        return False

    total = n * n * n
    grid = (C, 27, triton.cdiv(total, block_size))
    _stream_bfl_kernel_batch[grid](
        f_pre,
        f_out,
        solid_mask_u8,
        q_field,
        ex,
        ey,
        ez,
        opposite,
        n,
        total,
        block=block_size,
    )
    return True


if triton is not None:
    @triton.jit
    def _stream_kernel_batch(
        f_pre,
        f_out,
        ex,
        ey,
        ez,
        n: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        """Plain full-lattice periodic pull-stream for a batch of geometries.

        Writes ``f_out[k, x] = f_pre[k, x - e_k]`` for every voxel with
        ``torch.roll`` parity. This is exactly the ``streamed`` branch of
        ``_stream_bfl_kernel_batch`` with the q-dependent BFL branch removed, so
        per-voxel values are bit-identical to that kernel's non-active output.
        """
        c = tl.program_id(0)
        k = tl.program_id(1)
        pid = tl.program_id(2)
        offsets = pid * block + tl.arange(0, block)
        valid = offsets < total

        n2 = n * n
        x = offsets // n2
        rem = offsets - x * n2
        y = rem // n
        z = rem - y * n

        dxk = tl.load(ex + k)
        dyk = tl.load(ey + k)
        dzk = tl.load(ez + k)

        sx = (x - dxk + n) % n
        sy = (y - dyk + n) % n
        sz = (z - dzk + n) % n

        f_pre_c = f_pre + c * 27 * total
        streamed = tl.load(f_pre_c + k * total + (sx * n2 + sy * n + sz), mask=valid, other=0.0)
        tl.store(f_out + c * 27 * total + k * total + offsets, streamed, mask=valid)


if triton is not None:
    @triton.jit
    def _bfl_correct_kernel_batch(
        f_pre,
        f_out,
        ex,
        ey,
        ez,
        opposite,
        q_flat,
        active_flat,
        pair_start,
        pair_count,
        n: tl.constexpr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        """Sparse Bouzidi-Firdaouss-Lallemand boundary correction.

        Grid over ``C * 26`` (c, i) pairs; each pair corrects the active
        (boundary-link) voxels for incoming direction ``i``, overwriting the
        plain-streamed value at ``f_out[k, x]`` (``k = opposite[i]``) with the
        exact ``res_low``/``res_high``/``res`` formulas from
        ``_stream_bfl_kernel_batch``. ``q_flat``/``active_flat`` are compact
        per-pair concatenations indexed by ``pair_start``/``pair_count``; pair
        ``p`` is item ``p // 26``, direction ``p % 26 + 1``. Must run AFTER the
        plain full-lattice ``_stream_kernel_batch`` so ``f_out[k, x]`` already
        holds the periodically streamed value used by the q >= 0.5 branch.
        """
        pair = tl.program_id(0)
        pid = tl.program_id(1)

        c = pair // 26
        i = (pair % 26) + 1
        k = tl.load(opposite + i)
        dxk = tl.load(ex + k)
        dyk = tl.load(ey + k)
        dzk = tl.load(ez + k)
        dxi = tl.load(ex + i)
        dyi = tl.load(ey + i)
        dzi = tl.load(ez + i)

        start = tl.load(pair_start + pair)
        cnt = tl.load(pair_count + pair)
        idx = pid * block + tl.arange(0, block)
        valid = idx < cnt

        x = tl.load(active_flat + start + idx, mask=valid, other=0)
        qi = tl.load(q_flat + start + idx, mask=valid, other=0.0)

        n2 = n * n
        x3 = x // n2
        rem = x - x3 * n2
        y3 = rem // n
        z3 = rem - y3 * n

        f_pre_c = f_pre + c * 27 * total
        f_out_c = f_out + c * 27 * total

        # f_i_here = f_pre[c, i, x]
        f_i_here = tl.load(f_pre_c + i * total + x, mask=valid, other=0.0)

        # f_i_up = f_pre[c, i, x - e_i], zero-padded outside the domain.
        up_x = x3 - dxi
        up_y = y3 - dyi
        up_z = z3 - dzi
        up_in = (up_x >= 0) & (up_x < n) & (up_y >= 0) & (up_y < n) & (up_z >= 0) & (up_z < n)
        f_i_up = tl.load(
            f_pre_c + i * total + (up_x * n2 + up_y * n + up_z),
            mask=valid & up_in,
            other=0.0,
        )

        # streamed = f_out[k, x] already written by _stream_kernel_batch (the
        # periodically pulled value f_pre[k, x - e_k]).
        streamed = tl.load(f_out_c + k * total + x, mask=valid, other=0.0)

        res_low = (1.0 - 2.0 * qi) * f_i_up + 2.0 * qi * f_i_here
        inv_2q = 1.0 / (2.0 * qi)
        res_high = inv_2q * f_i_here + (1.0 - inv_2q) * streamed
        res = tl.where(qi < 0.5, res_low, res_high)

        tl.store(f_out_c + k * total + x, res, mask=valid)


def stream_bfl_d3q27_batch_compressed(
    f_pre: torch.Tensor,
    f_out: torch.Tensor,
    sparse: dict,
    ex: torch.Tensor,
    ey: torch.Tensor,
    ez: torch.Tensor,
    opposite: torch.Tensor,
    block_size: int = 256,
) -> bool:
    """Batched D3Q27 streaming + BFL via a plain stream kernel and a sparse
    boundary-correction kernel (Task 34 compressed workspace).

    ``f_pre``/``f_out`` are ``[C, 27, D, H, W]`` fp32 contiguous tensors.
    ``sparse`` is the compact active-voxel table produced by
    ``D3Q27Solver._build_bfl_sparse_tables``:

        q_flat       [N_active] fp32  boundary-link q values, concatenated
        active_flat  [N_active] int32 flat voxel offsets, concatenated
        pair_start   [C*26]     int32 start index per (c, i) pair
        pair_count   [C*26]     int32 active-voxel count per (c, i) pair

    Kernel 1 (plain full-lattice pull stream) runs first and writes every
    voxel; kernel 2 (sparse BFL correction) then overwrites only the
    boundary-link voxels with the identical per-voxel formulas the fused
    ``_stream_bfl_kernel_batch`` uses. The full-lattice ``[C, 27, D, H, W]``
    q-field never exists here. Returns True when the compressed path executed.
    """
    if triton is None or not f_pre.is_cuda:
        return False
    if f_pre.dim() != 5 or f_pre.shape[1] != 27:
        return False
    n = int(f_pre.shape[2])
    if f_pre.shape[2] != f_pre.shape[3] or f_pre.shape[2] != f_pre.shape[4]:
        return False
    if not (f_pre.is_contiguous() and f_out.is_contiguous()):
        return False
    C = int(f_pre.shape[0])

    total = n * n * n
    grid = (C, 27, triton.cdiv(total, block_size))
    _stream_kernel_batch[grid](
        f_pre,
        f_out,
        ex,
        ey,
        ez,
        n,
        total,
        block=block_size,
    )

    q_flat = sparse["q_flat"]
    active_flat = sparse["active_flat"]
    pair_start = sparse["pair_start"]
    pair_count = sparse["pair_count"]
    if q_flat.numel() > 0 and int(pair_count.max().item()) > 0:
        max_count = int(pair_count.max().item())
        n_pairs = int(pair_count.numel())
        grid_correct = (n_pairs, triton.cdiv(max_count, block_size))
        _bfl_correct_kernel_batch[grid_correct](
            f_pre,
            f_out,
            ex,
            ey,
            ez,
            opposite,
            q_flat,
            active_flat,
            pair_start,
            pair_count,
            n,
            total,
            block=block_size,
        )
    return True


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
