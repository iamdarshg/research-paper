#!/usr/bin/env python3
"""Meta-device model-capacity sweep for the 128^3 scaling study."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import aircraft_diffusion_cfd as adc  # noqa: E402

META = torch.device("meta")
BYTES_BF16_PLAN = 12.0   # weights bf16 + grads bf16 + AdamW m/v fp32
BYTES_FP32_PLAN = 16.0   # everything fp32
EMA_BYTES_PER_TEACHER = 2.0
SOLVER_WS_AT_96 = int(5.4 * 1024 ** 3)
A100_USABLE = int(78 * 1024 ** 3)
GIB = 1024 ** 3


def _count_parameters(module):
    return int(sum(p.numel() for p in module.parameters()))


def _build_stack(grid_res, latent_dim, dec_w, dec_d, cond_dim):
    unet_base = min(dec_w, 128)
    ch = [unet_base, unet_base + 48, unet_base + 96]
    mc = adc.ModelConfig(
        latent_dim=latent_dim,
        encoder_channels=ch,
        decoder_channels=list(reversed(ch)),
        conditioning_dim=cond_dim,
        base_grid_resolution=grid_res,
        grid_resolution=grid_res,
        coordinate_decoder_width=dec_w,
        coordinate_decoder_depth=dec_d,
        coordinate_fourier_bands=6,
        coordinate_chunk_size=32768,
        enable_gradient_checkpointing=True,
        use_torch_compile=False,
    )
    dc = adc.DiffusionConfig()
    teacher = adc.LatentDiffusionUNet(mc, dc).to(META)
    converter = adc.LatentTo3DConverter(
        latent_dim=latent_dim, grid_resolution=grid_res,
        coordinate_decoder_threshold=96, coordinate_chunk_size=32768,
        coordinate_decoder_width=dec_w, coordinate_decoder_depth=dec_d,
        coordinate_fourier_bands=6,
        enable_coordinate_gradient_checkpointing=True,
        enable_decoder_compile=False).to(META)
    se = [c // 2 for c in ch]
    sd = [c // 2 for c in reversed(ch)]
    sg = math.gcd(mc.attention_groups, se[0])
    for c in se + sd:
        sg = math.gcd(sg, c)
    sg = max(1, sg)
    skv = max(1, math.gcd(sg, mc.attention_kv_groups))
    student = adc.LatentDiffusionUNet(
        adc.ModelConfig(
            latent_dim=latent_dim, encoder_channels=se,
            decoder_channels=sd, conditioning_dim=cond_dim,
            attention_groups=sg, attention_kv_groups=skv,
            num_attention_layers=mc.num_attention_layers,
            enable_gradient_checkpointing=True,
            use_torch_compile=False), dc).to(META)
    return teacher, converter, student


def evaluate(label, grid_res, latent_dim, dec_w, dec_d, cond_dim):
    teacher, converter, student = _build_stack(grid_res, latent_dim, dec_w, dec_d, cond_dim)
    counts = {"teacher": _count_parameters(teacher),
              "converter": _count_parameters(converter),
              "student": _count_parameters(student)}
    total = sum(counts.values())
    voxels = grid_res ** 3

    wgo_bf16 = total * BYTES_BF16_PLAN
    wgo_fp32 = total * BYTES_FP32_PLAN
    ema = counts["teacher"] * EMA_BYTES_PER_TEACHER

    chunk_rows = 32768
    n_chunks = math.ceil(voxels / chunk_rows)
    coord_dim = 3 * (1 + 2 * 6)
    saved_boundary = n_chunks * chunk_rows * (latent_dim + coord_dim) * 4
    recompute_ws = chunk_rows * dec_w * 4 * 10
    logits = voxels * 4 * 4
    act = saved_boundary + recompute_ws + logits
    solver = int(SOLVER_WS_AT_96 * voxels / 96 ** 3)

    def verdict(tb):
        margin = A100_USABLE - tb
        if margin < 0: return "DOES_NOT_FIT"
        if margin < 8 * GIB: return "TIGHT"
        return "FITS"

    peak_bf16 = wgo_bf16 + ema + act + solver
    peak_fp32 = wgo_fp32 + ema + act + solver
    return {
        "label": label, "grid": grid_res, "voxels": voxels,
        "latent_dim": latent_dim, "dec_w": dec_w, "dec_d": dec_d,
        "p_teacher_M": round(counts["teacher"] / 1e6, 1),
        "p_converter_M": round(counts["converter"] / 1e6, 1),
        "p_student_M": round(counts["student"] / 1e6, 1),
        "p_total_B": round(total / 1e9, 3),
        "wgo_bf16_GiB": round(wgo_bf16 / GIB, 2),
        "wgo_fp32_GiB": round(wgo_fp32 / GIB, 2),
        "ema_GiB": round(ema / GIB, 2),
        "act_GiB": round(act / GIB, 2),
        "solver_GiB": round(solver / GIB, 2),
        "peak_bf16_GiB": round(peak_bf16 / GIB, 2),
        "peak_fp32_GiB": round(peak_fp32 / GIB, 2),
        "verdict_bf16": verdict(int(peak_bf16)),
        "verdict_fp32": verdict(int(peak_fp32)),
    }


def run_sweep(cond):
    rows = [evaluate("reference_96_w928_d5", 96, 192, 928, 5, cond)]
    for g in (96, 112, 128):
        for w, d, tag in ((2048, 8, "A"), (2560, 10, "B"),
                          (3072, 12, "C"), (3584, 14, "D")):
            rows.append(evaluate(f"{g}_w{w}_d{d}_{tag}", g, 512, w, d, cond))
    rows.append(evaluate("128_w4096_d16_4B", 128, 768, 4096, 16, cond))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    cond = adc.infer_conditioning_dim()
    print(f"conditioning_dim={cond}")
    rows = run_sweep(cond)
    hdr = list(rows[0].keys())
    lines = [",".join(hdr)] + [
        ",".join(str(r[k]) for k in hdr) for r in rows]
    table = "\n".join(lines)
    print(table)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
