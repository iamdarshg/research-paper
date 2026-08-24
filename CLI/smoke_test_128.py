#!/usr/bin/env python3
# Smoke test for 128^3 / 85M-param model on GPU.
import sys, argparse
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--latent-dim", type=int, default=512)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--output-dir", default="build/smoke_128")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    from aircraft_diffusion_cfd import LatentTo3DConverter, ModelConfig

    converter = LatentTo3DConverter(
        latent_dim=args.latent_dim, grid_resolution=args.grid,
        coordinate_decoder_width=args.width, coordinate_decoder_depth=args.depth,
        coordinate_chunk_size=8192, enable_coordinate_gradient_checkpointing=True,
    ).to(device).to(torch.bfloat16)

    total_params = sum(p.numel() for p in converter.parameters())
    print(f"Converter params: {total_params/1e6:.1f}M")
    print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

    latent = torch.randn(1, args.latent_dim, device=device, dtype=torch.bfloat16)
    idx = torch.randint(0, args.grid**3, (32768,), device=device)

    logits = converter.forward_flat_indices(latent, idx)
    print(f"Sparse forward OK: {logits.shape}")

    loss = logits.sum(); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(converter.parameters(), 1.0)
    has_g = any(p.grad is not None and p.grad.abs().sum() > 0 for p in converter.parameters())
    all_f = all(p.grad is None or torch.isfinite(p.grad).all() for p in converter.parameters())
    print(f"Backward OK: grad_norm={gn.item():.4f} has_grads={has_g} finite={all_f}")

    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    ckpt = od / "smoke.pt"
    torch.save(converter.state_dict(), ckpt)
    print(f"Checkpoint: {ckpt.stat().st_size/1024**2:.0f} MB")

    ok = has_g and all_f and total_params > 50e6
    status = "PASS" if ok else "FAIL"
    print(f"SMOKE TEST: {status}")
    return 0 if ok else 1

if __name__ == "__main__": raise SystemExit(main())