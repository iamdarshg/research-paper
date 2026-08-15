#!/usr/bin/env python
"""
Mission-profile stress probe for the interruption-safe recovery checkpoint.

NO geometry is supplied to the model.
Pipeline:
    DesignSpec -> 22-D condition vector -> 4-step consistency model
    -> latent -> trained LatentTo3DConverter -> 96^3 probability field
    -> frozen intrinsic threshold -> binary voxel field

Run from the research-paper repository root.

Optimizations vs the original:
  * Loads BOTH run-state checkpoints (`ckpt["model"]`) and plain saved
    checkpoints (`ckpt["diffusion_model"]` / `ckpt["consistency_model"]` /
    `ckpt["converter"]` / `ckpt["ema_model"]`), so post-fix recovery
    checkpoints evaluate directly.
  * `--threshold` overrides the checkpoint's embedded threshold (needed for a
    fixed-0.5 recovery evaluation).
  * The expensive matplotlib 3D voxel render is skipped unless `--render-voxels`
    is passed; fast max-projections are always kept.
  * `--fast` enables bf16 autocast inference (opt-in; changes sigmoid outputs at
    bf16 precision, so it is a speed-over-exact-parity tradeoff and is NOT the
    default).
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPO_ROOT / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

# ---------------------------------------------------------------------------
# Trusted checkpoint loading (security: CWE-502 safe-deserialization gate)
# ---------------------------------------------------------------------------
# The exception set torch's weights_only=True loader raises when it rejects a
# checkpoint that embeds non-whitelisted globals (run-state RNG, custom
# compatibility objects). Depending on how the pickle was produced this is any
# of these, not just pickle.UnpicklingError.
_WEIGHTS_ONLY_FALLBACK_EXCEPTIONS = (
    pickle.UnpicklingError,
    AttributeError,
    TypeError,
    ModuleNotFoundError,
    ImportError,
    EOFError,
)

# Only checkpoints under this root are ever eligible for the weights_only=False
# fallback. These are trusted local artifacts from our own runs at explicit
# paths, never untrusted input.
_TRUSTED_CHECKPOINT_ROOT = REPO_ROOT / "build"


def _is_trusted_checkpoint_path(path) -> bool:
    """True when ``path`` resolves inside the trusted build/ checkpoint root."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    try:
        trusted_root = _TRUSTED_CHECKPOINT_ROOT.resolve()
    except OSError:
        return False
    return resolved == trusted_root or trusted_root in resolved.parents


def _load_checkpoint_metadata(checkpoint: Path):
    """Load checkpoint metadata preferring the safe weights_only=True loader.

    ``weights_only=True`` rejects any checkpoint that embeds non-whitelisted
    globals by raising one of ``_WEIGHTS_ONLY_FALLBACK_EXCEPTIONS``. We fall back
    to the unsafe ``weights_only=False`` loader ONLY for a trusted local
    artifact that resolves under the build/ root, and we log a warning when we
    do. Untrusted paths re-raise: we never deserialize untrusted input.
    """
    try:
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    except _WEIGHTS_ONLY_FALLBACK_EXCEPTIONS as exc:
        if not _is_trusted_checkpoint_path(checkpoint):
            logging.getLogger(__name__).error(
                "weights_only=True rejected %s (%s); refusing weights_only=False "
                "fallback for an untrusted checkpoint path",
                checkpoint,
                exc,
            )
            raise
        logging.getLogger(__name__).warning(
            "weights_only=True rejected %s (%s); falling back to "
            "weights_only=False for trusted local checkpoint under %s",
            checkpoint,
            exc,
            _TRUSTED_CHECKPOINT_ROOT,
        )
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

from aircraft_diffusion_cfd import (  # noqa: E402
    ModelConfig,
    DiffusionConfig,
    ConsistencyModel,
    LatentTo3DConverter,
    DesignSpec,
    build_condition_vector,
)
from aircraft_validity import evaluate_aircraft_validity  # noqa: E402


# These stay at/inside the repo's sampled conditioning envelope.
# They are "F-16-like" and "Cessna-like" mission regimes, NOT literal
# full-scale aircraft dimensions.
PROFILES = {
    "f16_like": dict(
        target_speed=90.0,                 # top edge of sampled range
        wingspan_limit_m=1.20,             # compact
        thrust_to_weight_min=0.85,         # high-energy / fighter-like
        turn_rate_min_deg_s=28.0,          # maneuverability edge
        required_static_thrust_n=320.0,
        engine_diameter_mm=220,
        engine_length_mm=420,
        engine_count_min=1,
        engine_count_max=1,
        payload_mass_min_g=750,
        payload_mass_max_g=2500,
        takeoff_distance_min_m=80,
        takeoff_distance_max_m=160,
        wall_thickness_min_mm=1,
        wall_thickness_max_mm=2,
        part_count_min=2,
        part_count_max=10,
        manufacturing_method="composite_wet_layup",
    ),
    "cessna_like": dict(
        target_speed=30.0,                 # bottom edge of sampled range
        wingspan_limit_m=2.40,             # maximum span
        thrust_to_weight_min=0.28,         # low-power / efficient regime
        turn_rate_min_deg_s=10.0,
        required_static_thrust_n=90.0,
        engine_diameter_mm=180,
        engine_length_mm=300,
        engine_count_min=1,
        engine_count_max=1,
        payload_mass_min_g=1000,
        payload_mass_max_g=4000,
        takeoff_distance_min_m=180,
        takeoff_distance_max_m=500,
        wall_thickness_min_mm=1,
        wall_thickness_max_mm=3,
        part_count_min=2,
        part_count_max=12,
        manufacturing_method="sheet_balsa_tabbed",
    ),
}


def load_recovery_checkpoint(path: Path, device: torch.device, threshold_override=None):
    # Run-state checkpoints embed torch rng state and compatibility mappings, so
    # the safe weights_only=True loader rejects them; the fallback below permits
    # weights_only=False ONLY for trusted local checkpoints under build/
    # (see _load_checkpoint_metadata).
    ckpt = _load_checkpoint_metadata(path)

    if "model" in ckpt and "compatibility" in ckpt:
        # Interruption-safe run-state format.
        cfg = ckpt["compatibility"]["configuration"]
        model_cfg = ModelConfig(**cfg["model_config"])
        diffusion_cfg = DiffusionConfig(**cfg["diffusion_config"])
        training_cfg = cfg.get("training_config", {}) or {}
        consistency_state = ckpt["model"]["consistency_model"]
        converter_state = ckpt["model"]["converter"]
        threshold = float(ckpt["geometry_probability_threshold"])
        global_step = int(ckpt.get("global_step", -1))
    else:
        # Plain saved-checkpoint format (save_checkpoint / --resume-from).
        if "diffusion_model" not in ckpt or "converter" not in ckpt:
            raise RuntimeError(
                "Unrecognized checkpoint layout: expected run-state keys "
                "'model'+'compatibility' or plain keys 'diffusion_model'+'converter'."
            )
        model_cfg = ModelConfig(**ckpt["model_config"])
        diffusion_cfg = DiffusionConfig(**ckpt["diffusion_config"])
        training_cfg = ckpt.get("training_config", {}) or {}
        consistency_state = ckpt["consistency_model"]
        converter_state = ckpt["converter"]
        threshold = float(
            ckpt.get(
                "geometry_probability_threshold",
                float(training_cfg.get("geometry_materialization_threshold", 0.5)),
            )
        )
        global_step = int(ckpt.get("global_step", -1))

    if threshold_override is not None:
        threshold = float(threshold_override)

    if model_cfg.grid_resolution != 96:
        raise RuntimeError(
            f"Expected the trained 96^3 model; checkpoint says "
            f"{model_cfg.grid_resolution}^3."
        )

    consistency = ConsistencyModel(model_cfg, diffusion_cfg).to(device)

    converter = LatentTo3DConverter(
        model_cfg.latent_dim,
        model_cfg.grid_resolution,
        coordinate_decoder_threshold=int(
            training_cfg.get("coordinate_decoder_threshold", 96)
        ),
        coordinate_chunk_size=model_cfg.coordinate_chunk_size,
        coordinate_decoder_width=model_cfg.coordinate_decoder_width,
        coordinate_decoder_depth=model_cfg.coordinate_decoder_depth,
        coordinate_fourier_bands=model_cfg.coordinate_fourier_bands,
        enable_coordinate_gradient_checkpointing=False,
    ).to(device)

    consistency.load_state_dict(consistency_state, strict=True)
    converter.load_state_dict(converter_state, strict=True)

    consistency.eval()
    consistency.student_model.eval()
    consistency.teacher_model.eval()
    converter.eval()

    meta = {
        "global_step": global_step,
        "grid_resolution": int(model_cfg.grid_resolution),
        "latent_dim": int(model_cfg.latent_dim),
        "conditioning_dim": int(model_cfg.conditioning_dim),
        "geometry_probability_threshold": threshold,
    }
    return model_cfg, consistency, converter, threshold, meta


def plot_projections(vox: np.ndarray, title: str, path: Path):
    # Training geometry convention is [Z, Y, X].
    views = [
        (vox.max(axis=0), "Top / planform"),   # Y,X
        (vox.max(axis=1), "Side"),             # Z,X
        (vox.max(axis=2), "Front"),            # Z,Y
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (im, name) in zip(axes, views):
        ax.imshow(im, origin="lower", interpolation="nearest")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_voxels(vox: np.ndarray, title: str, path: Path):
    # Matplotlib voxels wants display dimensions [X,Y,Z]. This is the slowest
    # render in the probe (polygon generation over every occupied voxel); kept
    # behind --render-voxels.
    display = np.transpose(vox.astype(bool), (2, 1, 0))
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(display, edgecolor=None)
    ax.set_title(title + "\nMODEL OUTPUT — exact 96 × 96 × 96 field")
    ax.set_xlabel("X — longitudinal")
    ax.set_ylabel("Y — span")
    ax.set_zlabel("Z — vertical")
    ax.view_init(elev=23, azim=-58)
    ax.set_box_aspect((1.4, 1.15, 0.7))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def generate_one(
    profile_name,
    spec,
    seed,
    model_cfg,
    consistency,
    converter,
    threshold,
    device,
    output_dir,
    render_voxels,
    fast,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # THIS is the only input to the model apart from random noise.
    condition = build_condition_vector(spec).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(fast and device.type != "cpu"),
    ):
        latent = consistency.fast_inference(
            (1, model_cfg.latent_dim),
            num_steps=4,
            condition=condition,
        )
        logits = converter(latent).nan_to_num(0.0)
        probabilities = torch.sigmoid(logits).nan_to_num(0.0)

    probabilities = probabilities.float()
    binary = (probabilities > threshold).to(torch.uint8)

    prob_np = probabilities.squeeze(0).float().cpu().numpy()
    vox_np = binary.squeeze(0).cpu().numpy()

    case_dir = output_dir / profile_name / f"seed_{seed:03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "probabilities_96.npy", prob_np)
    np.save(case_dir / "voxels_96.npy", vox_np)
    np.savez_compressed(
        case_dir / "model_output_96.npz",
        probabilities=prob_np,
        voxels=vox_np,
        condition_vector=condition.squeeze(0).cpu().numpy(),
        threshold=np.array(threshold, dtype=np.float32),
    )

    validity = evaluate_aircraft_validity(vox_np, canonicalize=False)
    metrics = validity.get("metrics", {})
    summary = {
        "profile": profile_name,
        "seed": seed,
        "occupancy": float(vox_np.mean()),
        "occupied_voxels": int(vox_np.sum()),
        "probability_min": float(prob_np.min()),
        "probability_max": float(prob_np.max()),
        "probability_mean": float(prob_np.mean()),
        "validity_status": validity.get("status"),
        "failed_checks": validity.get("failed_checks", []),
        "validity_metrics": metrics,
    }

    with (case_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    plot_projections(
        vox_np,
        f"{profile_name} — seed {seed}",
        case_dir / "projections.png",
    )
    if render_voxels:
        plot_voxels(
            vox_np,
            f"{profile_name} — seed {seed}",
            case_dir / "voxels_3d.png",
        )

    print(
        f"{profile_name:12s} seed={seed:3d} "
        f"occ={summary['occupancy']:.5f} "
        f"valid={summary['validity_status']} "
        f"failed={summary['failed_checks']}"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--profile",
        choices=["f16_like", "cessna_like", "both"],
        default="both",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
    )
    ap.add_argument(
        "--output-dir",
        default="build/mission_profile_stress_probe",
    )
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the checkpoint's materialization threshold.")
    ap.add_argument("--render-voxels", action="store_true",
                    help="Also render the slow matplotlib 3D voxel plot.")
    ap.add_argument("--fast", action="store_true",
                    help="bf16 autocast inference (speed over exact parity).")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print("device:", device)

    model_cfg, consistency, converter, threshold, ckpt_meta = (
        load_recovery_checkpoint(checkpoint, device, args.threshold)
    )
    print("checkpoint:", json.dumps(ckpt_meta, indent=2))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    profile_names = (
        list(PROFILES)
        if args.profile == "both"
        else [args.profile]
    )

    all_summaries = []
    for profile_name in profile_names:
        spec = DesignSpec(**PROFILES[profile_name])
        with (out / f"{profile_name}_mission.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(PROFILES[profile_name], f, indent=2)

        for seed in args.seeds:
            all_summaries.append(
                generate_one(
                    profile_name,
                    spec,
                    seed,
                    model_cfg,
                    consistency,
                    converter,
                    threshold,
                    device,
                    out,
                    args.render_voxels,
                    args.fast,
                )
            )

    with (out / "all_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_metadata": ckpt_meta,
                "results": all_summaries,
                "claim_boundary": (
                    "Free-running mission-conditioned model outputs. "
                    "F-16-like/Cessna-like labels describe condition regimes, "
                    "not supplied geometry or guaranteed aircraft identity."
                ),
            },
            f,
            indent=2,
            default=float,
        )

    print("\nSaved actual model fields under:", out.resolve())


if __name__ == "__main__":
    main()
