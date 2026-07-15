#!/usr/bin/env python3
"""Measure direct-solver repeatability and SPSA directional variance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from aircraft_diffusion_cfd import (
    AdvancedCFDSimulator,
    AircraftDesignDataset,
    CFDConfig,
    LatentTo3DConverter,
    ModelConfig,
    _clear_direct_solver_geometry_caches,
    _direct_measured_objective_for_single,
)


def _stats(values: List[float]) -> Dict[str, float]:
    mean = fmean(values)
    std = pstdev(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": std,
        "cv": std / abs(mean) if mean != 0.0 else 0.0,
    }


def _delta(
    shape: torch.Size,
    *,
    coarse_grid_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    coarse_shape = tuple(max(1, min(coarse_grid_size, int(dim))) for dim in shape)
    coarse = torch.randint(
        0,
        2,
        (1, 1, *coarse_shape),
        generator=generator,
        device=device,
        dtype=torch.int8,
    ).float().mul(2.0).sub(1.0)
    expanded = F.interpolate(coarse, size=tuple(shape), mode="trilinear", align_corners=False)[0, 0]
    return (expanded / expanded.abs().mean().clamp_min(1.0e-6)).clamp(-2.0, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--repeat-runs", type=int, default=32)
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--solver-steps", type=int, default=5)
    parser.add_argument("--perturbation", type=float, default=0.15)
    parser.add_argument("--perturbation-grid-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.repeat_runs < 2 or args.directions < 2:
        raise ValueError("repeat-runs and directions must both be at least 2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=model_config.grid_resolution,
        latent_dim=model_config.latent_dim,
        manifest_path=args.manifest,
    )
    sample = dataset[args.sample_index]
    converter = LatentTo3DConverter(
        model_config.latent_dim,
        model_config.grid_resolution,
        coordinate_decoder_threshold=96,
        coordinate_chunk_size=model_config.coordinate_chunk_size,
        coordinate_decoder_width=model_config.coordinate_decoder_width,
        coordinate_decoder_depth=model_config.coordinate_decoder_depth,
        coordinate_fourier_bands=model_config.coordinate_fourier_bands,
    ).to(device)
    converter.load_state_dict(checkpoint["converter"])
    converter.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(converter(sample["latent"].unsqueeze(0).to(device)))[0]

    target_occupancy = float(sample["geometry"].float().mean().item())
    design_spec = sample["design_spec"]
    simulator = AdvancedCFDSimulator(
        CFDConfig(base_grid_resolution=model_config.grid_resolution, solver_type="D3Q27"),
        device,
    )
    objective_kwargs: Dict[str, Any] = {
        "design_spec": design_spec,
        "cfd_simulator": simulator,
        "cfd_steps": args.solver_steps,
        "connectivity_weight": 1.0,
        "aircraft_validity_weight": 1.0,
        "threshold": 0.5,
        "target_occupancy": target_occupancy,
    }

    repeat_components = []
    for _ in range(args.repeat_runs):
        repeat_components.append(
            _direct_measured_objective_for_single(
                probabilities,
                **objective_kwargs,
                return_components=True,
            )
        )
        _clear_direct_solver_geometry_caches(simulator)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    directional_derivatives: List[float] = []
    midpoint_losses: List[float] = []
    plus_losses: List[float] = []
    minus_losses: List[float] = []
    for _ in range(args.directions):
        delta = _delta(
            probabilities.shape,
            coarse_grid_size=args.perturbation_grid_size,
            generator=generator,
            device=device,
        )
        plus = float(
            _direct_measured_objective_for_single(
                (probabilities + args.perturbation * delta).clamp(0.0, 1.0),
                **objective_kwargs,
            )
        )
        _clear_direct_solver_geometry_caches(simulator)
        minus = float(
            _direct_measured_objective_for_single(
                (probabilities - args.perturbation * delta).clamp(0.0, 1.0),
                **objective_kwargs,
            )
        )
        _clear_direct_solver_geometry_caches(simulator)
        plus_losses.append(plus)
        minus_losses.append(minus)
        midpoint_losses.append(0.5 * (plus + minus))
        directional_derivatives.append((plus - minus) / (2.0 * args.perturbation))

    component_stats = {
        key: _stats([float(row[key]) for row in repeat_components])
        for key in repeat_components[0]
    }
    payload = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "sample_index": args.sample_index,
        "target_occupancy": target_occupancy,
        "solver_steps": args.solver_steps,
        "repeat_runs": args.repeat_runs,
        "directions": args.directions,
        "solver_evaluation_count": args.repeat_runs + (2 * args.directions),
        "identical_geometry_component_stats": component_stats,
        "plus_loss_stats": _stats(plus_losses),
        "minus_loss_stats": _stats(minus_losses),
        "perturbation_midpoint_loss_stats": _stats(midpoint_losses),
        "directional_derivative_stats": _stats(directional_derivatives),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
