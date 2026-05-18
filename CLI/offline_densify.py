#!/usr/bin/env python3
"""Offline RLVR-style dataset densification for conditioned aircraft voxels.

This module intentionally stays on the "generate -> verify -> accept -> write"
side of the house. It does not claim online RL or fully learned conditioning.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, label

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    CFDConfig,
    DesignSpec,
    OptimizedAircraftGenerator,
    build_condition_vector,
    sample_design_spec,
)


@dataclass
class RLVRBootstrapConfig:
    min_total_reward: float = 0.15
    min_connected_fraction: float = 0.90
    min_occupancy_ratio: float = 0.01
    max_occupancy_ratio: float = 0.35
    cfd_steps: int = 24
    cfd_top_k: int = 1
    enable_cfd: bool = False
    base_grid_resolution: int = 16
    num_candidates_per_condition: int = 6


def _largest_component_fraction(binary: np.ndarray) -> float:
    labeled, num_components = label(binary)
    if num_components <= 1:
        return 1.0 if binary.sum() > 0 else 0.0
    component_sizes = np.bincount(labeled.ravel())
    occupied = max(1, int(binary.sum()))
    largest = int(component_sizes[1:].max()) if component_sizes.size > 1 else 0
    return float(largest) / float(occupied)


def _bbox_dimensions(binary: np.ndarray) -> Dict[str, int]:
    coords = np.argwhere(binary)
    if coords.size == 0:
        return {"span_x": 0, "span_y": 0, "span_z": 0}
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = (maxs - mins + 1).astype(int)
    return {"span_x": int(dims[0]), "span_y": int(dims[1]), "span_z": int(dims[2])}


def _surface_to_volume(binary: np.ndarray) -> float:
    occupied = int(binary.sum())
    if occupied == 0:
        return float("inf")
    shell = np.logical_and(binary_dilation(binary), ~binary).sum()
    return float(shell) / float(occupied)


def _shell_fraction(binary: np.ndarray) -> float:
    occupied = int(binary.sum())
    if occupied == 0:
        return 1.0
    eroded = binary_erosion(binary)
    return 1.0 - (float(eroded.sum()) / float(occupied))


def _manufacturing_reward(metrics: Dict[str, float], design_spec: DesignSpec) -> float:
    shell_fraction = metrics["shell_fraction"]
    method = design_spec.manufacturing_method
    wall_mid_mm = 0.5 * (
        float(design_spec.wall_thickness_min_mm)
        + float(design_spec.wall_thickness_max_mm)
    )
    target_shell = np.clip(1.02 - 0.12 * wall_mid_mm, 0.55, 0.95)
    if method == "sheet_balsa_tabbed":
        target_shell = max(target_shell, 0.88)
    elif method == "fdm_pla_0p6mm":
        target_shell = min(target_shell, 0.74)
    elif method == "composite_wet_layup":
        target_shell = min(target_shell + 0.04, 0.86)

    deviation = abs(shell_fraction - target_shell)
    return float(np.clip(1.0 - deviation / 0.45, 0.0, 1.0))


def _maneuverability_reward(metrics: Dict[str, float], design_spec: DesignSpec) -> float:
    span = metrics["span_x_fraction"]
    length = metrics["span_y_fraction"]
    occupancy = metrics["occupancy_ratio"]
    proxy = (span / max(length, 1e-6)) * (1.0 - 0.55 * occupancy)
    target = float(design_spec.turn_rate_min_deg_s) / 30.0
    return float(np.clip(1.0 - abs(proxy - target) / max(target, 0.25), 0.0, 1.0))


def _thrust_reward(metrics: Dict[str, float], design_spec: DesignSpec) -> float:
    avg_payload_kg = 0.0005 * (
        float(design_spec.payload_mass_min_g) + float(design_spec.payload_mass_max_g)
    )
    estimated_vehicle_mass_kg = max(
        0.5,
        avg_payload_kg + 3.0 * metrics["occupancy_ratio"] + 0.3 * metrics["surface_to_volume"],
    )
    effective_twr = float(design_spec.required_static_thrust_n) / (
        estimated_vehicle_mass_kg * 9.81
    )
    target_twr = max(0.1, float(design_spec.thrust_to_weight_min))
    return float(np.clip(effective_twr / (1.25 * target_twr), 0.0, 1.0))


def _part_count_reward(metrics: Dict[str, float], design_spec: DesignSpec) -> float:
    inferred_parts = 1.0 + 2.5 * metrics["surface_to_volume"] + 0.25 * (
        float(design_spec.engine_count_min) + float(design_spec.engine_count_max)
    )
    lower = float(design_spec.part_count_min)
    upper = max(lower, float(design_spec.part_count_max))
    if lower <= inferred_parts <= upper:
        return 1.0
    if inferred_parts < lower:
        return float(np.clip(1.0 - (lower - inferred_parts) / max(lower, 1.0), 0.0, 1.0))
    return float(np.clip(1.0 - (inferred_parts - upper) / max(upper, 1.0), 0.0, 1.0))


def _hard_reject(metrics: Dict[str, float], config: RLVRBootstrapConfig) -> Optional[str]:
    if metrics["occupancy_ratio"] <= 0.0:
        return "empty_geometry"
    if metrics["occupancy_ratio"] < config.min_occupancy_ratio:
        return "too_sparse"
    if metrics["occupancy_ratio"] > config.max_occupancy_ratio:
        return "too_dense"
    if metrics["connected_fraction"] < config.min_connected_fraction:
        return "disconnected"
    if max(metrics["span_x_fraction"], metrics["span_y_fraction"]) > 0.98:
        return "touches_grid_boundary"
    return None


def score_candidate(
    geometry: torch.Tensor,
    design_spec: DesignSpec,
    config: Optional[RLVRBootstrapConfig] = None,
) -> Dict[str, Any]:
    config = config or RLVRBootstrapConfig()
    binary = (geometry.detach().cpu().numpy() > 0.5).astype(bool)
    bbox = _bbox_dimensions(binary)
    occupancy_ratio = float(binary.mean())
    connected_fraction = _largest_component_fraction(binary)
    shell_fraction = _shell_fraction(binary)
    surface_to_volume = _surface_to_volume(binary)
    metrics = {
        "occupancy_ratio": occupancy_ratio,
        "connected_fraction": connected_fraction,
        "shell_fraction": shell_fraction,
        "surface_to_volume": surface_to_volume,
        "span_x_fraction": bbox["span_x"] / max(1, binary.shape[0]),
        "span_y_fraction": bbox["span_y"] / max(1, binary.shape[1]),
        "span_z_fraction": bbox["span_z"] / max(1, binary.shape[2]),
    }

    rejection_reason = _hard_reject(metrics, config)
    reward_components = {
        "connectivity": connected_fraction,
        "occupancy": float(
            np.clip(
                1.0
                - abs(
                    occupancy_ratio
                    - 0.5 * (config.min_occupancy_ratio + config.max_occupancy_ratio)
                )
                / max(config.max_occupancy_ratio, 1e-6),
                0.0,
                1.0,
            )
        ),
        "manufacturing": _manufacturing_reward(metrics, design_spec),
        "maneuverability": _maneuverability_reward(metrics, design_spec),
        "thrust": _thrust_reward(metrics, design_spec),
        "part_count": _part_count_reward(metrics, design_spec),
    }
    total_reward = (
        0.25 * reward_components["connectivity"]
        + 0.10 * reward_components["occupancy"]
        + 0.20 * reward_components["manufacturing"]
        + 0.20 * reward_components["maneuverability"]
        + 0.15 * reward_components["thrust"]
        + 0.10 * reward_components["part_count"]
    )
    accepted = rejection_reason is None and total_reward >= config.min_total_reward
    return {
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "total_reward": float(total_reward),
        "reward_components": reward_components,
        "metrics": metrics,
    }


def generate_candidate_pool(
    num_samples: int,
    grid_size: int,
    latent_dim: int,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    dataset = AircraftDesignDataset(
        num_samples=num_samples,
        grid_size=grid_size,
        seed=seed,
        latent_dim=latent_dim,
    )
    return [dataset[i] for i in range(len(dataset))]


def _empty_artifact(latent_dim: int, grid_size: int, condition_dim: int) -> Dict[str, Any]:
    return {
        "latents": torch.zeros((0, latent_dim), dtype=torch.float32),
        "geometries": torch.zeros((0, grid_size, grid_size, grid_size), dtype=torch.float32),
        "condition_vectors": torch.zeros((0, condition_dim), dtype=torch.float32),
        "design_specs": [],
        "reward_records": [],
    }


def bootstrap_dataset(
    output_path: str,
    candidates: Optional[Iterable[Dict[str, Any]]] = None,
    config: Optional[RLVRBootstrapConfig] = None,
    num_samples: int = 32,
    grid_size: int = 16,
    latent_dim: int = 16,
    seed: int = 0,
) -> Dict[str, Any]:
    config = config or RLVRBootstrapConfig(base_grid_resolution=grid_size)
    records = list(candidates) if candidates is not None else generate_candidate_pool(
        num_samples=num_samples,
        grid_size=grid_size,
        latent_dim=latent_dim,
        seed=seed,
    )

    accepted_records: List[Dict[str, Any]] = []
    for record in records:
        reward = score_candidate(record["geometry"], record["design_spec"], config=config)
        if reward["accepted"]:
            accepted_records.append({**record, "reward": reward})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if accepted_records:
        latents = torch.stack([record["latent"].float() for record in accepted_records])
        geometries = torch.stack([record["geometry"].float() for record in accepted_records])
        condition_vectors = torch.stack(
            [record["condition_vector"].float() for record in accepted_records]
        )
        payload = {
            "latents": latents,
            "geometries": geometries,
            "condition_vectors": condition_vectors,
            "design_specs": [asdict(record["design_spec"]) for record in accepted_records],
            "reward_records": [record["reward"] for record in accepted_records],
        }
    else:
        condition_dim = records[0]["condition_vector"].numel() if records else 0
        payload = _empty_artifact(latent_dim, grid_size, condition_dim)

    torch.save(payload, output)
    return {
        "output_path": str(output),
        "num_candidates": len(records),
        "num_accepted": len(accepted_records),
    }


def _git_commit_sha(repo_root: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                text=True,
            )
            .strip()
        )
    except Exception:
        return None


def write_dataset_artifact(
    output_dir: str,
    accepted_records: List[Dict[str, Any]],
    config: RLVRBootstrapConfig,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": checkpoint_path,
        "commit_sha": _git_commit_sha(Path(__file__).resolve().parent.parent),
        "num_accepted": len(accepted_records),
        "config": asdict(config),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    jsonl_lines = []
    npz_payload: Dict[str, Any] = {
        "reward_total": np.array(
            [record["reward"]["total_reward"] for record in accepted_records],
            dtype=np.float32,
        ),
    }
    if accepted_records:
        npz_payload["voxels"] = np.stack(
            [(record["geometry"].detach().cpu().numpy() > 0.5).astype(np.uint8) for record in accepted_records]
        )
        npz_payload["condition_vectors"] = np.stack(
            [record["condition_vector"].detach().cpu().numpy().astype(np.float32) for record in accepted_records]
        )

    for idx, record in enumerate(accepted_records):
        reward = record["reward"]
        jsonl_lines.append(
            json.dumps(
                {
                    "sample_id": idx,
                    "design_spec": asdict(record["design_spec"]),
                    "acceptance_reason": "accepted",
                    "reward_total": reward["total_reward"],
                    "reward_components": reward["reward_components"],
                }
            )
        )

    np.savez(out_dir / "accepted-0000.npz", **npz_payload)
    (out_dir / "accepted.jsonl").write_text(
        "\n".join(jsonl_lines),
        encoding="utf-8",
    )
    return {
        "manifest": str(out_dir / "manifest.json"),
        "npz": str(out_dir / "accepted-0000.npz"),
        "jsonl": str(out_dir / "accepted.jsonl"),
    }


class OfflineAcceptedDataset(torch.utils.data.Dataset):
    """Load accepted densified samples without changing the main training path."""

    def __init__(self, artifact_path: str):
        payload = torch.load(artifact_path, map_location="cpu")
        self.latents = payload["latents"].float()
        self.geometries = payload["geometries"].float()
        self.condition_vectors = payload["condition_vectors"].float()
        self.design_specs = [
            DesignSpec(**design_spec) for design_spec in payload.get("design_specs", [])
        ]
        self.reward_records = payload.get("reward_records", [])

    def __len__(self) -> int:
        return int(self.geometries.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        reward = self.reward_records[idx] if idx < len(self.reward_records) else None
        design_spec = (
            self.design_specs[idx]
            if idx < len(self.design_specs)
            else sample_design_spec()
        )
        return {
            "latent": self.latents[idx],
            "geometry": self.geometries[idx],
            "condition_vector": self.condition_vectors[idx],
            "design_spec": design_spec,
            "reward_total": 0.0 if reward is None else reward["total_reward"],
            "reward_record": reward,
        }


def densify_from_checkpoint(
    checkpoint_path: str,
    output_path: str,
    num_conditions: int = 4,
    config: Optional[RLVRBootstrapConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    config = config or RLVRBootstrapConfig()
    generator = OptimizedAircraftGenerator(checkpoint_path, device=device)
    candidate_records: List[Dict[str, Any]] = []
    for condition_seed in range(num_conditions):
        rng = np.random.default_rng(condition_seed)
        for _ in range(config.num_candidates_per_condition):
            design_spec = sample_design_spec()
            design_spec.target_speed = float(rng.uniform(35.0, 85.0))
            condition_vector = build_condition_vector(design_spec)
            latent = generator.consistency_model.fast_inference(
                (1, generator.model_config.latent_dim),
                num_steps=4,
                condition=condition_vector.unsqueeze(0).to(generator.device),
            )
            voxel_grid = torch.sigmoid(generator.converter(latent)).squeeze(0).detach().cpu().float()
            candidate_records.append(
                {
                    "latent": latent.squeeze(0).detach().cpu().float(),
                    "geometry": voxel_grid,
                    "condition_vector": condition_vector,
                    "design_spec": design_spec,
                }
            )
    return bootstrap_dataset(output_path, candidates=candidate_records, config=config)
