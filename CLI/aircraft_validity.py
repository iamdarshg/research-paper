#!/usr/bin/env python3
"""Aircraft-specific voxel validity checks beyond generic connectivity.

These are screening heuristics only. NASA's CFD V&V guidance distinguishes
implementation checks from validation against physical reality:
https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from report_metadata import apply_report_metadata


def _as_tensor(voxels: Any) -> torch.Tensor:
    if isinstance(voxels, torch.Tensor):
        tensor = voxels.detach().cpu().float()
    else:
        tensor = torch.as_tensor(voxels, dtype=torch.float32)
    if tensor.ndim == 4:
        tensor = tensor.max(dim=0).values
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid or channel-first 4D grid, got shape {tuple(tensor.shape)}")
    return (tensor > 0.5).float()


def _extent(indices: torch.Tensor, axis: int, resolution: int) -> float:
    occupied = torch.nonzero(indices > 0.5, as_tuple=False)
    if occupied.numel() == 0:
        return 0.0
    return float((occupied[:, axis].max() - occupied[:, axis].min() + 1).item() / resolution)


def _crop_to_occupied_bbox(grid: torch.Tensor) -> torch.Tensor:
    occupied = torch.nonzero(grid > 0.5, as_tuple=False)
    if occupied.numel() == 0:
        return grid
    mins = occupied.min(dim=0).values
    maxs = occupied.max(dim=0).values + 1
    return grid[
        mins[0]:maxs[0],
        mins[1]:maxs[1],
        mins[2]:maxs[2],
    ]


def _center_in_canvas(grid: torch.Tensor, canvas_shape: torch.Size) -> torch.Tensor:
    canvas = torch.zeros(tuple(canvas_shape), dtype=grid.dtype)
    starts = []
    for size, canvas_size in zip(grid.shape, canvas_shape):
        starts.append(max(0, (int(canvas_size) - int(size)) // 2))
    z0, y0, x0 = starts
    z1, y1, x1 = z0 + grid.shape[0], y0 + grid.shape[1], x0 + grid.shape[2]
    canvas[z0:z1, y0:y1, x0:x1] = grid
    return canvas


def _band_bounds(length: int, start_ratio: float, end_ratio: float) -> tuple[int, int]:
    start = min(length - 1, max(0, int(length * start_ratio)))
    end = max(start + 1, min(length, int(length * end_ratio)))
    return start, end


def _heuristic_metrics(grid: torch.Tensor) -> Dict[str, float]:
    occupied = float(grid.sum().item())
    total = float(grid.numel())
    occupancy_ratio = occupied / max(total, 1.0)
    occupied_indices = torch.nonzero(grid > 0.5, as_tuple=False)

    flipped = torch.flip(grid, dims=[1])
    voxel_asymmetry = torch.abs(grid - flipped).sum().item() / max(occupied, 1.0)
    voxel_symmetry_score = max(0.0, 1.0 - float(voxel_asymmetry))
    span_profile = grid.sum(dim=(0, 2))
    span_profile_asymmetry = torch.abs(span_profile - torch.flip(span_profile, dims=[0])).sum().item() / max(occupied, 1.0)
    symmetry_score = max(0.0, 1.0 - float(span_profile_asymmetry))

    res_z, res_y, res_x = grid.shape
    thickness_fraction = _extent(grid, axis=0, resolution=res_z)
    span_fraction = _extent(grid, axis=1, resolution=res_y)
    length_fraction = _extent(grid, axis=2, resolution=res_x)

    center_start, center_end = _band_bounds(res_y, 0.42, 0.58)
    left_start, left_end = _band_bounds(res_y, 0.00, 0.35)
    right_start, right_end = _band_bounds(res_y, 0.65, 1.00)
    low_end_start, low_end_end = _band_bounds(res_x, 0.00, 0.28)
    high_end_start, high_end_end = _band_bounds(res_x, 0.72, 1.00)

    center_band = grid[:, center_start:center_end, :]
    left_band = grid[:, left_start:left_end, :]
    right_band = grid[:, right_start:right_end, :]
    low_end_band = grid[:, :, low_end_start:low_end_end]
    high_end_band = grid[:, :, high_end_start:high_end_end]
    center_low_end_band = center_band[:, :, low_end_start:low_end_end]
    center_high_end_band = center_band[:, :, high_end_start:high_end_end]

    center_fraction = float(center_band.sum().item() / max(occupied, 1.0))
    left_fraction = float(left_band.sum().item() / max(occupied, 1.0))
    right_fraction = float(right_band.sum().item() / max(occupied, 1.0))
    low_end_fraction = float(low_end_band.sum().item() / max(occupied, 1.0))
    high_end_fraction = float(high_end_band.sum().item() / max(occupied, 1.0))
    tail_fraction = min(low_end_fraction, high_end_fraction)
    center_band_occupied = float(center_band.sum().item())
    center_low_end_fraction = float(center_low_end_band.sum().item() / max(center_band_occupied, 1.0))
    center_high_end_fraction = float(center_high_end_band.sum().item() / max(center_band_occupied, 1.0))

    center_density = float(center_band.mean().item()) if center_band.numel() else 0.0
    left_density = float(left_band.mean().item()) if left_band.numel() else 0.0
    right_density = float(right_band.mean().item()) if right_band.numel() else 0.0
    wing_density = max(left_density, right_density, 1e-6)
    longitudinal_profile = grid.sum(dim=(0, 1))
    occupied_profile = longitudinal_profile[longitudinal_profile > 0]
    longitudinal_profile_cv = 0.0
    if occupied_profile.numel() > 1:
        longitudinal_profile_cv = float(
            occupied_profile.float().std(unbiased=False).item()
            / max(occupied_profile.float().mean().item(), 1e-6)
        )

    occupied_bbox_fill_ratio = 0.0
    planform_fill_ratio = 0.0
    side_projection_fill_ratio = 0.0
    mean_longitudinal_slice_fill_ratio = 0.0
    max_longitudinal_slice_fill_ratio = 0.0
    center_spine_coverage = 0.0
    if occupied_indices.numel() > 0:
        mins = occupied_indices.min(dim=0).values
        maxs = occupied_indices.max(dim=0).values + 1
        bbox_shape = (maxs - mins).float()
        bbox_volume = float(torch.prod(bbox_shape).item())
        occupied_bbox_fill_ratio = occupied / max(bbox_volume, 1.0)
        crop = grid[
            mins[0]:maxs[0],
            mins[1]:maxs[1],
            mins[2]:maxs[2],
        ] > 0.5
        planform_fill_ratio = float(crop.any(dim=0).float().mean().item())
        side_projection_fill_ratio = float(crop.any(dim=1).float().mean().item())
        longitudinal_slice_fills: List[float] = []
        for x_idx in range(crop.shape[2]):
            slice_grid = crop[:, :, x_idx]
            if bool(slice_grid.any().item()):
                longitudinal_slice_fills.append(float(slice_grid.float().mean().item()))
        if longitudinal_slice_fills:
            mean_longitudinal_slice_fill_ratio = float(np.mean(longitudinal_slice_fills))
            max_longitudinal_slice_fill_ratio = float(np.max(longitudinal_slice_fills))
        occupied_x_profile = grid.sum(dim=(0, 1)) > 0
        center_x_profile = center_band.sum(dim=(0, 1)) > 0
        center_spine_coverage = float(
            torch.logical_and(occupied_x_profile, center_x_profile).sum().item()
            / max(float(occupied_x_profile.sum().item()), 1.0)
        )

    return {
        "occupancy_ratio": occupancy_ratio,
        "symmetry_score": symmetry_score,
        "voxel_symmetry_score": voxel_symmetry_score,
        "thickness_fraction_z": thickness_fraction,
        "span_fraction_y": span_fraction,
        "length_fraction_x": length_fraction,
        "center_body_fraction": center_fraction,
        "left_wing_fraction": left_fraction,
        "right_wing_fraction": right_fraction,
        "center_body_density": center_density,
        "left_wing_density": left_density,
        "right_wing_density": right_density,
        "center_body_density_ratio": center_density / wing_density,
        "longitudinal_profile_cv": longitudinal_profile_cv,
        "occupied_bbox_fill_ratio": occupied_bbox_fill_ratio,
        "planform_fill_ratio": planform_fill_ratio,
        "side_projection_fill_ratio": side_projection_fill_ratio,
        "mean_longitudinal_slice_fill_ratio": mean_longitudinal_slice_fill_ratio,
        "max_longitudinal_slice_fill_ratio": max_longitudinal_slice_fill_ratio,
        "center_low_end_fraction": center_low_end_fraction,
        "center_high_end_fraction": center_high_end_fraction,
        "center_spine_coverage": center_spine_coverage,
        "low_end_fraction": low_end_fraction,
        "high_end_fraction": high_end_fraction,
        "tail_fraction": tail_fraction,
    }


def _orientation_score(metrics: Dict[str, float]) -> float:
    wing_fraction = min(metrics["left_wing_fraction"], metrics["right_wing_fraction"])
    wing_density = min(metrics["left_wing_density"], metrics["right_wing_density"])
    centerline_bonus = min(metrics["center_body_density_ratio"], 4.0)
    missing_wing_penalty = -6.0 if wing_fraction < 0.02 else 0.0
    return (
        4.0 * metrics["symmetry_score"]
        + 18.0 * wing_fraction
        + 8.0 * wing_density
        + 1.5 * centerline_bonus
        + 2.0 * metrics["span_fraction_y"]
        + 1.5 * metrics["length_fraction_x"]
        - 2.0 * metrics["thickness_fraction_z"]
        + missing_wing_penalty
    )


def _canonicalize_aircraft_grid(grid: torch.Tensor) -> tuple[torch.Tensor, Dict[str, Any]]:
    cropped = _crop_to_occupied_bbox(grid)
    if float(cropped.sum().item()) <= 0.0:
        return grid, {"permutation": [0, 1, 2], "score": 0.0}

    best_grid = _center_in_canvas(cropped, grid.shape)
    best_metrics = _heuristic_metrics(best_grid)
    best_perm = (0, 1, 2)
    best_score = _orientation_score(best_metrics)

    for perm in itertools.permutations(range(3)):
        oriented = cropped.permute(*perm).contiguous()
        centered = _center_in_canvas(oriented, grid.shape)
        metrics = _heuristic_metrics(centered)
        score = _orientation_score(metrics)
        if score > best_score:
            best_grid = centered
            best_metrics = metrics
            best_perm = perm
            best_score = score

    return best_grid, {
        "permutation": list(best_perm),
        "score": float(best_score),
        "metrics": best_metrics,
    }


def evaluate_aircraft_validity(voxels: Any) -> Dict[str, Any]:
    # Heuristic shape checks are intentionally separated from claim evidence.
    # NASA-STD-7009B treats model/simulation credibility as a lifecycle product,
    # not a single geometric proxy: https://standards.nasa.gov/standard/nasa/nasa-std-7009
    raw_grid = _as_tensor(voxels)
    grid, canonicalization = _canonicalize_aircraft_grid(raw_grid)
    metrics = canonicalization.get("metrics") or _heuristic_metrics(grid)

    checks = {
        "nonempty_occupancy": 0.002 <= metrics["occupancy_ratio"] <= 0.50,
        "symmetry": metrics["symmetry_score"] >= 0.55,
        "span_sanity": (
            metrics["span_fraction_y"] >= 0.35
            and metrics["length_fraction_x"] >= 0.35
            and metrics["thickness_fraction_z"] <= 0.35
        ),
        "wing_body_balance": (
            metrics["center_body_fraction"] >= 0.10
            and metrics["left_wing_fraction"] >= 0.05
            and metrics["right_wing_fraction"] >= 0.05
        ),
        "body_centerline_dominance": metrics["center_body_density_ratio"] >= 1.15,
        "longitudinal_profile_variation": metrics["longitudinal_profile_cv"] >= 0.18,
        "planform_sparsity": (
            metrics["planform_fill_ratio"] <= 0.75
            and metrics["occupied_bbox_fill_ratio"] <= 0.65
        ),
        "fuselage_end_presence": (
            min(metrics["center_low_end_fraction"], metrics["center_high_end_fraction"]) >= 0.015
            and metrics["center_spine_coverage"] >= 0.70
        ),
        "tail_body_plausibility": (
            metrics["tail_fraction"] <= 0.20
            and max(metrics["low_end_fraction"], metrics["high_end_fraction"]) <= 0.50
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "metrics": metrics,
        "canonicalization": canonicalization,
        "claim_boundary": "First-pass aircraft-specific heuristic validity, not structural or aerodynamic proof.",
    }


def evaluate_aircraft_validity_batch(paths: Iterable[Path]) -> Dict[str, Any]:
    sample_reports: List[Dict[str, Any]] = []
    for idx, raw_path in enumerate(paths):
        path = Path(raw_path)
        sample_report = evaluate_aircraft_validity(_load_voxels(path))
        sample_report["sample_index"] = idx
        sample_report["artifact_path"] = str(path.resolve())
        sample_reports.append(sample_report)

    failed = [
        report["sample_index"]
        for report in sample_reports
        if report.get("status") != "pass"
    ]
    if not sample_reports:
        status = "blocked"
    else:
        status = "pass" if not failed else "fail"

    return {
        "status": status,
        "sample_count": len(sample_reports),
        "passed_sample_count": len(sample_reports) - len(failed),
        "failed_sample_indices": failed,
        "samples": sample_reports,
        "claim_boundary": "Batch aggregation of first-pass validity heuristics; not CFD or structural validation.",
    }


def _load_voxels(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return torch.as_tensor(np.load(path), dtype=torch.float32)
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("geometry", "voxels", "voxel_grid", "geometries"):
                if key in payload:
                    payload = payload[key]
                    break
        return torch.as_tensor(payload, dtype=torch.float32)
    raise ValueError(f"Unsupported voxel artifact: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run first-pass aircraft-specific voxel validity checks.")
    parser.add_argument("--input", action="append", default=[], help="Path to a .npy/.pt voxel artifact. May be repeated.")
    parser.add_argument("--input-dir", default=None, help="Directory containing .npy/.pt/.pth voxel artifacts.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--manifest", default=None, help="Optional manifest path for evidence lineage metadata.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for evidence lineage metadata.")
    parser.add_argument("--run-id", default=None, help="Optional run identifier shared across report artifacts.")
    parser.add_argument("--protocol-config", default=None, help="Optional protocol config path for evidence lineage metadata.")
    args = parser.parse_args()

    paths = [Path(value) for value in args.input]
    input_errors: List[str] = []
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if input_dir.exists():
            paths.extend(
                sorted(
                    path
                    for path in input_dir.iterdir()
                    if path.suffix.lower() in {".npy", ".pt", ".pth"}
                )
            )
        else:
            input_errors.append(f"input_dir does not exist: {input_dir}")
    report = evaluate_aircraft_validity_batch(paths)
    if input_errors:
        report["status"] = "blocked"
        report["errors"] = input_errors
    apply_report_metadata(
        report,
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        protocol_path=args.protocol_config,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
