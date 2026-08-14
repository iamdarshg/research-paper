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
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from scipy.ndimage import label as connected_component_labels

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
    largest_component_fraction = 0.0
    if occupied > 0.0:
        # largest_component_fraction is invariant under the solid-bbox crop:
        # the crop contains every solid cell, and scipy's local-connectivity
        # labeling neither loses nor creates a connection outside it. Cropping
        # shrinks the label work to the aircraft occupancy instead of the full
        # 96^3 lattice. All other metrics stay on the full grid.
        mins = occupied_indices.min(dim=0).values
        maxs = occupied_indices.max(dim=0).values + 1
        occupied_mask = grid[
            mins[0]:maxs[0],
            mins[1]:maxs[1],
            mins[2]:maxs[2],
        ].numpy() > 0.5
        labeled, component_count = connected_component_labels(occupied_mask)
        if component_count > 0:
            # np.bincount coerces its complete input to int64. Counting only
            # foreground labels keeps the temporary proportional to aircraft
            # occupancy instead of the full 96^3 lattice.
            component_sizes = np.bincount(labeled[occupied_mask])[1:]
            largest_component_fraction = float(component_sizes.max() / occupied)

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
    normalization_boundary_fraction = 0.0
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
        z_low, z_high = _band_bounds(res_z, 0.30, 0.70)
        y_low, y_high = _band_bounds(res_y, 0.05, 0.95)
        x_low, x_high = _band_bounds(res_x, 0.05, 0.95)
        boundary_occupied = (
            (occupied_indices[:, 0] < z_low)
            | (occupied_indices[:, 0] >= z_high)
            | (occupied_indices[:, 1] < y_low)
            | (occupied_indices[:, 1] >= y_high)
            | (occupied_indices[:, 2] < x_low)
            | (occupied_indices[:, 2] >= x_high)
        )
        normalization_boundary_fraction = float(
            boundary_occupied.sum().item() / max(occupied, 1.0)
        )

    return {
        "occupancy_ratio": occupancy_ratio,
        "largest_component_fraction": largest_component_fraction,
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
        "normalization_boundary_fraction": normalization_boundary_fraction,
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


def _lower_bound_violation(value: float, lower_bound: float) -> float:
    return float(np.clip((lower_bound - value) / max(abs(lower_bound), 1.0e-6), 0.0, 1.0))


def _upper_bound_violation(
    value: float,
    upper_bound: float,
    natural_ceiling: float = 1.0,
) -> float:
    scale = max(natural_ceiling - upper_bound, 1.0e-6)
    return float(np.clip((value - upper_bound) / scale, 0.0, 1.0))


def _validity_violation_scores(
    metrics: Dict[str, float],
    occupancy_upper_bound: float,
) -> Dict[str, float]:
    """Return continuous distances to the same gates used for pass/fail."""
    strict_normalization_violation = max(
        _upper_bound_violation(metrics["span_fraction_y"], 0.90),
        _upper_bound_violation(metrics["length_fraction_x"], 0.90),
        _upper_bound_violation(metrics["thickness_fraction_z"], 0.40),
    )
    normalization_violation = 0.0
    if strict_normalization_violation > 0.0:
        normalization_violation = float(
            np.clip(
                0.10 * strict_normalization_violation
                + math.sqrt(max(metrics["normalization_boundary_fraction"], 0.0)),
                0.0,
                1.0,
            )
        )

    return {
        "nonempty_occupancy": max(
            _lower_bound_violation(metrics["occupancy_ratio"], 0.002),
            _upper_bound_violation(metrics["occupancy_ratio"], 0.50),
        ),
        "grounded_occupancy_envelope": _upper_bound_violation(
            metrics["occupancy_ratio"], occupancy_upper_bound
        ),
        "dominant_connected_airframe": _lower_bound_violation(
            metrics["largest_component_fraction"], 0.70
        ),
        "symmetry": _lower_bound_violation(metrics["symmetry_score"], 0.55),
        "span_sanity": max(
            _lower_bound_violation(metrics["span_fraction_y"], 0.35),
            _lower_bound_violation(metrics["length_fraction_x"], 0.35),
            _upper_bound_violation(metrics["thickness_fraction_z"], 0.35),
        ),
        "wing_body_balance": max(
            _lower_bound_violation(metrics["center_body_fraction"], 0.10),
            _lower_bound_violation(metrics["left_wing_fraction"], 0.05),
            _lower_bound_violation(metrics["right_wing_fraction"], 0.05),
        ),
        "body_centerline_dominance": _lower_bound_violation(
            metrics["center_body_density_ratio"], 1.15
        ),
        "longitudinal_profile_variation": _lower_bound_violation(
            metrics["longitudinal_profile_cv"], 0.18
        ),
        "planform_sparsity": max(
            _upper_bound_violation(metrics["planform_fill_ratio"], 0.75),
            _upper_bound_violation(metrics["occupied_bbox_fill_ratio"], 0.65),
        ),
        "normalization_margin": normalization_violation,
        "fuselage_end_presence": max(
            _lower_bound_violation(metrics["center_low_end_fraction"], 0.015),
            _lower_bound_violation(metrics["center_high_end_fraction"], 0.015),
            _lower_bound_violation(metrics["center_spine_coverage"], 0.70),
        ),
        "tail_body_plausibility": max(
            _upper_bound_violation(metrics["tail_fraction"], 0.20),
            _upper_bound_violation(
                max(metrics["low_end_fraction"], metrics["high_end_fraction"]),
                0.50,
            ),
        ),
    }


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


def canonicalize_aircraft_voxels(voxels: Any) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return the binary, centered canonical orientation used by validity checks.

    The caller must persist this returned grid when it intends to train on a
    canonicalized corpus.  Merely recording the selected permutation while
    retaining the original array leaves a mixed-orientation training signal.
    """
    raw_grid = _as_tensor(voxels)
    return _canonicalize_aircraft_grid(raw_grid)


def evaluate_aircraft_validity(
    voxels: Any,
    *,
    canonicalize: bool = True,
) -> Dict[str, Any]:
    # Heuristic shape checks are intentionally separated from claim evidence.
    # NASA-STD-7009B treats model/simulation credibility as a lifecycle product,
    # not a single geometric proxy: https://standards.nasa.gov/standard/nasa/nasa-std-7009
    if canonicalize:
        grid, canonicalization = canonicalize_aircraft_voxels(voxels)
        metrics = canonicalization.get("metrics") or _heuristic_metrics(grid)
    else:
        grid = _as_tensor(voxels)
        metrics = _heuristic_metrics(grid)
        canonicalization = {
            "permutation": [0, 1, 2],
            "score": float(_orientation_score(metrics)),
            "metrics": metrics,
            "status": "preserved_input_frame",
        }
    occupancy_upper_bound = 0.04 if min(grid.shape) < 64 else 0.02

    checks = {
        "nonempty_occupancy": 0.002 <= metrics["occupancy_ratio"] <= 0.50,
        "grounded_occupancy_envelope": metrics["occupancy_ratio"] <= occupancy_upper_bound,
        "dominant_connected_airframe": metrics["largest_component_fraction"] >= 0.70,
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
        "normalization_margin": (
            metrics["span_fraction_y"] <= 0.90
            and metrics["length_fraction_x"] <= 0.90
            and metrics["thickness_fraction_z"] <= 0.40
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
    violation_scores = _validity_violation_scores(metrics, occupancy_upper_bound)
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "violation_scores": violation_scores,
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
