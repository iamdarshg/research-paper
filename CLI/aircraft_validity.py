#!/usr/bin/env python3
"""Aircraft-specific voxel validity checks beyond generic connectivity.

These are screening heuristics only. NASA's CFD V&V guidance distinguishes
implementation checks from validation against physical reality:
https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


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


def evaluate_aircraft_validity(voxels: Any) -> Dict[str, Any]:
    # Heuristic shape checks are intentionally separated from claim evidence.
    # NASA-STD-7009B treats model/simulation credibility as a lifecycle product,
    # not a single geometric proxy: https://standards.nasa.gov/standard/nasa/nasa-std-7009
    grid = _as_tensor(voxels)
    res_z, res_y, res_x = grid.shape
    occupied = float(grid.sum().item())
    total = float(grid.numel())
    occupancy_ratio = occupied / max(total, 1.0)

    flipped = torch.flip(grid, dims=[1])
    asymmetry = torch.abs(grid - flipped).sum().item() / max(occupied, 1.0)
    symmetry_score = max(0.0, 1.0 - float(asymmetry))

    span_fraction = _extent(grid, axis=1, resolution=res_y)
    length_fraction = _extent(grid, axis=2, resolution=res_x)

    center_band = grid[:, int(res_y * 0.42):max(int(res_y * 0.58), int(res_y * 0.42) + 1), :]
    left_band = grid[:, :int(res_y * 0.35), :]
    right_band = grid[:, int(res_y * 0.65):, :]
    rear_band = grid[:, :, :max(1, int(res_x * 0.28))]

    center_fraction = float(center_band.sum().item() / max(occupied, 1.0))
    left_fraction = float(left_band.sum().item() / max(occupied, 1.0))
    right_fraction = float(right_band.sum().item() / max(occupied, 1.0))
    tail_fraction = float(rear_band.sum().item() / max(occupied, 1.0))

    checks = {
        "nonempty_occupancy": 0.005 <= occupancy_ratio <= 0.50,
        "symmetry": symmetry_score >= 0.80,
        "span_sanity": span_fraction >= 0.35 and length_fraction >= 0.35,
        "wing_body_balance": center_fraction >= 0.10 and left_fraction >= 0.05 and right_fraction >= 0.05,
        "tail_body_plausibility": 0.01 <= tail_fraction <= 0.45,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not failed else "fail",
        "checks": checks,
        "failed_checks": failed,
        "metrics": {
            "occupancy_ratio": occupancy_ratio,
            "symmetry_score": symmetry_score,
            "span_fraction_y": span_fraction,
            "length_fraction_x": length_fraction,
            "center_body_fraction": center_fraction,
            "left_wing_fraction": left_fraction,
            "right_wing_fraction": right_fraction,
            "tail_fraction": tail_fraction,
        },
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
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
