#!/usr/bin/env python3
"""Render grid-size loss curves from training_metrics.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(paths: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history") or []
        for row in history:
            item = dict(row)
            item["metrics_path"] = str(path)
            item["coordinate_decoder_threshold"] = payload.get("training_config", {}).get("coordinate_decoder_threshold")
            item["direct_solver_loss_weight"] = payload.get("training_config", {}).get("direct_solver_loss_weight")
            item["direct_solver_steps"] = payload.get("training_config", {}).get("direct_solver_steps")
            item["direct_solver_perturbation_grid_size"] = payload.get("training_config", {}).get("direct_solver_perturbation_grid_size")
            item["direct_connectivity_weight"] = payload.get("training_config", {}).get("direct_connectivity_weight")
            rows.append(item)
    if not rows:
        raise ValueError("No history rows found in supplied training metrics.")
    return rows


def _latest_epoch_by_grid(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        grid = int(row["grid_size"])
        if grid not in latest or int(row.get("epoch", 0)) >= int(latest[grid].get("epoch", 0)):
            latest[grid] = row
    return [latest[grid] for grid in sorted(latest)]


def render(rows: List[Dict[str, Any]], output: Path) -> None:
    latest = _latest_epoch_by_grid(rows)
    grids = np.asarray([int(row["grid_size"]) for row in latest])
    optimizer = np.asarray([float(row.get("optimization_loss", row.get("loss", np.nan))) for row in latest])
    reconstruction = np.asarray([float(row.get("geometry_reconstruction", np.nan)) for row in latest])
    direct_solver_eval = np.asarray([float(row.get("direct_solver_eval_loss", 0.0)) for row in latest])

    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 220,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    axes[0].plot(grids, optimizer, marker="o", color="#496A81", label="optimizer loss")
    axes[0].plot(grids, reconstruction, marker="s", color="#4F9D69", label="geometry BCE")
    axes[0].set_title("Coordinate-decoder training loss")
    axes[0].set_xlabel("voxel grid edge length")
    axes[0].set_ylabel("epoch mean loss")
    axes[0].set_xticks(grids)
    axes[0].legend(frameon=False)

    width = max(1.0, min(7.0, (float(grids.max()) - float(grids.min())) / max(len(grids), 1) * 0.28))
    axes[1].bar(grids, direct_solver_eval, width=width, color="#7B6D8D", label="scheduled solver eval")
    axes[1].set_title("Measured solver-in-loop objective")
    axes[1].set_xlabel("voxel grid edge length")
    axes[1].set_ylabel("scheduled mean loss")
    axes[1].set_xticks(grids)
    axes[1].legend(frameon=False)

    fig.suptitle("Grid-size loss sweep with direct solver-in-loop optimization", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = _load_metrics(args.metrics)
    render(rows, args.output)
    summary = {
        "figure": str(args.output.resolve()),
        "rows": _latest_epoch_by_grid(rows),
        "claim_boundary": (
            "Local smoke sweep only. Resolution changes are isolated from decoder-family changes "
            "by forcing the coordinate decoder at every grid size. direct_solver_eval_loss is the "
            "scheduled measured internal-solver objective included in optimization through SPSA; "
            "it is not external aerodynamic validation. If multiple metrics files are supplied for "
            "the same grid and epoch, the later path on the command line is selected so continuation "
            "runs can replace an earlier checkpoint deliberately."
        ),
    }
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
