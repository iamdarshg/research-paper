#!/usr/bin/env python3
"""Render Airshow corpus, training, and generated-geometry figures for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


TRAINING_METRICS = [
    {
        "epoch": 1,
        "loss": 14.938825494713253,
        "mse": 0.9822448200649685,
        "geometry_reconstruction": 0.2227125730779436,
        "consistency": 0.010249742534425524,
        "connectivity": 2.5519284748368793,
        "aerodynamic": 11.171689775254992,
    },
    {
        "epoch": 2,
        "loss": 20.219982828034293,
        "mse": 0.8463106320963966,
        "geometry_reconstruction": 0.07974993081556427,
        "consistency": 0.002187567080060641,
        "connectivity": 0.04037682910760244,
        "aerodynamic": 19.251358032226562,
    },
    {
        "epoch": 3,
        "loss": 21.590529081556532,
        "mse": 0.7996996521949769,
        "geometry_reconstruction": 0.07781737653745545,
        "consistency": 0.0010912266456418566,
        "connectivity": 0.0014880951907899644,
        "aerodynamic": 20.710432773166232,
    },
]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure_style() -> None:
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


def render_corpus_summary(report: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "airshow_corpus_summary.png"
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)

    funnel_labels = ["public docs", "eligible", "converted", "404 rejects"]
    funnel_values = [
        int(report["all_public_model_documents"]),
        int(report["candidate_documents_after_license_and_geometry_filter"]),
        int(report["record_count"]),
        int(report["failure_count"]),
    ]
    colors = ["#496A81", "#6EA6A1", "#4F9D69", "#B35C44"]
    axes[0].bar(funnel_labels, funnel_values, color=colors)
    axes[0].set_title("Airshow corpus funnel")
    axes[0].set_ylabel("records")
    axes[0].tick_params(axis="x", rotation=28)
    for idx, value in enumerate(funnel_values):
        axes[0].text(idx, value + 4, str(value), ha="center", va="bottom", fontsize=7)

    license_counts = report["license_counts"]
    license_labels = ["CC0", "CC BY", "CC BY-SA"]
    license_values = [
        int(license_counts.get("No Rights Reserved (CC0)", 0)),
        int(license_counts.get("Attribution (CC BY)", 0)),
        int(license_counts.get("Attribution Share Alike (CC BY-SA)", 0)),
    ]
    axes[1].barh(license_labels, license_values, color=["#4F9D69", "#E0A458", "#7B6D8D"])
    axes[1].set_title("Admitted licenses")
    axes[1].set_xlabel("records")
    for idx, value in enumerate(license_values):
        axes[1].text(value + 3, idx, str(value), va="center", fontsize=7)

    split_counts = report["split_counts"]
    split_labels = ["train", "val", "test", "holdout"]
    split_values = [int(split_counts.get(label, 0)) for label in split_labels]
    axes[2].bar(split_labels, split_values, color=["#496A81", "#6EA6A1", "#E0A458", "#7B6D8D"])
    axes[2].set_title("Deterministic splits")
    axes[2].set_ylabel("records")
    for idx, value in enumerate(split_values):
        axes[2].text(idx, value + 4, str(value), ha="center", va="bottom", fontsize=7)

    fig.suptitle("Public VSP Airshow corpus: traceable inputs and admitted records", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_training_losses(output_dir: Path) -> Path:
    path = output_dir / "airshow_training_losses.png"
    epochs = np.asarray([row["epoch"] for row in TRAINING_METRICS])
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.45), constrained_layout=True)

    axes[0].plot(epochs, [row["loss"] for row in TRAINING_METRICS], marker="o", color="#496A81", label="total")
    axes[0].plot(epochs, [row["aerodynamic"] for row in TRAINING_METRICS], marker="s", color="#B35C44", label="aero term")
    axes[0].set_title("Smoke training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_xticks(epochs)
    axes[0].legend(frameon=False)

    small_terms = ["mse", "geometry_reconstruction", "consistency", "connectivity"]
    palette = ["#4F9D69", "#E0A458", "#7B6D8D", "#6EA6A1"]
    for term, color in zip(small_terms, palette):
        axes[1].plot(epochs, [row[term] for row in TRAINING_METRICS], marker="o", color=color, label=term.replace("_", " "))
    axes[1].set_title("Non-aero terms")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_xticks(epochs)
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle("Three-epoch Airshow smoke training diagnostics", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_flight_metrics(report: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "airshow_flight_path_metrics.png"
    cases = report["cases"]
    labels = [case["case_id"].replace("_", "\n") for case in cases]
    occupancy = [case["geometry_summary"]["occupancy_ratio"] for case in cases]
    cd_values = [case["cfd_metrics"]["drag_coefficient"] for case in cases]
    ld_values = [case["cfd_metrics"]["lift_to_drag"] for case in cases]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    series = [
        ("Occupancy", occupancy, "#496A81", "fraction"),
        ("Raw D3Q27 Cd", cd_values, "#B35C44", "coefficient"),
        ("Raw D3Q27 L/D", ld_values, "#4F9D69", "ratio"),
    ]
    for axis, (title, values, color, ylabel) in zip(axes, series):
        axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        for idx, value in enumerate(values):
            axis.text(idx, value + max(values) * 0.035, f"{value:.4f}", ha="center", va="bottom", fontsize=6)
    fig.suptitle("Generated flight-path smoke checks: all validity results fail span sanity", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _projection_image(voxels: np.ndarray, axis: int) -> np.ndarray:
    projected = voxels.max(axis=axis)
    return np.flipud(projected.T if axis == 0 else projected)


def render_generated_geometry(report: Dict[str, Any], output_dir: Path) -> Path:
    path = output_dir / "airshow_generated_geometry.png"
    cases = report["cases"]
    fig = plt.figure(figsize=(7.1, 5.4), constrained_layout=True)
    subfigs = fig.subfigures(len(cases), 1, hspace=0.07)
    if len(cases) == 1:
        subfigs = [subfigs]

    for subfig, case in zip(subfigs, cases):
        voxel_path = Path(case["artifact_paths"]["voxels_npy"])
        voxels = np.load(voxel_path) > 0.5
        title = case["case_id"].replace("_", " ")
        subfig.suptitle(
            f"{title}: occ={case['geometry_summary']['occupancy_ratio']:.5f}, validity={case['validity']['status']} ({', '.join(case['validity']['failed_checks'])})",
            fontsize=9,
        )

        ax3d = subfig.add_subplot(1, 4, 1, projection="3d")
        projection_axes = [subfig.add_subplot(1, 4, index) for index in range(2, 5)]
        filled = np.argwhere(voxels)
        if filled.size:
            ax3d.voxels(voxels, facecolors="#4F9D69", edgecolor="#254236", linewidth=0.18, alpha=0.88)
        ax3d.set_title("voxel render", fontsize=8)
        ax3d.set_axis_off()
        ax3d.view_init(elev=23, azim=42)
        ax3d.set_box_aspect((1, 1, 1))

        projection_specs = [("front", 0), ("top", 1), ("side", 2)]
        for axis, (label, projection_axis) in zip(projection_axes, projection_specs):
            axis.imshow(_projection_image(voxels, projection_axis), cmap="Greys", interpolation="nearest")
            axis.set_title(label, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
                spine.set_color("#666666")

    fig.suptitle("Generated Airshow-checkpoint geometries and projections", fontsize=11)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-report", default="build/airshow_grounded_corpus_20260620/corpus_report.json")
    parser.add_argument("--flight-report", default="build/airshow_training_20260620/flight_path_tests/flight_path_results.json")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    _figure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_report = _load_json(Path(args.corpus_report))
    flight_report = _load_json(Path(args.flight_report))
    paths: List[Path] = [
        render_corpus_summary(corpus_report, output_dir),
        render_training_losses(output_dir),
        render_flight_metrics(flight_report, output_dir),
        render_generated_geometry(flight_report, output_dir),
    ]
    print(json.dumps({"figures": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
