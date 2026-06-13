"""Render low-Mach OpenFOAM/LBM comparison CSV as an interactive Plotly page."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def matched_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["openfoam_grid"] == df["lbm_grid"]].copy()


def marker_text(df: pd.DataFrame) -> list[str]:
    return [
        (
            f"Mach {row.mach}<br>"
            f"OpenFOAM grid {int(row.openfoam_grid)}^3<br>"
            f"LBM grid {int(row.lbm_grid)}^3<br>"
            f"OpenFOAM Cd {row.openfoam_cd:.6g}<br>"
            f"LBM Cd {row.lbm_cd:.6g}<br>"
            f"Cd error {row.cd_error_percent:.3g}%<br>"
            f"OpenFOAM Cl {row.openfoam_cl:.6g}<br>"
            f"LBM Cl {row.lbm_cl:.6g}<br>"
            f"Cl error {row.cl_error_percent:.3g}%<br>"
            f"LBM converged {row.lbm_converged}"
        )
        for row in df.itertuples(index=False)
    ]


def add_error_trace(fig, df: pd.DataFrame, metric: str, row: int, col: int, title: str) -> None:
    error_col = f"{metric.lower()}_error_percent"
    fig.add_trace(
        go.Scatter3d(
            x=df["mach"],
            y=df["lbm_grid"],
            z=df[error_col],
            mode="markers",
            marker={
                "size": 6,
                "color": df["openfoam_grid"],
                "colorscale": "Viridis",
                "colorbar": {"title": "OF grid"},
                "symbol": "circle",
            },
            text=marker_text(df),
            hoverinfo="text",
            name=title,
        ),
        row=row,
        col=col,
    )


def add_value_lines(fig, df: pd.DataFrame, value: str, row: int, col: int) -> None:
    matched = matched_rows(df)
    for grid, group in matched.groupby("lbm_grid"):
        group = group.sort_values("mach")
        fig.add_trace(
            go.Scatter(
                x=group["mach"],
                y=group[f"openfoam_{value}"],
                mode="lines+markers",
                name=f"OpenFOAM {value.upper()} {int(grid)}^3",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=group["mach"],
                y=group[f"lbm_{value}"],
                mode="lines+markers",
                name=f"LBM {value.upper()} {int(grid)}^3",
                line={"dash": "dash"},
            ),
            row=row,
            col=col,
        )


def render(input_csv: Path, output_html: Path) -> None:
    df = pd.read_csv(input_csv)
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(
            "Cd error vs Mach and LBM grid",
            "Cl error vs Mach and LBM grid",
            "Matched-grid Cd values",
            "Matched-grid compute time",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    add_error_trace(fig, df, "Cd", 1, 1, "Cd error")
    add_error_trace(fig, df, "Cl", 1, 2, "Cl error")
    add_value_lines(fig, df, "cd", 2, 1)

    matched = matched_rows(df)
    for grid, group in matched.groupby("lbm_grid"):
        group = group.sort_values("mach")
        fig.add_trace(
            go.Scatter(
                x=group["mach"],
                y=group["openfoam_seconds"],
                mode="lines+markers",
                name=f"OpenFOAM seconds {int(grid)}^3",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=group["mach"],
                y=group["lbm_seconds"],
                mode="lines+markers",
                name=f"LBM seconds {int(grid)}^3",
                line={"dash": "dash"},
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title={
            "text": (
                "Low-Mach OpenFOAM vs Internal D3Q27 LBM Sweep<br>"
                "<sup>OpenFOAM is the reference. High Cl percentage errors are amplified because reference Cl is near zero.</sup>"
            ),
            "x": 0.5,
        },
        height=950,
        width=1350,
        legend={"orientation": "h", "y": -0.12},
        margin={"l": 40, "r": 30, "t": 95, "b": 120},
    )
    fig.update_scenes(
        xaxis_title="Mach",
        yaxis_title="LBM grid resolution",
        zaxis_title="Error (%)",
    )
    fig.update_xaxes(title_text="Mach", row=2, col=1)
    fig.update_yaxes(title_text="Cd", row=2, col=1)
    fig.update_xaxes(title_text="Mach", row=2, col=2)
    fig.update_yaxes(title_text="Seconds", row=2, col=2)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()
    render(args.input_csv, args.output_html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
