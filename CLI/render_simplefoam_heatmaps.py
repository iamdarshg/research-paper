"""Render simpleFoam low-Mach suite rows as interactive Plotly heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def pivot(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index="mach", columns="grid", values=value, aggfunc="first").sort_index()


def hover_grid(df: pd.DataFrame, value: str) -> list[list[str]]:
    table = pivot(df, value)
    hover: list[list[str]] = []
    for mach in table.index:
        row_hover = []
        for grid in table.columns:
            rows = df[(df["mach"] == mach) & (df["grid"] == grid)]
            if rows.empty:
                row_hover.append("No run")
                continue
            record = rows.iloc[0]
            row_hover.append(
                f"Mach {mach}<br>"
                f"Grid {int(grid)}^3<br>"
                f"Status {record.get('openfoam_status')}<br>"
                f"OpenFOAM Cd {record.get('openfoam_cd')}<br>"
                f"LBM Cd {record.get('lbm_cd')}<br>"
                f"Cd error {record.get('cd_error_percent')}%<br>"
                f"OpenFOAM seconds {record.get('openfoam_seconds')}<br>"
                f"LBM seconds {record.get('lbm_seconds')}<br>"
                f"Rough converged {record.get('of_rough_converged')}<br>"
                f"U residual max {record.get('of_u_initial_residual_max')}<br>"
                f"Local continuity {record.get('of_continuity_local')}"
            )
        hover.append(row_hover)
    return hover


def add_heatmap(fig, df: pd.DataFrame, value: str, title: str, row: int, col: int, colorscale: str) -> None:
    table = pivot(df, value)
    fig.add_trace(
        go.Heatmap(
            x=[f"{int(col)}^3" for col in table.columns],
            y=[f"{float(idx):.2f}" for idx in table.index],
            z=table.values,
            colorscale=colorscale,
            colorbar={"title": title},
            text=hover_grid(df, value),
            hoverinfo="text",
            name=title,
        ),
        row=row,
        col=col,
    )


def render(input_csv: Path, output_html: Path) -> None:
    df = pd.read_csv(input_csv)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cd error (%)",
            "Cl error (%)",
            "OpenFOAM compute time (s)",
            "Completed cells",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )
    add_heatmap(fig, df, "cd_error_percent", "Cd error %", 1, 1, "Viridis")
    add_heatmap(fig, df, "cl_error_percent", "Cl error %", 1, 2, "Plasma")
    add_heatmap(fig, df, "openfoam_seconds", "Seconds", 2, 1, "Cividis")

    completed = df[(df["openfoam_status"] == "completed") & (df.get("of_rough_converged", False).astype(str) == "True")].copy()
    unconverged = df[(df["openfoam_status"] == "completed") & (df.get("of_rough_converged", False).astype(str) != "True")].copy()
    failed = df[df["openfoam_status"] != "completed"].copy()
    if not completed.empty:
        fig.add_trace(
            go.Scatter(
                x=completed["grid"],
                y=completed["mach"],
                mode="markers",
                marker={"size": 9, "color": completed["cd_error_percent"], "colorscale": "Viridis", "showscale": True},
                text=[
                    f"Grid {int(r.grid)}^3<br>Mach {r.mach}<br>Cd error {r.cd_error_percent}<br>Seconds {r.openfoam_seconds}<br>Rough converged True"
                    for r in completed.itertuples(index=False)
                ],
                hoverinfo="text",
                name="completed",
            ),
            row=2,
            col=2,
        )
    if not unconverged.empty:
        fig.add_trace(
            go.Scatter(
                x=unconverged["grid"],
                y=unconverged["mach"],
                mode="markers",
                marker={"size": 9, "color": "orange", "symbol": "triangle-up"},
                text=[
                    (
                        f"Grid {int(r.grid)}^3<br>Mach {r.mach}<br>Status unconverged<br>"
                        f"U residual {getattr(r, 'of_u_initial_residual_max', None)}<br>"
                        f"Local continuity {getattr(r, 'of_continuity_local', None)}"
                    )
                    for r in unconverged.itertuples(index=False)
                ],
                hoverinfo="text",
                name="completed but unconverged",
            ),
            row=2,
            col=2,
        )
    if not failed.empty:
        fig.add_trace(
            go.Scatter(
                x=failed["grid"],
                y=failed["mach"],
                mode="markers",
                marker={"size": 10, "color": "crimson", "symbol": "x"},
                text=[
                    f"Grid {int(r.grid)}^3<br>Mach {r.mach}<br>Status {r.openfoam_status}<br>Stage {r.openfoam_failed_stage}"
                    for r in failed.itertuples(index=False)
                ],
                hoverinfo="text",
                name="failed/timeout",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title={
            "text": (
                "simpleFoam Low-Mach Error Suite<br>"
                "<sup>OpenFOAM simpleFoam reference; cells appear only as bounded runs complete.</sup>"
            ),
            "x": 0.5,
        },
        height=950,
        width=1350,
        margin={"l": 60, "r": 40, "t": 100, "b": 80},
    )
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        fig.layout[axis].title = "Grid resolution"
    for axis in ("yaxis", "yaxis2", "yaxis3"):
        fig.layout[axis].title = "Mach"
    fig.update_xaxes(title_text="Grid resolution", row=2, col=2)
    fig.update_yaxes(title_text="Mach", row=2, col=2)
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
