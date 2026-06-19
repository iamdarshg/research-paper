from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _copy_comparison_rows(source: Path | None, destination: Path) -> tuple[int, str]:
    if source is None or not source.exists():
        destination.write_text("", encoding="utf-8")
        return 0, "No comparison CSV was available; Cd-error plot is blocked by missing paired OpenFOAM/LBM rows."

    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            handle.write("")

    finite_rows = sum(1 for row in rows if _finite_number(row.get("cd_error_percent")))
    if finite_rows == 0:
        return 0, "Cd-error plot is blocked because no finite paired Cd error rows exist."
    return finite_rows, "Cd-error rows are available for plotting."


def _load_grid_speed_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_lbm_cases(grid_speed_summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not grid_speed_summary:
        return []
    out = []
    for case in grid_speed_summary.get("lbm_cases", []):
        out.append(
            {
                "mach": case.get("mach"),
                "grid": case.get("grid"),
                "validity_regime": case.get("validity_regime") or case.get("validity"),
                "claim_grade": case.get("claim_grade"),
                "training_drag_source": case.get("training_drag_source"),
                "lbm_converged": case.get("lbm_converged"),
            }
        )
    return out


def write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Compressibility Evidence Report",
        "",
        "## Solver Status",
        "",
        "- Internal D3Q27 status: low-Mach weakly compressible/isothermal.",
        "- High-Mach internal D3Q27 status: experimental and unvalidated.",
        "- Compressible LBM implementation in this pass: not implemented, because the current solver lacks thermal state, perfect-gas EOS coupling, compressible boundary conditions, and shock/steep-gradient validation.",
        "",
        "## Commands Recorded",
        "",
    ]
    if summary.get("commands"):
        for item in summary["commands"]:
            lines.append(f"- `{item['command']}`: {item['outcome']}")
    else:
        lines.append("- No commands were recorded in this evidence summary.")

    lines.extend(
        [
            "",
            "## LBM Metadata Probe",
            "",
        ]
    )
    if summary.get("lbm_cases"):
        lines.append("| Mach | Grid | Validity regime | Claim grade | Training drag source | Converged |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for case in summary["lbm_cases"]:
            lines.append(
                "| {mach} | {grid} | `{validity}` | `{claim}` | `{source}` | {conv} |".format(
                    mach=case.get("mach"),
                    grid=case.get("grid"),
                    validity=case.get("validity_regime"),
                    claim=case.get("claim_grade"),
                    source=case.get("training_drag_source"),
                    conv=case.get("lbm_converged"),
                )
            )
    else:
        lines.append("No LBM metadata probe summary was available.")

    lines.extend(
        [
            "",
            "## Evidence Gates",
            "",
            f"- Audit artifact: `{summary['audit_artifact']}`",
            f"- Comparison rows: `{summary['comparison_csv']}`",
            f"- Plot status: {summary['plot_status']}",
            "",
            "## Claim Boundary",
            "",
            "Raw internal low-Mach LBM remains separate from calibrated/surrogate/training paths. Internal Mach > 0.3 results must not be cited as validated compressible CFD.",
        ]
    )
    (output_dir / "compressibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "build" / "solver_diagnostics" / "compressibility_evidence_20260612",
    )
    parser.add_argument(
        "--audit-artifact",
        type=Path,
        default=REPO / "build" / "solver_diagnostics" / "compressibility_audit_20260612" / "solver_compressibility_audit.md",
    )
    parser.add_argument("--grid-speed-summary", type=Path, default=None)
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--command", action="append", default=[], help="Recorded command/outcome pair as command :: outcome")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "comparison_rows.csv"
    finite_rows, plot_status = _copy_comparison_rows(args.comparison_csv, comparison_csv)
    grid_speed_summary = _load_grid_speed_summary(args.grid_speed_summary)
    summary = {
        "status": "path_b_gated_low_mach_internal_solver",
        "audit_artifact": str(args.audit_artifact),
        "grid_speed_summary": str(args.grid_speed_summary) if args.grid_speed_summary else None,
        "comparison_csv": str(comparison_csv),
        "finite_cd_error_rows": finite_rows,
        "plot_status": plot_status,
        "lbm_cases": _summarize_lbm_cases(grid_speed_summary),
        "commands": [
            {"command": item.split(" :: ", 1)[0], "outcome": item.split(" :: ", 1)[1] if " :: " in item else "recorded"}
            for item in args.command
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.output_dir, summary)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
