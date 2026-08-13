#!/usr/bin/env python3
"""Live, read-only training dashboard for epoch metrics and resource telemetry."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich import box
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from watch_training_progress import snapshot


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _latest_update(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - 262144))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("kind") != "optimizer_update":
            continue
        done = int(record.get("completed_in_epoch", 0))
        total = int(record.get("total_in_epoch", 0))
        losses = record.get("losses", {})
        gradients = record.get("student_gradients", {})
        return {
            "done": done,
            "total": total,
            "fraction": done / max(total, 1),
            "timing": f"global step {int(record.get('global_step', 0))}",
            "run_state_checkpoint_path": record.get("run_state_checkpoint_path"),
            "resumed_from_update": record.get("resumed_from_update"),
            "remaining_in_epoch": int(record.get("remaining_in_epoch", max(total - done, 0))),
            "metrics": {
                "opt_loss": losses.get("optimization"),
                "mse": losses.get("mse"),
                "clean_geom": losses.get("clean_geometry"),
                "geom": losses.get("geometry"),
                "gen_geom": losses.get("generation_geometry"),
                "consistency": losses.get("consistency"),
                "latent_recon": losses.get("latent_reconstruction"),
                "direct_solver": losses.get("direct_solver"),
                "grad_data": (gradients.get("data") or {}).get("applied_norm"),
                "grad_cons": (gradients.get("consistency") or {}).get(
                    "applied_norm"
                ),
                "grad_direct": (gradients.get("direct") or {}).get(
                    "applied_norm"
                ),
            },
        }
    return None


def _latest_live_batch(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            prefix = handle.read(4)
            handle.seek(0, 2)
            end = handle.tell()
            if prefix.startswith(b"\xff\xfe"):
                start = max(2, end - 524288)
                start += start % 2
                handle.seek(start)
                text = handle.read().decode("utf-16-le", errors="replace")
            elif prefix.startswith(b"\xfe\xff"):
                start = max(2, end - 524288)
                start += start % 2
                handle.seek(start)
                text = handle.read().decode("utf-16-be", errors="replace")
            else:
                handle.seek(max(0, end - 262144))
                text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None

    lines = [ANSI_ESCAPE.sub("", line).strip() for line in re.split(r"[\r\n]+", text)]
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if "opt_loss=" not in line:
            continue
        progress = re.search(r"(?P<done>\d+)/(?P<total>\d+)", line)
        if progress is None:
            continue
        record = line
        for continuation in lines[index + 1:index + 5]:
            if continuation.startswith(("Training with optimizations", "Running D3Q27")):
                break
            record += " " + continuation
            if "]" in continuation:
                break
        values: Dict[str, float] = {}
        for key, raw_value in re.findall(
            r"([A-Za-z_][A-Za-z0-9_]*)=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            record,
        ):
            try:
                values[key] = float(raw_value)
            except ValueError:
                continue
        done = int(progress.group("done"))
        total = int(progress.group("total"))
        timing = re.search(r"\[(?P<timing>[^\]]+)", line)
        return {
            "done": done,
            "total": total,
            "fraction": done / max(total, 1),
            "timing": timing.group("timing") if timing else "-",
            "metrics": values,
        }
    return None


def _recent_telemetry(path: Path, limit: int = 12) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - 262144))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:]:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _number(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"


def _state(value: str) -> Text:
    colour = {
        "running": "green",
        "improving": "green",
        "complete": "green",
        "passed": "green",
        "regressing": "red",
        "failed": "red",
        "partial": "yellow",
        "waiting_for_first_completed_epoch": "yellow",
        "unavailable": "dim",
    }.get(value, "cyan")
    return Text(value.replace("_", " "), style=f"bold {colour}")


def _metric_table(metrics: Optional[Dict[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold cyan")
    table.add_column("Metric", ratio=2)
    table.add_column("Value", justify="right")
    if not metrics:
        table.add_row("Epoch metrics", "Waiting for epoch 1 to complete")
        return table
    rows = [
        ("Epoch", metrics.get("epoch"), 0),
        ("Optimisation loss", metrics.get("optimization_loss", metrics.get("loss")), 5),
        ("Geometry reconstruction", metrics.get("clean_geometry_reconstruction"), 5),
        ("Generation reconstruction", metrics.get("generation_reconstruction"), 5),
        ("Direct solver loss", metrics.get("direct_solver_loss"), 5),
        ("Direct aero loss", metrics.get("direct_aero_loss"), 5),
        ("Connectivity loss", metrics.get("direct_connectivity_loss"), 5),
        ("Aircraft validity loss", metrics.get("direct_aircraft_validity_loss"), 5),
        ("SPSA gradient norm", metrics.get("direct_spsa_gradient_norm"), 5),
        ("Solver calls", metrics.get("direct_solver_call_count"), 0),
    ]
    for name, value, digits in rows:
        table.add_row(name, _number(value, digits))
    return table


def _live_batch_table(live_batch: Optional[Dict[str, Any]]) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold green")
    table.add_column("Live batch", ratio=2)
    table.add_column("Value", justify="right")
    if not live_batch:
        table.add_row("Status", "Preparing data / promotion baseline")
        table.add_row("Losses", "Waiting for first optimizer update")
        return table

    done = int(live_batch["done"])
    total = int(live_batch["total"])
    width = 32
    filled = min(width, max(0, round(width * float(live_batch["fraction"]))))
    progress = f"[green]{'#' * filled}[/green][dim]{'-' * (width - filled)}[/dim]"
    metrics = live_batch["metrics"]
    run_state_path = live_batch.get("run_state_checkpoint_path")
    run_state_age = "-"
    if run_state_path:
        try:
            run_state_age = f"{max(0.0, time.time() - Path(run_state_path).stat().st_mtime):.0f}s"
        except OSError:
            run_state_age = "unavailable"
    rows = [
        ("Progress", f"{done}/{total} ({100.0 * done / max(total, 1):.1f}%)"),
        ("Batch bar", progress),
        ("Timing / ETA", str(live_batch.get("timing", "-"))),
        ("Optimisation loss", _number(metrics.get("opt_loss"), 5)),
        ("Direct solver loss", _number(metrics.get("direct_solver"), 5)),
        ("Clean geometry", _number(metrics.get("clean_geom"), 5)),
        ("Generated geometry", _number(metrics.get("gen_geom"), 5)),
        ("Consistency", _number(metrics.get("consistency"), 5)),
        ("Latent reconstruction", _number(metrics.get("latent_recon"), 5)),
        ("Data gradient", _number(metrics.get("grad_data"), 5)),
        ("Consistency gradient", _number(metrics.get("grad_cons"), 5)),
        ("Direct gradient", _number(metrics.get("grad_direct"), 5)),
        ("MSE", _number(metrics.get("mse"), 5)),
        ("Run mode", "exact resume" if live_batch.get("resumed_from_update") else "fresh run"),
        ("Remaining in epoch", str(live_batch.get("remaining_in_epoch", "-"))),
        ("Run-state checkpoint", str(run_state_path or "-")),
        ("Checkpoint age", run_state_age),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


def _resource_table(
    telemetry: Optional[Dict[str, Any]],
    recent: Optional[list[Dict[str, Any]]] = None,
) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, expand=True, header_style="bold magenta")
    table.add_column("Resource", ratio=2)
    table.add_column("Current", justify="right")
    if not telemetry:
        table.add_row("Telemetry", "Waiting for resource sample")
        return table
    process = telemetry.get("process", {})
    system = telemetry.get("system", {})
    gpu = (telemetry.get("gpu") or [{}])[0]
    gpu_utilization = []
    for sample in recent or []:
        sample_gpu = (sample.get("gpu") or [{}])[0]
        value = sample_gpu.get("utilization_gpu_percent")
        if isinstance(value, (int, float)):
            gpu_utilization.append(float(value))
    rolling_utilization = (
        f"{sum(gpu_utilization) / len(gpu_utilization):.1f}% "
        f"({min(gpu_utilization):.0f}-{max(gpu_utilization):.0f}%)"
        if gpu_utilization
        else "-"
    )
    rows = [
        ("Elapsed", f"{float(telemetry.get('elapsed_s', 0)) / 3600:.2f} h"),
        ("CUDA utilisation (latest)", f"{_number(gpu.get('utilization_gpu_percent'), 1)} %"),
        ("CUDA utilisation (rolling)", rolling_utilization),
        ("GPU memory controller", f"{_number(gpu.get('utilization_memory_percent'), 1)} %"),
        ("GPU memory", f"{_number(gpu.get('memory_used_mb'), 0)} / {_number(gpu.get('memory_total_mb'), 0)} MB"),
        ("GPU power", f"{_number(gpu.get('power_draw_w'), 1)} W"),
        ("GPU engine source", "nvidia-smi CUDA; not Task Manager 3D"),
        ("Trainer CPU", f"{_number(process.get('cpu_percent'), 1)} %"),
        ("Trainer RSS", f"{_number(process.get('rss_mb'), 0)} MB"),
        ("System RAM", f"{_number(system.get('memory_percent'), 1)} %"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    return table


def render(
    item: Dict[str, Any],
    live_batch: Optional[Dict[str, Any]] = None,
    recent_telemetry: Optional[list[Dict[str, Any]]] = None,
) -> Group:
    labels = item["labels"]
    state = Table.grid(expand=True)
    for _ in range(4):
        state.add_column(justify="center")
    state.add_row(
        Text("TRAINER", style="bold"),
        Text("LOSS TREND", style="bold"),
        Text("SOLVER", style="bold"),
        Text("PROMOTION", style="bold"),
    )
    state.add_row(
        _state(labels["trainer_state"]),
        _state(labels["loss_trend"]),
        _state(labels["solver_coverage"]),
        _state(labels["promotion"]),
    )
    title = "Aircraft Training Monitor"
    subtitle = f"epoch state: {labels['epoch_state'].replace('_', ' ')} | refreshed {item['timestamp']}"
    return Group(
        Panel(Align.center(Text(title, style="bold white")), subtitle=subtitle, border_style="blue"),
        Panel(state, border_style="cyan"),
        Panel(_live_batch_table(live_batch), title="Within-Epoch Progress", border_style="green"),
        Panel(_metric_table(item["epoch_metrics"]), title="Losses", border_style="cyan"),
        Panel(
            _resource_table(item["resource_telemetry"], recent_telemetry),
            title="Resources",
            border_style="magenta",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", required=True)
    parser.add_argument("--telemetry-path", required=True)
    parser.add_argument("--console-log", default=None)
    parser.add_argument("--updates-path", default=None)
    parser.add_argument("--trainer-pid", type=int, default=None)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    console_log = Path(args.console_log) if args.console_log else None
    updates_path = Path(args.updates_path) if args.updates_path else None

    def view() -> Group:
        return render(
            snapshot(Path(args.history_path), Path(args.telemetry_path), args.trainer_pid),
            _latest_update(updates_path) or _latest_live_batch(console_log),
            _recent_telemetry(Path(args.telemetry_path)),
        )

    if args.once:
        from rich.console import Console
        Console().print(view())
        return 0
    with Live(view(), refresh_per_second=max(1, int(1 / max(0.1, args.refresh))), screen=True) as live:
        while True:
            live.update(view())
            import time
            time.sleep(max(0.5, args.refresh))


if __name__ == "__main__":
    raise SystemExit(main())
