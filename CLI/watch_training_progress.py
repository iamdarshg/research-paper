#!/usr/bin/env python3
"""Join completed epoch losses with live resource telemetry without touching training."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            handle.seek(max(0, end - 65536))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    except OSError:
        return None
    return None


def _latest_epoch(path: Path) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        return None, {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {}
    history = payload.get("history", []) if isinstance(payload, dict) else []
    latest = history[-1] if isinstance(history, list) and history else None
    previous = history[-2] if isinstance(history, list) and len(history) > 1 else None
    if not isinstance(latest, dict):
        return None, {}
    labels = {
        "epoch_state": "completed",
        "loss_trend": "first_epoch" if previous is None else "flat",
        "solver_coverage": "unknown",
        "promotion": "not_reported",
    }
    current_loss = latest.get("optimization_loss", latest.get("loss"))
    previous_loss = previous.get("optimization_loss", previous.get("loss")) if previous else None
    if isinstance(current_loss, (int, float)) and isinstance(previous_loss, (int, float)):
        if current_loss < previous_loss:
            labels["loss_trend"] = "improving"
        elif current_loss > previous_loss:
            labels["loss_trend"] = "regressing"
    coverage = latest.get("direct_solver_iteration_coverage")
    if isinstance(coverage, (int, float)):
        labels["solver_coverage"] = "complete" if coverage >= 0.999 else "partial"
    promoted = latest.get("promotion_gate_passed")
    if isinstance(promoted, (int, float)):
        labels["promotion"] = "passed" if promoted >= 0.5 else "failed"
    return latest, labels


def _trainer_state(pid: Optional[int]) -> str:
    if pid is None:
        return "not_checked"
    try:
        return "running" if psutil.pid_exists(pid) else "exited"
    except psutil.Error:
        return "unknown"


def snapshot(history_path: Path, telemetry_path: Path, trainer_pid: Optional[int]) -> Dict[str, Any]:
    epoch, labels = _latest_epoch(history_path)
    telemetry = _latest_jsonl(telemetry_path)
    if epoch is None:
        labels = {
            "epoch_state": "waiting_for_first_completed_epoch",
            "loss_trend": "unavailable",
            "solver_coverage": "unavailable",
            "promotion": "unavailable",
        }
    return {
        "kind": "training_progress_snapshot",
        "timestamp": _now_iso(),
        "labels": {**labels, "trainer_state": _trainer_state(trainer_pid)},
        "epoch_metrics": epoch,
        "resource_telemetry": telemetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-path", required=True)
    parser.add_argument("--telemetry-path", required=True)
    parser.add_argument("--output", required=True, help="Append-only labelled snapshot JSONL.")
    parser.add_argument("--trainer-pid", type=int, default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        item = snapshot(Path(args.history_path), Path(args.telemetry_path), args.trainer_pid)
        line = json.dumps(item, sort_keys=True)
        print(line, flush=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
