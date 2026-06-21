#!/usr/bin/env python3
"""Run a command while sampling process, system, and GPU resource use."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import psutil


GPU_QUERY = [
    "timestamp",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
    "power.draw",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_csv_row(row: str) -> List[str]:
    return [part.strip() for part in row.strip().split(",")]


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.replace("MiB", "").replace("W", "").replace("%", "").strip()
    if cleaned in {"", "[N/A]", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _query_gpu() -> List[Dict[str, Any]]:
    if shutil.which("nvidia-smi") is None:
        return []
    query = ",".join(GPU_QUERY)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    rows: List[Dict[str, Any]] = []
    for line in output.splitlines():
        parts = _split_csv_row(line)
        if len(parts) < len(GPU_QUERY):
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "name": parts[1],
                "memory_total_mb": _parse_float(parts[2]),
                "memory_used_mb": _parse_float(parts[3]),
                "utilization_gpu_percent": _parse_float(parts[4]),
                "utilization_memory_percent": _parse_float(parts[5]),
                "power_draw_w": _parse_float(parts[6]),
            }
        )
    return rows


def _query_gpu_process_memory() -> Dict[int, float]:
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return {}

    usage: Dict[int, float] = {}
    for line in output.splitlines():
        parts = _split_csv_row(line)
        if len(parts) < 2:
            continue
        try:
            usage[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return usage


def _process_tree(root: psutil.Process) -> List[psutil.Process]:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    alive: List[psutil.Process] = []
    for process in processes:
        try:
            if process.is_running():
                alive.append(process)
        except psutil.Error:
            continue
    return alive


def _prime_cpu_counters(processes: Iterable[psutil.Process]) -> None:
    for process in processes:
        try:
            process.cpu_percent(interval=None)
        except psutil.Error:
            continue


def _sample_processes(root: psutil.Process) -> Dict[str, Any]:
    processes = _process_tree(root)
    rss_total = 0
    vms_total = 0
    cpu_percent = 0.0
    process_rows: List[Dict[str, Any]] = []
    gpu_process_memory = _query_gpu_process_memory()

    for process in processes:
        try:
            memory = process.memory_info()
            cpu = process.cpu_percent(interval=None)
            row = {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": float(cpu),
                "rss_mb": float(memory.rss / (1024 * 1024)),
                "vms_mb": float(memory.vms / (1024 * 1024)),
                "gpu_memory_mb": gpu_process_memory.get(process.pid),
            }
        except psutil.Error:
            continue
        rss_total += int(memory.rss)
        vms_total += int(memory.vms)
        cpu_percent += float(cpu)
        process_rows.append(row)

    gpu_memory_total = sum(
        float(row["gpu_memory_mb"] or 0.0)
        for row in process_rows
    )
    return {
        "process_count": len(process_rows),
        "cpu_percent": cpu_percent,
        "rss_mb": float(rss_total / (1024 * 1024)),
        "vms_mb": float(vms_total / (1024 * 1024)),
        "gpu_process_memory_mb": gpu_memory_total,
        "processes": process_rows,
    }


def _summarize(samples: List[Dict[str, Any]], return_code: int, elapsed_s: float) -> Dict[str, Any]:
    def series(path: List[str]) -> List[float]:
        values: List[float] = []
        for sample in samples:
            value: Any = sample
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def stats(values: List[float]) -> Dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    gpu_used: List[float] = []
    gpu_util: List[float] = []
    gpu_mem_util: List[float] = []
    gpu_power: List[float] = []
    for sample in samples:
        for gpu in sample.get("gpu", []):
            for values, key in [
                (gpu_used, "memory_used_mb"),
                (gpu_util, "utilization_gpu_percent"),
                (gpu_mem_util, "utilization_memory_percent"),
                (gpu_power, "power_draw_w"),
            ]:
                value = gpu.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))

    return {
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "sample_count": len(samples),
        "process_cpu_percent": stats(series(["process", "cpu_percent"])),
        "process_rss_mb": stats(series(["process", "rss_mb"])),
        "process_gpu_memory_mb": stats(series(["process", "gpu_process_memory_mb"])),
        "system_cpu_percent": stats(series(["system", "cpu_percent"])),
        "system_memory_used_mb": stats(series(["system", "memory_used_mb"])),
        "gpu_memory_used_mb": stats(gpu_used),
        "gpu_utilization_percent": stats(gpu_util),
        "gpu_memory_utilization_percent": stats(gpu_mem_util),
        "gpu_power_draw_w": stats(gpu_power),
    }


def run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("No command supplied after --")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / args.samples_name
    summary_path = output_dir / args.summary_name
    metadata_path = output_dir / args.metadata_name

    cwd = Path(args.cwd).resolve() if args.cwd else Path.cwd()
    metadata = {
        "created_at": _now_iso(),
        "cwd": str(cwd),
        "command": command,
        "command_display": " ".join(shlex.quote(part) for part in command),
        "interval_s": args.interval,
        "python": sys.executable,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[monitor] command: {metadata['command_display']}")
    print(f"[monitor] samples: {samples_path}")
    print(f"[monitor] summary: {summary_path}")

    start = time.monotonic()
    process = subprocess.Popen(command, cwd=str(cwd), env=os.environ.copy())
    root = psutil.Process(process.pid)
    _prime_cpu_counters(_process_tree(root))

    samples: List[Dict[str, Any]] = []
    with samples_path.open("w", encoding="utf-8") as handle:
        while True:
            return_code = process.poll()
            virtual_memory = psutil.virtual_memory()
            sample = {
                "timestamp": _now_iso(),
                "elapsed_s": time.monotonic() - start,
                "process": _sample_processes(root) if root.is_running() else {},
                "system": {
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_total_mb": float(virtual_memory.total / (1024 * 1024)),
                    "memory_used_mb": float(virtual_memory.used / (1024 * 1024)),
                    "memory_percent": float(virtual_memory.percent),
                },
                "gpu": _query_gpu(),
            }
            samples.append(sample)
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
            handle.flush()
            if return_code is not None:
                break
            time.sleep(max(0.1, float(args.interval)))

    elapsed = time.monotonic() - start
    final_return_code = process.returncode if process.returncode is not None else process.wait()
    summary = {
        **metadata,
        **_summarize(samples, int(final_return_code), elapsed),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return int(final_return_code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Directory for resource samples and summary.")
    parser.add_argument("--interval", type=float, default=5.0, help="Sampling interval in seconds.")
    parser.add_argument("--cwd", default=None, help="Working directory for the monitored command.")
    parser.add_argument("--samples-name", default="resource_samples.jsonl")
    parser.add_argument("--summary-name", default="resource_summary.json")
    parser.add_argument("--metadata-name", default="resource_metadata.json")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after --")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
