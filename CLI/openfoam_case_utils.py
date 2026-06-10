"""Utilities for local OpenFOAM diagnostic cases.

These helpers are intentionally small and script-friendly. They keep WSL
staging and latest-time pressure-force extraction out of one-off notebooks and
chat snippets.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable, Optional


def wsl_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def tar_directory_bytes(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(root.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(root)))
    return buffer.getvalue()


def extract_tar_bytes(data: bytes, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        tar.extractall(target)


def copy_windows_dir_to_wsl(src: Path, wsl_dst: str, *, distro: str = "Ubuntu-24.04") -> None:
    """Copy a Windows directory into a WSL path via tar over stdin.

    This avoids path-space issues like ``/mnt/c/Users/Darsh Gupta`` being
    accidentally collapsed by shell quoting.
    """
    src = Path(src).resolve()
    subprocess.run(
        ["wsl", "-d", distro, "--", "bash", "-lc", f"rm -rf {wsl_quote(wsl_dst)} && mkdir -p {wsl_quote(wsl_dst)}"],
        check=True,
        text=True,
        capture_output=True,
    )
    proc = subprocess.run(
        ["wsl", "-d", distro, "--", "bash", "-lc", f"tar -C {wsl_quote(wsl_dst)} -xf -"],
        input=tar_directory_bytes(src),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore"))


def copy_wsl_dir_to_windows(wsl_src: str, dst: Path, *, distro: str = "Ubuntu-24.04") -> None:
    proc = subprocess.run(
        ["wsl", "-d", distro, "--", "bash", "-lc", f"tar -C {wsl_quote(wsl_src)} -cf - ."],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore"))
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    extract_tar_bytes(proc.stdout, dst)


def run_openfoam_wsl(
    wsl_case: str,
    command: str,
    *,
    distro: str = "Ubuntu-24.04",
    bashrc: str = "/usr/share/openfoam/etc/bashrc",
    timeout: int = 1200,
) -> subprocess.CompletedProcess[str]:
    shell = f"cd {wsl_quote(wsl_case)} && source {wsl_quote(bashrc)} >/dev/null 2>&1 && {command}"
    return subprocess.run(
        ["wsl", "-d", distro, "--", "bash", "-lc", shell],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def numeric_time_dirs(case: Path) -> list[Path]:
    times: list[tuple[float, Path]] = []
    for path in Path(case).iterdir():
        if not path.is_dir():
            continue
        try:
            times.append((float(path.name), path))
        except ValueError:
            continue
    return [path for _, path in sorted(times, key=lambda item: item[0])]


def latest_numeric_time_dir(case: Path) -> Path:
    dirs = numeric_time_dirs(case)
    if not dirs:
        raise RuntimeError(f"No numeric OpenFOAM time directories found in {case}")
    return dirs[-1]


def temporarily_hide_force_files(case: Path) -> list[tuple[Path, Path]]:
    """Hide existing force files so manual pressure integration uses fields.

    ``run_internal_benchmark.pressure_force_from_case`` prefers existing
    ``postProcessing/**/force*.dat`` files. For long continuation runs, those
    files can be stale. This helper temporarily moves them aside.
    """
    moved: list[tuple[Path, Path]] = []
    for path in Path(case).glob("postProcessing/**/force*.dat"):
        hidden = path.with_suffix(path.suffix + ".stale")
        path.rename(hidden)
        moved.append((hidden, path))
    return moved


def restore_hidden_force_files(moved: Iterable[tuple[Path, Path]]) -> None:
    for hidden, original in moved:
        if hidden.exists() and not original.exists():
            hidden.rename(original)


def path_from_env_temp(name: str) -> Path:
    return Path(os.environ["TEMP"]) / name
