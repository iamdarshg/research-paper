#!/usr/bin/env python3
"""Export an interruption-safe resume bundle (issue #39).

Packages the artifacts a free-cloud session needs to continue training after a
disconnect:

  * the latest valid checkpoint (resume_manifest.json's ``latest_checkpoint``,
    falling back to the highest ``checkpoint_updates_*.pt``),
  * ``resume_manifest.json``, ``updates.jsonl``, ``history.json``,
  * ``initial_geometry_promotion.json`` when present,
  * log files when ``--include-logs`` is given,

into a deterministic ``.tar.gz`` (every member is written with ``mtime=0`` and
``gzip`` header mtime is zeroed) that embeds a ``bundle_manifest.json`` with
per-file SHA-256 digests plus the bundle manifest's own self-hash
(``bundle_sha256``: SHA-256 of the canonical JSON serialization of the
manifest itself, i.e. a content-addressed identity of the bundle descriptor).
The archive's on-disk SHA-256 is printed for the operator.

The importer (``import_resume_bundle.py``) strictly re-verifies every
per-file digest and the manifest self-hash before extracting, so a single
corrupted byte is caught before any file is touched.

Exit codes: 0 success; 1 missing/invalid inputs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
REQUIRED_RUN_FILES = ("resume_manifest.json", "updates.jsonl", "history.json")
LOG_SUFFIXES = (".log", ".out", ".err")


def _sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_latest_checkpoint(run_dir: Path) -> Path | None:
    """Latest valid checkpoint: manifest pointer first, then highest cadence."""
    manifest_path = run_dir / "resume_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            latest = Path(payload["latest_checkpoint"])
            if not latest.is_absolute():
                latest = run_dir / latest
            if latest.is_file():
                return latest.resolve()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    cadence = sorted(run_dir.glob("checkpoint_updates_*.pt"))
    if cadence:
        return cadence[-1].resolve()
    return None


def _collect_members(run_dir: Path, include_logs: bool) -> list[tuple[str, Path]]:
    """Ordered (archive-relative-name, source-path) member list."""
    members: list[tuple[str, Path]] = []
    checkpoint = _select_latest_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            "no checkpoint found: resume_manifest.json has no existing "
            "latest_checkpoint and no checkpoint_updates_*.pt files exist"
        )
    members.append((checkpoint.name, checkpoint))
    for required in REQUIRED_RUN_FILES:
        path = run_dir / required
        if not path.is_file():
            raise FileNotFoundError(
                f"run directory {run_dir} is missing required file {required!r}"
            )
        members.append((path.name, path))
    initial_promotion = run_dir / "initial_geometry_promotion.json"
    if initial_promotion.is_file():
        members.append((initial_promotion.name, initial_promotion))
    if include_logs:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in LOG_SUFFIXES:
                members.append((path.relative_to(run_dir).as_posix(), path))
    return members


def _tar_member_info(name: str, size: int) -> tarfile.TarInfo:
    """TarInfo with normalized ownership and a zeroed mtime."""
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _build_tar_bytes(members: list[tuple[str, Path]], manifest_bytes: bytes) -> bytes:
    """Deterministic uncompressed tar with mtime=0 members (manifest last)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, path in members:
            info = _tar_member_info(name, path.stat().st_size)
            with path.open("rb") as handle:
                tar.addfile(info, handle)
        info = _tar_member_info(BUNDLE_MANIFEST_NAME, len(manifest_bytes))
        tar.addfile(info, io.BytesIO(manifest_bytes))
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Package a monitored-training run directory into a verifiable, "
            "deterministic resume bundle for issue #39."
        )
    )
    parser.add_argument("--input", required=True, help="Monitored training run directory.")
    parser.add_argument("--output", required=True, help="Output .tar.gz bundle path.")
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Also bundle *.log / *.out / *.err files found under --input.",
    )
    args = parser.parse_args()

    run_dir = Path(args.input).resolve()
    if not run_dir.is_dir():
        print(f"error: input run directory does not exist: {run_dir}", file=sys.stderr)
        return 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        members = _collect_members(run_dir, include_logs=args.include_logs)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    files = {name: _sha256_file(path) for name, path in members}
    bundle_manifest = {
        "schema_version": 1,
        # NOTE: no wall-clock timestamp is embedded on purpose — the manifest
        # is part of the archive, so a timestamp would break byte-for-byte
        # reproducibility (same inputs -> identical bundle). The export time
        # is printed to stdout instead.
        "source_run_dir": str(run_dir),
        "latest_checkpoint": members[0][0],
        "files": files,
        "bundle_sha256": None,
    }
    # The bundle's own SHA-256: self-hash of the canonical manifest blob.
    # (An archive cannot embed its own complete-file hash; this content-addressed
    # identity is recomputed by the importer to detect manifest tampering.)
    canonical = json.dumps({**bundle_manifest, "bundle_sha256": ""}, sort_keys=True, indent=2)
    bundle_manifest["bundle_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_bytes = json.dumps(bundle_manifest, sort_keys=True, indent=2).encode("utf-8") + b"\n"

    tar_bytes = _build_tar_bytes(members, manifest_bytes)
    # Zero the gzip header mtime so identical inputs produce identical archives.
    compressed = gzip.compress(tar_bytes, compresslevel=9, mtime=0)
    output.write_bytes(compressed)

    print(f"Bundle written: {output}")
    print(f"  members: {len(files) + 1} ({BUNDLE_MANIFEST_NAME} + {len(files)} run files)")
    print(f"  latest checkpoint: {bundle_manifest['latest_checkpoint']}")
    print(f"  bundle manifest sha256 (self-hash): {bundle_manifest['bundle_sha256']}")
    print(f"  archive file sha256: {_sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
