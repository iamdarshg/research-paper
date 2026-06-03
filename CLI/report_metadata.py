"""Helpers for attaching consistent lineage metadata to report artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def resolve_path(value: str | None) -> str | None:
    if not value:
        return None
    return str(Path(value).resolve())


def file_sha256(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).resolve()
    if not path.exists() or not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_report_metadata(
    report: Dict[str, Any],
    *,
    run_id: str | None = None,
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    protocol_path: str | None = None,
    checkpoint_hash: str | None = None,
    manifest_hash: str | None = None,
    protocol_hash: str | None = None,
) -> Dict[str, Any]:
    checkpoint_path = resolve_path(checkpoint_path)
    manifest_path = resolve_path(manifest_path)
    protocol_path = resolve_path(protocol_path)

    if run_id:
        report["run_id"] = str(run_id)
    if checkpoint_path:
        report["checkpoint_path"] = checkpoint_path
    if manifest_path:
        report["manifest_path"] = manifest_path
    if protocol_path:
        report["protocol_path"] = protocol_path

    checkpoint_hash = checkpoint_hash or file_sha256(checkpoint_path)
    manifest_hash = manifest_hash or file_sha256(manifest_path)
    protocol_hash = protocol_hash or file_sha256(protocol_path)

    if checkpoint_hash:
        report["checkpoint_hash"] = checkpoint_hash
    if manifest_hash:
        report["manifest_hash"] = manifest_hash
    if protocol_hash:
        report["protocol_hash"] = protocol_hash

    return report
