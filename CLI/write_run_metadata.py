#!/usr/bin/env python3
"""Write shared run metadata for evidence reports.

This helper is run after training in the first-training protocol so final
evidence consistency hashes the real `.pt` checkpoint instead of the deterministic
reference checkpoint card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_run_metadata(
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
    run_id_prefix: str = "first-training",
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    output_path = Path(output_path)

    for label, path in (
        ("checkpoint", checkpoint_path),
        ("manifest", manifest_path),
        ("protocol", protocol_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    checkpoint_hash = _sha256_file(checkpoint_path)
    manifest_hash = _sha256_file(manifest_path)
    protocol_hash = _sha256_file(protocol_path)
    report: Dict[str, Any] = {
        "run_id": f"{run_id_prefix}-{checkpoint_hash.split(':', 1)[1][:12]}",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest_hash,
        "protocol_path": str(protocol_path),
        "protocol_hash": protocol_hash,
        "claim_boundary": (
            "This metadata links reports to a first real training checkpoint. "
            "It does not establish publication-scale model validity."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Write shared run metadata for evidence reports.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file to hash.")
    parser.add_argument("--manifest", required=True, help="Grounded manifest file to hash.")
    parser.add_argument("--protocol", required=True, help="Protocol YAML file to hash.")
    parser.add_argument("--output", required=True, help="Output run metadata JSON path.")
    parser.add_argument("--run-id-prefix", default="first-training", help="Prefix for generated run_id.")
    args = parser.parse_args()

    report = build_run_metadata(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        protocol_path=args.protocol,
        output_path=args.output,
        run_id_prefix=args.run_id_prefix,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
