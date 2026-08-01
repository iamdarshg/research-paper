#!/usr/bin/env python3
"""Import and verify an issue #39 resume bundle.

Verifies, BEFORE extracting anything:

  * ``bundle_manifest.json`` exists inside the archive,
  * every per-file SHA-256 in the manifest matches the archived bytes
    (a single corrupted byte fails the import with a clear error),
  * the manifest's own ``bundle_sha256`` self-hash recomputes.

Then extracts all members into ``--output`` and prints the exact resume
command for the next free-cloud training session::

    python CLI/run_monitored_training.py \\
        --manifest <resume_manifest.json manifest, else docs/dataset/minimal_grounded_manifest.jsonl> \\
        --resume-from <extracted latest checkpoint> \\
        --max-optimizer-updates 5 --checkpoint-every-updates 1 \\
        --fixed-validation-seeds <resume_manifest.json seeds, else 0,1,2,3,4,5> \\
        --save-dir <output dir>

Exit codes: 0 success; 1 missing/invalid inputs; 2 checksum verification
failure (nothing is extracted on failure).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
DEFAULT_MANIFEST = "docs/dataset/minimal_grounded_manifest.jsonl"
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5]


def _verify_and_extract(bundle: Path, out_dir: Path) -> dict:
    """Verify every member hash + manifest self-hash, then extract."""
    with tarfile.open(bundle, "r:gz") as tar:
        names = set(tar.getnames())
        if BUNDLE_MANIFEST_NAME not in names:
            raise ValueError(
                f"bundle {bundle} has no {BUNDLE_MANIFEST_NAME}; not an issue-#39 bundle"
            )
        manifest = json.loads(
            tar.extractfile(BUNDLE_MANIFEST_NAME).read().decode("utf-8")
        )
        expected_self_hash = manifest.get("bundle_sha256")
        canonical = json.dumps(
            {**manifest, "bundle_sha256": ""}, sort_keys=True, indent=2
        )
        recomputed_self_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if expected_self_hash and expected_self_hash != recomputed_self_hash:
            raise ValueError(
                "bundle manifest self-hash mismatch: manifest was modified "
                f"(expected {expected_self_hash}, recomputed {recomputed_self_hash})"
            )

        files = manifest.get("files") or {}
        if not files:
            raise ValueError("bundle manifest lists no files")

        # Verify every per-file SHA-256 against the archived bytes first.
        for name, expected_digest in files.items():
            if name not in names:
                raise ValueError(
                    f"checksum verification failed: {name!r} is listed in the "
                    "bundle manifest but missing from the archive"
                )
            member = tar.getmember(name)
            digest = hashlib.sha256()
            stream = tar.extractfile(member)
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected_digest:
                raise ValueError(
                    f"checksum verification failed for {name!r}: "
                    f"expected sha256 {expected_digest}, got {actual} "
                    "(bundle is corrupt or was tampered with)"
                )

        # All good: extract (filter="data" rejects absolute paths / traversal).
        out_dir.mkdir(parents=True, exist_ok=True)
        tar.extractall(out_dir, filter="data")
    return manifest


def _resume_command(out_dir: Path, manifest: dict) -> str:
    """Build the exact next-session resume command."""
    resume_manifest_path = out_dir / "resume_manifest.json"
    seeds = DEFAULT_SEEDS
    source_manifest = DEFAULT_MANIFEST
    latest_checkpoint_name = manifest.get("latest_checkpoint")
    if resume_manifest_path.is_file():
        try:
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            resume_manifest = {}
        stored_seeds = resume_manifest.get("fixed_validation_seeds")
        if stored_seeds:
            seeds = [int(seed) for seed in stored_seeds]
        # Forward-compatible manifest path field; the current trainer writes
        # only corpus_manifest_hash, so this normally stays at the default.
        source_manifest = (
            resume_manifest.get("manifest")
            or resume_manifest.get("manifest_path")
            or DEFAULT_MANIFEST
        )
        latest_checkpoint_name = resume_manifest.get(
            "latest_checkpoint", latest_checkpoint_name
        )
    if latest_checkpoint_name:
        checkpoint_path = out_dir / Path(latest_checkpoint_name).name
    else:
        cadence = sorted(out_dir.glob("checkpoint_updates_*.pt"))
        checkpoint_path = cadence[-1] if cadence else out_dir / "best_geometry_model.pt"
    seeds_arg = ",".join(str(seed) for seed in seeds)
    return (
        "python CLI/run_monitored_training.py "
        f"--manifest {source_manifest} "
        f"--resume-from {checkpoint_path} "
        "--max-optimizer-updates 5 "
        "--checkpoint-every-updates 1 "
        f"--fixed-validation-seeds {seeds_arg} "
        f"--save-dir {out_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and extract an issue #39 resume bundle."
    )
    parser.add_argument("--input", required=True, help="Bundle .tar.gz to import.")
    parser.add_argument("--output", required=True, help="Directory to extract into.")
    parser.add_argument(
        "--print-resume-command",
        action="store_true",
        help="Print the exact resume command after extraction (also printed by default).",
    )
    args = parser.parse_args()

    bundle = Path(args.input).resolve()
    if not bundle.is_file():
        print(f"error: bundle file does not exist: {bundle}", file=sys.stderr)
        return 1
    out_dir = Path(args.output).resolve()

    try:
        manifest = _verify_and_extract(bundle, out_dir)
    except (tarfile.TarError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2 if "checksum" in str(exc) or "self-hash" in str(exc) else 1

    verified = len(manifest.get("files") or {})
    print(f"Verified {verified} file(s) against bundle manifest sha256 digests: OK")
    print(f"Extracted bundle to {out_dir}")
    command = _resume_command(out_dir, manifest)
    print("Resume command:")
    print(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
