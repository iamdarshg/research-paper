#!/usr/bin/env python3
"""Upload corpus artifacts to GCS, then stop/delete this VM automatically."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import requests

PROJECT = os.environ.get("GCP_PROJECT", "project-ba289c9c-3c25-4e15-9dc")
ZONE = os.environ.get("GCP_ZONE", "asia-south1-c")
BUCKET = os.environ.get("GCP_BUCKET", "research-paper-corpus-20261030")
INSTANCE = os.environ.get("GCP_INSTANCE", "research-paper-corpus-builder")
PREFIX = os.environ.get("GCP_PREFIX", "runs/corpus-builder")
CORPUS_ROOT = Path(os.environ.get("CORPUS_ROOT", "/home/iamdarshg/corpus_build"))
LOG = Path("/var/log/corpus-build-setup.log")


def token() -> str:
    r = requests.get(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"}, timeout=20)
    r.raise_for_status()
    return r.json()["access_token"]


def upload_file(path: Path, name: str) -> None:
    subprocess.run(["gsutil", "-q", "cp", str(path), f"gs://{BUCKET}/{PREFIX}/{name}"], check=True)


def upload_tree(root: Path) -> None:
    if root.exists():
        subprocess.run(["gsutil", "-m", "-q", "cp", "-r", str(root), f"gs://{BUCKET}/{PREFIX}/"], check=True)


def stop_self() -> None:
    url = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{INSTANCE}/stop"
    r = requests.post(url, headers={"Authorization": f"Bearer {token()}"}, timeout=30)
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"stop failed: {r.status_code} {r.text[:500]}")


def delete_self() -> None:
    url = f"https://compute.googleapis.com/compute/v1/projects/{PROJECT}/zones/{ZONE}/instances/{INSTANCE}"
    r = requests.delete(url, headers={"Authorization": f"Bearer {token()}"}, timeout=30)
    if r.status_code not in (200, 202, 204):
        raise RuntimeError(f"delete failed: {r.status_code} {r.text[:500]}")


def main() -> int:
    marker = CORPUS_ROOT / "UPLOAD_STATUS.json"
    try:
        upload_tree(CORPUS_ROOT)
        if LOG.exists():
            upload_file(LOG, "corpus-build-setup.log")
        marker.write_text(json.dumps({"status": "uploaded", "bucket": BUCKET, "prefix": PREFIX}, indent=2))
        upload_file(marker, "UPLOAD_STATUS.json")
        delete_self()
        return 0
    except Exception as exc:
        marker.write_text(json.dumps({"status": "upload_failed", "error": repr(exc)}, indent=2))
        try:
            upload_file(marker, "UPLOAD_STATUS.json")
            if LOG.exists(): upload_file(LOG, "corpus-build-setup.log")
        finally:
            try: stop_self()
            except Exception: pass
        print(repr(exc), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())