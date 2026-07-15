#!/usr/bin/env python3
"""Render exact public OpenVSP sources into a canonical voxel corpus.

Only source `.vsp3` files from the exact-CAD catalog are admitted. Preview
meshes are never used: OpenVSP reads each source project and exports the STL
that is then canonicalized and screened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import requests

from aircraft_validity import canonicalize_aircraft_voxels, evaluate_aircraft_validity
from build_aircraftverse_corpus import (
    CONDITIONING_FIELDS,
    CorpusBuildError,
    _load_stl_mesh,
    _sha256_file,
    voxelize_mesh,
)
from validate_manifest import DEFAULT_UNIQUE_GEOMETRY_TARGET, validate_manifest_file


PREPROCESSING_VERSION = "openvsp-source-render-canonical-voxelizer-v1"
AIRSHOW_COLLECTION = "vsp_airshow_public_models"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _selection_key(seed: int, source_id: str) -> str:
    return _sha256_bytes(f"{int(seed)}:{source_id}".encode("utf-8"))


def select_exact_vsp_records(
    catalog_path: Path,
    *,
    target_count: int,
    seed: int,
    selection_offset: int = 0,
) -> List[Dict[str, Any]]:
    payload = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    catalog_records = payload.get("records", [])
    candidates = [
        dict(record)
        for record in catalog_records
        if isinstance(record, Mapping)
        and record.get("source_collection") == AIRSHOW_COLLECTION
        and record.get("file_format") == "vsp3"
        and record.get("exact_cad_url")
        and record.get("source_id")
    ]
    candidates.sort(key=lambda record: _selection_key(seed, str(record["source_id"])))
    offset = max(0, int(selection_offset))
    return candidates[offset:offset + max(0, int(target_count))]


def _split_for_id(source_id: str) -> str:
    bucket = int(_sha256_bytes(source_id.encode("utf-8"))[:8], 16) % 10
    return "train" if bucket < 7 else "val" if bucket < 8 else "test" if bucket < 9 else "holdout"


def _atomic_save_npy(path: Path, voxels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.save(temporary_path, voxels.astype(np.uint8, copy=False))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".jsonl", mode="w", encoding="utf-8", delete=False) as handle:
        temporary_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True, ensure_ascii=True) + "\n")
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _download_file(session: requests.Session, url: str, destination: Path, timeout_seconds: int) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def render_vsp_to_stl(
    *,
    vsp_executable: Path,
    script_path: Path,
    source_vsp: Path,
    scratch_dir: Path,
    timeout_seconds: int,
) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    input_path = scratch_dir / "input.vsp3"
    output_path = scratch_dir / "output.stl"
    shutil.copyfile(source_vsp, input_path)
    output_path.unlink(missing_ok=True)
    completed = subprocess.run(
        [str(vsp_executable), "-script", str(script_path.resolve())],
        cwd=scratch_dir,
        capture_output=True,
        text=True,
        timeout=max(1, int(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-2000:]
        raise CorpusBuildError("openvsp_export_failed", detail or "OpenVSP did not produce STL output.")
    return output_path


def _null_conditioning_fields() -> Dict[str, Any]:
    return {field_name: None for field_name in CONDITIONING_FIELDS}


def _record_from_render(
    source: Mapping[str, Any],
    *,
    output_dir: Path,
    vsp_path: Path,
    stl_path: Path,
    grid_size: int,
) -> Dict[str, Any]:
    mesh = _load_stl_mesh(stl_path.read_bytes())
    raw_voxels = voxelize_mesh(mesh, grid_size)
    canonical_voxels, canonicalization = canonicalize_aircraft_voxels(raw_voxels)
    canonical_np = canonical_voxels.numpy().astype(np.uint8)
    validity = evaluate_aircraft_validity(canonical_np)
    if validity["status"] != "pass":
        raise CorpusBuildError(
            "aircraft_validity_failed",
            ", ".join(str(value) for value in validity["failed_checks"]),
        )

    source_id = str(source["source_id"])
    voxel_path = output_dir / "voxels" / f"{source_id}.npy"
    _atomic_save_npy(voxel_path, canonical_np)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return {
        "sample_id": source_id,
        "source_id": source_id,
        "source_collection": AIRSHOW_COLLECTION,
        "source_page": source.get("source_page"),
        "source_url": source.get("exact_cad_url"),
        "source_license": source.get("source_license"),
        "source_license_id": source.get("source_license_id"),
        "name": source.get("name"),
        "display_name": source.get("display_name"),
        "manufacturer": source.get("manufacturer"),
        "split": _split_for_id(source_id),
        "design_family": "public_openvsp_whole_aircraft_model",
        "geometry_path": f"voxels/{voxel_path.name}",
        "geometry_provenance": "Exact public OpenVSP .vsp3 source rendered to STL by OpenVSP, then normalized, canonicalized, and voxelized.",
        "preprocessing_version": PREPROCESSING_VERSION,
        "units": "OpenVSP source units; normalized voxel lattice",
        "original_units": "OpenVSP project units retained in source VSP3",
        "geometry_sha256": _sha256_file(vsp_path),
        "rendered_stl_sha256": _sha256_file(stl_path),
        "voxel_sha256": _sha256_file(voxel_path),
        "source_native_vsp3_path": str(vsp_path.resolve()),
        "source_native_metadata": dict(source),
        "design_spec": _null_conditioning_fields(),
        "design_spec_availability": {field_name: False for field_name in CONDITIONING_FIELDS},
        "design_spec_provenance": {field_name: "unavailable_in_public_openvsp_catalog_metadata" for field_name in CONDITIONING_FIELDS},
        "conditioning_mode": "unconditioned_source_metadata_only",
        "mesh_metrics": {
            "source_vertices": int(len(mesh.vertices)),
            "source_faces": int(len(mesh.faces)),
            "source_extents": extents.tolist(),
            "source_is_watertight": bool(mesh.is_watertight),
            "occupancy_ratio": float(canonical_np.mean()),
            "occupied_voxels": int(canonical_np.sum()),
        },
        "canonicalization": canonicalization,
        "aircraft_validity": validity,
        "date_built": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Source VSP3 provenance and rendered geometry establish a reproducible public CAD corpus entry, not certification or flight-test evidence.",
    }


def build_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    raw_dir = output_dir / "raw_vsp3"
    render_dir = output_dir / "rendered_stl"
    scratch_dir = output_dir / "scratch"
    manifest_path = output_dir / "manifest.jsonl"
    report_path = output_dir / "report.json"
    rejection_path = output_dir / "rejections.jsonl"
    vsp_executable = Path(args.vsp_executable).resolve()
    script_path = Path(args.vsp_script).resolve()
    if not vsp_executable.exists() or not script_path.exists():
        raise FileNotFoundError("OpenVSP executable or export script is missing.")

    selected = select_exact_vsp_records(
        Path(args.catalog),
        target_count=args.target_count,
        seed=args.selection_seed,
        selection_offset=args.selection_offset,
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "research-paper-exact-openvsp-corpus/1.0"})
    records: List[Dict[str, Any]] = []
    rejections: List[Dict[str, str]] = []
    seen_vsp_hashes: set[str] = set()
    seen_voxel_hashes: set[str] = set()
    rejection_counts: Counter[str] = Counter()

    for index, source in enumerate(selected, start=1):
        source_id = str(source["source_id"])
        vsp_path = raw_dir / f"{source_id}.vsp3"
        stl_path = render_dir / f"{source_id}.stl"
        try:
            print(f"[{index}/{len(selected)}] render {source_id}", flush=True)
            _download_file(session, str(source["exact_cad_url"]), vsp_path, args.timeout_seconds)
            rendered_path = render_vsp_to_stl(
                vsp_executable=vsp_executable,
                script_path=script_path,
                source_vsp=vsp_path,
                scratch_dir=scratch_dir,
                timeout_seconds=args.timeout_seconds,
            )
            stl_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rendered_path, stl_path)
            record = _record_from_render(
                source,
                output_dir=output_dir,
                vsp_path=vsp_path,
                stl_path=stl_path,
                grid_size=args.grid_size,
            )
            if record["geometry_sha256"] in seen_vsp_hashes:
                raise CorpusBuildError("duplicate_vsp3", "Duplicate exact VSP3 content.")
            if record["voxel_sha256"] in seen_voxel_hashes:
                raise CorpusBuildError("duplicate_canonical_voxel", "Duplicate canonical voxel content.")
            records.append(record)
            seen_vsp_hashes.add(record["geometry_sha256"])
            seen_voxel_hashes.add(record["voxel_sha256"])
        except (CorpusBuildError, requests.RequestException, subprocess.SubprocessError, OSError) as exc:
            code = exc.code if isinstance(exc, CorpusBuildError) else type(exc).__name__
            rejection_counts[code] += 1
            rejections.append({"source_id": source_id, "code": str(code), "message": str(exc)})
            print(f"[{index}/{len(selected)}] reject {source_id}: {code}: {exc}", flush=True)

    records.sort(key=lambda record: str(record["source_id"]))
    _atomic_write_jsonl(manifest_path, records)
    _atomic_write_jsonl(rejection_path, rejections)
    basic_validation = validate_manifest_file(str(manifest_path), level="basic")
    claim_validation = validate_manifest_file(
        str(manifest_path), level="claim-bearing", unique_geometry_target=DEFAULT_UNIQUE_GEOMETRY_TARGET
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(selected),
        "accepted_count": len(records),
        "unique_vsp3_count": len(seen_vsp_hashes),
        "unique_canonical_voxel_count": len(seen_voxel_hashes),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "manifest_path": str(manifest_path),
        "rejection_path": str(rejection_path),
        "grid_size": int(args.grid_size),
        "openvsp_executable": str(vsp_executable),
        "openvsp_export_script": str(script_path),
        "basic_validation": basic_validation,
        "claim_validation": claim_validation,
        "claim_boundary": "Exact VSP3 sources are rendered through OpenVSP; passing corpus gates does not establish aircraft certification or experimental flight validation.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/dataset/exact_cad_source_catalog_20260624.json")
    parser.add_argument("--output-dir", default="build/openvsp_airshow_exact_20260713")
    parser.add_argument("--vsp-executable", required=True)
    parser.add_argument("--vsp-script", default="CLI/openvsp_export_stl.vspscript")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--target-count", type=int, default=359)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--selection-offset", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    report = build_corpus(args)
    print(json.dumps({key: report[key] for key in ("selected_count", "accepted_count", "rejection_counts")}, indent=2, sort_keys=True))
    return 0 if report["accepted_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
