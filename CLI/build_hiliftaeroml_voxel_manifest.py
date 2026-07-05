#!/usr/bin/env python3
"""Stream HiLiftAeroML STL surfaces into a local voxel manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import trimesh

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CLI.build_aircraft_flight_path_manifest import build_flight_path_manifest
from CLI.build_airshow_corpus import (
    _build_design_spec,
    _design_spec_provenance,
    _mesh_metrics,
    _split_for_id,
    voxelize_mesh,
)


HILIFT_COLLECTION = "hiliftaeroml_crm_hl_surface_runs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _load_jsonl_count(path: Optional[Path]) -> int:
    return len(_load_jsonl_records(path))


def load_exact_catalog_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a top-level records list")
    return [record for record in records if isinstance(record, dict)]


def select_hilift_surface_records(
    catalog_records: Sequence[Dict[str, Any]],
    *,
    existing_manifest_count: int,
    target_total_records: int,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    needed = max(0, int(target_total_records) - int(existing_manifest_count))
    if max_records is not None and max_records > 0:
        needed = min(needed, int(max_records))
    candidates = [
        record
        for record in catalog_records
        if record.get("source_collection") == HILIFT_COLLECTION
        and record.get("exact_cad_url")
        and record.get("file_format") == "stl"
    ]
    candidates.sort(
        key=lambda record: (
            str(record.get("geometry_variant_id", "")),
            int(record.get("angle_of_attack_deg", 0)),
            str(record.get("source_id", "")),
        )
    )
    if needed <= 0:
        return []
    return candidates[:needed]


def select_unique_hilift_variants(
    catalog_records: Sequence[Dict[str, Any]],
    *,
    existing_variant_ids: Optional[set[str]] = None,
    existing_content_hashes: Optional[set[str]] = None,
    target_unique_geometries: int = 600,
    max_records: Optional[int] = None,
) -> List[Dict[str, Any]]:
    existing_variants = {str(value) for value in (existing_variant_ids or set()) if value}
    seen_hashes = {str(value) for value in (existing_content_hashes or set()) if value}
    needed = max(0, int(target_unique_geometries) - len(existing_variants))
    if max_records is not None and max_records > 0:
        needed = min(needed, int(max_records))
    if needed <= 0:
        return []

    candidates = [
        record
        for record in catalog_records
        if record.get("source_collection") == HILIFT_COLLECTION
        and record.get("exact_cad_url")
        and record.get("file_format") == "stl"
        and record.get("geometry_variant_id")
    ]
    candidates.sort(
        key=lambda record: (
            str(record["geometry_variant_id"]),
            0 if int(record.get("angle_of_attack_deg", 0)) == 4 else 1,
            int(record.get("angle_of_attack_deg", 0)),
            str(record.get("source_id", "")),
        )
    )

    selected: List[Dict[str, Any]] = []
    selected_variants: set[str] = set()
    for record in candidates:
        variant_id = str(record["geometry_variant_id"])
        if variant_id in existing_variants or variant_id in selected_variants:
            continue
        content_hash = record.get("geometry_sha256")
        if content_hash and str(content_hash) in seen_hashes:
            continue
        selected.append(record)
        selected_variants.add(variant_id)
        if content_hash:
            seen_hashes.add(str(content_hash))
        if len(selected) >= needed:
            break
    return selected


def parse_force_moment_csv(text: str) -> Dict[str, float]:
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return {}
    output: Dict[str, float] = {}
    for key, value in rows[0].items():
        if value in {None, ""}:
            continue
        try:
            output[key] = float(value)
        except ValueError:
            continue
    return output


def artifact_stem_for_record(source_record: Dict[str, Any], *, cache_by_geometry_variant: bool) -> str:
    if cache_by_geometry_variant and source_record.get("geometry_variant_id"):
        return str(source_record["geometry_variant_id"])
    return str(source_record["source_id"])


def download_to_path(session: Any, url: str, path: Path, *, timeout: int) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def fetch_optional_text(session: Any, url: Optional[str], *, timeout: int) -> str:
    if not url:
        return ""
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def load_stl_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
            and len(geometry.vertices) > 0
            and len(geometry.faces) > 0
        ]
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise ValueError(f"STL did not contain a usable mesh: {path}")


def build_manifest_record(
    source_record: Dict[str, Any],
    *,
    manifest_path: Path,
    voxel_path: Path,
    voxel_sha256: str,
    geometry_sha256: str,
    metrics: Dict[str, Any],
    force_moment: Dict[str, float],
) -> Dict[str, Any]:
    variant_id = str(source_record.get("geometry_variant_id") or source_record["source_id"])
    design_spec = _build_design_spec(metrics, variant_id)
    rel_voxel = voxel_path.resolve().relative_to(manifest_path.resolve().parent)
    angle = int(source_record.get("angle_of_attack_deg", 0))
    return {
        "sample_id": str(source_record["source_id"]),
        "source_id": str(source_record["source_id"]),
        "source": "HiLiftAeroML exact STL surface run",
        "source_collection": source_record.get("source_collection"),
        "source_page": source_record.get("source_page"),
        "source_url": source_record.get("exact_cad_url"),
        "step_cad_url": source_record.get("step_cad_url"),
        "force_moment_url": source_record.get("force_moment_url"),
        "geometry_values_url": source_record.get("geometry_values_url"),
        "geometry_variant_id": variant_id,
        "angle_of_attack_deg": angle,
        "geometry_uniqueness": source_record.get("geometry_uniqueness"),
        "source_license": source_record.get("source_license"),
        "license_training_status": source_record.get("license_training_status"),
        "geometry_kind": source_record.get("geometry_kind"),
        "split": _split_for_id(variant_id),
        "design_family": "hiliftaeroml_crm_hl",
        "units": "source_stl_units_not_declared_normalized_to_voxel_lattice",
        "geometry_path": rel_voxel.as_posix(),
        "geometry_provenance": (
            "HiLiftAeroML exact STL surface run downloaded from the catalog exact_cad_url, "
            "voxelized into a centered normalized 96^3 occupancy lattice, and recorded without "
            "claiming that repeated AoA surface files are distinct aircraft geometries."
        ),
        "preprocessing_version": "hiliftaeroml-stl-stream-voxelizer-v1",
        "geometry_sha256": geometry_sha256,
        "voxel_sha256": voxel_sha256,
        "design_spec": design_spec,
        "design_spec_provenance": _design_spec_provenance(),
        "response_metrics": {
            "drag_coefficient": force_moment.get("cd"),
            "lift_coefficient": force_moment.get("cl"),
            "moment_coefficient": force_moment.get("cm"),
            "occupancy_ratio": metrics.get("occupancy_ratio"),
            "occupied_voxels": metrics.get("occupied_voxels"),
            "angle_of_attack_deg": angle,
        },
        "response_metrics_provenance": {
            "drag_coefficient": "direct_from_hiliftaeroml_force_mom_csv_cd",
            "lift_coefficient": "direct_from_hiliftaeroml_force_mom_csv_cl",
            "moment_coefficient": "direct_from_hiliftaeroml_force_mom_csv_cm",
            "occupancy_ratio": "direct_voxel_count_from_preprocessed_public_stl",
            "occupied_voxels": "direct_voxel_count_from_preprocessed_public_stl",
            "angle_of_attack_deg": "direct_from_hiliftaeroml_run_identifier",
        },
        "mesh_metrics": metrics,
        "claim_boundary": (
            "Exact public HiLiftAeroML surface geometry and force/moment labels. Repeated "
            "geometry_variant_id records across AoA must be treated as repeated geometry with "
            "different flow conditions, not independent aircraft designs."
        ),
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def build_hilift_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    import requests

    output_root = Path(args.output_root).resolve()
    raw_dir = output_root / "raw_stl"
    voxel_dir = output_root / "voxels"
    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)

    existing_records = _load_jsonl_records(
        Path(args.existing_manifest).resolve() if args.existing_manifest else None
    )
    existing_count = len(existing_records)
    existing_variant_ids = {
        str(record.get("geometry_variant_id") or record.get("source_id"))
        for record in existing_records
        if record.get("geometry_variant_id") or record.get("source_id")
    }
    existing_content_hashes = {
        str(record["geometry_sha256"])
        for record in existing_records
        if record.get("geometry_sha256")
    }
    catalog_records = load_exact_catalog_records(Path(args.catalog).resolve())
    selected = select_unique_hilift_variants(
        catalog_records=catalog_records,
        existing_variant_ids=existing_variant_ids,
        existing_content_hashes=existing_content_hashes,
        target_unique_geometries=args.target_unique_geometries,
        max_records=args.max_records,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "research-paper-hiliftaeroml-voxel-builder/1.0"})
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    artifact_cache: Dict[str, Dict[str, Any]] = {}

    for index, source_record in enumerate(selected, start=1):
        source_id = str(source_record["source_id"])
        artifact_stem = artifact_stem_for_record(
            source_record,
            cache_by_geometry_variant=not args.no_cache_by_geometry_variant,
        )
        raw_path = raw_dir / f"{artifact_stem}.stl"
        voxel_path = voxel_dir / f"{artifact_stem}.npy"
        try:
            if artifact_stem in artifact_cache and voxel_path.exists():
                print(f"[{index}/{len(selected)}] reuse {artifact_stem} for {source_id}")
                cached = artifact_cache[artifact_stem]
                geometry_sha256 = cached["geometry_sha256"]
                voxel_sha256 = cached["voxel_sha256"]
                metrics = cached["metrics"]
            else:
                print(f"[{index}/{len(selected)}] download {source_id}")
                download_to_path(session, str(source_record["exact_cad_url"]), raw_path, timeout=args.timeout_seconds)
                geometry_sha256 = sha256_file(raw_path)
                mesh = load_stl_mesh(raw_path)
                voxels = voxelize_mesh(mesh, args.grid_size)
                np.save(voxel_path, voxels.astype(np.float32))
                voxel_sha256 = sha256_file(voxel_path)
                metrics = _mesh_metrics(mesh, voxels)
                artifact_cache[artifact_stem] = {
                    "geometry_sha256": geometry_sha256,
                    "voxel_sha256": voxel_sha256,
                    "metrics": metrics,
                }
                if args.delete_raw_stl:
                    raw_path.unlink(missing_ok=True)
            force_text = fetch_optional_text(
                session,
                source_record.get("force_moment_url"),
                timeout=args.timeout_seconds,
            )
            force_moment = parse_force_moment_csv(force_text) if force_text else {}
            records.append(
                build_manifest_record(
                    source_record,
                    manifest_path=manifest_path,
                    voxel_path=voxel_path,
                    voxel_sha256=voxel_sha256,
                    geometry_sha256=geometry_sha256,
                    metrics=metrics,
                    force_moment=force_moment,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep going and report source failures.
            failures.append(
                {
                    "source_id": source_id,
                    "url": str(source_record.get("exact_cad_url")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(selected)}] failed {source_id}: {exc}", file=sys.stderr)

    _write_jsonl(manifest_path, records)
    merged_manifest = None
    merge_report = None
    if args.existing_manifest and args.combined_manifest:
        merge_report = build_flight_path_manifest(
            [Path(args.existing_manifest), manifest_path],
            output_manifest=Path(args.combined_manifest),
            report_path=Path(args.combined_report) if args.combined_report else None,
            run_id=args.run_id,
        )
        merged_manifest = str(Path(args.combined_manifest).resolve())

    produced_variant_ids = {
        str(record["geometry_variant_id"])
        for record in records
        if record.get("geometry_variant_id")
    }
    combined_unique_geometry_count = len(existing_variant_ids | produced_variant_ids)
    report = {
        "run_id": args.run_id,
        "catalog": str(Path(args.catalog).resolve()),
        "existing_manifest": str(Path(args.existing_manifest).resolve()) if args.existing_manifest else None,
        "existing_manifest_count": existing_count,
        "target_total_records": args.target_total_records,
        "target_unique_geometries": args.target_unique_geometries,
        "existing_unique_geometry_count": len(existing_variant_ids),
        "combined_unique_geometry_count": combined_unique_geometry_count,
        "unique_geometry_target_met": combined_unique_geometry_count >= args.target_unique_geometries,
        "selected_count": len(selected),
        "record_count": len(records),
        "failure_count": len(failures),
        "grid_size": args.grid_size,
        "manifest": str(manifest_path),
        "combined_manifest": merged_manifest,
        "delete_raw_stl": bool(args.delete_raw_stl),
        "cache_by_geometry_variant": not bool(args.no_cache_by_geometry_variant),
        "unique_voxel_artifacts": len({record.get("geometry_path") for record in records}),
        "failures": failures,
        "merge_report": merge_report,
        "claim_boundary": (
            "HiLiftAeroML surface records are exact public STL surfaces, but repeated AoA files "
            "for the same geometry_variant_id are repeated geometry with different flow labels."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="docs/dataset/exact_cad_source_catalog_20260624.json")
    parser.add_argument("--existing-manifest", default="build/expanded_aircraft_corpus_20260622/manifest.jsonl")
    parser.add_argument("--output-root", default="build/hiliftaeroml_g96_stream_20260624")
    parser.add_argument("--manifest", default="build/hiliftaeroml_g96_stream_20260624/manifest.jsonl")
    parser.add_argument("--report", default="build/hiliftaeroml_g96_stream_20260624/report.json")
    parser.add_argument("--combined-manifest", default="build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl")
    parser.add_argument("--combined-report", default="build/expanded_aircraft_hilift_corpus_20260624/flight_path_manifest_report.json")
    parser.add_argument("--run-id", default="hiliftaeroml-g96-stream-20260624")
    parser.add_argument("--target-total-records", type=_positive_int, default=752)
    parser.add_argument("--target-unique-geometries", type=_positive_int, default=600)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--grid-size", type=_positive_int, default=96)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=300)
    parser.add_argument("--delete-raw-stl", action="store_true")
    parser.add_argument("--no-cache-by-geometry-variant", action="store_true")
    args = parser.parse_args(argv)
    report = build_hilift_manifest(args)
    return 0 if report["record_count"] > 0 and report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
