#!/usr/bin/env python3
"""Build a provenance-preserving voxel corpus from public VSP Airshow models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import trimesh


AIRSHOW_URL = "https://airshow.openvsp.org/"
FIRESTORE_TEMPLATE = (
    "https://firestore.googleapis.com/v1/projects/{project_id}/"
    "databases/(default)/documents/models"
)

LICENSE_NAMES = {
    "1": "No Rights Reserved (CC0)",
    "2": "Attribution (CC BY)",
    "3": "Attribution Share Alike (CC BY-SA)",
    "4": "Attribution Non-Commercial (CC BY-NC)",
    "5": "Attribution Non-Commercial No Derivatives (CC BY-NC-ND)",
}

UNIT_NAMES = {
    "1": "millimeters",
    "2": "meters",
    "3": "feet",
    "4": "inches",
    "5": "centimeters",
    "6": "dimensionless",
}

MANUFACTURING_METHODS = (
    "foam_core_hotwire",
    "fdm_pla_0p4mm",
    "fdm_pla_0p6mm",
    "sheet_balsa_tabbed",
    "composite_wet_layup",
)


class CorpusBuildError(RuntimeError):
    """Raised when a public model cannot be converted into a training record."""


def _firestore_value(payload: Dict[str, Any]) -> Any:
    if "stringValue" in payload:
        return payload["stringValue"]
    if "integerValue" in payload:
        return int(payload["integerValue"])
    if "doubleValue" in payload:
        return float(payload["doubleValue"])
    if "booleanValue" in payload:
        return bool(payload["booleanValue"])
    if "timestampValue" in payload:
        return payload["timestampValue"]
    if "arrayValue" in payload:
        return [_firestore_value(item) for item in payload["arrayValue"].get("values", [])]
    if "mapValue" in payload:
        return {
            key: _firestore_value(value)
            for key, value in payload["mapValue"].get("fields", {}).items()
        }
    return None


def _extract_airshow_config(html: str) -> Dict[str, str]:
    config: Dict[str, str] = {}
    for key in ("apiKey", "projectId", "storageBucket"):
        match = re.search(rf"{key}:\"([^\"]+)\"", html)
        if not match:
            raise CorpusBuildError(f"Could not find Airshow Firebase {key} in public page.")
        config[key] = match.group(1)
    return config


def fetch_airshow_config(session: requests.Session, source_url: str) -> Dict[str, str]:
    response = session.get(source_url, timeout=30)
    response.raise_for_status()
    return _extract_airshow_config(response.text)


def fetch_model_documents(
    session: requests.Session,
    *,
    api_key: str,
    project_id: str,
    page_size: int,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    endpoint = FIRESTORE_TEMPLATE.format(project_id=project_id)
    page_token: Optional[str] = None
    while True:
        params = {"key": api_key, "pageSize": str(page_size)}
        if page_token:
            params["pageToken"] = page_token
        response = session.get(endpoint, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        for document in payload.get("documents", []):
            item = {"id": document["name"].split("/")[-1], "document_name": document["name"]}
            item.update(
                {
                    key: _firestore_value(value)
                    for key, value in document.get("fields", {}).items()
                }
            )
            documents.append(item)
        page_token = payload.get("nextPageToken")
        if not page_token:
            return documents


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _parse_index_list(raw: str) -> Iterable[List[int]]:
    current: List[int] = []
    for token in raw.replace(",", " ").split():
        value = int(token)
        if value == -1:
            if current:
                yield current
            current = []
        else:
            current.append(value)
    if current:
        yield current


def parse_x3d_indexed_faces(text: str) -> trimesh.Trimesh:
    """Parse X3D IndexedFaceSet geometry into one trimesh mesh."""
    root = ET.fromstring(text)
    vertices: List[List[float]] = []
    faces: List[List[int]] = []

    for indexed_face_set in root.iter():
        if _local_name(indexed_face_set.tag) != "IndexedFaceSet":
            continue
        coordinate = None
        for child in indexed_face_set.iter():
            if _local_name(child.tag) == "Coordinate":
                coordinate = child
                break
        if coordinate is None:
            continue

        point_payload = coordinate.attrib.get("point", "")
        coord_array = np.fromstring(point_payload, sep=" ", dtype=np.float64)
        if coord_array.size == 0 or coord_array.size % 3 != 0:
            continue
        coords = coord_array.reshape((-1, 3))
        offset = len(vertices)
        vertices.extend(coords.tolist())

        for polygon in _parse_index_list(indexed_face_set.attrib.get("coordIndex", "")):
            if len(polygon) < 3:
                continue
            for index in range(1, len(polygon) - 1):
                faces.append(
                    [
                        offset + polygon[0],
                        offset + polygon[index],
                        offset + polygon[index + 1],
                    ]
                )

    if not vertices or not faces:
        raise CorpusBuildError("X3D file did not contain usable IndexedFaceSet geometry.")
    return trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _scene_to_mesh(scene_or_mesh: Any) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = [
            geometry
            for geometry in scene_or_mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
            and len(geometry.vertices) > 0
            and len(geometry.faces) > 0
        ]
        if geometries:
            return trimesh.util.concatenate(geometries)
    raise CorpusBuildError("Downloaded mesh was empty or unsupported.")


def load_public_geometry(path: Path) -> trimesh.Trimesh:
    text = path.read_text(encoding="utf-8", errors="ignore").lstrip()
    if text.startswith("<"):
        return parse_x3d_indexed_faces(text)
    return _scene_to_mesh(trimesh.load(path, force="scene"))


def _centered_slices(raw_shape: Sequence[int], grid_size: int) -> Tuple[Tuple[slice, ...], Tuple[slice, ...]]:
    source_slices = []
    dest_slices = []
    for raw_dim in raw_shape:
        if raw_dim <= grid_size:
            source_start = 0
            dest_start = (grid_size - raw_dim) // 2
            length = raw_dim
        else:
            source_start = (raw_dim - grid_size) // 2
            dest_start = 0
            length = grid_size
        source_slices.append(slice(source_start, source_start + length))
        dest_slices.append(slice(dest_start, dest_start + length))
    return tuple(source_slices), tuple(dest_slices)


def voxelize_mesh(mesh: trimesh.Trimesh, grid_size: int) -> np.ndarray:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise CorpusBuildError("Mesh has no vertices or faces.")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if not np.isfinite(extents).all() or float(extents.max()) <= 0.0:
        raise CorpusBuildError("Mesh extents are invalid.")

    working = mesh.copy()
    center = np.asarray(working.bounds, dtype=np.float64).mean(axis=0)
    working.apply_translation(-center)
    working.apply_scale(0.8 / float(extents.max()))
    voxel_matrix = working.voxelized(pitch=1.0 / float(grid_size)).matrix.astype(np.float32)
    if voxel_matrix.sum() <= 0:
        raise CorpusBuildError("Voxelizer produced an empty grid.")

    final = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    source_slices, dest_slices = _centered_slices(voxel_matrix.shape, grid_size)
    final[dest_slices] = voxel_matrix[source_slices]
    if final.sum() <= 0:
        raise CorpusBuildError("Centered voxel crop is empty.")
    return final


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _safe_console_text(value: Any, *, stream: Any = None) -> str:
    text = str(value)
    encoding = getattr(stream or sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _split_for_id(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "val"
    if bucket < 9:
        return "test"
    return "holdout"


def _safe_float(value: float, lower: float, upper: float) -> float:
    return float(min(upper, max(lower, value)))


def _build_design_spec(metrics: Dict[str, Any], source_id: str) -> Dict[str, Any]:
    occupancy = float(metrics["occupancy_ratio"])
    extents = np.asarray(metrics["source_extents"], dtype=np.float64)
    ordered = np.sort(extents)
    span_ratio = float(ordered[-2] / ordered[-1]) if ordered[-1] > 0 else 0.5
    thickness_ratio = float(ordered[0] / ordered[-1]) if ordered[-1] > 0 else 0.15
    slenderness = float(ordered[-1] / max(ordered[-2], 1e-6))
    method = MANUFACTURING_METHODS[int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:2], 16) % len(MANUFACTURING_METHODS)]

    target_speed = _safe_float(18.0 + 22.0 * slenderness + 120.0 * thickness_ratio, 12.0, 90.0)
    wingspan_limit = _safe_float(1.0 + 5.0 * span_ratio, 1.0, 8.0)
    payload_min = int(_safe_float(250.0 + 8000.0 * occupancy, 150.0, 6000.0))
    payload_max = int(_safe_float(payload_min * 2.2, payload_min + 250.0, 15000.0))
    thrust = _safe_float(80.0 + 900.0 * occupancy + 2.0 * target_speed, 80.0, 2000.0)
    turn_rate = _safe_float(8.0 + 25.0 * span_ratio, 6.0, 38.0)

    return {
        "target_speed_mps": round(target_speed, 3),
        "wingspan_limit_m": round(wingspan_limit, 3),
        "thrust_to_weight_min": round(_safe_float(0.28 + 0.9 * thickness_ratio, 0.25, 0.95), 3),
        "turn_rate_min_deg_s": round(turn_rate, 3),
        "required_static_thrust_n": round(thrust, 3),
        "engine_diameter_mm": int(_safe_float(90 + 500 * thickness_ratio, 80, 900)),
        "engine_length_mm": int(_safe_float(180 + 1000 * thickness_ratio, 150, 1600)),
        "engine_count_min": 1,
        "engine_count_max": 2,
        "payload_mass_min_g": payload_min,
        "payload_mass_max_g": payload_max,
        "takeoff_distance_min_m": int(_safe_float(55 + 250 * (1.0 - occupancy), 40, 260)),
        "takeoff_distance_max_m": int(_safe_float(110 + 500 * (1.0 - occupancy), 80, 620)),
        "wall_thickness_min_mm": 1,
        "wall_thickness_max_mm": 3 if method in {"composite_wet_layup", "foam_core_hotwire"} else 2,
        "part_count_min": 1,
        "part_count_max": 10 if method == "composite_wet_layup" else 8,
        "manufacturing_method": method,
    }


def _design_spec_provenance() -> Dict[str, str]:
    return {
        "target_speed_mps": "inferred_from_voxelized_airshow_geometry_ratio_for_conditioning_only",
        "wingspan_limit_m": "inferred_from_normalized_mesh_span_ratio_for_conditioning_only",
        "thrust_to_weight_min": "inferred_from_normalized_mesh_thickness_ratio_for_conditioning_only",
        "turn_rate_min_deg_s": "inferred_from_normalized_mesh_span_ratio_for_conditioning_only",
        "required_static_thrust_n": "inferred_from_voxel_occupancy_and_target_speed_for_conditioning_only",
        "engine_diameter_mm": "inferred_from_normalized_mesh_thickness_ratio_for_conditioning_only",
        "engine_length_mm": "inferred_from_normalized_mesh_thickness_ratio_for_conditioning_only",
        "engine_count_min": "repo_supported_conditioning_default_not_source_metadata",
        "engine_count_max": "repo_supported_conditioning_default_not_source_metadata",
        "payload_mass_min_g": "inferred_from_voxel_occupancy_for_conditioning_only",
        "payload_mass_max_g": "inferred_from_voxel_occupancy_for_conditioning_only",
        "takeoff_distance_min_m": "inferred_from_voxel_occupancy_for_conditioning_only",
        "takeoff_distance_max_m": "inferred_from_voxel_occupancy_for_conditioning_only",
        "wall_thickness_min_mm": "repo_supported_manufacturing_default_not_source_metadata",
        "wall_thickness_max_mm": "repo_supported_manufacturing_default_not_source_metadata",
        "part_count_min": "repo_supported_manufacturing_default_not_source_metadata",
        "part_count_max": "repo_supported_manufacturing_default_not_source_metadata",
        "manufacturing_method": "deterministic_repo_supported_category_for_conditioning_only",
    }


def _mesh_metrics(mesh: trimesh.Trimesh, voxels: np.ndarray) -> Dict[str, Any]:
    extents = np.asarray(mesh.extents, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    occupied = int(voxels.sum())
    return {
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "source_is_watertight": bool(mesh.is_watertight),
        "source_bounds": bounds.tolist(),
        "source_extents": extents.tolist(),
        "occupied_voxels": occupied,
        "occupancy_ratio": float(occupied / voxels.size),
    }


def _download(session: requests.Session, url: str, path: Path) -> bytes:
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    response = session.get(url, timeout=90)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def _candidate_records(
    models: List[Dict[str, Any]],
    allowed_licenses: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = set(allowed_licenses)
    candidates = []
    for model in models:
        license_id = str(model.get("license", ""))
        geometry_url = str(model.get("newX3dUrl") or model.get("x3dUrl") or "")
        if license_id not in allowed or not geometry_url:
            continue
        candidates.append(model)
    candidates.sort(key=lambda item: str(item.get("id", "")))
    return candidates


def build_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_console_output()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_geometry"
    voxel_dir = output_dir / "voxels"
    manifest_path = output_dir / "manifest.jsonl"
    provenance_path = output_dir / "provenance.json"
    report_path = output_dir / "corpus_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "research-paper-airshow-corpus-builder/1.0"})
    config = fetch_airshow_config(session, args.source_url)
    models = fetch_model_documents(
        session,
        api_key=config["apiKey"],
        project_id=config["projectId"],
        page_size=args.page_size,
    )
    candidates = _candidate_records(models, args.allowed_licenses)
    if args.max_records:
        candidates = candidates[: args.max_records]

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    license_counts: Counter[str] = Counter()

    started_at = datetime.now(timezone.utc).isoformat()
    for index, model in enumerate(candidates, start=1):
        source_id = str(model["id"])
        geometry_url = str(model.get("newX3dUrl") or model.get("x3dUrl"))
        raw_path = raw_dir / f"{source_id}.x3d"
        voxel_path = voxel_dir / f"{source_id}.npy"
        try:
            raw_bytes = _download(session, geometry_url, raw_path)
            mesh = load_public_geometry(raw_path)
            voxels = voxelize_mesh(mesh, args.grid_size)
            np.save(voxel_path, voxels.astype(np.float32))
            metrics = _mesh_metrics(mesh, voxels)
            design_spec = _build_design_spec(metrics, source_id)
            record = {
                "sample_id": f"airshow_{source_id}",
                "source_id": source_id,
                "source": "VSP Airshow public Firestore/storage export",
                "source_page": args.source_url,
                "source_url": model.get("newVspUrl") or model.get("vspUrl") or geometry_url,
                "visual_geometry_url": geometry_url,
                "airshow_firestore_document": model.get("document_name"),
                "name": model.get("name") or model.get("displayName") or source_id,
                "display_name": model.get("displayName"),
                "manufacturer": model.get("manufacturer"),
                "uploaded_by": model.get("uploadedBy"),
                "date": model.get("date"),
                "date_accessed": started_at[:10],
                "downloads": model.get("downloads"),
                "units": UNIT_NAMES.get(str(model.get("units")), str(model.get("units") or "unknown")),
                "source_license_id": str(model.get("license")),
                "source_license": LICENSE_NAMES.get(str(model.get("license")), str(model.get("license"))),
                "split": _split_for_id(source_id),
                "design_family": "vsp_airshow_public_model",
                "geometry_path": str(voxel_path.relative_to(output_dir)).replace("\\", "/"),
                "geometry_provenance": (
                    "Public VSP Airshow preview geometry downloaded from the model document's "
                    "newX3dUrl/x3dUrl, parsed from X3D IndexedFaceSet geometry, normalized to "
                    f"fit a centered {args.grid_size}^3 voxel lattice, and saved as a training voxel grid."
                ),
                "preprocessing_version": "vsp-airshow-x3d-voxelizer-v1",
                "preprocessing_hash": hashlib.sha256(
                    json.dumps(
                        {
                            "script": "CLI/build_airshow_corpus.py",
                            "grid_size": args.grid_size,
                            "allowed_licenses": list(args.allowed_licenses),
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                "geometry_sha256": _sha256_bytes(raw_bytes),
                "voxel_sha256": _sha256_file(voxel_path),
                "design_spec": design_spec,
                "design_spec_provenance": _design_spec_provenance(),
                "response_metrics": {
                    "occupancy_ratio": metrics["occupancy_ratio"],
                    "occupied_voxels": metrics["occupied_voxels"],
                },
                "response_metrics_provenance": {
                    "occupancy_ratio": "direct_voxel_count_from_preprocessed_public_geometry",
                    "occupied_voxels": "direct_voxel_count_from_preprocessed_public_geometry",
                },
                "mesh_metrics": metrics,
                "claim_boundary": (
                    "Grounded public geometry for model-training smoke evidence. Airshow names, "
                    "manufacturers, licenses, and URLs are source metadata; mission/design-spec "
                    "fields are deterministic conditioning inferences and not source factual claims."
                ),
            }
            records.append(record)
            license_counts[record["source_license"]] += 1
            print(f"[{index}/{len(candidates)}] ok {source_id} {_safe_console_text(record['name'])}")
        except Exception as exc:  # noqa: BLE001 - keep conversion failures in the report.
            failures.append({"source_id": source_id, "url": geometry_url, "error": str(exc)})
            print(f"[{index}/{len(candidates)}] skip {source_id}: {_safe_console_text(exc, stream=sys.stderr)}", file=sys.stderr)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    split_counts = Counter(record["split"] for record in records)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_url": args.source_url,
        "firestore_project_id": config["projectId"],
        "storage_bucket": config["storageBucket"],
        "all_public_model_documents": len(models),
        "candidate_documents_after_license_and_geometry_filter": len(candidates),
        "record_count": len(records),
        "failure_count": len(failures),
        "grid_size": args.grid_size,
        "allowed_license_ids": list(args.allowed_licenses),
        "allowed_license_names": [LICENSE_NAMES.get(item, item) for item in args.allowed_licenses],
        "license_counts": dict(license_counts),
        "split_counts": dict(split_counts),
        "manifest_path": str(manifest_path),
        "provenance_path": str(provenance_path),
        "failures": failures,
    }
    provenance = {
        "report": report,
        "records": [
            {
                key: record.get(key)
                for key in (
                    "sample_id",
                    "source_id",
                    "name",
                    "manufacturer",
                    "uploaded_by",
                    "source_license",
                    "source_url",
                    "visual_geometry_url",
                    "geometry_sha256",
                    "voxel_sha256",
                    "split",
                )
            }
            for record in records
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    _configure_console_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=AIRSHOW_URL)
    parser.add_argument("--output-dir", default="build/airshow_grounded_corpus")
    parser.add_argument("--grid-size", type=_positive_int, default=16)
    parser.add_argument("--page-size", type=_positive_int, default=1000)
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap after filtering; 0 means no cap.")
    parser.add_argument(
        "--allowed-licenses",
        nargs="+",
        default=["1", "2", "3"],
        help="Airshow license ids to admit. Defaults to CC0, CC BY, and CC BY-SA.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)
    report = build_corpus(args)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["record_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
