#!/usr/bin/env python3
"""Build a fail-closed, provenance-preserving corpus from AircraftVerse.

The builder reads local ZIP members directly and only downloads a shard when a
local archive has not been supplied.  It never treats source-native metadata
as a fixed-wing mission label: unavailable conditioning fields remain null and
are masked in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import requests
import trimesh

from aircraft_validity import canonicalize_aircraft_voxels, evaluate_aircraft_validity
from validate_manifest import DEFAULT_UNIQUE_GEOMETRY_TARGET, validate_manifest_file


ZENODO_RECORD_API = "https://zenodo.org/api/records/6525446"
ZENODO_RECORD_URL = "https://zenodo.org/records/6525446"
RECORD_VERSION = "1.0.0"
PREPROCESSING_VERSION = "aircraftverse-canonical-voxelizer-v2"
REQUIRED_MEMBERS = (
    "cadfile.stl",
    "Geom.stp",
    "design_tree.json",
    "design_low_level.json",
    "design_seq.json",
    "output.json",
)
POSITIVE_PERFORMANCE_FIELDS = (
    "Mass",
    "Max_Distance",
    "Hover_Time",
    "Max_Speed",
    "Power_MFD",
    "Power_MxSpd",
    "Speed_MFD",
)
RATIO_FIELD_NAMES = {
    "battery_current_ratio",
    "batterycurrentratio",
    "motor_current_ratio",
    "motorcurrentratio",
    "motor_power_ratio",
    "motorpowerratio",
    "control_utilization",
    "controlutilization",
}
CONDITIONING_FIELDS = (
    "target_speed_mps",
    "wingspan_limit_m",
    "thrust_to_weight_min",
    "turn_rate_min_deg_s",
    "required_static_thrust_n",
    "engine_diameter_mm",
    "engine_length_mm",
    "engine_count_min",
    "engine_count_max",
    "payload_mass_min_g",
    "payload_mass_max_g",
    "takeoff_distance_min_m",
    "takeoff_distance_max_m",
    "wall_thickness_min_mm",
    "wall_thickness_max_mm",
    "part_count_min",
    "part_count_max",
    "manufacturing_method",
)


class CorpusBuildError(RuntimeError):
    """A stable, reportable reason a source design cannot enter the corpus."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_md5_prefix(value: str) -> str:
    return str(value).lower().removeprefix("md5:")


def verify_archive_checksum(
    path: Path,
    *,
    expected_md5: str,
    expected_size: Optional[int] = None,
) -> None:
    if not path.exists():
        raise CorpusBuildError("archive_missing", f"Archive does not exist: {path}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise CorpusBuildError(
            "archive_size_mismatch",
            f"Archive size {path.stat().st_size} does not match source size {expected_size}.",
        )
    actual_md5 = _md5_file(path)
    if actual_md5 != _strip_md5_prefix(expected_md5):
        raise CorpusBuildError(
            "archive_checksum_mismatch",
            f"Archive MD5 {actual_md5} does not match source checksum {_strip_md5_prefix(expected_md5)}.",
        )


def _fetch_zenodo_record() -> Dict[str, Any]:
    response = requests.get(ZENODO_RECORD_API, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise CorpusBuildError("record_metadata_invalid", "Zenodo record did not contain a file catalog.")
    return payload


def _download_resumable(url: str, path: Path, *, expected_md5: str, expected_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = path.stat().st_size if path.exists() else 0
    if existing_size == int(expected_size):
        verify_archive_checksum(path, expected_md5=expected_md5, expected_size=expected_size)
        return
    if existing_size > int(expected_size):
        raise CorpusBuildError(
            "archive_oversize",
            f"Existing archive {path} is larger than the pinned source object.",
        )

    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    try:
        if response.status_code not in {200, 206}:
            response.raise_for_status()
        append = existing_size > 0 and response.status_code == 206
        if existing_size > 0 and response.status_code == 200:
            # The host did not honour the range request. Restart atomically in
            # the same location rather than appending duplicate bytes.
            existing_size = 0
        mode = "ab" if append else "wb"
        with path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    finally:
        response.close()

    verify_archive_checksum(path, expected_md5=expected_md5, expected_size=expected_size)


def _selection_key(seed: int, shard_key: str, design_id: str) -> str:
    return _sha256_bytes(f"{int(seed)}:{shard_key}:{design_id}".encode("utf-8"))


def deterministic_design_order(
    design_ids: Iterable[str],
    *,
    seed: int,
    shard_key: str,
) -> List[str]:
    return sorted(set(str(design_id) for design_id in design_ids), key=lambda design_id: _selection_key(seed, shard_key, design_id))


def _split_for_source_id(source_id: str) -> str:
    bucket = int(_sha256_bytes(source_id.encode("utf-8"))[:8], 16) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "val"
    if bucket < 9:
        return "test"
    return "holdout"


def _as_finite_positive(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError("performance_invalid", f"{field_name} is missing or non-numeric.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise CorpusBuildError("performance_invalid", f"{field_name} must be finite and strictly positive.")
    return number


def _normalized_field_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum() or character == "_")


def validate_source_performance(performance: Mapping[str, Any]) -> Dict[str, float]:
    """Validate only explicit source feasibility fields, without inventing labels."""
    if not isinstance(performance, Mapping):
        raise CorpusBuildError("performance_invalid", "output.json must contain an object.")

    interferences = performance.get("Interferences", performance.get("interferences"))
    if interferences is None:
        raise CorpusBuildError("performance_invalid", "Interferences is required for source feasibility.")
    try:
        interference_count = float(interferences)
    except (TypeError, ValueError) as exc:
        raise CorpusBuildError("performance_invalid", "Interferences must be numeric.") from exc
    if not math.isfinite(interference_count) or interference_count != 0.0:
        raise CorpusBuildError("source_interference", "Source design reports component interferences.")

    validated = {
        field_name: _as_finite_positive(performance.get(field_name), field_name)
        for field_name in POSITIVE_PERFORMANCE_FIELDS
    }
    for key, raw_value in performance.items():
        normalized = _normalized_field_name(str(key))
        if normalized not in RATIO_FIELD_NAMES:
            continue
        try:
            ratio = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise CorpusBuildError("performance_invalid", f"{key} is not numeric.") from exc
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise CorpusBuildError("performance_invalid", f"{key} must be in [0, 1].")
        validated[str(key)] = ratio
    return validated


def _load_json_member(archive: zipfile.ZipFile, member_name: str) -> Any:
    try:
        with archive.open(member_name) as handle:
            return json.load(handle)
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorpusBuildError("member_json_invalid", f"Cannot parse {member_name}.") from exc


def _design_prefixes(archive: zipfile.ZipFile) -> Dict[str, str]:
    prefixes: Dict[str, str] = {}
    for name in archive.namelist():
        normalized = name.replace("\\", "/").strip("/")
        if not normalized.endswith("/cadfile.stl"):
            continue
        prefix = normalized[: -len("/cadfile.stl")]
        design_id = prefix.rsplit("/", 1)[-1]
        if design_id:
            prefixes[design_id] = prefix
    return prefixes


def _load_stl_mesh(stl_bytes: bytes) -> trimesh.Trimesh:
    try:
        loaded = trimesh.load(io.BytesIO(stl_bytes), file_type="stl", force="mesh")
    except Exception as exc:  # trimesh raises multiple parser-specific exception types.
        raise CorpusBuildError("geometry_invalid", "STL failed to parse.") from exc
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise CorpusBuildError("geometry_invalid", "STL scene contained no meshes.")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise CorpusBuildError("geometry_invalid", "STL did not produce a triangle mesh.")
    mesh = loaded.copy()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise CorpusBuildError("geometry_invalid", "STL contains no vertices or faces.")
    if not np.isfinite(np.asarray(mesh.vertices, dtype=np.float64)).all():
        raise CorpusBuildError("geometry_invalid", "STL has non-finite vertex coordinates.")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if not np.isfinite(extents).all() or np.any(extents <= 0.0):
        raise CorpusBuildError("geometry_invalid", "STL extents must be finite and positive on all axes.")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0:
        raise CorpusBuildError("geometry_invalid", "STL contains no triangles.")
    areas = trimesh.triangles.area(triangles)
    if not np.isfinite(areas).all():
        raise CorpusBuildError("geometry_invalid", "STL contains non-finite triangles.")
    # CAD tessellators commonly emit a small number of zero-area faces at
    # seams. Remove only those faces, then re-check the remaining assembly.
    nondegenerate = areas > np.finfo(np.float64).eps
    if not bool(nondegenerate.all()):
        mesh.update_faces(nondegenerate)
        mesh.remove_unreferenced_vertices()
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise CorpusBuildError("geometry_invalid", "STL contains only degenerate triangles.")
        cleaned_areas = trimesh.triangles.area(np.asarray(mesh.triangles, dtype=np.float64))
        if not np.isfinite(cleaned_areas).all() or np.any(cleaned_areas <= 0.0):
            raise CorpusBuildError("geometry_invalid", "STL cleanup left degenerate triangles.")
    return mesh


def voxelize_mesh(mesh: trimesh.Trimesh, grid_size: int) -> np.ndarray:
    if grid_size < 16:
        raise CorpusBuildError("grid_invalid", "grid_size must be at least 16.")
    extents = np.asarray(mesh.extents, dtype=np.float64)
    working = mesh.copy()
    center = np.asarray(working.bounds, dtype=np.float64).mean(axis=0)
    working.apply_translation(-center)
    working.apply_scale(0.80 / float(extents.max()))
    try:
        source_voxels = working.voxelized(pitch=1.0 / float(grid_size)).matrix.astype(np.float32)
    except Exception as exc:
        raise CorpusBuildError("voxelization_failed", "trimesh voxelization failed.") from exc
    if source_voxels.size == 0 or float(source_voxels.sum()) <= 0.0:
        raise CorpusBuildError("voxelization_failed", "Voxelization produced no occupied cells.")

    output = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    source_slices: List[slice] = []
    destination_slices: List[slice] = []
    for source_size in source_voxels.shape:
        length = min(int(source_size), int(grid_size))
        source_start = max(0, (int(source_size) - length) // 2)
        destination_start = max(0, (int(grid_size) - length) // 2)
        source_slices.append(slice(source_start, source_start + length))
        destination_slices.append(slice(destination_start, destination_start + length))
    output[tuple(destination_slices)] = source_voxels[tuple(source_slices)]
    return output


def _declared_linear_dimensions(payload: Any, *, parent_key: str = "") -> Iterator[float]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = _normalized_field_name(str(key))
            yield from _declared_linear_dimensions(value, parent_key=normalized)
        return
    if isinstance(payload, list):
        for value in payload:
            yield from _declared_linear_dimensions(value, parent_key=parent_key)
        return
    if not any(token in parent_key for token in ("length", "diameter", "span", "width", "height", "radius", "chord")):
        return
    try:
        value = float(payload)
    except (TypeError, ValueError):
        return
    if math.isfinite(value) and value > 0.0:
        yield value


def _validate_mesh_scale(mesh: trimesh.Trimesh, low_level: Mapping[str, Any]) -> Dict[str, float]:
    declared = list(_declared_linear_dimensions(low_level))
    if not declared:
        raise CorpusBuildError("specification_invalid", "No positive declared component dimension is recoverable.")
    largest_declared = max(declared)
    largest_extent = float(np.asarray(mesh.extents, dtype=np.float64).max())
    ratio = largest_extent / largest_declared
    if not 0.5 <= ratio <= 50.0:
        raise CorpusBuildError(
            "geometry_scale_invalid",
            f"Assembly-to-declared-dimension ratio {ratio:.6g} is outside [0.5, 50].",
        )
    return {
        "largest_declared_linear_dimension": float(largest_declared),
        "largest_mesh_extent": largest_extent,
        "mesh_to_declared_extent_ratio": float(ratio),
    }


def _mesh_metrics(mesh: trimesh.Trimesh, voxels: np.ndarray) -> Dict[str, Any]:
    occupied = int(voxels.sum())
    return {
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "source_is_watertight": bool(mesh.is_watertight),
        "source_bounds": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "source_extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
        "occupied_voxels": occupied,
        "occupancy_ratio": float(occupied / max(int(voxels.size), 1)),
    }


def _null_conditioning_spec() -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, str]]:
    spec = {field_name: None for field_name in CONDITIONING_FIELDS}
    availability = {field_name: False for field_name in CONDITIONING_FIELDS}
    provenance = {field_name: "unavailable_in_AircraftVerse_source_metadata" for field_name in CONDITIONING_FIELDS}
    return spec, availability, provenance


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


def _record_from_design(
    *,
    archive: zipfile.ZipFile,
    archive_path: Path,
    archive_metadata: Mapping[str, Any],
    prefix: str,
    design_id: str,
    output_dir: Path,
    grid_size: int,
) -> Dict[str, Any]:
    members = {name.replace("\\", "/").strip("/") for name in archive.namelist()}
    member_paths = {member: f"{prefix}/{member}" for member in REQUIRED_MEMBERS}
    missing = [member for member, member_path in member_paths.items() if member_path not in members]
    if missing:
        raise CorpusBuildError("member_missing", f"Missing required members: {', '.join(missing)}.")

    low_level = _load_json_member(archive, member_paths["design_low_level.json"])
    design_tree = _load_json_member(archive, member_paths["design_tree.json"])
    design_seq = _load_json_member(archive, member_paths["design_seq.json"])
    performance = _load_json_member(archive, member_paths["output.json"])
    if not isinstance(low_level, Mapping) or not isinstance(performance, Mapping):
        raise CorpusBuildError("member_json_invalid", "Low-level design and output members must be objects.")
    validated_performance = validate_source_performance(performance)

    with archive.open(member_paths["cadfile.stl"]) as handle:
        stl_bytes = handle.read()
    with archive.open(member_paths["Geom.stp"]) as handle:
        step_bytes = handle.read()
    if not stl_bytes or not step_bytes:
        raise CorpusBuildError("member_empty", "STL and STEP members must both be non-empty.")

    mesh = _load_stl_mesh(stl_bytes)
    scale_report = _validate_mesh_scale(mesh, low_level)
    raw_voxels = voxelize_mesh(mesh, grid_size)
    canonical_voxels, canonicalization = canonicalize_aircraft_voxels(raw_voxels)
    canonical_np = canonical_voxels.numpy().astype(np.uint8)
    validity = evaluate_aircraft_validity(canonical_np)
    if validity["status"] != "pass":
        raise CorpusBuildError(
            "aircraft_validity_failed",
            "Aircraft validity checks failed: " + ", ".join(validity["failed_checks"]),
        )

    source_id = f"aircraftverse:{archive_metadata['key']}:{design_id}"
    voxel_path = output_dir / "voxels" / f"{archive_metadata['key']}.{design_id}.npy"
    _atomic_save_npy(voxel_path, canonical_np)
    design_spec, availability, spec_provenance = _null_conditioning_spec()
    stl_hash = _sha256_bytes(stl_bytes)
    step_hash = _sha256_bytes(step_bytes)
    mesh_metrics = {**_mesh_metrics(mesh, canonical_np), **scale_report}
    return {
        "sample_id": source_id,
        "source_id": source_id,
        "source_collection": "AircraftVerse",
        "source_design_id": design_id,
        "source_archive_key": archive_metadata["key"],
        "source_url": ZENODO_RECORD_URL,
        "source_record_api": ZENODO_RECORD_API,
        "source_record_version": RECORD_VERSION,
        "source_license": "CC BY 4.0 (Zenodo API); CC BY-SA stated by accompanying AircraftVerse paper; stricter attribution/share-alike handling retained.",
        "split": _split_for_source_id(source_id),
        "design_family": "aircraftverse_source_native_aerial_vehicle",
        "geometry_path": f"voxels/{voxel_path.name}",
        "geometry_provenance": "Source-native AircraftVerse STL whole-assembly CAD, normalized, canonicalized, and voxelized by this builder.",
        "preprocessing_version": PREPROCESSING_VERSION,
        "units": "source_native_units; normalized voxel lattice",
        "original_units": "AircraftVerse source-native CAD units",
        "geometry_sha256": stl_hash,
        "step_sha256": step_hash,
        "voxel_sha256": _sha256_file(voxel_path),
        "archive_md5": _strip_md5_prefix(str(archive_metadata["checksum"])),
        "archive_sha256": _sha256_file(archive_path),
        "archive_size_bytes": int(archive_metadata["size"]),
        "archive_url": archive_metadata["url"],
        "member_sha256": {
            "cadfile.stl": stl_hash,
            "Geom.stp": step_hash,
            "design_tree.json": _sha256_bytes(json.dumps(design_tree, sort_keys=True).encode("utf-8")),
            "design_low_level.json": _sha256_bytes(json.dumps(low_level, sort_keys=True).encode("utf-8")),
            "design_seq.json": _sha256_bytes(json.dumps(design_seq, sort_keys=True).encode("utf-8")),
            "output.json": _sha256_bytes(json.dumps(performance, sort_keys=True).encode("utf-8")),
        },
        "source_native_design_tree": design_tree,
        "source_native_low_level": low_level,
        "source_native_design_sequence": design_seq,
        "source_native_performance": performance,
        "validated_source_performance": validated_performance,
        "design_spec": design_spec,
        "design_spec_availability": availability,
        "design_spec_provenance": spec_provenance,
        "conditioning_mode": "unconditioned_source_metadata_only",
        "mesh_metrics": mesh_metrics,
        "canonicalization": canonicalization,
        "aircraft_validity": validity,
        "date_built": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": "Source CAD and source feasibility outputs support corpus membership only; this record is not flight-test, structural, or CFD validation.",
    }


def _archive_metadata_from_record(record: Mapping[str, Any], shard_key: str) -> Dict[str, Any]:
    for item in record.get("files", []):
        if isinstance(item, Mapping) and item.get("key") == shard_key:
            links = item.get("links") or {}
            return {
                "key": shard_key,
                "checksum": str(item.get("checksum") or ""),
                "size": int(item.get("size") or 0),
                "url": str(links.get("self") or links.get("download") or ""),
            }
    raise CorpusBuildError("archive_metadata_missing", f"Zenodo record does not contain {shard_key}.")


def _local_archive_metadata(path: Path) -> Dict[str, Any]:
    return {
        "key": path.name,
        "checksum": f"md5:{_md5_file(path)}",
        "size": path.stat().st_size,
        "url": str(path.resolve()),
    }


def _load_existing_records(manifest_path: Path) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "aircraftverse_manifest.jsonl"
    report_path = output_dir / "corpus_report.json"
    ledger_path = output_dir / "rejections.jsonl"
    target_count = int(args.target_count)
    grid_size = int(args.grid_size)
    selection_seed = int(args.selection_seed)

    existing_records = _load_existing_records(manifest_path) if bool(args.resume) else []
    accepted_records = list(existing_records)
    accepted_geometry_hashes = {str(record.get("geometry_sha256")) for record in accepted_records if record.get("geometry_sha256")}
    accepted_voxel_hashes = {str(record.get("voxel_sha256")) for record in accepted_records if record.get("voxel_sha256")}
    rejection_counts: Counter[str] = Counter()
    rejections: List[Dict[str, Any]] = []
    archive_reports: List[Dict[str, Any]] = []

    local_archives = [Path(value).resolve() for value in (args.archive or [])]
    if local_archives:
        archive_entries = [(path, _local_archive_metadata(path)) for path in local_archives]
    else:
        record = _fetch_zenodo_record()
        requested_shards = list(args.shard or ["AircraftVerse_1.zip", "AircraftVerse_2.zip", "AircraftVerse_3.zip"])
        archive_entries = []
        for shard_key in requested_shards:
            metadata = _archive_metadata_from_record(record, shard_key)
            archive_path = output_dir / "archives" / shard_key
            _download_resumable(
                metadata["url"],
                archive_path,
                expected_md5=metadata["checksum"],
                expected_size=metadata["size"],
            )
            archive_entries.append((archive_path, metadata))

    for archive_path, metadata in archive_entries:
        if len(accepted_records) >= target_count:
            break
        verify_archive_checksum(
            archive_path,
            expected_md5=metadata["checksum"],
            expected_size=metadata["size"],
        )
        accepted_before = len(accepted_records)
        with zipfile.ZipFile(archive_path) as archive:
            prefixes = _design_prefixes(archive)
            for design_id in deterministic_design_order(prefixes, seed=selection_seed, shard_key=str(metadata["key"])):
                if len(accepted_records) >= target_count:
                    break
                try:
                    record = _record_from_design(
                        archive=archive,
                        archive_path=archive_path,
                        archive_metadata=metadata,
                        prefix=prefixes[design_id],
                        design_id=design_id,
                        output_dir=output_dir,
                        grid_size=grid_size,
                    )
                    if record["geometry_sha256"] in accepted_geometry_hashes:
                        raise CorpusBuildError("duplicate_geometry", "Duplicate source STL hash.")
                    if record["voxel_sha256"] in accepted_voxel_hashes:
                        raise CorpusBuildError("duplicate_voxel", "Duplicate canonical voxel hash.")
                    accepted_records.append(record)
                    accepted_geometry_hashes.add(record["geometry_sha256"])
                    accepted_voxel_hashes.add(record["voxel_sha256"])
                except CorpusBuildError as exc:
                    rejection_counts[exc.code] += 1
                    rejections.append(
                        {
                            "archive_key": metadata["key"],
                            "source_design_id": design_id,
                            "code": exc.code,
                            "message": str(exc),
                        }
                    )
        archive_reports.append(
            {
                "archive_key": metadata["key"],
                "archive_path": str(archive_path),
                "accepted_count": len(accepted_records) - accepted_before,
                "archive_sha256": _sha256_file(archive_path),
            }
        )

    accepted_records.sort(key=lambda record: str(record["source_id"]))
    _atomic_write_jsonl(manifest_path, accepted_records)
    _atomic_write_jsonl(ledger_path, rejections)
    basic_validation = validate_manifest_file(str(manifest_path), level="basic")
    claim_validation = validate_manifest_file(
        str(manifest_path),
        level="claim-bearing",
        unique_geometry_target=max(DEFAULT_UNIQUE_GEOMETRY_TARGET, target_count),
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_record": ZENODO_RECORD_URL,
        "record_version": RECORD_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "grid_size": grid_size,
        "selection_seed": selection_seed,
        "target_count": target_count,
        "accepted_count": len(accepted_records),
        "unique_geometry_count": len(accepted_geometry_hashes),
        "unique_voxel_count": len(accepted_voxel_hashes),
        "rejected_count": len(rejections),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "archives": archive_reports,
        "manifest_path": str(manifest_path),
        "rejection_ledger_path": str(ledger_path),
        "basic_validation": basic_validation,
        "claim_validation": claim_validation,
        "claim_boundary": "A passing claim validation establishes manifest completeness and distinct CAD identities, not aircraft flightworthiness or experimental validation.",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="build/aircraftverse_corpus")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--target-count", type=int, default=625)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--archive", action="append", default=[], help="Use a local archive instead of downloading a shard; repeatable.")
    parser.add_argument("--shard", action="append", default=[], help="Zenodo archive key to download; repeatable.")
    parser.add_argument("--resume", action="store_true", help="Reuse accepted records from an existing manifest.")
    args = parser.parse_args(argv)
    report = build_corpus(args)
    print(json.dumps({key: report[key] for key in ("accepted_count", "unique_geometry_count", "rejected_count", "claim_validation")}, indent=2, sort_keys=True))
    return 0 if report["accepted_count"] >= args.target_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
