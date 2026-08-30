#!/usr/bin/env python3
"""Build the deterministic, relocatable final training corpus.

The input manifest is an immutable source of admitted original records. All
published geometry is copied into a private staging directory and named by
its canonical semantic content hash. No output directory is published until
counts, metadata, paths, shapes, dtypes, and hashes have all been checked.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

if __package__:
    from .perturb_corpus import (
        PERTURBATION_GENERATOR_VERSION,
        TRANSFORMS,
        iter_transform_candidates,
        validate as validate_perturbation,
    )
    from .procedural_aircraft_generator import (
        AIRCRAFT_TYPES,
        PROCEDURAL_GENERATOR_VERSION,
        PROCEDURAL_MIN_ACCEPTED_PER_TYPE,
        iter_procedural_samples,
    )
    from .validate_manifest import DEFAULT_UNIQUE_GEOMETRY_TARGET, validate_manifest_file
else:
    from perturb_corpus import (
        PERTURBATION_GENERATOR_VERSION,
        TRANSFORMS,
        iter_transform_candidates,
        validate as validate_perturbation,
    )
    from procedural_aircraft_generator import (
        AIRCRAFT_TYPES,
        PROCEDURAL_GENERATOR_VERSION,
        PROCEDURAL_MIN_ACCEPTED_PER_TYPE,
        iter_procedural_samples,
    )
    from validate_manifest import DEFAULT_UNIQUE_GEOMETRY_TARGET, validate_manifest_file


SOURCE_GRID_SHAPE = (96, 96, 96)
GRID_SHAPE = SOURCE_GRID_SHAPE
NORMALIZED_VOXEL_UNITS = "normalized voxel lattice; occupancy is dimensionless"
BUILDER_VERSION = "final-training-corpus-v2"
PERTURBATION_METADATA_VERSION = "final-training-corpus-v1-perturbation-v1"
PROCEDURAL_METADATA_VERSION = "final-training-corpus-v1-procedural-v1"

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

DEFAULT_PERTURBATION_BATCHES = (
    ("wing_dihedral_up", "tail_widen_30"),
    ("wing_dihedral_down", "tail_widen_50", "nose_thin", "airfoil_thicken"),
)
DEFAULT_EXPECTED_ORIGINAL_COUNT = 1069
DEFAULT_EXPECTED_PERTURBATION_COUNT = 4958
DEFAULT_EXPECTED_PROCEDURAL_COUNT = 2000
DEFAULT_EXPECTED_TOTAL_COUNT = 8027

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OUTPUT_GEOMETRY_RE = re.compile(r"^voxels/[0-9a-f]{64}\.npy$")


def load_jsonl_manifest(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects in stable line order, rejecting malformed records."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def resolve_geometry_path(record: Mapping[str, Any], manifest_path: Path) -> Path:
    """Resolve a source geometry reference without accepting a missing file."""
    geometry_ref = record.get("geometry_path")
    if not geometry_ref:
        raise ValueError(f"Record {record.get('source_id')} has no geometry_path")
    candidate = Path(str(geometry_ref))
    if not candidate.is_absolute():
        candidate = Path(manifest_path).parent / candidate
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Record {record.get('source_id')} geometry does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Record {record.get('source_id')} geometry is not a file: {candidate}")
    return candidate


def canonicalize_voxels(voxels: np.ndarray) -> np.ndarray:
    """Canonicalize a cubic voxel array to detached binary uint8 storage."""
    array = np.asarray(voxels)
    if array.ndim != 3 or len(set(array.shape)) != 1 or array.shape[0] <= 0:
        raise ValueError(f"voxel array must be a non-empty cube, got {tuple(array.shape)}")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError("voxel array contains non-finite values")
    try:
        canonical = (array > 0.5).astype(np.uint8, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"voxel array cannot be thresholded to binary uint8: {exc}") from exc
    return np.ascontiguousarray(canonical)


def resample_cubic_voxels(voxels: np.ndarray, target_grid_size: int) -> np.ndarray:
    """Nearest-neighbour resample with an explicit deterministic index map."""
    canonical = canonicalize_voxels(voxels)
    target = int(target_grid_size)
    if target <= 0:
        raise ValueError("target_grid_size must be positive")
    source = int(canonical.shape[0])
    if source == target:
        return canonical
    indices = np.floor(np.arange(target, dtype=np.float64) * source / target).astype(np.intp)
    indices = np.minimum(indices, source - 1)
    return np.ascontiguousarray(canonical[np.ix_(indices, indices, indices)])


def canonical_content_hash(voxels: np.ndarray) -> str:
    canonical = canonicalize_voxels(voxels)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_canonical_hash(record: Mapping[str, Any]) -> str:
    declared: dict[str, str] = {}
    for field in ("canonical_content_sha256", "voxel_sha256"):
        if field not in record:
            continue
        value = record[field]
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value.lower()):
            raise ValueError(
                f"Record {record.get('source_id')} has an invalid {field}; expected 64 lowercase hex characters"
            )
        declared[field] = value.lower()
    if "canonical_content_sha256" not in declared:
        raise ValueError(
            f"Record {record.get('source_id')} must declare a 64-character canonical_content_sha256"
        )
    return declared["canonical_content_sha256"]


def _declared_voxel_file_hash(record: Mapping[str, Any]) -> str | None:
    value = record.get("voxel_sha256")
    if value is None:
        return None
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value.lower()):
        raise ValueError(
            f"Record {record.get('source_id')} has an invalid voxel_sha256; expected 64 lowercase hex characters"
        )
    return value.lower()


def _load_canonical_file(path: Path) -> np.ndarray:
    try:
        loaded = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load voxel geometry {path}: {exc}") from exc
    if not isinstance(loaded, np.ndarray):
        close = getattr(loaded, "close", None)
        if close:
            close()
        raise ValueError(f"Voxel geometry must be a .npy array, got {type(loaded).__name__} at {path}")
    return canonicalize_voxels(loaded)


def preflight_source_records(
    source_manifest: Path,
    *,
    expected_original_count: int | None = None,
) -> list[dict[str, Any]]:
    """Validate every raw source row before any duplicate filtering or admission."""
    source_manifest = Path(source_manifest)
    source_records = load_jsonl_manifest(source_manifest)
    if expected_original_count is not None and len(source_records) != expected_original_count:
        raise ValueError(
            f"raw source record count {len(source_records)} does not equal expected {expected_original_count}"
        )

    source_ids: set[str] = set()
    source_hashes: set[str] = set()
    entries: list[dict[str, Any]] = []
    for source_record_index, parent in enumerate(source_records):
        source_id = str(parent.get("source_id") or parent.get("sample_id") or "")
        if not source_id:
            raise ValueError(f"source record {source_record_index} has no source_id")
        if source_id in source_ids:
            raise ValueError(f"duplicate source_id in raw source manifest: {source_id}")

        source_path = resolve_geometry_path(parent, source_manifest)
        canonical = _load_canonical_file(source_path)
        if tuple(canonical.shape) != SOURCE_GRID_SHAPE:
            raise ValueError(
                f"source record {source_record_index} must have shape {SOURCE_GRID_SHAPE}, "
                f"got {tuple(canonical.shape)}"
            )
        actual_hash = canonical_content_hash(canonical)
        declared_hash = _declared_canonical_hash(parent)
        if declared_hash != actual_hash:
            raise ValueError(
                f"source record {source_record_index} canonical hash mismatch: declared {declared_hash}, got {actual_hash}"
            )
        declared_voxel_hash = _declared_voxel_file_hash(parent)
        if declared_voxel_hash is not None:
            actual_voxel_hash = _file_sha256(source_path)
            if declared_voxel_hash != actual_voxel_hash:
                raise ValueError(
                    f"source record {source_record_index} voxel file hash mismatch: "
                    f"declared {declared_voxel_hash}, got {actual_voxel_hash}"
                )
        if actual_hash in source_hashes:
            raise ValueError(
                f"duplicate canonical source content in raw source manifest: {actual_hash}"
            )

        source_ids.add(source_id)
        source_hashes.add(actual_hash)
        entries.append(
            {
                "record": parent,
                "source_path": source_path,
                "source_hash": actual_hash,
                "source_record_index": source_record_index,
            }
        )
    return entries


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    path.write_bytes(rendered.encode("utf-8"))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n" for record in records
    )
    path.write_bytes(rendered.encode("utf-8"))


def _canonical_npy_sha256(voxels: np.ndarray) -> str:
    payload = io.BytesIO()
    np.save(payload, canonicalize_voxels(voxels), allow_pickle=False)
    return hashlib.sha256(payload.getvalue()).hexdigest()


def _write_canonical_voxel(voxels: np.ndarray, content_hash: str, voxel_root: Path) -> tuple[str, str]:
    if not _HASH_RE.fullmatch(content_hash):
        raise ValueError(f"invalid content hash for output geometry: {content_hash!r}")
    canonical = canonicalize_voxels(voxels)
    actual_hash = canonical_content_hash(canonical)
    if actual_hash != content_hash:
        raise ValueError(f"canonical hash changed before staging: expected {content_hash}, got {actual_hash}")

    voxel_root.mkdir(parents=True, exist_ok=True)
    destination = voxel_root / f"{content_hash}.npy"
    if destination.exists():
        existing = _load_canonical_file(destination)
        if canonical_content_hash(existing) != content_hash:
            raise ValueError(f"content-addressed output collision at {destination}")
        return f"voxels/{destination.name}", _file_sha256(destination)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(voxel_root), prefix=f".{content_hash}.", suffix=".npy.tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            if _mark_sparse_file(handle):
                _write_sparse_npy(handle, canonical)
            else:
                np.save(handle, canonical, allow_pickle=False)
            handle.flush()
        os.replace(str(temporary_path), str(destination))
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    staged = _load_canonical_file(destination)
    if canonical_content_hash(staged) != content_hash:
        raise ValueError(f"staged content hash mismatch at {destination}")
    voxel_file_hash = _file_sha256(destination)
    expected_file_hash = _canonical_npy_sha256(canonical)
    if voxel_file_hash != expected_file_hash:
        raise ValueError(
            f"staged voxel file hash mismatch at {destination}: {voxel_file_hash} != {expected_file_hash}"
        )
    return f"voxels/{destination.name}", voxel_file_hash


def _mark_sparse_file(handle: Any) -> bool:
    """Mark a Windows file sparse; return false where the API is unavailable."""
    if os.name != "nt":
        return False
    try:
        import msvcrt

        fsctl_set_sparse = 0x900C4
        device_io_control = ctypes.windll.kernel32.DeviceIoControl
        device_io_control.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.c_void_p,
        ]
        device_io_control.restype = ctypes.c_int
        returned = ctypes.c_ulong(0)
        result = device_io_control(
            ctypes.c_void_p(msvcrt.get_osfhandle(handle.fileno())),
            fsctl_set_sparse,
            None,
            0,
            None,
            0,
            ctypes.byref(returned),
            None,
        )
        return bool(result)
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _write_sparse_npy(handle: Any, array: np.ndarray, *, chunk_bytes: int = 4096) -> None:
    """Write a deterministic NPY header and leave zero data ranges as holes."""
    header_buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header_buffer,
        np.lib.format.header_data_from_array_1_0(array),
    )
    handle.write(header_buffer.getvalue())
    data_offset = handle.tell()
    flat = np.ascontiguousarray(array).reshape(-1)
    for start in range(0, flat.size, chunk_bytes):
        end = min(start + chunk_bytes, flat.size)
        chunk = flat[start:end]
        if np.any(chunk):
            handle.seek(data_offset + start)
            handle.write(chunk.tobytes(order="C"))
    if flat.nbytes:
        # Establish the logical end without writing over the final element;
        # the last voxel may itself be occupied.
        handle.seek(data_offset + flat.nbytes)
        handle.truncate()


def _null_conditioning_metadata(reason: str) -> dict[str, Any]:
    return {
        "design_spec": {field: None for field in CONDITIONING_FIELDS},
        "design_spec_availability": {field: False for field in CONDITIONING_FIELDS},
        "design_spec_provenance": {field: reason for field in CONDITIONING_FIELDS},
    }


def _identity_canonicalization(status: str) -> dict[str, Any]:
    """Declare that persisted geometry already uses the corpus canonical axes."""
    return {
        "permutation": [0, 1, 2],
        "status": status,
        "method": "identity; geometry generated or resampled in canonical z-y-x frame",
    }


def _is_http_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def _contains_local_path_reference(value: str) -> bool:
    if _is_http_url(value):
        return False
    if value.lower().startswith(("file://", "file:/")):
        return True
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("\\\\")
        or value.startswith("/")
        or bool(re.match(r"^[A-Za-z]:[\\/]", value))
        or bool(re.search(r"(?:[A-Za-z]:[\\/]|\\\\)", value))
    )


def _looks_like_absolute_path(value: str) -> bool:
    return _contains_local_path_reference(value)


def _lexical_absolute_path(path: Path | str, *, role: str) -> Path:
    raw = os.fspath(path)
    if isinstance(raw, bytes):
        raw = os.fsdecode(raw)
    if not str(raw).strip() or str(raw).strip() in {".", ".."}:
        raise ValueError(f"unsafe empty/current-directory {role} target: {path!r}")
    candidate = Path(os.path.abspath(raw))
    if candidate == Path(candidate.anchor):
        raise ValueError(f"unsafe filesystem-root {role} target: {path!r}")
    return candidate


def _is_reparse_point(path: Path) -> bool:
    try:
        if Path(path).is_symlink():
            return True
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT


def _assert_no_reparse_components(path: Path, *, role: str) -> None:
    candidate = _lexical_absolute_path(path, role=role)
    current = Path(candidate.anchor)
    for part in candidate.parts:
        if part == candidate.anchor:
            continue
        current = current / part
        if not os.path.lexists(current):
            break
        if _is_reparse_point(current):
            raise ValueError(f"{role} contains a symlink/junction/reparse point: {current}")


def _safe_output_target(output_dir: Path | str) -> Path:
    candidate = _lexical_absolute_path(output_dir, role="output")
    _assert_no_reparse_components(candidate, role="output")
    if os.path.lexists(candidate):
        raise FileExistsError(f"refusing to replace existing output directory: {candidate}")
    parent = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)
    _assert_no_reparse_components(parent, role="output parent")
    if os.path.lexists(candidate):
        raise FileExistsError(f"output target appeared during preparation: {candidate}")
    return candidate


def _assert_no_reparse_tree(root: Path) -> None:
    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                child = Path(entry.path)
                if _is_reparse_point(child):
                    raise ValueError(f"staging tree contains a symlink/junction/reparse point: {child}")
                if entry.is_dir(follow_symlinks=False):
                    visit(child)

    visit(root)


def _safe_cleanup_staging(staging_dir: Path | str, expected_parent: Path | str) -> None:
    """Delete only an owned staging child, quarantining it before recursion."""
    stage = _lexical_absolute_path(staging_dir, role="staging")
    parent = _lexical_absolute_path(expected_parent, role="staging parent")
    if stage.parent != parent:
        raise ValueError(f"staging target is outside its expected parent: {stage}")
    if not re.fullmatch(r"\.[^\\/]+\.staging-[A-Za-z0-9_-]+", stage.name):
        raise ValueError(f"staging target has an unexpected name: {stage.name}")
    _assert_no_reparse_components(parent, role="staging parent")
    if not os.path.lexists(stage):
        return
    if _is_reparse_point(stage):
        raise ValueError(f"refusing to recurse into a reparse staging target: {stage}")

    quarantine = parent / f".{stage.name}.cleanup-{uuid.uuid4().hex}"
    if os.path.lexists(quarantine):
        raise ValueError(f"unexpected cleanup quarantine collision: {quarantine}")
    os.replace(str(stage), str(quarantine))
    if _is_reparse_point(quarantine):
        # Unlink the reparse entry itself; never recurse through it.
        quarantine.unlink(missing_ok=True)
        raise ValueError(f"staging target changed to a reparse point during cleanup: {quarantine}")
    _assert_no_reparse_tree(quarantine)
    shutil.rmtree(quarantine)


def _strip_local_path_fields(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key != "flight_path" and (
                child_key.endswith("_path") or child_key in {"stl_path", "source_manifest_path"}
            ):
                continue
            if isinstance(child_value, str) and _contains_local_path_reference(child_value):
                continue
            cleaned[child_key] = _strip_local_path_fields(child_value, key=child_key)
        return cleaned
    if isinstance(value, list):
        return [
            _strip_local_path_fields(item, key=key)
            for item in value
            if not (isinstance(item, str) and _contains_local_path_reference(item))
        ]
    return value


def _assert_no_absolute_paths(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_no_absolute_paths(child, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_absolute_paths(child, context=f"{context}[{index}]")
    elif isinstance(value, str) and _contains_local_path_reference(value):
        raise ValueError(f"{context} contains an absolute local path: {value}")


def _build_original_record(
    parent: Mapping[str, Any],
    *,
    source_manifest_name: str,
    source_manifest_hash: str,
    source_record_index: int,
    content_hash: str,
    voxel_file_hash: str | None = None,
    geometry_path: str,
) -> dict[str, Any]:
    source_id = str(parent.get("source_id") or parent.get("sample_id") or "")
    if not source_id:
        raise ValueError(f"source record {source_record_index} has no source_id")
    output = _strip_local_path_fields(dict(parent))
    if not isinstance(output, dict):
        raise ValueError("source record metadata must be a JSON object")
    if isinstance(parent.get("canonicalization"), Mapping):
        output["source_canonicalization"] = dict(parent["canonicalization"])
    source_units = parent.get("units")
    if source_units and source_units != NORMALIZED_VOXEL_UNITS:
        output["source_units"] = source_units
    output.update(
        {
            "source_id": source_id,
            "source_type": "original",
            "corpus_batch": "original",
            "source_manifest_name": source_manifest_name,
            "source_manifest_sha256": source_manifest_hash,
            "source_manifest_record_index": source_record_index,
            "geometry_path": geometry_path,
            "canonical_content_sha256": content_hash,
            "voxel_sha256": voxel_file_hash or content_hash,
            "units": NORMALIZED_VOXEL_UNITS,
            "conditioning_mode": "unconditioned_source_metadata_only",
            "canonicalization": _identity_canonicalization(
                "resampled_from_source_canonical_frame"
            ),
        }
    )
    output.update(
        _null_conditioning_metadata("unavailable; not used as conditioning without field-level source evidence")
    )
    output.setdefault("geometry_provenance", "Admitted source voxel geometry; provenance retained from source manifest.")
    output.setdefault("preprocessing_version", "final-training-corpus-v1-original-admission-v1")
    output.setdefault("design_family", "grounded_original")
    _assert_no_absolute_paths(output, context=f"original record {source_record_index}")
    return output


def build_perturbation_record(
    parent: Mapping[str, Any],
    *,
    transform: str,
    parent_record_index: int,
    parent_hash: str,
    child_hash: str,
    voxel_file_hash: str | None = None,
    geometry_path: str,
) -> dict[str, Any]:
    parent_source_id = str(parent.get("source_id") or parent.get("sample_id") or "")
    if not parent_source_id:
        raise ValueError("perturbation parent has no source_id")
    parent_split = parent.get("split")
    if not isinstance(parent_split, str) or not parent_split:
        raise ValueError(f"perturbation parent {parent_source_id} has no usable split")
    if not _HASH_RE.fullmatch(parent_hash) or not _HASH_RE.fullmatch(child_hash):
        raise ValueError("perturbation parent and child hashes must be 64-character lowercase hex")
    if not _OUTPUT_GEOMETRY_RE.fullmatch(geometry_path):
        raise ValueError(f"generated geometry path is not a relative content-addressed path: {geometry_path}")
    record = {
        "source_id": f"perturb:{transform}:{parent_source_id}",
        "source_type": "perturbation_expanded",
        "corpus_batch": "perturbation",
        "parent_source_id": parent_source_id,
        "parent_record_index": parent_record_index,
        "parent_canonical_content_sha256": parent_hash,
        "transform": transform,
        "geometry_path": geometry_path,
        "canonical_content_sha256": child_hash,
        "voxel_sha256": voxel_file_hash or child_hash,
        "split": parent_split,
        "conditioning_mode": "unconditioned_source_metadata_only",
        "geometry_provenance": (
            "Deterministic voxel-space transform of parent; not independent CAD or an aerodynamic, "
            "mission, or manufacturing measurement."
        ),
        "preprocessing_version": PERTURBATION_METADATA_VERSION,
        "units": NORMALIZED_VOXEL_UNITS,
        "design_family": "generated_perturbation",
        "canonicalization": _identity_canonicalization(
            "generated_in_parent_canonical_frame"
        ),
    }
    record.update(_null_conditioning_metadata("unavailable; not inherited or inferred for generated geometry"))
    _assert_no_absolute_paths(record, context=f"perturbation {record['source_id']}")
    return record


def build_procedural_record(
    *,
    aircraft_type: str,
    accepted_index: int,
    attempt: int,
    seed: int,
    child_hash: str,
    voxel_file_hash: str | None = None,
    geometry_path: str,
) -> dict[str, Any]:
    if not _HASH_RE.fullmatch(child_hash):
        raise ValueError("procedural child hash must be 64-character lowercase hex")
    if not _OUTPUT_GEOMETRY_RE.fullmatch(geometry_path):
        raise ValueError(f"generated geometry path is not a relative content-addressed path: {geometry_path}")
    record = {
        "source_id": f"proc:{aircraft_type}:{accepted_index}",
        "source_type": "procedural",
        "corpus_batch": "procedural",
        "aircraft_type": aircraft_type,
        "accepted_index": accepted_index,
        "attempt": attempt,
        "generator_seed": seed,
        "geometry_path": geometry_path,
        "canonical_content_sha256": child_hash,
        "voxel_sha256": voxel_file_hash or child_hash,
        "split": "train",
        "conditioning_mode": "unconditioned_source_metadata_only",
        "geometry_provenance": "Deterministic procedural voxel geometry; NOT real CAD or measured aircraft data.",
        "preprocessing_version": PROCEDURAL_METADATA_VERSION,
        "units": NORMALIZED_VOXEL_UNITS,
        "design_family": f"generated_procedural_{aircraft_type}",
        "canonicalization": _identity_canonicalization(
            "generated_in_procedural_canonical_frame"
        ),
    }
    record.update(_null_conditioning_metadata("unavailable; not inherited or inferred for generated geometry"))
    _assert_no_absolute_paths(record, context=f"procedural {record['source_id']}")
    return record


def _safe_published_geometry_path(output_root: Path, geometry_path: Any) -> Path:
    if not isinstance(geometry_path, str) or not _OUTPUT_GEOMETRY_RE.fullmatch(geometry_path):
        raise ValueError(f"published geometry path must be relative voxels/<sha256>.npy: {geometry_path!r}")
    candidate = (output_root / Path(geometry_path)).resolve()
    try:
        candidate.relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"published geometry path escapes corpus directory: {geometry_path}") from exc
    if candidate.parent != (output_root / "voxels").resolve():
        raise ValueError(f"published geometry path is not directly under voxels/: {geometry_path}")
    if _is_reparse_point(candidate):
        raise ValueError(f"published geometry path is a symlink/junction/reparse point: {geometry_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"published geometry file does not exist: {geometry_path}")
    return candidate


def _validate_record_metadata(record: Mapping[str, Any], *, index: int) -> None:
    for field in ("source_id", "geometry_provenance", "preprocessing_version", "units", "design_family"):
        if not record.get(field):
            raise ValueError(f"record {index} missing claim-bearing field {field}")
    if record.get("units") != NORMALIZED_VOXEL_UNITS:
        raise ValueError(f"record {index} does not declare normalized voxel units")
    design_spec = record.get("design_spec")
    if not isinstance(design_spec, dict) or set(design_spec) != set(CONDITIONING_FIELDS):
        raise ValueError(f"record {index} design_spec does not contain exactly all conditioning fields")
    if any(value is not None for value in design_spec.values()):
        raise ValueError(f"record {index} has a non-null conditioning value")
    availability = record.get("design_spec_availability")
    if availability != {field: False for field in CONDITIONING_FIELDS}:
        raise ValueError(f"record {index} does not mark every conditioning field unavailable")
    provenance = record.get("design_spec_provenance")
    if not isinstance(provenance, dict) or set(provenance) != set(CONDITIONING_FIELDS):
        raise ValueError(f"record {index} lacks field-level conditioning provenance")
    if any(not value for value in provenance.values()):
        raise ValueError(f"record {index} has empty field-level conditioning provenance")
    _assert_no_absolute_paths(record, context=f"record {index}")


def _validation_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "record_count": report["record_count"],
        "unique_geometry_count": report["unique_geometry_count"],
        "duplicate_geometry_record_count": report["duplicate_geometry_record_count"],
        "unique_geometry_target": report["unique_geometry_target"],
        "unique_geometry_target_met": report["unique_geometry_target_met"],
        "error_count": len(report.get("errors", [])),
    }


def validate_published_corpus(
    output_dir: Path,
    *,
    unique_geometry_target: int = DEFAULT_UNIQUE_GEOMETRY_TARGET,
    expected_total_count: int | None = None,
) -> dict[str, Any]:
    """Run independent path, array, hash, and claim-bearing checks on output."""
    output_root = _lexical_absolute_path(output_dir, role="published output")
    _assert_no_reparse_components(output_root, role="published output")
    manifest_path = output_root / "combined_training_manifest.jsonl"
    voxel_root = output_root / "voxels"
    build_spec_path = output_root / "build_spec.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"published manifest does not exist: {manifest_path}")
    if not voxel_root.is_dir():
        raise FileNotFoundError(f"published voxel directory does not exist: {voxel_root}")
    if not build_spec_path.is_file():
        raise FileNotFoundError(f"published build specification does not exist: {build_spec_path}")
    build_spec = json.loads(build_spec_path.read_text(encoding="utf-8"))
    declared_shape = build_spec.get("grid_shape")
    if (
        not isinstance(declared_shape, list)
        or len(declared_shape) != 3
        or len(set(declared_shape)) != 1
        or not isinstance(declared_shape[0], int)
        or declared_shape[0] <= 0
    ):
        raise ValueError(f"published build specification has invalid grid_shape: {declared_shape!r}")
    expected_shape = tuple(declared_shape)

    records = load_jsonl_manifest(manifest_path)
    if expected_total_count is not None and len(records) != expected_total_count:
        raise ValueError(f"published record count {len(records)} does not equal expected {expected_total_count}")

    unique_hashes: set[str] = set()
    expected_files: set[str] = set()
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    record_ids: set[str] = set()
    for index, record in enumerate(records):
        _validate_record_metadata(record, index=index)
        source_id = str(record["source_id"])
        if source_id in record_ids:
            raise ValueError(f"record {index} duplicates source_id {source_id}")
        record_ids.add(source_id)
        resolved_geometry = _safe_published_geometry_path(output_root, record.get("geometry_path"))
        expected_files.add(resolved_geometry.relative_to(output_root).as_posix())
        try:
            loaded = np.load(str(resolved_geometry), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"record {index} geometry cannot be loaded: {resolved_geometry}: {exc}") from exc
        if not isinstance(loaded, np.ndarray):
            raise ValueError(f"record {index} geometry is not a numpy array: {resolved_geometry}")
        if tuple(loaded.shape) != expected_shape:
            raise ValueError(f"record {index} geometry shape mismatch: {tuple(loaded.shape)}")
        if loaded.dtype != np.dtype(np.uint8):
            raise ValueError(f"record {index} geometry dtype mismatch: {loaded.dtype}")
        if not np.all((loaded == 0) | (loaded == 1)):
            raise ValueError(f"record {index} geometry is not binary uint8")
        actual_hash = canonical_content_hash(loaded)
        declared_hash = _declared_canonical_hash(record)
        if declared_hash != actual_hash:
            raise ValueError(f"record {index} canonical hash mismatch: declared {declared_hash}, got {actual_hash}")
        voxel_hash = _declared_voxel_file_hash(record)
        actual_voxel_hash = _file_sha256(resolved_geometry)
        if voxel_hash is None or voxel_hash != actual_voxel_hash:
            raise ValueError(
                f"record {index} voxel file hash mismatch: declared {voxel_hash}, got {actual_voxel_hash}"
            )
        if resolved_geometry.stem != actual_hash:
            raise ValueError(f"record {index} content-addressed filename mismatch: {resolved_geometry.name}")
        if actual_hash in unique_hashes:
            raise ValueError(f"record {index} duplicates canonical geometry hash {actual_hash}")
        unique_hashes.add(actual_hash)
        shape_counts["x".join(str(edge) for edge in expected_shape)] += 1
        dtype_counts["uint8"] += 1
        split_counts[str(record.get("split"))] += 1

    original_parent_splits: dict[str, str] = {}
    original_split_counts: Counter[str] = Counter()
    descendants_by_parent_split: Counter[str] = Counter()
    cross_split_violations = 0
    procedural_train_count = 0
    for index, record in enumerate(records):
        source_type = record.get("source_type")
        if source_type == "original":
            split = record.get("split")
            if not isinstance(split, str) or not split:
                raise ValueError(f"original record {index} has no split for parent isolation")
            original_parent_splits[str(record["source_id"])] = split
            original_split_counts[split] += 1
        elif source_type == "perturbation_expanded":
            parent_source_id = record.get("parent_source_id")
            if parent_source_id not in original_parent_splits:
                raise ValueError(f"perturbation record {index} references unknown original parent {parent_source_id}")
            parent_split = original_parent_splits[parent_source_id]
            descendants_by_parent_split[parent_split] += 1
            if record.get("split") != parent_split:
                cross_split_violations += 1
                raise ValueError(
                    f"perturbation record {index} split {record.get('split')} does not inherit parent split {parent_split}"
                )
        elif source_type == "procedural":
            if record.get("split") != "train":
                raise ValueError(f"procedural record {index} must remain in train")
            procedural_train_count += 1

    parent_split_counts = {
        "original_parents_by_split": dict(original_split_counts),
        "descendants_by_parent_split": dict(descendants_by_parent_split),
        "procedural_train": procedural_train_count,
        "cross_split_violations": cross_split_violations,
    }

    _assert_no_reparse_tree(voxel_root)
    actual_files = {
        path.relative_to(output_root).as_posix()
        for path in voxel_root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"published voxel file set mismatch: missing={missing[:3]}, extra={extra[:3]}")

    basic_report = validate_manifest_file(
        str(manifest_path), level="basic", unique_geometry_target=unique_geometry_target
    )
    claim_report = validate_manifest_file(
        str(manifest_path), level="claim-bearing", unique_geometry_target=unique_geometry_target
    )
    if basic_report["status"] != "pass":
        raise ValueError(f"basic manifest validation failed: {basic_report.get('errors', [])[:3]}")
    if claim_report["status"] != "pass":
        raise ValueError(f"claim-bearing manifest validation failed: {claim_report.get('errors', [])[:3]}")
    if len(unique_hashes) < unique_geometry_target:
        raise ValueError(f"unique geometry count {len(unique_hashes)} is below target {unique_geometry_target}")

    return {
        "record_count": len(records),
        "unique_geometry_count": len(unique_hashes),
        "shape_counts": dict(shape_counts),
        "dtype_counts": dict(dtype_counts),
        "split_counts": dict(split_counts),
        "parent_split_counts": parent_split_counts,
        "basic_validation": _validation_summary(basic_report),
        "claim_validation": _validation_summary(claim_report),
    }


def _git_commit_identity() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(MODULE_DIR.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to record the builder git commit identity") from exc
    commit = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"git returned an invalid builder commit identity: {commit!r}")
    return commit


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for distribution in ("scipy", "torch"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"unable to record dependency version for {distribution}") from exc
    return versions


def _storage_metadata(voxel_root: Path) -> dict[str, Any]:
    files = sorted(path for path in voxel_root.glob("*.npy") if path.is_file())
    return {
        "geometry_format": "NumPy NPY 1.0",
        "logical_geometry_files": len(files),
        "logical_geometry_bytes": sum(path.stat().st_size for path in files),
        "sparse_allocation": "best effort on Windows; dense fallback elsewhere",
        "filesystem_compression": "not controlled by builder",
    }


def _assert_procedural_family_minimums(stats: Mapping[str, Any], requested_count: int) -> None:
    minimum_total = sum(PROCEDURAL_MIN_ACCEPTED_PER_TYPE.values())
    if requested_count < minimum_total:
        return
    per_type = stats.get("per_type")
    if not isinstance(per_type, Mapping):
        raise ValueError("procedural generator did not return per-family acceptance statistics")
    missing = {
        aircraft_type: minimum
        for aircraft_type, minimum in PROCEDURAL_MIN_ACCEPTED_PER_TYPE.items()
        if int(per_type.get(aircraft_type, 0)) < minimum
    }
    if missing:
        raise ValueError(f"procedural family minimums not met: {missing}")


def _replay_geometry(
    output_root: Path,
    record: Mapping[str, Any],
    expected_hash: str,
    index: int,
    expected_shape: tuple[int, int, int],
) -> None:
    resolved_geometry = _safe_published_geometry_path(output_root, record.get("geometry_path"))
    try:
        loaded = np.load(str(resolved_geometry), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"replay record {index} geometry cannot be loaded: {resolved_geometry}: {exc}") from exc
    if not isinstance(loaded, np.ndarray) or tuple(loaded.shape) != expected_shape:
        raise ValueError(f"replay record {index} geometry shape mismatch")
    if loaded.dtype != np.dtype(np.uint8) or not np.all((loaded == 0) | (loaded == 1)):
        raise ValueError(f"replay record {index} geometry is not binary uint8")
    actual_hash = canonical_content_hash(loaded)
    if actual_hash != expected_hash or resolved_geometry.stem != expected_hash:
        raise ValueError(
            f"replay record {index} content mismatch: expected {expected_hash}, got {actual_hash}"
        )


def _compare_replay_record(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    output_root: Path,
    index: int,
    expected_shape: tuple[int, int, int],
) -> None:
    if dict(actual) != dict(expected):
        differing_keys = sorted(
            key for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)
        )
        raise ValueError(f"replay record {index} identity mismatch in keys: {differing_keys[:5]}")
    _replay_geometry(
        output_root,
        actual,
        expected["canonical_content_sha256"],
        index,
        expected_shape,
    )


def replay_published_corpus(
    output_dir: Path,
    source_manifest: Path,
    *,
    perturbation_batches: Sequence[Sequence[str]] | None = None,
    procedural_count: int | None = None,
    procedural_seed: int | None = None,
    expected_original_count: int | None = None,
    expected_perturbation_count: int | None = None,
    expected_procedural_count: int | None = None,
    expected_total_count: int | None = None,
    unique_geometry_target: int | None = None,
) -> dict[str, Any]:
    """Replay every identity and content hash in place, without a second corpus copy."""
    output_root = _lexical_absolute_path(output_dir, role="published output")
    _assert_no_reparse_components(output_root, role="published output")
    manifest_path = output_root / "combined_training_manifest.jsonl"
    build_spec_path = output_root / "build_spec.json"
    if not manifest_path.is_file() or not build_spec_path.is_file():
        raise FileNotFoundError("published corpus is missing its manifest or build specification")

    source_manifest = Path(source_manifest).resolve()
    build_spec = json.loads(build_spec_path.read_text(encoding="utf-8"))
    declared_shape = build_spec.get("grid_shape")
    if (
        not isinstance(declared_shape, list)
        or len(declared_shape) != 3
        or len(set(declared_shape)) != 1
        or not isinstance(declared_shape[0], int)
        or declared_shape[0] <= 0
    ):
        raise ValueError(f"published build specification has invalid grid_shape: {declared_shape!r}")
    expected_shape = tuple(declared_shape)
    target_grid_size = int(expected_shape[0])
    published_batches = build_spec.get("perturbation_batches")
    if not isinstance(published_batches, list):
        raise ValueError("published build spec is missing perturbation_batches")
    batches = _normalize_batches(
        published_batches if perturbation_batches is None else perturbation_batches
    )
    procedural_count = int(
        build_spec.get("procedural_count") if procedural_count is None else procedural_count
    )
    procedural_seed = int(
        build_spec.get("procedural_seed") if procedural_seed is None else procedural_seed
    )
    if procedural_count < 0:
        raise ValueError("procedural_count must be non-negative")
    published_counts = build_spec.get("expected_counts")
    if not isinstance(published_counts, dict):
        raise ValueError("published build spec is missing expected_counts")
    expected_original_count = int(
        published_counts.get("original") if expected_original_count is None else expected_original_count
    )
    expected_perturbation_count = int(
        published_counts.get("perturbation")
        if expected_perturbation_count is None
        else expected_perturbation_count
    )
    expected_procedural_count = int(
        published_counts.get("procedural")
        if expected_procedural_count is None
        else expected_procedural_count
    )
    expected_total_count = int(
        published_counts.get("total") if expected_total_count is None else expected_total_count
    )
    _check_expected_counts(
        expected_original_count=expected_original_count,
        expected_perturbation_count=expected_perturbation_count,
        expected_procedural_count=expected_procedural_count,
        expected_total_count=expected_total_count,
    )
    if unique_geometry_target is not None and unique_geometry_target != expected_total_count:
        raise ValueError(
            "unique_geometry_target must equal expected_total_count for a full deterministic replay"
        )
    expected_spec = {
        "builder_version": BUILDER_VERSION,
        "source_manifest_sha256": _file_sha256(source_manifest),
        "perturbation_batches": [list(batch) for batch in batches],
        "procedural_count": procedural_count,
        "procedural_seed": procedural_seed,
        "target_grid_size": target_grid_size,
        "expected_counts": {
            "original": expected_original_count,
            "perturbation": expected_perturbation_count,
            "procedural": expected_procedural_count,
            "total": expected_total_count,
        },
    }
    for key, expected_value in expected_spec.items():
        if build_spec.get(key) != expected_value:
            raise ValueError(f"build spec mismatch for {key}: {build_spec.get(key)!r} != {expected_value!r}")
    if build_spec.get("builder_commit") != _git_commit_identity():
        raise ValueError("published build spec builder_commit does not match the current source commit")
    if build_spec.get("dependency_versions") != _dependency_versions():
        raise ValueError("published build spec dependency_versions do not match the replay runtime")

    records = load_jsonl_manifest(manifest_path)
    if len(records) != expected_total_count:
        raise ValueError(f"published record count {len(records)} does not equal expected {expected_total_count}")
    source_entries = preflight_source_records(
        source_manifest,
        expected_original_count=expected_original_count,
    )
    replay_index = 0
    seen_hashes: set[str] = set()
    replay_counts: Counter[str] = Counter()
    parent_split_counts: Counter[str] = Counter()

    for entry in source_entries:
        parent = entry["record"]
        source_output = resample_cubic_voxels(
            _load_canonical_file(entry["source_path"]), target_grid_size
        )
        output_hash = canonical_content_hash(source_output)
        parent_output = _build_original_record(
            parent,
            source_manifest_name=source_manifest.name,
            source_manifest_hash=build_spec["source_manifest_sha256"],
            source_record_index=entry["source_record_index"],
            content_hash=output_hash,
            voxel_file_hash=_canonical_npy_sha256(source_output),
            geometry_path=f"voxels/{output_hash}.npy",
        )
        _compare_replay_record(
            records[replay_index],
            parent_output,
            output_root=output_root,
            index=replay_index,
            expected_shape=expected_shape,
        )
        replay_index += 1
        replay_counts["original"] += 1
        parent_split_counts[str(parent.get("split"))] += 0
        if output_hash in seen_hashes:
            raise ValueError(f"resampling produced duplicate original geometry {output_hash}")
        seen_hashes.add(output_hash)

    per_transform: dict[str, dict[str, int]] = {}
    for batch in batches:
        for transform in batch:
            per_transform.setdefault(
                transform,
                {"candidates": 0, "accepted": 0, "rejected_invalid": 0, "rejected_duplicate": 0},
            )
        for entry in source_entries:
            parent = entry["record"]
            source_voxels = _load_canonical_file(entry["source_path"])
            published_parent_hash = canonical_content_hash(
                resample_cubic_voxels(source_voxels, target_grid_size)
            )
            for transform, transformed_source, _source_child_hash in iter_transform_candidates(source_voxels, batch):
                transformed = resample_cubic_voxels(transformed_source, target_grid_size)
                child_hash = canonical_content_hash(transformed)
                transform_stats = per_transform[transform]
                transform_stats["candidates"] += 1
                if child_hash in seen_hashes:
                    transform_stats["rejected_duplicate"] += 1
                    continue
                if not validate_perturbation(transformed_source):
                    transform_stats["rejected_invalid"] += 1
                    continue
                expected_record = build_perturbation_record(
                    parent,
                    transform=transform,
                    parent_record_index=entry["source_record_index"],
                    parent_hash=published_parent_hash,
                    child_hash=child_hash,
                    voxel_file_hash=_canonical_npy_sha256(transformed),
                    geometry_path=f"voxels/{child_hash}.npy",
                )
                _compare_replay_record(
                    records[replay_index],
                    expected_record,
                    output_root=output_root,
                    index=replay_index,
                    expected_shape=expected_shape,
                )
                replay_index += 1
                replay_counts["perturbation"] += 1
                per_transform[transform]["accepted"] += 1
                parent_split_counts[str(parent.get("split"))] += 1
                seen_hashes.add(child_hash)

    procedural_stats: dict[str, Any] = {}
    procedural_seen_hashes = set(seen_hashes)
    for sample in iter_procedural_samples(
        procedural_count,
        procedural_seed,
        seen_hashes=procedural_seen_hashes,
        stats=procedural_stats,
    ):
        procedural_voxels = resample_cubic_voxels(sample["voxels"], target_grid_size)
        child_hash = canonical_content_hash(procedural_voxels)
        if child_hash in seen_hashes:
            raise ValueError(f"resampling produced duplicate procedural geometry {child_hash}")
        seen_hashes.add(child_hash)
        expected_record = build_procedural_record(
            aircraft_type=sample["aircraft_type"],
            accepted_index=sample["accepted_index"],
            attempt=sample["attempt"],
            seed=sample["seed"],
            child_hash=child_hash,
            voxel_file_hash=_canonical_npy_sha256(procedural_voxels),
            geometry_path=f"voxels/{child_hash}.npy",
        )
        _compare_replay_record(
            records[replay_index],
            expected_record,
            output_root=output_root,
            index=replay_index,
            expected_shape=expected_shape,
        )
        replay_index += 1
        replay_counts["procedural"] += 1
    _assert_procedural_family_minimums(procedural_stats, procedural_count)

    if replay_index != len(records) or len(seen_hashes) != expected_total_count:
        raise ValueError(
            f"replay count/uniqueness mismatch: records={replay_index}, unique={len(seen_hashes)}, expected={expected_total_count}"
        )
    for category, expected_count in {
        "original": expected_original_count,
        "perturbation": expected_perturbation_count,
        "procedural": expected_procedural_count,
    }.items():
        if replay_counts[category] != expected_count:
            raise ValueError(f"replay {category} count {replay_counts[category]} != expected {expected_count}")
    return {
        "status": "pass",
        "record_count": replay_index,
        "recomputed_record_count": replay_index,
        "recomputed_geometry_count": len(seen_hashes),
        "counts": dict(replay_counts),
        "per_transform": per_transform,
        "procedural": procedural_stats,
        "parent_split_counts": {
            "descendants_by_parent_split": dict(parent_split_counts),
            "cross_split_violations": 0,
        },
    }


def _normalize_batches(perturbation_batches: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    normalized: list[tuple[str, ...]] = []
    for batch_index, batch in enumerate(perturbation_batches):
        if isinstance(batch, str):
            raise ValueError(f"perturbation batch {batch_index} must be a sequence, not a string")
        transforms = tuple(str(transform).strip() for transform in batch if str(transform).strip())
        for transform in transforms:
            if transform not in TRANSFORMS:
                raise ValueError(f"unknown perturbation transform: {transform}")
        normalized.append(transforms)
    return tuple(normalized)


def _check_expected_counts(
    *,
    expected_original_count: int,
    expected_perturbation_count: int,
    expected_procedural_count: int,
    expected_total_count: int,
) -> None:
    expected_sum = expected_original_count + expected_perturbation_count + expected_procedural_count
    if expected_total_count != expected_sum:
        raise ValueError(
            "expected_total_count must equal expected_original_count + expected_perturbation_count "
            f"+ expected_procedural_count, got {expected_total_count} vs {expected_sum}"
        )
    if any(value < 0 for value in (expected_original_count, expected_perturbation_count, expected_procedural_count)):
        raise ValueError("expected record counts must be non-negative")


def rebuild_final_training_corpus(
    source_manifest: Path,
    output_dir: Path,
    *,
    perturbation_batches: Sequence[Sequence[str]] = DEFAULT_PERTURBATION_BATCHES,
    procedural_count: int = DEFAULT_EXPECTED_PROCEDURAL_COUNT,
    procedural_seed: int = 42,
    expected_original_count: int = DEFAULT_EXPECTED_ORIGINAL_COUNT,
    expected_perturbation_count: int = DEFAULT_EXPECTED_PERTURBATION_COUNT,
    expected_procedural_count: int = DEFAULT_EXPECTED_PROCEDURAL_COUNT,
    expected_total_count: int = DEFAULT_EXPECTED_TOTAL_COUNT,
    target_grid_size: int = SOURCE_GRID_SHAPE[0],
) -> dict[str, Any]:
    """Rebuild and atomically publish the complete final corpus."""
    source_manifest = Path(source_manifest).resolve()
    output_dir = _safe_output_target(output_dir)
    _check_expected_counts(
        expected_original_count=expected_original_count,
        expected_perturbation_count=expected_perturbation_count,
        expected_procedural_count=expected_procedural_count,
        expected_total_count=expected_total_count,
    )
    if procedural_count < 0:
        raise ValueError("procedural_count must be non-negative")
    target_grid_size = int(target_grid_size)
    if target_grid_size <= 0:
        raise ValueError("target_grid_size must be positive")
    target_shape = (target_grid_size,) * 3
    batches = _normalize_batches(perturbation_batches)

    output_parent = output_dir.parent
    staging_dir: Path | None = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(output_parent)))
    _assert_no_reparse_components(staging_dir, role="staging")
    try:
        source_manifest_hash = _file_sha256(source_manifest)
        source_entries = preflight_source_records(
            source_manifest,
            expected_original_count=expected_original_count,
        )
        records: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        dropped_counts: Counter[str] = Counter()

        for entry in source_entries:
            source_record_index = entry["source_record_index"]
            parent = entry["record"]
            source_path = resolve_geometry_path(parent, source_manifest)
            canonical = _load_canonical_file(source_path)
            source_hash = canonical_content_hash(canonical)
            if source_hash != entry["source_hash"]:
                raise ValueError(f"source record {source_record_index} changed during rebuild")
            canonical = resample_cubic_voxels(canonical, target_grid_size)
            actual_hash = canonical_content_hash(canonical)
            if actual_hash in seen_hashes:
                raise ValueError(f"resampling produced duplicate original geometry {actual_hash}")
            geometry_path, voxel_file_hash = _write_canonical_voxel(
                canonical, actual_hash, staging_dir / "voxels"
            )
            output_record = _build_original_record(
                parent,
                source_manifest_name=source_manifest.name,
                source_manifest_hash=source_manifest_hash,
                source_record_index=source_record_index,
                content_hash=actual_hash,
                voxel_file_hash=voxel_file_hash,
                geometry_path=geometry_path,
            )
            records.append(output_record)
            seen_hashes.add(actual_hash)

        per_transform: dict[str, dict[str, int]] = {}
        batch_counts: dict[str, dict[str, Any]] = {}
        for batch_index, batch in enumerate(batches, start=1):
            batch_key = f"batch_{batch_index}"
            batch_stats: dict[str, Any] = {
                "transforms": list(batch),
                "candidates": 0,
                "accepted": 0,
                "rejected_invalid": 0,
                "rejected_duplicate": 0,
            }
            for transform in batch:
                per_transform.setdefault(
                    transform,
                    {"candidates": 0, "accepted": 0, "rejected_invalid": 0, "rejected_duplicate": 0},
                )
            for entry in source_entries:
                parent = entry["record"]
                source_voxels = _load_canonical_file(entry["source_path"])
                published_parent_hash = canonical_content_hash(
                    resample_cubic_voxels(source_voxels, target_grid_size)
                )
                for transform, transformed_source, _source_child_hash in iter_transform_candidates(source_voxels, batch):
                    transformed = resample_cubic_voxels(transformed_source, target_grid_size)
                    child_hash = canonical_content_hash(transformed)
                    batch_stats["candidates"] += 1
                    transform_stats = per_transform[transform]
                    transform_stats["candidates"] += 1
                    if child_hash in seen_hashes:
                        batch_stats["rejected_duplicate"] += 1
                        transform_stats["rejected_duplicate"] += 1
                        dropped_counts["duplicate_canonical_geometry"] += 1
                        continue
                    if not validate_perturbation(transformed_source):
                        batch_stats["rejected_invalid"] += 1
                        transform_stats["rejected_invalid"] += 1
                        continue
                    geometry_path, voxel_file_hash = _write_canonical_voxel(
                        transformed, child_hash, staging_dir / "voxels"
                    )
                    generated_record = build_perturbation_record(
                        parent,
                        transform=transform,
                        parent_record_index=entry["source_record_index"],
                        parent_hash=published_parent_hash,
                        child_hash=child_hash,
                        voxel_file_hash=voxel_file_hash,
                        geometry_path=geometry_path,
                    )
                    records.append(generated_record)
                    seen_hashes.add(child_hash)
                    batch_stats["accepted"] += 1
                    transform_stats["accepted"] += 1
            batch_counts[batch_key] = batch_stats

        procedural_stats: dict[str, Any] = {}
        procedural_seen_hashes = set(seen_hashes)
        for sample in iter_procedural_samples(
            procedural_count,
            procedural_seed,
            seen_hashes=procedural_seen_hashes,
            stats=procedural_stats,
        ):
            procedural_voxels = resample_cubic_voxels(sample["voxels"], target_grid_size)
            child_hash = canonical_content_hash(procedural_voxels)
            if child_hash in seen_hashes:
                raise ValueError(f"resampling produced duplicate procedural geometry {child_hash}")
            geometry_path, voxel_file_hash = _write_canonical_voxel(
                procedural_voxels, child_hash, staging_dir / "voxels"
            )
            records.append(
                build_procedural_record(
                    aircraft_type=sample["aircraft_type"],
                    accepted_index=sample["accepted_index"],
                    attempt=sample["attempt"],
                    seed=sample["seed"],
                    child_hash=child_hash,
                    voxel_file_hash=voxel_file_hash,
                    geometry_path=geometry_path,
                )
            )
            seen_hashes.add(child_hash)
        _assert_procedural_family_minimums(procedural_stats, procedural_count)
        dropped_counts["duplicate_canonical_geometry"] += int(procedural_stats.get("rejected_duplicate", 0))

        admitted_counts = {
            "original": len(source_entries),
            "perturbation": sum(int(stats["accepted"]) for stats in per_transform.values()),
            "procedural": int(procedural_stats.get("accepted", 0)),
        }
        if admitted_counts["original"] != expected_original_count:
            raise ValueError(f"original record count {admitted_counts['original']} does not equal expected {expected_original_count}")
        if admitted_counts["perturbation"] != expected_perturbation_count:
            raise ValueError(
                f"perturbation record count {admitted_counts['perturbation']} does not equal expected {expected_perturbation_count}"
            )
        if admitted_counts["procedural"] != expected_procedural_count:
            raise ValueError(
                f"procedural record count {admitted_counts['procedural']} does not equal expected {expected_procedural_count}"
            )
        if procedural_stats.get("accepted", 0) != procedural_count:
            raise ValueError(
                f"procedural generator accepted {procedural_stats.get('accepted', 0)} of requested {procedural_count}"
            )
        if len(records) != expected_total_count or len(seen_hashes) != expected_total_count:
            raise ValueError(
                f"final record/unique count {len(records)}/{len(seen_hashes)} does not equal expected {expected_total_count}"
            )

        manifest_path = staging_dir / "combined_training_manifest.jsonl"
        _write_jsonl(manifest_path, records)
        build_spec = {
            "builder_version": BUILDER_VERSION,
            "source_manifest_name": source_manifest.name,
            "source_manifest_sha256": source_manifest_hash,
            "builder_commit": _git_commit_identity(),
            "dependency_versions": _dependency_versions(),
            "output_manifest_name": "combined_training_manifest.jsonl",
            "source_grid_shape": list(SOURCE_GRID_SHAPE),
            "target_grid_size": target_grid_size,
            "grid_shape": list(target_shape),
            "voxel_dtype": "uint8",
            "geometry_naming": "voxels/<canonical_content_sha256>.npy",
            "perturbation_batches": [list(batch) for batch in batches],
            "procedural_count": procedural_count,
            "procedural_seed": procedural_seed,
            "expected_counts": {
                "original": expected_original_count,
                "perturbation": expected_perturbation_count,
                "procedural": expected_procedural_count,
                "total": expected_total_count,
            },
            "generator_versions": {
                "perturbation": PERTURBATION_GENERATOR_VERSION,
                "procedural": PROCEDURAL_GENERATOR_VERSION,
            },
            "conditioning_policy": "complete null design_spec with false field availability; generated records are unconditioned",
            "split_policy": "perturbation descendants inherit parent split; procedural records are train",
            "storage": _storage_metadata(staging_dir / "voxels"),
        }
        _assert_no_absolute_paths(build_spec, context="build_spec")
        _write_json(staging_dir / "build_spec.json", build_spec)

        replay = replay_published_corpus(
            staging_dir,
            source_manifest,
            perturbation_batches=batches,
            procedural_count=procedural_count,
            procedural_seed=procedural_seed,
            expected_original_count=expected_original_count,
            expected_perturbation_count=expected_perturbation_count,
            expected_procedural_count=expected_procedural_count,
            expected_total_count=expected_total_count,
            unique_geometry_target=expected_total_count,
        )
        audit = validate_published_corpus(
            staging_dir,
            unique_geometry_target=expected_total_count,
            expected_total_count=expected_total_count,
        )
        _write_json(staging_dir / "validation" / "basic.json", audit["basic_validation"])
        _write_json(staging_dir / "validation" / "claim-bearing.json", audit["claim_validation"])

        output_manifest_hash = _file_sha256(manifest_path)
        report = {
            "builder_version": BUILDER_VERSION,
            "source_manifest_name": source_manifest.name,
            "source_manifest_sha256": source_manifest_hash,
            "output_manifest_name": "combined_training_manifest.jsonl",
            "output_manifest_sha256": output_manifest_hash,
            "record_count": len(records),
            "unique_geometry_count": len(seen_hashes),
            "source_counts": {
                "original_input": len(source_entries),
                "original_admitted": len(source_entries),
            },
            "batch_counts": {
                **batch_counts,
                "perturbation_candidates": sum(int(stats["candidates"]) for stats in batch_counts.values()),
                "perturbation_accepted": admitted_counts["perturbation"],
            },
            "per_transform": per_transform,
            "procedural": procedural_stats,
            "admitted_counts": admitted_counts,
            "dropped_counts": dict(dropped_counts),
            "split_counts": audit["split_counts"],
            "shape_counts": audit["shape_counts"],
            "dtype_counts": audit["dtype_counts"],
            "storage": build_spec["storage"],
            "deterministic_replay": replay,
            "parent_split_counts": audit["parent_split_counts"],
            "basic_validation": audit["basic_validation"],
            "claim_validation": audit["claim_validation"],
            "claim_boundary": (
                "Original records retain source provenance. Perturbation and procedural records are deterministic "
                "generated voxel variants, not independent CAD, aerodynamic measurements, mission labels, or "
                "manufacturing ground truth."
            ),
            "scientific_caveats": [
                "All generated records use complete-null conditioning metadata and are unconditioned source-metadata-only inputs.",
                "Every perturbation descendant inherits its original parent split; parent-grouped counts are recorded so evaluation families cannot be mistaken for independent samples.",
            ],
        }
        _assert_no_absolute_paths(report, context="report")
        _write_json(staging_dir / "report.json", report)

        os.replace(str(staging_dir), str(output_dir))
        _assert_no_reparse_components(output_dir, role="published output")
        staging_dir = None
        return report
    finally:
        if staging_dir is not None and staging_dir.exists():
            try:
                _safe_cleanup_staging(staging_dir, output_parent)
            except Exception:
                # Never mask the build failure or recurse after ownership is uncertain.
                pass


def _parse_batches(values: Sequence[str] | None) -> tuple[tuple[str, ...], ...]:
    if values is None:
        return DEFAULT_PERTURBATION_BATCHES
    return tuple(
        tuple(transform.strip() for transform in value.split(",") if transform.strip())
        for value in values
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--perturb-batch", action="append", default=None)
    parser.add_argument("--procedural-count", type=int, default=DEFAULT_EXPECTED_PROCEDURAL_COUNT)
    parser.add_argument("--procedural-seed", type=int, default=42)
    parser.add_argument("--target-grid-size", type=int, default=SOURCE_GRID_SHAPE[0])
    parser.add_argument("--expected-originals", type=int, default=DEFAULT_EXPECTED_ORIGINAL_COUNT)
    parser.add_argument("--expected-perturbations", type=int, default=DEFAULT_EXPECTED_PERTURBATION_COUNT)
    parser.add_argument("--expected-procedural", type=int, default=DEFAULT_EXPECTED_PROCEDURAL_COUNT)
    parser.add_argument("--expected-total", type=int, default=DEFAULT_EXPECTED_TOTAL_COUNT)
    args = parser.parse_args(argv)
    report = rebuild_final_training_corpus(
        Path(args.source_manifest),
        Path(args.output_dir),
        perturbation_batches=_parse_batches(args.perturb_batch),
        procedural_count=args.procedural_count,
        procedural_seed=args.procedural_seed,
        target_grid_size=args.target_grid_size,
        expected_original_count=args.expected_originals,
        expected_perturbation_count=args.expected_perturbations,
        expected_procedural_count=args.expected_procedural,
        expected_total_count=args.expected_total,
    )
    print(
        json.dumps(
            {
                "record_count": report["record_count"],
                "unique_geometry_count": report["unique_geometry_count"],
                "admitted_counts": report["admitted_counts"],
                "claim_validation": report["claim_validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
