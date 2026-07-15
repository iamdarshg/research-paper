"""Filter a grounded manifest to records passing aircraft-validity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from aircraft_validity import canonicalize_aircraft_voxels, evaluate_aircraft_validity


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _geometry_path(record: Dict[str, Any], base_dir: Path) -> Path:
    geometry_ref = record.get("geometry_path")
    if not geometry_ref:
        raise ValueError("record is missing geometry_path; STL-only manifests are not supported by this filter")
    return (base_dir / str(geometry_ref)).resolve()


def _record_for_output_manifest(record: Dict[str, Any], geometry_path: Path, output_manifest: Path) -> Dict[str, Any]:
    output_record = dict(record)
    output_record["geometry_path"] = os.path.relpath(
        geometry_path.resolve(),
        output_manifest.resolve().parent,
    ).replace("\\", "/")
    return output_record


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_stats(sample_reports: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = [
        "occupancy_ratio",
        "symmetry_score",
        "span_fraction_y",
        "length_fraction_x",
        "thickness_fraction_z",
        "center_body_fraction",
        "left_wing_fraction",
        "right_wing_fraction",
        "center_body_density_ratio",
        "longitudinal_profile_cv",
        "tail_fraction",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = sorted(float(report["validity"]["metrics"][key]) for report in sample_reports)
        if not values:
            continue
        last = len(values) - 1
        stats[key] = {
            "min": values[0],
            "p25": values[round(last * 0.25)],
            "median": values[round(last * 0.50)],
            "p75": values[round(last * 0.75)],
            "max": values[-1],
        }
    return stats


def filter_manifest_by_aircraft_validity(
    manifest_path: Path,
    output_manifest: Path,
    output_report: Path,
    canonical_geometry_dir: Path | None = None,
) -> Dict[str, Any]:
    records = _load_jsonl(manifest_path)
    base_dir = manifest_path.resolve().parent
    kept_records: List[Dict[str, Any]] = []
    sample_reports: List[Dict[str, Any]] = []
    failed_checks: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    kept_split_counts: Counter[str] = Counter()
    accepted_canonical_hashes: set[str] = set()
    duplicate_canonical_geometry_count = 0

    for idx, record in enumerate(records):
        split_counts[str(record.get("split", ""))] += 1
        path = _geometry_path(record, base_dir)
        raw_voxels = np.load(path)
        canonical_voxels, canonicalization = canonicalize_aircraft_voxels(raw_voxels)
        validity = evaluate_aircraft_validity(canonical_voxels)
        status = str(validity["status"])
        for check_name in validity["failed_checks"]:
            failed_checks[str(check_name)] += 1

        sample_report = {
            "sample_index": idx,
            "sample_id": record.get("sample_id"),
            "source_id": record.get("source_id"),
            "name": record.get("name") or record.get("display_name"),
            "split": record.get("split"),
            "geometry_path": record.get("geometry_path"),
            "status": status,
            "failed_checks": validity["failed_checks"],
            "validity": validity,
        }
        sample_reports.append(sample_report)

        if status == "pass":
            if canonical_geometry_dir is None:
                output_record = _record_for_output_manifest(record, path, output_manifest)
            else:
                canonical_geometry_dir.mkdir(parents=True, exist_ok=True)
                sample_id = str(record.get("sample_id") or record.get("source_id") or idx)
                destination = canonical_geometry_dir / f"{sample_id}.npy"
                np.save(destination, canonical_voxels.numpy().astype(np.uint8))
                output_record = _record_for_output_manifest(record, destination, output_manifest)
                output_record["voxel_sha256"] = _sha256_file(destination)
                output_record["canonicalization"] = canonicalization
            canonical_hash = hashlib.sha256(
                canonical_voxels.numpy().astype(np.uint8).tobytes()
            ).hexdigest()
            if canonical_hash in accepted_canonical_hashes:
                duplicate_canonical_geometry_count += 1
                sample_report["status"] = "duplicate"
                sample_report["duplicate_canonical_geometry"] = True
                continue
            accepted_canonical_hashes.add(canonical_hash)
            kept_records.append(output_record)
            kept_split_counts[str(record.get("split", ""))] += 1

    _write_jsonl(output_manifest, kept_records)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "canonical_geometry_dir": (
            str(canonical_geometry_dir.resolve()) if canonical_geometry_dir is not None else None
        ),
        "source_record_count": len(records),
        "kept_record_count": len(kept_records),
        "rejected_record_count": len(records) - len(kept_records),
        "duplicate_canonical_geometry_count": duplicate_canonical_geometry_count,
        "source_split_counts": dict(split_counts),
        "kept_split_counts": dict(kept_split_counts),
        "failed_check_counts": dict(failed_checks),
        "metric_stats": _metric_stats(sample_reports),
        "samples": sample_reports,
        "claim_boundary": (
            "Source-geometry screen using the same first-pass aircraft-validity heuristics "
            "as generated-output checks. Passing this filter is not aerodynamic, structural, "
            "or physical validation."
        ),
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input grounded JSONL manifest.")
    parser.add_argument("--output-manifest", required=True, help="Filtered manifest path.")
    parser.add_argument("--output-report", required=True, help="JSON report path.")
    parser.add_argument(
        "--canonical-geometry-dir",
        default=None,
        help="Optional directory where passing grids are persisted in canonical orientation.",
    )
    args = parser.parse_args(argv)

    report = filter_manifest_by_aircraft_validity(
        Path(args.manifest),
        Path(args.output_manifest),
        Path(args.output_report),
        Path(args.canonical_geometry_dir) if args.canonical_geometry_dir else None,
    )
    print(json.dumps({key: report[key] for key in (
        "source_record_count",
        "kept_record_count",
        "rejected_record_count",
        "failed_check_counts",
        "kept_split_counts",
    )}, indent=2, sort_keys=True))
    return 0 if report["kept_record_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
