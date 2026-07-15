#!/usr/bin/env python3
"""Validate grounded dataset manifests for smoke wiring or claim-bearing workflows.

The claim-bearing level follows dataset-documentation expectations from
Datasheets for Datasets: motivation, composition, collection/preprocessing, and
recommended uses must be explicit before downstream claims are credible.
https://doi.org/10.48550/arXiv.1803.09010
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from report_metadata import apply_report_metadata


ALLOWED_LEVELS = {"basic", "claim-bearing"}
ALLOWED_SPLITS = {"train", "val", "validation", "holdout", "test"}
DEFAULT_UNIQUE_GEOMETRY_TARGET = 600
SCHEMA_PATH = Path(__file__).resolve().parent / "conditioning_schema.yaml"


def _load_structured_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must contain JSON objects")
                records.append(payload)
        return records

    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("records", payload))

    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of sample records")

    records: List[Dict[str, Any]] = []
    for idx, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"{path} record {idx} must be an object")
        records.append(record)
    return records


def _required_design_spec_fields() -> List[str]:
    payload = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    scalar_features = [feature["name"] for feature in payload.get("scalar_features", [])]
    categorical_features = list((payload.get("categorical_features") or {}).keys())
    return scalar_features + categorical_features


def validate_manifest_records(
    records: List[Dict[str, Any]],
    *,
    manifest_path: str,
    level: str = "basic",
    unique_geometry_target: int = DEFAULT_UNIQUE_GEOMETRY_TARGET,
) -> Dict[str, Any]:
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"level must be one of {sorted(ALLOWED_LEVELS)}")

    errors: List[str] = []
    warnings: List[str] = []
    base_dir = Path(manifest_path).resolve().parent
    required_design_fields = _required_design_spec_fields()

    if not records:
        errors.append("Manifest contains zero records.")

    for idx, record in enumerate(records):
        record_prefix = f"record {idx}"

        geometry_ref = record.get("geometry_path") or record.get("stl_path")
        if not geometry_ref:
            errors.append(f"{record_prefix}: missing geometry_path or stl_path")
        else:
            resolved_path = (base_dir / geometry_ref).resolve()
            if not resolved_path.exists():
                errors.append(f"{record_prefix}: referenced geometry file does not exist: {geometry_ref}")

        split = record.get("split")
        if not split:
            errors.append(f"{record_prefix}: missing split")
        elif str(split) not in ALLOWED_SPLITS:
            errors.append(f"{record_prefix}: unsupported split {split!r}")

        if level != "claim-bearing":
            continue

        for key in ("source_id", "geometry_provenance", "preprocessing_version", "units", "design_family"):
            if not record.get(key):
                errors.append(f"{record_prefix}: missing required claim-bearing field {key}")

        design_spec = record.get("design_spec")
        if not isinstance(design_spec, dict):
            errors.append(f"{record_prefix}: missing required claim-bearing design_spec object")
            continue

        for field_name in required_design_fields:
            if field_name not in design_spec:
                errors.append(f"{record_prefix}: design_spec missing required field {field_name}")

    geometry_identities = {
        str(
            record.get("geometry_sha256")
            or record.get("voxel_sha256")
            or record.get("geometry_variant_id")
            or record.get("geometry_path")
            or record.get("stl_path")
            or record.get("source_id")
            or f"record-{idx}"
        )
        for idx, record in enumerate(records)
    }
    unique_geometry_count = len(geometry_identities)
    duplicate_geometry_record_count = len(records) - unique_geometry_count
    unique_geometry_target_met = unique_geometry_count >= unique_geometry_target

    if errors:
        status = "blocked"
    elif level == "claim-bearing" and not unique_geometry_target_met:
        status = "fail"
    else:
        status = "pass"
    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "level": level,
        "status": status,
        "record_count": len(records),
        "unique_geometry_count": unique_geometry_count,
        "duplicate_geometry_record_count": duplicate_geometry_record_count,
        "unique_geometry_target": unique_geometry_target,
        "unique_geometry_target_met": unique_geometry_target_met,
        "errors": errors,
        "warnings": warnings,
        "required_design_spec_fields": required_design_fields,
        "allowed_splits": sorted(ALLOWED_SPLITS),
    }


def validate_manifest_file(
    manifest_path: str,
    *,
    level: str = "basic",
    unique_geometry_target: int = DEFAULT_UNIQUE_GEOMETRY_TARGET,
) -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    records = _load_structured_records(path)
    return validate_manifest_records(
        records,
        manifest_path=str(path),
        level=level,
        unique_geometry_target=unique_geometry_target,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a grounded dataset manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a manifest file (.json, .jsonl, .yaml)")
    parser.add_argument(
        "--level",
        default="basic",
        choices=sorted(ALLOWED_LEVELS),
        help="Validation level: basic wiring checks or stricter claim-bearing checks.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report output path.")
    parser.add_argument(
        "--unique-geometry-target",
        type=int,
        default=DEFAULT_UNIQUE_GEOMETRY_TARGET,
        help="Minimum distinct geometries required for claim-bearing validation.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run identifier shared across report artifacts.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path for evidence lineage metadata.")
    parser.add_argument("--protocol-config", default=None, help="Optional protocol config path for evidence lineage metadata.")
    args = parser.parse_args()

    report = validate_manifest_file(
        args.manifest,
        level=args.level,
        unique_geometry_target=args.unique_geometry_target,
    )
    apply_report_metadata(
        report,
        run_id=args.run_id,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        protocol_path=args.protocol_config,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
