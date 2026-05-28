#!/usr/bin/env python3
"""Validate grounded dataset manifests for smoke wiring or claim-bearing workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


ALLOWED_LEVELS = {"basic", "claim-bearing"}
ALLOWED_SPLITS = {"train", "val", "validation", "holdout", "test"}
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

    status = "pass" if not errors else "blocked"
    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "level": level,
        "status": status,
        "record_count": len(records),
        "errors": errors,
        "warnings": warnings,
        "required_design_spec_fields": required_design_fields,
        "allowed_splits": sorted(ALLOWED_SPLITS),
    }


def validate_manifest_file(manifest_path: str, *, level: str = "basic") -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    records = _load_structured_records(path)
    return validate_manifest_records(records, manifest_path=str(path), level=level)


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
    args = parser.parse_args()

    report = validate_manifest_file(args.manifest, level=args.level)
    rendered = json.dumps(report, indent=2)
    print(rendered)

    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
