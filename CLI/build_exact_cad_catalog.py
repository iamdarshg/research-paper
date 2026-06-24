#!/usr/bin/env python3
"""Build a provenance-preserving catalog of exact aircraft CAD sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from CLI.build_airshow_corpus import (
    AIRSHOW_URL,
    LICENSE_NAMES,
    fetch_airshow_config,
    fetch_model_documents,
)


ALLOWED_AIRSHOW_LICENSES = ("1", "2", "3")
HILIFTAEROML_BASE_URL = "https://huggingface.co/datasets/nvidia/HiLiftAeroML/resolve/main"
HILIFTAEROML_SOURCE_PAGE = "https://huggingface.co/datasets/nvidia/HiLiftAeroML"
NASA_UAM_SOURCE_PAGE = "https://www.nasa.gov/reference/uam-refs/"

NASA_UAM_OPENVSP_ARCHIVES: Sequence[Dict[str, str]] = (
    {
        "vehicle": "Tiltduct",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/6-pax-tiltduct-vsp.zip?emrc=3e9bd5",
    },
    {
        "vehicle": "Tiltwing",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/nasa-tiltwing-6pax-vsp.zip?emrc=cf6844",
    },
    {
        "vehicle": "Multi-Tiltrotor",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/nasa-multi-tiltrotor.zip?emrc=f98326",
    },
    {
        "vehicle": "Quadrotor",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/quadrotor-vsp.zip?emrc=847667",
    },
    {
        "vehicle": "Lift + Cruise",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/liftpcruisevsp.zip?emrc=ce58ad",
    },
    {
        "vehicle": "Lift + Cruise Legacy",
        "zip_url": "https://sites-e.larc.nasa.gov/sacd/wp-content/uploads/sites/167/2022/02/LiftPCruiseVSP.zip",
    },
    {
        "vehicle": "Side-by-Side",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/sbs-vsp.zip?emrc=5abdc2",
    },
    {
        "vehicle": "Quiet Single Main Rotor",
        "zip_url": "https://www.nasa.gov/wp-content/uploads/2026/03/qsmr-vsp.zip?emrc=c332aa",
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip().lower())
    return slug.strip("_.-") or "unnamed"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _vsp_url(model: Dict[str, Any]) -> str:
    return str(model.get("newVspUrl") or model.get("vspUrl") or "").strip()


def _preview_url(model: Dict[str, Any]) -> str:
    return str(model.get("newX3dUrl") or model.get("x3dUrl") or "").strip()


def build_airshow_exact_cad_records(
    models: Iterable[Dict[str, Any]],
    *,
    allowed_licenses: Sequence[str] = ALLOWED_AIRSHOW_LICENSES,
    source_page: str = AIRSHOW_URL,
) -> List[Dict[str, Any]]:
    """Return exact OpenVSP source-file records from Airshow model metadata."""
    allowed = {str(item) for item in allowed_licenses}
    records: List[Dict[str, Any]] = []
    for model in models:
        license_id = str(model.get("license", ""))
        exact_url = _vsp_url(model)
        if license_id not in allowed or not exact_url:
            continue
        source_id = str(model.get("id") or _slug(exact_url))
        records.append(
            {
                "source_id": f"airshow_{source_id}",
                "source_collection": "vsp_airshow_public_models",
                "source_page": source_page,
                "exact_cad_url": exact_url,
                "preview_geometry_url": _preview_url(model) or None,
                "file_format": "vsp3",
                "cad_system": "OpenVSP",
                "geometry_kind": "whole_aircraft_or_aircraft_like_parametric_model",
                "name": model.get("name") or model.get("displayName") or source_id,
                "display_name": model.get("displayName"),
                "manufacturer": model.get("manufacturer"),
                "source_license_id": license_id,
                "source_license": LICENSE_NAMES.get(license_id, license_id),
                "license_training_status": "admitted_by_default_catalog_filter",
                "downloads": model.get("downloads"),
                "date": model.get("date"),
                "airshow_firestore_document": model.get("document_name"),
                "candidate_status": "ready_exact_cad_url",
                "claim_boundary": (
                    "Exact OpenVSP source file URL from the public VSP Airshow model document. "
                    "Model names and uploaded metadata are source metadata; the catalog does not "
                    "claim real-aircraft certification accuracy or flight-test validation."
                ),
            }
        )
    return sorted(records, key=lambda item: item["source_id"])


def build_nasa_uam_records(archives: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one record per .vsp3 member found in official NASA UAM ZIP metadata."""
    records: List[Dict[str, Any]] = []
    for archive in archives:
        vehicle = str(archive["vehicle"])
        vehicle_slug = _slug(vehicle)
        members = sorted(
            str(member)
            for member in archive.get("members", [])
            if str(member).lower().endswith(".vsp3")
        )
        for member in members:
            member_slug = _slug(Path(member).with_suffix("").as_posix())
            records.append(
                {
                    "source_id": f"nasa_uam_{vehicle_slug}_{member_slug}",
                    "source_collection": "nasa_uam_reference_vehicles",
                    "source_page": NASA_UAM_SOURCE_PAGE,
                    "archive_url": archive["zip_url"],
                    "archive_member": member,
                    "archive_sha256": archive.get("zip_sha256"),
                    "archive_size_bytes": archive.get("zip_size_bytes"),
                    "exact_cad_url": archive["zip_url"],
                    "file_format": "vsp3",
                    "cad_system": "OpenVSP",
                    "geometry_kind": "nasa_reference_vehicle_parametric_model",
                    "vehicle": vehicle,
                    "source_license": "NASA public reference vehicle release",
                    "license_training_status": "public_nasa_download_usage_to_be_bounded_in_citation",
                    "candidate_status": "ready_exact_cad_archive",
                    "claim_boundary": (
                        "Official NASA UAM reference vehicle OpenVSP archive member. Treat as "
                        "representative reference geometry, not as a production aircraft CAD file."
                    ),
                }
            )
    return sorted(records, key=lambda item: item["source_id"])


def build_hiliftaeroml_records(
    *,
    geometry_count: int = 180,
    canonical_aoa: int = 4,
    base_url: str = HILIFTAEROML_BASE_URL,
) -> List[Dict[str, Any]]:
    """Build canonical STEP records for HiLiftAeroML geometry variants."""
    records: List[Dict[str, Any]] = []
    for index in range(1, geometry_count + 1):
        variant = f"geo_LHC{index:03d}"
        run_id = f"{variant}_AoA_{canonical_aoa}"
        records.append(
            {
                "source_id": f"hiliftaeroml_{variant}",
                "source_collection": "hiliftaeroml_crm_hl_variants",
                "source_page": HILIFTAEROML_SOURCE_PAGE,
                "exact_cad_url": f"{base_url}/{run_id}/{run_id}.stp",
                "surface_mesh_url": f"{base_url}/{run_id}/{run_id}.stl",
                "force_moment_url": f"{base_url}/{run_id}/force_mom_{run_id}.csv",
                "geometry_values_url": f"{base_url}/{run_id}/geo_values_{run_id}.csv",
                "file_format": "stp",
                "cad_system": "STEP surface geometry definition",
                "geometry_kind": "crm_hl_parametric_variant",
                "geometry_variant_id": variant,
                "canonical_aoa_deg": canonical_aoa,
                "available_aoa_degrees": [4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
                "available_flow_solution_count": 10,
                "source_license": "CC-BY-4.0",
                "license_training_status": "permissive_attribution_required",
                "candidate_status": "ready_exact_cad_url_large_payload",
                "claim_boundary": (
                    "Exact STEP CAD URL for a HiLiftAeroML CRM-HL geometry variant. The full "
                    "dataset is large; this catalog stores canonical per-variant CAD URLs and "
                    "does not clone the Hugging Face repository."
                ),
            }
        )
    return records


def build_local_nasa_crm_records(catalog_path: Path) -> List[Dict[str, Any]]:
    if not catalog_path.exists():
        return []
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for source in payload.get("sources", []):
        if not source.get("enabled", True) or source.get("candidate_status") != "ready":
            continue
        source_id = str(source["source_id"])
        records.append(
            {
                "source_id": f"local_{source_id}",
                "source_collection": "local_nasa_crm_ready_catalog",
                "source_page": source.get("source_page"),
                "exact_cad_url": source.get("source_url"),
                "archive_member": source.get("archive_member"),
                "file_format": source.get("file_format"),
                "cad_system": "STEP/IGES/CAD archive as listed in NASA CRM source catalog",
                "geometry_kind": source.get("geometry_kind"),
                "configuration": source.get("configuration"),
                "design_family": source.get("design_family"),
                "source_license": source.get("source_license"),
                "license_training_status": "already_promoted_in_repo_ready_catalog",
                "candidate_status": "ready_local_catalog_record",
                "claim_boundary": source.get("usage_terms_note") or payload.get("claim_boundary"),
            }
        )
    return sorted(records, key=lambda item: item["source_id"])


def summarize_crm_candidate_sweep(candidate_path: Path) -> Dict[str, Any]:
    if not candidate_path.exists():
        return {"candidate_group_count": 0, "ready_group_count": 0, "apparent_record_count": 0}
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    ready = [item for item in candidates if item.get("candidate_status") == "ready"]
    return {
        "candidate_group_count": len(candidates),
        "ready_group_count": len(ready),
        "apparent_record_count": sum(int(item.get("apparent_record_count") or 0) for item in candidates),
        "ready_apparent_record_count": sum(int(item.get("apparent_record_count") or 0) for item in ready),
        "source_path": str(candidate_path),
    }


def inspect_zip_archive(
    session: Any,
    archive: Dict[str, str],
    *,
    verify_tls: bool,
    timeout: int = 60,
) -> Dict[str, Any]:
    response = session.get(archive["zip_url"], timeout=timeout, verify=verify_tls)
    response.raise_for_status()
    payload = response.content
    with zipfile.ZipFile(BytesIO(payload)) as handle:
        members = handle.namelist()
    return {
        **archive,
        "zip_size_bytes": len(payload),
        "zip_sha256": _sha256_bytes(payload),
        "members": members,
    }


def collect_nasa_uam_archive_metadata(
    session: Any,
    archives: Iterable[Dict[str, str]],
    *,
    verify_tls: bool,
) -> Dict[str, Any]:
    inspected: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for archive in archives:
        try:
            inspected.append(inspect_zip_archive(session, archive, verify_tls=verify_tls))
        except Exception as exc:  # noqa: BLE001 - source availability is reported, not fatal.
            failures.append(
                {
                    "vehicle": str(archive.get("vehicle")),
                    "zip_url": str(archive.get("zip_url")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return {"archives": inspected, "failures": failures}


def fetch_airshow_records(
    *,
    source_url: str,
    page_size: int,
    allowed_licenses: Sequence[str],
) -> Dict[str, Any]:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "research-paper-exact-cad-catalog/1.0"})
    config = fetch_airshow_config(session, source_url)
    models = fetch_model_documents(
        session,
        api_key=config["apiKey"],
        project_id=config["projectId"],
        page_size=page_size,
    )
    records = build_airshow_exact_cad_records(
        models,
        allowed_licenses=allowed_licenses,
        source_page=source_url,
    )
    license_counts = Counter(str(model.get("license", "")) for model in models)
    return {
        "records": records,
        "metadata": {
            "all_public_model_documents": len(models),
            "exact_vsp_url_documents": sum(1 for model in models if _vsp_url(model)),
            "license_qualified_exact_vsp_records": len(records),
            "license_counts": dict(sorted(license_counts.items())),
            "firestore_project_id": config.get("projectId"),
            "storage_bucket": config.get("storageBucket"),
        },
    }


def fetch_nasa_uam_records(*, verify_tls: bool) -> Dict[str, Any]:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "research-paper-exact-cad-catalog/1.0"})
    metadata = collect_nasa_uam_archive_metadata(
        session,
        NASA_UAM_OPENVSP_ARCHIVES,
        verify_tls=verify_tls,
    )
    inspected = metadata["archives"]
    records = build_nasa_uam_records(inspected)
    return {
        "records": records,
        "metadata": {
            "archive_count": len(inspected),
            "archive_failure_count": len(metadata["failures"]),
            "vsp3_member_count": len(records),
            "archives": [
                {
                    "vehicle": item["vehicle"],
                    "zip_url": item["zip_url"],
                    "zip_size_bytes": item["zip_size_bytes"],
                    "zip_sha256": item["zip_sha256"],
                    "vsp3_members": [
                        member for member in item["members"] if member.lower().endswith(".vsp3")
                    ],
                }
                for item in inspected
            ],
            "failures": metadata["failures"],
        },
    }


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "record_count": len(records),
        "source_collection_counts": dict(
            sorted(Counter(record["source_collection"] for record in records).items())
        ),
        "file_format_counts": dict(sorted(Counter(record["file_format"] for record in records).items())),
        "license_counts": dict(
            sorted(Counter(str(record.get("source_license") or "unknown") for record in records).items())
        ),
        "candidate_status_counts": dict(
            sorted(Counter(record["candidate_status"] for record in records).items())
        ),
    }


def _markdown_table(mapping: Dict[str, Any], *, key_label: str, value_label: str) -> str:
    lines = [f"| {key_label} | {value_label} |", "| --- | ---: |"]
    lines.extend(f"| `{key}` | {value} |" for key, value in mapping.items())
    return "\n".join(lines)


def render_markdown_report(report: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> str:
    summary = report["summary"]
    crm_candidate = report["source_metadata"].get("nasa_crm_candidate_sweep", {})
    lines = [
        "# Exact CAD Source Sweep",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Summary",
        "",
        f"- Exact CAD catalog records: `{summary['record_count']}`",
        "- This is a source catalog, not a checked-in binary CAD mirror.",
        "- Large assets, especially HiLiftAeroML STEP/STL/flow fields, are intentionally referenced by URL.",
        "- Third-party generic CAD marketplaces are excluded from the claim-bearing lane until license and provenance review.",
        "",
        _markdown_table(summary["source_collection_counts"], key_label="Source collection", value_label="Records"),
        "",
        _markdown_table(summary["file_format_counts"], key_label="CAD format", value_label="Records"),
        "",
        "## Source Lanes",
        "",
        "### VSP Airshow",
        "",
        "The live Airshow sweep catalogs license-qualified exact OpenVSP `.vsp3` URLs. These are exact "
        "parametric model source files, but they remain community/user-contributed models rather than "
        "certified aircraft CAD.",
        "",
        f"- Public model documents observed: `{report['source_metadata']['airshow'].get('all_public_model_documents', 0)}`",
        f"- Documents with exact VSP URLs: `{report['source_metadata']['airshow'].get('exact_vsp_url_documents', 0)}`",
        f"- License-qualified exact VSP records: `{report['source_metadata']['airshow'].get('license_qualified_exact_vsp_records', 0)}`",
        "",
        "### NASA UAM Reference Vehicles",
        "",
        "The NASA UAM lane inspects official NASA OpenVSP ZIP archives and records each `.vsp3` member. "
        "These are representative public reference vehicles, not production vehicle drawings.",
        "",
        f"- OpenVSP archives inspected: `{report['source_metadata']['nasa_uam'].get('archive_count', 0)}`",
        f"- Archive fetch failures recorded: `{report['source_metadata']['nasa_uam'].get('archive_failure_count', 0)}`",
        f"- `.vsp3` members found: `{report['source_metadata']['nasa_uam'].get('vsp3_member_count', 0)}`",
        "",
        "### HiLiftAeroML",
        "",
        "The HiLiftAeroML lane catalogs one canonical STEP URL per CRM-HL geometry variant. The dataset "
        "also provides STL surfaces and force/moment CSVs for ten angles of attack per variant.",
        "",
        f"- Geometry variants cataloged: `{report['source_metadata']['hiliftaeroml'].get('geometry_variant_count', 0)}`",
        f"- Canonical AoA used for CAD URLs: `{report['source_metadata']['hiliftaeroml'].get('canonical_aoa_deg', 0)} deg`",
        "",
        "### NASA CRM Local And Candidate Sources",
        "",
        "The local CRM lane mirrors the repository's existing ready source catalog. The candidate sweep is "
        "kept separate because some entries are component libraries or format-conversion candidates rather "
        "than already-promoted training examples.",
        "",
        f"- Ready local CRM catalog records: `{report['source_metadata']['local_nasa_crm'].get('record_count', 0)}`",
        f"- CRM candidate groups already tracked: `{crm_candidate.get('candidate_group_count', 0)}`",
        f"- Ready CRM candidate groups: `{crm_candidate.get('ready_group_count', 0)}`",
        f"- Apparent candidate CAD records across groups: `{crm_candidate.get('apparent_record_count', 0)}`",
        "",
        "## Recommended Ingestion Order",
        "",
        "1. Pull HiLiftAeroML selected STEP, STL, force/moment CSV, and geometry-values CSV files first; it is the strongest exact-CAD-plus-flow-label source.",
        "2. Pull NASA UAM `.vsp3` archives next; they add configuration diversity with official NASA provenance and small payloads.",
        "3. Pull license-qualified Airshow `.vsp3` files in batches, preserving upload metadata and license IDs.",
        "4. Promote CRM candidate-sweep component libraries only after deciding whether component-level STEP files should train whole-aircraft generation or remain validation/context assets.",
        "",
        "## Ground-Truth Boundary",
        "",
        "This catalog can support exact geometry provenance. It does not by itself prove that every model is a physically valid aircraft, that Airshow community models match real aircraft dimensions, or that inferred mission metadata is factual. Those claims still require solver validation, unit normalization, duplicate/family split control, and source-specific metadata review.",
        "",
        "## Catalog Files",
        "",
        f"- JSON catalog: `{report['catalog_path']}`",
        f"- Machine report: `{report['machine_report_path']}`",
        "",
        "## Source References",
        "",
        f"- NASA UAM reference vehicles: {NASA_UAM_SOURCE_PAGE}",
        f"- VSP Airshow: {AIRSHOW_URL}",
        "- OpenVSP Airshow announcement: https://openvsp.org/blogs/announcements/2024/08/22/openvsp-airshow-is-live",
        f"- HiLiftAeroML dataset: {HILIFTAEROML_SOURCE_PAGE}",
        "- HiLiftAeroML overview: https://caemldatasets.org/hiliftaeroml/",
        "- NASA Common Research Model original CAD: https://commonresearchmodel.larc.nasa.gov/geometry/original-cad-files/",
        "- AIAA DPW-6 CRM CAD geometry notes: https://www.aiaa-dpw.org/Workshop6/DPW6-geom.html",
    ]
    if records:
        examples = records[:5]
        lines.extend(["", "## First Records", ""])
        for record in examples:
            lines.append(
                f"- `{record['source_id']}` `{record['file_format']}` from `{record['source_collection']}`"
            )
    return "\n".join(lines) + "\n"


def build_catalog(args: argparse.Namespace) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    source_metadata: Dict[str, Any] = {}

    local_records = build_local_nasa_crm_records(Path(args.crm_catalog))
    records.extend(local_records)
    source_metadata["local_nasa_crm"] = {"record_count": len(local_records), "source_path": args.crm_catalog}

    if not args.skip_airshow:
        airshow = fetch_airshow_records(
            source_url=args.airshow_url,
            page_size=args.page_size,
            allowed_licenses=args.allowed_airshow_licenses,
        )
        records.extend(airshow["records"])
        source_metadata["airshow"] = airshow["metadata"]
    else:
        source_metadata["airshow"] = {"skipped": True}

    if not args.skip_nasa_uam:
        nasa_uam = fetch_nasa_uam_records(verify_tls=not args.allow_insecure_tls)
        records.extend(nasa_uam["records"])
        source_metadata["nasa_uam"] = nasa_uam["metadata"]
    else:
        source_metadata["nasa_uam"] = {"skipped": True}

    hilift_records = build_hiliftaeroml_records(
        geometry_count=args.hiliftaeroml_geometry_count,
        canonical_aoa=args.hiliftaeroml_canonical_aoa,
    )
    records.extend(hilift_records)
    source_metadata["hiliftaeroml"] = {
        "geometry_variant_count": len(hilift_records),
        "canonical_aoa_deg": args.hiliftaeroml_canonical_aoa,
        "source_page": HILIFTAEROML_SOURCE_PAGE,
    }
    source_metadata["nasa_crm_candidate_sweep"] = summarize_crm_candidate_sweep(
        Path(args.crm_candidate_sweep)
    )

    records = sorted(records, key=lambda item: (item["source_collection"], item["source_id"]))
    generated_at = _utc_now()
    catalog = {
        "schema_version": 1,
        "generated_at": generated_at,
        "claim_boundary": (
            "Exact CAD source catalog. URL-level records are not a binary mirror and do not imply "
            "aerodynamic or mission-label validity without separate solver and metadata validation."
        ),
        "records": records,
    }
    summary = summarize_records(records)

    output_catalog = Path(args.output_catalog)
    machine_report_path = Path(args.machine_report)
    output_report = Path(args.output_report)
    output_catalog.parent.mkdir(parents=True, exist_ok=True)
    machine_report_path.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_catalog.write_text(json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")

    report = {
        "generated_at": generated_at,
        "catalog_path": str(output_catalog),
        "machine_report_path": str(machine_report_path),
        "summary": summary,
        "source_metadata": source_metadata,
    }
    machine_report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    output_report.write_text(render_markdown_report(report, records), encoding="utf-8")
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-catalog", default="docs/dataset/exact_cad_source_catalog_20260624.json")
    parser.add_argument("--output-report", default="docs/dataset/exact_cad_source_sweep_20260624.md")
    parser.add_argument("--machine-report", default="docs/dataset/exact_cad_source_report_20260624.json")
    parser.add_argument("--crm-catalog", default="docs/dataset/nasa_crm_source_catalog.json")
    parser.add_argument("--crm-candidate-sweep", default="docs/dataset/nasa_crm_source_candidates.json")
    parser.add_argument("--airshow-url", default=AIRSHOW_URL)
    parser.add_argument("--page-size", type=_positive_int, default=1000)
    parser.add_argument("--allowed-airshow-licenses", nargs="+", default=list(ALLOWED_AIRSHOW_LICENSES))
    parser.add_argument("--hiliftaeroml-geometry-count", type=_positive_int, default=180)
    parser.add_argument("--hiliftaeroml-canonical-aoa", type=int, default=4)
    parser.add_argument("--skip-airshow", action="store_true")
    parser.add_argument("--skip-nasa-uam", action="store_true")
    parser.add_argument(
        "--allow-insecure-tls",
        action="store_true",
        help="Use only when local certificate stores fail on NASA HTTPS inspection.",
    )
    args = parser.parse_args(argv)
    report = build_catalog(args)
    print(json.dumps(report["summary"], indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["summary"]["record_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
