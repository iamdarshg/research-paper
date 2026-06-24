#!/usr/bin/env python3
"""Build a public whole-aircraft evidence package from official NASA CRM assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import trimesh


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aircraft_diffusion_cfd import AircraftDesignDataset, CFDConfig, AdvancedCFDSimulator
from aircraft_validity import evaluate_aircraft_validity
from condition_feasibility import MIN_WALL_BY_METHOD_MM, validate_condition_feasibility


ACCESS_DATE = str(date.today())
GRID_SIZE = 32
SIMULATION_STEPS = 20
PREPROCESSING_VERSION = "nasa-crm-whole-aircraft-context-v2"
MANUFACTURING_METHOD = "composite_wet_layup"

CRM_HL_ASSEMBLED_PAGE = (
    "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/assembled-geometry/"
)
CRM_HL_REFERENCE_PAGE = (
    "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/reference-geometry/"
)
CRM_HL_NTF_PAGE = (
    "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/model-specific-geometry/"
)
CRM_HL_BARE_PAGE = "https://commonresearchmodel.larc.nasa.gov/crm-hl-reference-geometry/crm-hl-bare-cad-model/"
CRM_HS_STP_PAGE = "https://commonresearchmodel.larc.nasa.gov/geometry/stp-files/"
CRM_HS_DPW6_PAGE = "https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/"
NASA_DATA_POLICY = "https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy"


@dataclass(frozen=True)
class AssetSpec:
    source_id: str
    source_url: str
    source_page: str
    configuration: str
    design_family: str
    split: str
    source_license: str
    usage_terms_note: str
    geometry_kind: str
    file_format: str
    archive_member: str | None = None
    requires_mirror: bool = False
    scale_model_fraction: float | None = None
    reason_for_inclusion: str = ""
    scale_inference_note: str = ""
    candidate_status: str = "ready"
    enabled: bool = True


DEFAULT_SOURCE_CATALOG_PATH = REPO_ROOT / "docs" / "dataset" / "nasa_crm_source_catalog.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_module_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ModuleNotFoundError:
        return "not_installed"
    return str(getattr(module, "__version__", ""))


def load_source_catalog(
    path: Path,
    *,
    candidate_status: str = "ready",
    source_ids: set[str] | None = None,
    limit: int | None = None,
) -> List[AssetSpec]:
    payload = load_json(path)
    entries = payload.get("sources", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError(f"Source catalog must contain a list under 'sources': {path}")

    selected: List[AssetSpec] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Source catalog entries must be objects: {path}")
        if not entry.get("enabled", True):
            continue
        entry_source_id = str(entry["source_id"])
        if source_ids and entry_source_id not in source_ids:
            continue
        entry_status = str(entry.get("candidate_status", "ready"))
        if candidate_status != "all" and entry_status != candidate_status:
            continue
        selected.append(
            AssetSpec(
                source_id=entry_source_id,
                source_url=str(entry["source_url"]),
                source_page=str(entry["source_page"]),
                configuration=str(entry["configuration"]),
                design_family=str(entry["design_family"]),
                split=str(entry["split"]),
                source_license=str(
                    entry.get(
                        "source_license",
                        "not_stated_public_nasa_download_bounded_by_nasa_data_use_policy",
                    )
                ),
                usage_terms_note=str(
                    entry.get(
                        "usage_terms_note",
                        "Publicly downloadable NASA CRM geometry. File-level license is not stated on the source page; "
                        "usage is bounded here to public NASA data-use policy context and research/validation packaging.",
                    )
                ),
                geometry_kind=str(entry.get("geometry_kind", "whole_aircraft_unknown")),
                file_format=str(entry.get("file_format", "step_zip")),
                archive_member=entry.get("archive_member"),
                requires_mirror=bool(entry.get("requires_mirror", False)),
                scale_model_fraction=entry.get("scale_model_fraction"),
                reason_for_inclusion=str(entry.get("reason_for_inclusion", "")),
                scale_inference_note=str(entry.get("scale_inference_note", "")),
                candidate_status=entry_status,
                enabled=bool(entry.get("enabled", True)),
            )
        )
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_existing_manifest_records(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("source_id"):
                records[str(record["source_id"])] = record
    return records


def download_file(url: str, destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(30, 300))
    response.raise_for_status()
    destination.write_bytes(response.content)


def extract_step(zip_path: Path, destination_dir: Path, member: str | None) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".stp", ".step")) and "__MACOSX" not in name
        ]
        if member is None:
            if len(members) != 1:
                raise ValueError(f"{zip_path} contains {len(members)} STEP members; expected exactly one")
            member = members[0]
        if member not in members:
            raise FileNotFoundError(f"{member} not found in {zip_path}")
        output_path = destination_dir / Path(member).name
        if not output_path.exists():
            output_path.write_bytes(archive.read(member))
        return output_path


def export_step_to_stl(step_path: Path, stl_path: Path) -> None:
    try:
        import cadquery as cq
        from cadquery import exporters
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cadquery is required only for STEP-to-STL export. Install cadquery to build "
            "the NASA CRM whole-aircraft geometry package, or use existing exported STL artifacts."
        ) from exc

    shape = cq.importers.importStep(str(step_path))
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    exporters.export(shape, str(stl_path), tolerance=0.5, angularTolerance=0.8)


def build_full_stl(source_stl_path: Path, full_stl_path: Path, *, requires_mirror: bool) -> Dict[str, Any]:
    mesh = trimesh.load_mesh(source_stl_path)
    source_bounds = mesh.bounds.tolist()
    source_extents = (mesh.bounds[1] - mesh.bounds[0]).tolist()

    if not requires_mirror:
        if source_stl_path.resolve() != full_stl_path.resolve():
            shutil.copyfile(source_stl_path, full_stl_path)
        full_mesh = mesh
        mirror_info = {
            "mode": "copy_fullspan_source",
            "mirror_axis_index": None,
            "mirror_axis_label": None,
        }
    else:
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        mins = vertices.min(axis=0)
        plane_axis = int(np.argmin(np.abs(mins)))

        mirrored = vertices.copy()
        mirrored[:, plane_axis] *= -1.0
        mirrored_faces = faces[:, ::-1] + len(vertices)
        full_mesh = trimesh.Trimesh(
            vertices=np.vstack([vertices, mirrored]),
            faces=np.vstack([faces, mirrored_faces]),
            process=False,
        )
        full_mesh.remove_unreferenced_vertices()
        full_stl_path.parent.mkdir(parents=True, exist_ok=True)
        full_mesh.export(full_stl_path)
        mirror_info = {
            "mode": "mirror_semispan_source",
            "mirror_axis_index": plane_axis,
            "mirror_axis_label": ("x", "y", "z")[plane_axis],
        }

    return {
        **mirror_info,
        "source_bounds": source_bounds,
        "source_extents": source_extents,
        "source_is_watertight": bool(mesh.is_watertight),
        "source_vertices": int(len(mesh.vertices)),
        "source_faces": int(len(mesh.faces)),
        "full_bounds": full_mesh.bounds.tolist(),
        "full_extents": (full_mesh.bounds[1] - full_mesh.bounds[0]).tolist(),
        "full_is_watertight": bool(full_mesh.is_watertight),
        "full_vertices": int(len(full_mesh.vertices)),
        "full_faces": int(len(full_mesh.faces)),
    }


def run_local_analysis(
    full_stl_path: Path,
    voxel_path: Path,
    *,
    grid_size: int = GRID_SIZE,
    simulation_steps: int = SIMULATION_STEPS,
    analysis_device: str = "cpu",
) -> Dict[str, Any]:
    dataset = AircraftDesignDataset(num_samples=0, grid_size=grid_size)
    voxels = dataset._voxelize_stl(str(full_stl_path), grid_size)
    voxel_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(voxel_path, voxels.detach().cpu().numpy())

    if analysis_device == "auto":
        analysis_device = "cuda" if torch.cuda.is_available() else "cpu"
    if analysis_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("analysis_device='cuda' requested, but CUDA is not available")
    simulator = AdvancedCFDSimulator(CFDConfig(base_grid_resolution=grid_size), torch.device(analysis_device))
    cfd = simulator.simulate_aerodynamics(voxels, steps=simulation_steps)
    validity = evaluate_aircraft_validity(voxels)

    return {
        "validity": validity,
        "cfd": {key: float(value) for key, value in cfd.items() if isinstance(value, (int, float))},
        "occupancy_ratio": float((voxels > 0.5).float().mean().item()),
        "voxel_sha256": sha256_file(voxel_path),
    }


def represented_dimensions_m(
    full_extents: Iterable[float],
    *,
    scale_model_fraction: float | None,
) -> Dict[str, float]:
    extents_mm = [float(value) for value in full_extents]
    actual_length_m = extents_mm[0] / 1000.0
    actual_span_m = extents_mm[1] / 1000.0
    actual_thickness_m = extents_mm[2] / 1000.0
    if scale_model_fraction:
        factor = 1.0 / scale_model_fraction
    else:
        factor = 1.0
    return {
        "actual_length_m": actual_length_m,
        "actual_span_m": actual_span_m,
        "actual_thickness_m": actual_thickness_m,
        "represented_length_m": actual_length_m * factor,
        "represented_span_m": actual_span_m * factor,
        "represented_thickness_m": actual_thickness_m * factor,
        "represented_scale_factor": factor,
    }


def build_response_metrics(
    analysis: Dict[str, Any],
    dimensions: Dict[str, float],
    configuration: str,
) -> Dict[str, float]:
    validity_metrics = analysis["validity"]["metrics"]
    drag_coefficient = float(analysis["cfd"].get("drag_coefficient", 1.0))
    config_factor = 1.0
    if "landing" in configuration:
        config_factor = 1.12
    elif "takeoff" in configuration:
        config_factor = 1.04
    elif "cruise" in configuration:
        config_factor = 0.92

    payload_proxy = (
        dimensions["represented_length_m"]
        * dimensions["represented_span_m"]
        * dimensions["represented_thickness_m"]
        * max(analysis["occupancy_ratio"], 1e-6)
        * 1000.0
    )
    thrust_proxy = config_factor / max(drag_coefficient, 1e-6)
    maneuverability_proxy = (
        validity_metrics["symmetry_score"]
        * (validity_metrics["span_fraction_y"] / max(validity_metrics["length_fraction_x"], 1e-6))
        * config_factor
    )
    structural_proxy = (
        validity_metrics["center_body_density_ratio"]
        * validity_metrics["thickness_fraction_z"]
        * max(analysis["occupancy_ratio"], 1e-6)
        * dimensions["represented_span_m"]
    )
    return {
        "payload_response": round(float(payload_proxy), 6),
        "thrust_response": round(float(thrust_proxy), 6),
        "maneuverability_response": round(float(maneuverability_proxy), 6),
        "structural_response": round(float(structural_proxy), 6),
    }


def build_design_spec(
    asset: AssetSpec,
    dimensions: Dict[str, float],
    response_metrics: Dict[str, float],
) -> Dict[str, Any]:
    represented_span_m = dimensions["represented_span_m"]
    represented_length_m = dimensions["represented_length_m"]
    represented_thickness_m = dimensions["represented_thickness_m"]

    if "landing" in asset.configuration:
        target_speed_mps = 72.0
        thrust_to_weight_min = 0.27
        turn_rate_min_deg_s = 4.2
        takeoff_distance_min_m = 1150
        takeoff_distance_max_m = 1850
        payload_factor = 0.92
    elif "takeoff" in asset.configuration:
        target_speed_mps = 84.0
        thrust_to_weight_min = 0.31
        turn_rate_min_deg_s = 3.7
        takeoff_distance_min_m = 1350
        takeoff_distance_max_m = 2250
        payload_factor = 1.0
    elif "reference_solid" in asset.configuration:
        target_speed_mps = 92.0
        thrust_to_weight_min = 0.29
        turn_rate_min_deg_s = 3.4
        takeoff_distance_min_m = 1450
        takeoff_distance_max_m = 2300
        payload_factor = 0.96
    else:
        target_speed_mps = 236.0
        thrust_to_weight_min = 0.29
        turn_rate_min_deg_s = 2.4
        takeoff_distance_min_m = 1800
        takeoff_distance_max_m = 2800
        payload_factor = 1.08

    payload_mass_max_g = int(round(response_metrics["payload_response"] * 36.0 * payload_factor))
    payload_mass_min_g = int(round(payload_mass_max_g * 0.45))
    engine_diameter_mm = int(round(represented_span_m * 55.0))
    engine_length_mm = int(round(engine_diameter_mm * 2.05))
    required_static_thrust_n = round(represented_span_m * 3800.0, 3)
    wall_min = round(max(MIN_WALL_BY_METHOD_MM[MANUFACTURING_METHOD], 1.2 + represented_thickness_m * 0.08), 3)
    wall_max = round(wall_min + 1.8, 3)
    part_count_min = 14 if "cruise" in asset.configuration else 18
    part_count_max = 36 if "cruise" in asset.configuration else 44

    design_spec = {
        "target_speed_mps": round(target_speed_mps, 3),
        "wingspan_limit_m": round(represented_span_m * 1.02, 3),
        "thrust_to_weight_min": round(thrust_to_weight_min, 3),
        "turn_rate_min_deg_s": round(turn_rate_min_deg_s, 3),
        "required_static_thrust_n": required_static_thrust_n,
        "engine_diameter_mm": engine_diameter_mm,
        "engine_length_mm": engine_length_mm,
        "engine_count_min": 2,
        "engine_count_max": 2,
        "payload_mass_min_g": payload_mass_min_g,
        "payload_mass_max_g": payload_mass_max_g,
        "takeoff_distance_min_m": takeoff_distance_min_m,
        "takeoff_distance_max_m": takeoff_distance_max_m,
        "wall_thickness_min_mm": wall_min,
        "wall_thickness_max_mm": wall_max,
        "part_count_min": part_count_min,
        "part_count_max": part_count_max,
        "manufacturing_method": MANUFACTURING_METHOD,
    }
    feasibility = validate_condition_feasibility(design_spec)
    if feasibility["status"] != "pass":
        raise ValueError(f"Design spec for {asset.source_id} failed feasibility: {feasibility}")
    return design_spec


def design_spec_provenance(asset: AssetSpec) -> Dict[str, str]:
    span_provenance = "direct_from_full_aircraft_stl_span_mm"
    if asset.scale_model_fraction is not None:
        span_provenance = (
            f"scale_corrected_from_model_span_mm_using_official_scale_fraction_{asset.scale_model_fraction:g}"
        )
    elif asset.scale_inference_note:
        span_provenance = "direct_from_full_aircraft_stl_span_mm_after_geometry_extent_scale_check"
    return {
        "target_speed_mps": "bounded_inference_from_configuration_family_not_flight_test_data",
        "wingspan_limit_m": span_provenance,
        "thrust_to_weight_min": "bounded_inference_from_transport_configuration_family",
        "turn_rate_min_deg_s": "bounded_inference_from_transport_configuration_family",
        "required_static_thrust_n": "inferred_from_represented_span_transport_proxy",
        "engine_diameter_mm": "inferred_from_represented_span_transport_twin_engine_proxy",
        "engine_length_mm": "inferred_from_engine_diameter_transport_proxy",
        "engine_count_min": "inferred_from_nasa_crm_twin_engine_family",
        "engine_count_max": "inferred_from_nasa_crm_twin_engine_family",
        "payload_mass_min_g": "inferred_from_represented_dimensions_transport_proxy",
        "payload_mass_max_g": "inferred_from_represented_dimensions_transport_proxy",
        "takeoff_distance_min_m": "bounded_inference_from_configuration_family",
        "takeoff_distance_max_m": "bounded_inference_from_configuration_family",
        "wall_thickness_min_mm": "mapped_to_repo_supported_manufacturing_category_not_real_aircraft_build_rule",
        "wall_thickness_max_mm": "mapped_to_repo_supported_manufacturing_category_not_real_aircraft_build_rule",
        "part_count_min": "mapped_to_repo_supported_manufacturing_category_not_real_aircraft_bom",
        "part_count_max": "mapped_to_repo_supported_manufacturing_category_not_real_aircraft_bom",
        "manufacturing_method": "closest_repo_supported_category_for_smooth_transport_shell_geometry",
    }


def write_analysis_report(
    report_path: Path,
    record: Dict[str, Any],
    *,
    grid_size: int = GRID_SIZE,
    simulation_steps: int = SIMULATION_STEPS,
    analysis_device: str = "cpu",
) -> None:
    payload = {
        "sample_id": record["sample_id"],
        "source_id": record["source_id"],
        "configuration": record["configuration"],
        "split": record["split"],
        "design_family": record["design_family"],
        "source_url": record["source_url"],
        "source_page": record["source_page"],
        "geometry_paths": {
            "step_path": record["artifacts"]["step_path"],
            "source_stl_path": record["artifacts"]["source_stl_path"],
            "full_stl_path": record["artifacts"]["full_stl_path"],
            "voxel_path": record["artifacts"]["voxel_path"],
        },
        "hashes": record["hashes"],
        "mirror_info": record["mirror_info"],
        "represented_dimensions_m": record["represented_dimensions_m"],
        "geometry_scale_note": record.get("geometry_scale_note"),
        "solver_config": {
            "grid_size": grid_size,
            "steps": simulation_steps,
            "solver_type": "D3Q27",
            "device": analysis_device,
        },
        "local_cfd": record["analysis"]["cfd"],
        "local_validity": record["analysis"]["validity"],
        "response_metrics": record["response_metrics"],
        "response_metrics_provenance": record["response_metrics_provenance"],
        "claim_boundary": (
            "Local whole-aircraft geometry-plus-internal-solver report for manifest grounding. "
            "It is not a wind-tunnel match or certification-grade CFD study."
        ),
    }
    write_json(report_path, payload)


def build_manifest_record(
    asset: AssetSpec,
    *,
    dataset_root: Path,
    manifest_path: Path,
    preprocessing_hash: str,
    step_path: Path,
    source_stl_path: Path,
    full_stl_path: Path,
    voxel_path: Path,
    analysis_report_path: Path,
    mirror_info: Dict[str, Any],
    analysis: Dict[str, Any],
    represented_dims: Dict[str, float],
) -> Dict[str, Any]:
    response_metrics = build_response_metrics(analysis, represented_dims, asset.configuration)
    design_spec = build_design_spec(asset, represented_dims, response_metrics)
    geometry_mode = "mirrored_semispan_step_to_full_stl" if asset.requires_mirror else "fullspan_step_to_stl"
    geometry_provenance = (
        f"Official NASA CRM STEP geometry downloaded from {asset.source_page}, "
        f"exported to STL with CadQuery/OpenCascade, processed as {geometry_mode}, "
        "and voxelized with AircraftDesignDataset._voxelize_stl."
    )
    record = {
        "sample_id": asset.source_id,
        "source_id": asset.source_id,
        "source_url": asset.source_url,
        "source_page": asset.source_page,
        "source_license": asset.source_license,
        "usage_terms": asset.usage_terms_note,
        "split": asset.split,
        "configuration": asset.configuration,
        "design_family": asset.design_family,
        "geometry_kind": asset.geometry_kind,
        "file_format": asset.file_format,
        "geometry_path": str(voxel_path.relative_to(manifest_path.parent)).replace("\\", "/"),
        "stl_path": str(full_stl_path.relative_to(manifest_path.parent)).replace("\\", "/"),
        "geometry_provenance": geometry_provenance,
        "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing_hash": preprocessing_hash,
        "units": "mm_inferred_from_geometry_scale",
        "original_units": "mm_inferred_from_geometry_scale",
        "design_spec": design_spec,
        "design_spec_provenance": design_spec_provenance(asset),
        "response_metrics": response_metrics,
        "response_metrics_provenance": {
            "payload_response": "local_geometry_analysis_represented_volume_proxy",
            "thrust_response": "local_internal_cfd_drag_inverse_proxy",
            "maneuverability_response": "local_geometry_validity_symmetry_planform_proxy",
            "structural_response": "local_geometry_validity_centerline_thickness_proxy",
        },
        "analysis_report_path": str(analysis_report_path.relative_to(manifest_path.parent)).replace("\\", "/"),
        "date_accessed": ACCESS_DATE,
        "reference_page": CRM_HL_REFERENCE_PAGE if "crm_hl" in asset.source_id else asset.source_page,
        "scale_model_fraction": asset.scale_model_fraction,
        "catalog_candidate_status": asset.candidate_status,
        "represented_dimensions_m": represented_dims,
        "geometry_scale_note": asset.scale_inference_note or None,
        "artifacts": {
            "step_path": str(step_path.resolve()),
            "source_stl_path": str(source_stl_path.resolve()),
            "full_stl_path": str(full_stl_path.resolve()),
            "voxel_path": str(voxel_path.resolve()),
        },
        "hashes": {
            "zip_sha256": sha256_file(Path(dataset_root / "raw" / Path(asset.source_url).name)),
            "step_sha256": sha256_file(step_path),
            "source_stl_sha256": sha256_file(source_stl_path),
            "full_stl_sha256": sha256_file(full_stl_path),
            "voxel_sha256": analysis["voxel_sha256"],
        },
        "mirror_info": mirror_info,
        "analysis": analysis,
        "reason_for_inclusion": asset.reason_for_inclusion,
        "claim_boundary": (
            "Public whole-aircraft geometry record with bounded local analysis. "
            "Design-spec values beyond direct size measurements are explicit inferences."
        ),
    }
    return record


def build_refinement_study(records: List[Dict[str, Any]], report_path: Path) -> None:
    chosen_ids = []
    for preferred in ("crm_hl_reference_ldg", "crm_hs_dpw6_cf"):
        for record in records:
            if record["source_id"] == preferred:
                chosen_ids.append(preferred)
                break
    chosen_ids = list(dict.fromkeys(chosen_ids))
    if not chosen_ids:
        return

    dataset = AircraftDesignDataset(num_samples=0, grid_size=GRID_SIZE)
    cases: Dict[str, Any] = {}
    for source_id in chosen_ids:
        record = next(item for item in records if item["source_id"] == source_id)
        stl_path = Path(record["artifacts"]["full_stl_path"])
        ladder = []
        for grid_size, steps in ((24, 15), (32, 20), (40, 25)):
            voxels = dataset._voxelize_stl(str(stl_path), grid_size)
            simulator = AdvancedCFDSimulator(CFDConfig(base_grid_resolution=grid_size), torch.device("cpu"))
            cfd = simulator.simulate_aerodynamics(voxels, steps=steps)
            validity = evaluate_aircraft_validity(voxels)
            ladder.append(
                {
                    "grid_size": grid_size,
                    "steps": steps,
                    "occupancy_ratio": float((voxels > 0.5).float().mean().item()),
                    "drag_coefficient": float(cfd.get("drag_coefficient", 0.0)),
                    "lift_coefficient": float(cfd.get("lift_coefficient", 0.0)),
                    "symmetry_score": float(validity["metrics"].get("symmetry_score", 0.0)),
                }
            )
        cases[source_id] = {
            "configuration": record["configuration"],
            "source_page": record["source_page"],
            "grid_ladder": ladder,
        }

    write_json(
        report_path,
        {
            "study": "nasa_crm_whole_aircraft_grid_refinement",
            "generated": ACCESS_DATE,
            "cases": cases,
            "claim_boundary": (
                "Internal-solver consistency study only. It checks grid/step sensitivity for representative "
                "public whole-aircraft geometries and does not establish external coefficient agreement."
            ),
            "source_context": {
                "crm_hl_assembled_page": CRM_HL_ASSEMBLED_PAGE,
                "crm_hs_dpw6_page": CRM_HS_DPW6_PAGE,
            },
        },
    )


def build_report(
    *,
    report_path: Path,
    manifest_path: Path,
    provenance_path: Path,
    records: List[Dict[str, Any]],
    refinement_path: Path,
    grid_size: int = GRID_SIZE,
    simulation_steps: int = SIMULATION_STEPS,
) -> None:
    counts: Dict[str, int] = {}
    families: Dict[str, int] = {}
    pages: Dict[str, int] = {}
    for record in records:
        counts[record["split"]] = counts.get(record["split"], 0) + 1
        families[record["design_family"]] = families.get(record["design_family"], 0) + 1
        pages[record["source_page"]] = pages.get(record["source_page"], 0) + 1

    lines = [
        "# NASA CRM Whole-Aircraft Corpus Report",
        "",
        f"- Generated: `{ACCESS_DATE}`",
        f"- Manifest: `{manifest_path}`",
        f"- Provenance ledger: `{provenance_path}`",
        f"- Record count: `{len(records)}`",
        "",
        "## Sources",
        "",
        f"- CRM-HL assembled geometry page: `{CRM_HL_ASSEMBLED_PAGE}`",
        f"- CRM-HL reference geometry page: `{CRM_HL_REFERENCE_PAGE}`",
        f"- CRM-HL model-specific geometry page: `{CRM_HL_NTF_PAGE}`",
        f"- CRM-HL bare model page: `{CRM_HL_BARE_PAGE}`",
        f"- High-speed CRM STP page: `{CRM_HS_STP_PAGE}`",
        f"- DPW6 geometry page: `{CRM_HS_DPW6_PAGE}`",
        f"- NASA data-use policy context: `{NASA_DATA_POLICY}`",
        "",
        "## Included Corpus",
        "",
        "- All records are public NASA CRM whole-aircraft or semispan aircraft-like CAD assets converted to STL and voxelized locally.",
        "- Semispan geometries were mirrored into full-aircraft STL artifacts before aircraft-validity analysis.",
        "- Scale-model NTF assets were retained, but represented design-envelope fields were scale-corrected from the official 2.7% or 5.2% factors.",
        "- One exception was required: the `NASA5p2` landing STEP extents matched full-scale CRM-HL dimensions, so it was treated as full-scale geometry rather than scale-corrected a second time.",
        f"- Split counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Design-family counts: `{json.dumps(families, sort_keys=True)}`",
        f"- Source-page counts: `{json.dumps(pages, sort_keys=True)}`",
        "",
        "## Preprocessing",
        "",
        "- Intake is driven by the checked-in `docs/dataset/nasa_crm_source_catalog.json` source catalog so new ready entries can be added without modifying builder code.",
        "- Official STEP files were downloaded from NASA CRM pages, hashed, and extracted locally from zip archives.",
        "- CAD triangulation used CadQuery/OpenCascade with fixed STL export tolerances.",
        f"- Full-aircraft STL artifacts were voxelized at `{grid_size}^3` with `AircraftDesignDataset._voxelize_stl`.",
        f"- Local analysis reports used the repo internal `D3Q27` solver for `{simulation_steps}` steps with fixed settings.",
        "",
        "## Validation",
        "",
        "- Every manifest record has a local aircraft-validity report and grounded response metrics derived from reproducible local analysis.",
        f"- Representative grid-refinement report: `{refinement_path}`",
        "- These reports are internal consistency evidence, not publication-grade aerodynamic coefficient validation.",
        "",
        "## Exclusions And Boundaries",
        "",
        "- A broken CRM65 icing-page zip was excluded because the downloaded `.stp` payload was only web-metadata text, not geometry.",
        "- DPW7 IGES-only assets were not included because this builder currently standardizes on STEP-based conversion for reproducibility on this host.",
        "- Manufacturing fields are mapped to the repo's supported conditioning schema categories; they are not claims about real transport-aircraft factory processes.",
        "- Target speed, payload, thrust, and takeoff fields are bounded inferences from configuration family and represented dimensions, not imported flight-test values.",
        "",
        "## Gate Support",
        "",
        "- Supports `validate_manifest.py --level claim-bearing`.",
        "- Supports `CLI/aircraft_validity.py`-style whole-aircraft geometry checks on public-source records.",
        "- Supports `condition_feasibility.py` because every record contains complete manufacturing fields.",
        "- Does not by itself satisfy the final protocol `min_grounded_records >= 20` requirement, because this package contains fewer than 20 records.",
        "- Does not by itself establish external CFD validation or paper-level aerodynamic accuracy claims.",
        "",
        "## Split Limits",
        "",
        "- This package is deterministic, but source-family leakage cannot be fully removed because every record comes from the NASA CRM ecosystem.",
        "- The split therefore separates high-lift reference, wind-tunnel model, and high-speed DPW contexts rather than pretending they are unrelated aircraft families.",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a public NASA CRM whole-aircraft manifest and evidence package.")
    parser.add_argument(
        "--source-catalog",
        default=str(DEFAULT_SOURCE_CATALOG_PATH),
        help="JSON source catalog path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "docs" / "dataset" / "nasa_crm_whole_aircraft"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "docs" / "dataset" / "nasa_crm_whole_aircraft_manifest.jsonl"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--provenance",
        default=str(REPO_ROOT / "docs" / "dataset" / "nasa_crm_whole_aircraft_provenance.json"),
        help="Output provenance ledger path.",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "docs" / "dataset" / "NASA_CRM_WHOLE_AIRCRAFT_REPORT.md"),
        help="Output report path.",
    )
    parser.add_argument(
        "--candidate-status",
        default="ready",
        help="Only ingest catalog entries with this candidate_status. Use 'all' to disable filtering.",
    )
    parser.add_argument(
        "--source-id",
        action="append",
        default=[],
        help="Optional source_id to ingest. Repeat to ingest multiple specific entries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected catalog entries to process.",
    )
    parser.add_argument(
        "--reuse-existing-records",
        action="store_true",
        help="Reuse existing manifest records by source_id instead of rebuilding them.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE,
        help="Voxel grid edge length for generated NASA CRM records.",
    )
    parser.add_argument(
        "--simulation-steps",
        type=int,
        default=SIMULATION_STEPS,
        help="Internal D3Q27 solver steps for per-record local analysis.",
    )
    parser.add_argument(
        "--analysis-device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="Device used for per-record internal solver analysis.",
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Skip the representative grid-refinement study refresh.",
    )
    args = parser.parse_args()
    if args.grid_size <= 0:
        raise SystemExit("--grid-size must be positive")
    if args.simulation_steps <= 0:
        raise SystemExit("--simulation-steps must be positive")

    source_catalog_path = Path(args.source_catalog).resolve()
    output_root = Path(args.output_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    provenance_path = Path(args.provenance).resolve()
    report_path = Path(args.report).resolve()
    selected_source_ids = set(args.source_id or [])
    assets = load_source_catalog(
        source_catalog_path,
        candidate_status=args.candidate_status,
        source_ids=selected_source_ids or None,
        limit=args.limit,
    )

    raw_root = output_root / "raw"
    step_root = output_root / "step"
    stl_source_root = output_root / "stl_source"
    stl_full_root = output_root / "stl_full"
    voxel_subdir = "voxels" if int(args.grid_size) == GRID_SIZE else f"voxels_g{int(args.grid_size)}"
    analysis_subdir = "analysis" if int(args.grid_size) == GRID_SIZE else f"analysis_g{int(args.grid_size)}"
    voxel_root = output_root / voxel_subdir
    analysis_root = output_root / "reports" / analysis_subdir
    refinement_root = output_root / "reports" / "refinement"
    for path in (raw_root, step_root, stl_source_root, stl_full_root, voxel_root, analysis_root, refinement_root):
        path.mkdir(parents=True, exist_ok=True)

    preprocessing_payload = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "grid_size": int(args.grid_size),
        "simulation_steps": int(args.simulation_steps),
        "voxelizer": "AircraftDesignDataset._voxelize_stl",
        "solver": "AdvancedCFDSimulator(D3Q27)",
        "cad_export_tolerance": 0.5,
        "cad_export_angular_tolerance": 0.8,
        "source_catalog_path": str(source_catalog_path),
        "candidate_status_filter": args.candidate_status,
    }
    preprocessing_hash = hashlib.sha256(
        json.dumps(preprocessing_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    existing_records = load_existing_manifest_records(manifest_path) if args.reuse_existing_records else {}
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    reused_count = 0
    built_count = 0
    for asset in assets:
        try:
            if asset.source_id in existing_records:
                records.append(existing_records[asset.source_id])
                reused_count += 1
                continue
            zip_path = raw_root / Path(asset.source_url).name
            if not zip_path.exists():
                download_file(asset.source_url, zip_path)

            step_dir = step_root / asset.source_id
            step_path = extract_step(zip_path, step_dir, asset.archive_member)
            source_stl_path = stl_source_root / f"{asset.source_id}.stl"
            full_stl_path = stl_full_root / f"{asset.source_id}_full.stl"
            voxel_path = voxel_root / f"{asset.source_id}_full.npy"
            analysis_report_path = analysis_root / f"{asset.source_id}.json"

            if not source_stl_path.exists():
                export_step_to_stl(step_path, source_stl_path)
            mirror_info = build_full_stl(
                source_stl_path,
                full_stl_path,
                requires_mirror=asset.requires_mirror,
            )
            analysis = run_local_analysis(
                full_stl_path,
                voxel_path,
                grid_size=int(args.grid_size),
                simulation_steps=int(args.simulation_steps),
                analysis_device=str(args.analysis_device),
            )
            represented_dims = represented_dimensions_m(
                mirror_info["full_extents"],
                scale_model_fraction=asset.scale_model_fraction,
            )
            record = build_manifest_record(
                asset,
                dataset_root=output_root,
                manifest_path=manifest_path,
                preprocessing_hash=preprocessing_hash,
                step_path=step_path,
                source_stl_path=source_stl_path,
                full_stl_path=full_stl_path,
                voxel_path=voxel_path,
                analysis_report_path=analysis_report_path,
                mirror_info=mirror_info,
                analysis=analysis,
                represented_dims=represented_dims,
            )
            write_analysis_report(
                analysis_report_path,
                record,
                grid_size=int(args.grid_size),
                simulation_steps=int(args.simulation_steps),
                analysis_device=str(args.analysis_device),
            )
            records.append(record)
            built_count += 1
        except Exception as exc:  # pragma: no cover - defensive packaging path
            errors.append(
                {
                    "source_id": asset.source_id,
                    "source_url": asset.source_url,
                    "source_page": asset.source_page,
                    "configuration": asset.configuration,
                    "error": repr(exc),
                }
            )

    records.sort(key=lambda item: item["source_id"])
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    refinement_path = refinement_root / "grid_refinement.json"
    if records and not args.skip_refinement:
        build_refinement_study(records, refinement_path)

    provenance = {
        "schema_version": 2,
        "generated": ACCESS_DATE,
        "claim_boundary": (
            "This ledger documents a public NASA CRM whole-aircraft evidence package with local proxy metrics. "
            "It is bounded by public geometry availability and internal-solver validation limits."
        ),
        "build_environment": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", ""),
            "numpy": getattr(np, "__version__", ""),
            "trimesh": getattr(trimesh, "__version__", ""),
            "cadquery": optional_module_version("cadquery"),
        },
        "source_catalog": {
            "catalog_path": str(source_catalog_path),
            "selected_source_ids": sorted(selected_source_ids),
            "candidate_status_filter": args.candidate_status,
            "selected_record_count": len(assets),
            "crm_hl_assembled_page": CRM_HL_ASSEMBLED_PAGE,
            "crm_hl_reference_page": CRM_HL_REFERENCE_PAGE,
            "crm_hl_model_specific_page": CRM_HL_NTF_PAGE,
            "crm_hl_bare_page": CRM_HL_BARE_PAGE,
            "crm_hs_stp_page": CRM_HS_STP_PAGE,
            "crm_hs_dpw6_page": CRM_HS_DPW6_PAGE,
            "nasa_data_use_policy": NASA_DATA_POLICY,
        },
        "preprocessing": preprocessing_payload | {"preprocessing_hash": preprocessing_hash},
        "reuse_existing_records": bool(args.reuse_existing_records),
        "built_record_count": built_count,
        "record_count": len(records),
        "records": records,
        "errors": errors,
        "reused_record_count": reused_count,
        "skip_refinement": bool(args.skip_refinement),
    }
    write_json(provenance_path, provenance)
    build_report(
        report_path=report_path,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        records=records,
        refinement_path=refinement_path,
        grid_size=int(args.grid_size),
        simulation_steps=int(args.simulation_steps),
    )

    status = "pass" if len(records) == len(assets) else "partial"
    print(
        json.dumps(
            {
                "status": status,
                "record_count": len(records),
                "selected_catalog_records": len(assets),
                "built_record_count": built_count,
                "errors": errors,
                "manifest_path": str(manifest_path),
                "provenance_path": str(provenance_path),
                "report_path": str(report_path),
                "reused_record_count": reused_count,
                "skip_refinement": bool(args.skip_refinement),
                "source_catalog_path": str(source_catalog_path),
            },
            indent=2,
        )
    )
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
