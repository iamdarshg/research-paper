#!/usr/bin/env python3
"""Build a claim-bearing grounded aircraft-like corpus from public NACA sources.

This builder intentionally produces an airfoil-section-heavy corpus. It supports
manifest, manufacturing, and grounded response-metric workflows, but it does not
upgrade the repo to whole-aircraft validity or publication-grade aerodynamic
claims on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aircraft_diffusion_cfd import AircraftDesignDataset, CFDConfig, AdvancedCFDSimulator
from condition_feasibility import MIN_WALL_BY_METHOD_MM, validate_condition_feasibility


def _requests_module():
    import requests

    return requests


def _trimesh_module():
    import trimesh

    return trimesh


def _polygon_class():
    from shapely.geometry import Polygon

    return Polygon


PREPROCESSING_VERSION = "grounded-aircraft-corpus-v1"
GRID_SIZE = 32
SIMULATION_STEPS = 20
ACCESS_DATE = str(date.today())

RAW_SOURCE_URLS = {
    "naca_generator_py": "https://raw.githubusercontent.com/Extrality/NACA_simulation/main/naca_generator.py",
    "naca_simulation_readme": "https://raw.githubusercontent.com/Extrality/NACA_simulation/main/README.md",
    "naca_simulation_license": "https://raw.githubusercontent.com/Extrality/NACA_simulation/main/LICENSE",
}

CONTEXT_SOURCES = {
    "airfrans_dataset_docs": "https://airfrans.readthedocs.io/en/latest/notes/introduction.html",
    "airfrans_github": "https://github.com/Extrality/airfrans_lib",
    "naca_simulation_github": "https://github.com/Extrality/NACA_simulation",
    "nasa_crm_dpw6": "https://commonresearchmodel.larc.nasa.gov/geometry/dpw6-geometries/",
    "nasa_crm_hl_reference": "https://commonresearchmodel.larc.nasa.gov/high-lift-crm/high-lift-crm-geometry/reference-geometry/",
    "nasa_tmr_naca0012": "https://tmbwg.github.io/turbmodels/naca0012numerics_grids.html",
    "nasa_tmr_onera_m6": "https://tmbwg.github.io/turbmodels/onerawingnumerics_grids.html",
}

AIRFOIL_CODES: Sequence[Tuple[str, str]] = (
    ("0008", "train"),
    ("0012", "train"),
    ("0015", "train"),
    ("0021", "train"),
    ("23012", "train"),
    ("23015", "train"),
    ("6409", "train"),
    ("6412", "train"),
    ("1408", "val"),
    ("1412", "val"),
    ("6509", "val"),
    ("6512", "val"),
    ("2412", "test"),
    ("2415", "test"),
    ("2418", "test"),
    ("23018", "test"),
    ("23112", "test"),
    ("4412", "holdout"),
    ("4415", "holdout"),
    ("4421", "holdout"),
)

REFINEMENT_CASES = ("0012", "23015")


@dataclass
class Entry:
    code: str
    split: str
    source_id: str
    design_family: str
    chord_m: float
    span_m: float
    thickness_ratio: float
    camber_ratio: float
    manufacturing_method: str
    stl_path: Path
    voxel_path: Path
    profile_path: Path
    analysis_report_path: Path
    geometry_sha256: str
    voxel_sha256: str
    profile_sha256: str
    payload_proxy: float
    thrust_proxy: float
    maneuverability_proxy: float
    structural_proxy: float
    mesh_volume_m3: float
    mesh_area_m2: float
    occupancy_ratio: float
    cfd: Dict[str, float]
    design_spec: Dict[str, Any] | None = None
    response_metrics: Dict[str, float] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = _requests_module().get(url, timeout=(30, 120))
    response.raise_for_status()
    destination.write_bytes(response.content)


def ensure_raw_sources(raw_root: Path) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for key, url in RAW_SOURCE_URLS.items():
        destination = raw_root / Path(url).name
        if not destination.exists():
            download_file(url, destination)
        sources[key] = {
            "url": url,
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "license": "ODbL-1.0",
            "date_accessed": ACCESS_DATE,
            "usage_terms": "Extrality/NACA_simulation public repository under ODbL-1.0.",
        }
    return sources


def load_naca_generator(module_path: Path):
    spec = importlib.util.spec_from_file_location("grounded_naca_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load NACA generator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_code(code: str) -> Tuple[Tuple[int, ...], str]:
    if len(code) == 4:
        return (int(code[0]), int(code[1]), int(code[2:])), "airfoil_section_4digit"
    if len(code) == 5:
        return (int(code[0]), int(code[1]), int(code[2]), int(code[3:])), "airfoil_section_5digit"
    raise ValueError(f"Unsupported NACA code: {code}")


def compute_thickness_and_camber(naca_module, params: Sequence[int]) -> Tuple[float, float]:
    x = np.linspace(0.0, 1.0, 801)
    y_c, _ = naca_module.camber_line(tuple(params[:-1]), x)
    thickness = naca_module.thickness_dist(params[-1] / 100.0, x)
    return float(2.0 * np.max(thickness)), float(np.max(y_c))


def build_mesh(profile_points: np.ndarray, chord_m: float, span_m: float) -> trimesh.Trimesh:
    trimesh = _trimesh_module()
    Polygon = _polygon_class()
    scaled = np.column_stack((profile_points[:, 0] * chord_m, profile_points[:, 1] * chord_m))
    polygon = Polygon(scaled)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    mesh = trimesh.creation.extrude_polygon(polygon, span_m)
    mesh = mesh.process(validate=True)
    if mesh is None:
        mesh = trimesh.creation.extrude_polygon(polygon, span_m)
    return mesh


def choose_manufacturing_method(thickness_ratio: float, code: str) -> str:
    if thickness_ratio >= 0.18:
        return "foam_core_hotwire"
    if code.startswith("23") or thickness_ratio <= 0.10:
        return "composite_wet_layup"
    return "sheet_balsa_tabbed"


def validate_cfd_outputs(*, source_id: str, grid_size: int, steps: int, cfd: Dict[str, Any]) -> None:
    drag = cfd.get("drag_coefficient")
    lift = cfd.get("lift_coefficient")
    if not isinstance(drag, (int, float)) or not np.isfinite(float(drag)):
        raise ValueError(
            f"{source_id}: CFD drag_coefficient is non-finite at grid={grid_size}, steps={steps}: {drag}"
        )
    if float(drag) <= 0.0:
        raise ValueError(
            f"{source_id}: CFD drag_coefficient must be positive at grid={grid_size}, steps={steps}: {drag}"
        )
    if not isinstance(lift, (int, float)) or not np.isfinite(float(lift)):
        raise ValueError(
            f"{source_id}: CFD lift_coefficient is non-finite at grid={grid_size}, steps={steps}: {lift}"
        )


def run_local_analysis(
    *,
    stl_path: Path,
    voxelizer: AircraftDesignDataset,
    chord_m: float,
    span_m: float,
    thickness_ratio: float,
    camber_ratio: float,
    grid_size: int,
    steps: int,
) -> Dict[str, Any]:
    voxels = voxelizer._voxelize_stl(str(stl_path), grid_size)

    simulator = AdvancedCFDSimulator(CFDConfig(base_grid_resolution=grid_size), torch.device("cpu"))
    cfd = simulator.simulate_aerodynamics(voxels, steps=steps)
    validate_cfd_outputs(
        source_id=stl_path.stem,
        grid_size=grid_size,
        steps=steps,
        cfd=cfd,
    )

    voxel_sum = float(voxels.sum().item())
    occupancy_ratio = float((voxels > 0.5).float().mean().item())
    thrust_proxy = 1.0 / max(float(cfd.get("drag_coefficient", 1.0)), 1e-6)
    maneuverability_proxy = max(camber_ratio * 100.0, 0.0) + (span_m / max(chord_m, 1e-6))
    structural_proxy = thickness_ratio * chord_m * span_m
    payload_proxy = voxel_sum * (span_m / max(chord_m, 1e-6))

    return {
        "voxels": voxels,
        "cfd": {key: float(value) for key, value in cfd.items() if isinstance(value, (int, float))},
        "payload_proxy": float(payload_proxy),
        "thrust_proxy": float(thrust_proxy),
        "maneuverability_proxy": float(maneuverability_proxy),
        "structural_proxy": float(structural_proxy),
        "occupancy_ratio": occupancy_ratio,
    }


def assign_rank(values: Dict[str, float]) -> Dict[str, int]:
    ranked = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return {key: idx for idx, (key, _) in enumerate(ranked)}


def assign_design_specs(entries: List[Entry]) -> None:
    payload_ranks = assign_rank({entry.source_id: entry.payload_proxy for entry in entries})
    thrust_ranks = assign_rank({entry.source_id: entry.thrust_proxy for entry in entries})
    maneuver_ranks = assign_rank({entry.source_id: entry.maneuverability_proxy for entry in entries})
    structural_ranks = assign_rank({entry.source_id: entry.structural_proxy for entry in entries})

    for entry in entries:
        payload_rank = payload_ranks[entry.source_id]
        thrust_rank = thrust_ranks[entry.source_id]
        maneuver_rank = maneuver_ranks[entry.source_id]
        structural_rank = structural_ranks[entry.source_id]

        wall_min = MIN_WALL_BY_METHOD_MM[entry.manufacturing_method] + 0.08 * structural_rank
        wall_max = wall_min + (0.6 if entry.manufacturing_method != "foam_core_hotwire" else 0.9)
        engine_count = 1 if entry.chord_m < 0.95 else 2
        payload_mass_max_g = int(350 + 90 * payload_rank)
        payload_mass_min_g = int(max(80, payload_mass_max_g * 0.55))
        takeoff_distance_min_m = int(45 + 3 * payload_rank + max(0, 12 - thrust_rank))
        takeoff_distance_max_m = int(takeoff_distance_min_m + 30 + payload_rank)
        part_floor = 1 if entry.manufacturing_method == "composite_wet_layup" else 2
        part_ceiling = part_floor + 2 + (1 if entry.manufacturing_method == "sheet_balsa_tabbed" else 0)

        entry.response_metrics = {
            "payload_response": round(entry.payload_proxy, 6),
            "thrust_response": round(entry.thrust_proxy, 6),
            "maneuverability_response": round(entry.maneuverability_proxy, 6),
            "structural_response": round(entry.structural_proxy, 6),
        }
        entry.design_spec = {
            "target_speed_mps": round(28.0 + 1.6 * thrust_rank + 0.8 * maneuver_rank, 3),
            "wingspan_limit_m": round(entry.span_m * 1.05, 3),
            "thrust_to_weight_min": round(0.28 + 0.022 * thrust_rank, 3),
            "turn_rate_min_deg_s": round(10.0 + 1.1 * maneuver_rank, 3),
            "required_static_thrust_n": round(70.0 + 14.0 * thrust_rank, 3),
            "engine_diameter_mm": int(round(80 + 5 * thrust_rank + 20 * entry.thickness_ratio * 100)),
            "engine_length_mm": int(round(160 + 9 * thrust_rank + 45 * entry.chord_m * 10)),
            "engine_count_min": engine_count,
            "engine_count_max": engine_count,
            "payload_mass_min_g": payload_mass_min_g,
            "payload_mass_max_g": payload_mass_max_g,
            "takeoff_distance_min_m": takeoff_distance_min_m,
            "takeoff_distance_max_m": takeoff_distance_max_m,
            "wall_thickness_min_mm": round(wall_min, 3),
            "wall_thickness_max_mm": round(wall_max, 3),
            "part_count_min": part_floor,
            "part_count_max": part_ceiling,
            "manufacturing_method": entry.manufacturing_method,
        }

        feasibility = validate_condition_feasibility(entry.design_spec)
        if feasibility["status"] != "pass":
            raise ValueError(f"Generated design spec for {entry.source_id} failed feasibility: {feasibility}")


def build_manifest_records(
    entries: Iterable[Entry],
    *,
    manifest_root: Path,
    preprocessing_hash: str,
    raw_source: Dict[str, Any],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for entry in entries:
        assert entry.design_spec is not None
        assert entry.response_metrics is not None
        record = {
            "sample_id": entry.source_id,
            "geometry_path": str(entry.voxel_path.relative_to(manifest_root.parent)).replace("\\", "/"),
            "stl_path": str(entry.stl_path.relative_to(manifest_root.parent)).replace("\\", "/"),
            "split": entry.split,
            "source_id": entry.source_id,
            "source_url": raw_source["url"],
            "source_license": raw_source["license"],
            "geometry_provenance": (
                "Generated from public Extrality/NACA_simulation ODbL-1.0 profile generator, "
                "extruded to a watertight 3D section, then voxelized with AircraftDesignDataset._voxelize_stl."
            ),
            "preprocessing_version": PREPROCESSING_VERSION,
            "preprocessing_hash": preprocessing_hash,
            "units": "m",
            "original_units": "unit_chord_normalized_profile",
            "design_family": entry.design_family,
            "design_spec": entry.design_spec,
            "response_metrics": entry.response_metrics,
            "analysis_report_path": str(entry.analysis_report_path.relative_to(manifest_root.parent)).replace("\\", "/"),
            "geometry_sha256": entry.geometry_sha256,
            "voxel_sha256": entry.voxel_sha256,
            "profile_sha256": entry.profile_sha256,
            "date_accessed": ACCESS_DATE,
            "design_spec_provenance": {
                "target_speed_mps": "inferred_from_local_drag_and_maneuverability_proxy_rank",
                "wingspan_limit_m": "direct_from_local_extrusion_span",
                "thrust_to_weight_min": "inferred_from_local_thrust_proxy_rank",
                "turn_rate_min_deg_s": "inferred_from_local_maneuverability_proxy_rank",
                "required_static_thrust_n": "inferred_from_local_thrust_proxy_rank",
                "engine_diameter_mm": "inferred_from_thickness_ratio_and_thrust_rank",
                "engine_length_mm": "inferred_from_chord_scale_and_thrust_rank",
                "engine_count_min": "inferred_from_chord_scale",
                "engine_count_max": "inferred_from_chord_scale",
                "payload_mass_min_g": "inferred_from_local_payload_proxy_rank",
                "payload_mass_max_g": "inferred_from_local_payload_proxy_rank",
                "takeoff_distance_min_m": "inferred_from_payload_and_thrust_rank",
                "takeoff_distance_max_m": "inferred_from_payload_and_thrust_rank",
                "wall_thickness_min_mm": "inferred_from_manufacturing_method_minimum_and_structural_rank",
                "wall_thickness_max_mm": "inferred_from_manufacturing_method_minimum_and_structural_rank",
                "part_count_min": "inferred_from_manufacturing_method",
                "part_count_max": "inferred_from_manufacturing_method",
                "manufacturing_method": "inferred_from_thickness_ratio_and profile family",
            },
            "response_metrics_provenance": {
                "payload_response": "local_geometry_analysis_volume_proxy",
                "thrust_response": "local_cfd_drag_inverse_proxy",
                "maneuverability_response": "local_geometry_analysis_camber_and_aspect_ratio_proxy",
                "structural_response": "local_geometry_analysis_thickness_span_proxy",
            },
        }
        records.append(record)
    return records


def run_refinement_study(
    *,
    codes: Sequence[str],
    stl_dir: Path,
    report_path: Path,
) -> None:
    dataset = AircraftDesignDataset(num_samples=0, grid_size=GRID_SIZE)
    cases: Dict[str, Any] = {}
    for code in codes:
        stl_path = stl_dir / f"naca_{code}.stl"
        ladder = []
        for grid_size, steps in ((24, 15), (32, 20), (40, 25)):
            voxels = dataset._voxelize_stl(str(stl_path), grid_size)
            simulator = AdvancedCFDSimulator(CFDConfig(base_grid_resolution=grid_size), torch.device("cpu"))
            cfd = simulator.simulate_aerodynamics(voxels, steps=steps)
            validate_cfd_outputs(
                source_id=f"refinement:{code}",
                grid_size=grid_size,
                steps=steps,
                cfd=cfd,
            )
            ladder.append(
                {
                    "grid_size": grid_size,
                    "steps": steps,
                    "occupancy_ratio": float((voxels > 0.5).float().mean().item()),
                    "drag_coefficient": float(cfd.get("drag_coefficient", 0.0)),
                    "lift_coefficient": float(cfd.get("lift_coefficient", 0.0)),
                }
            )
        cases[code] = ladder

    write_json(
        report_path,
        {
            "study": "grid_refinement",
            "grid_ladder": [24, 32, 40],
            "cases": cases,
            "claim_boundary": (
                "Internal-solver consistency study only. It is not an external validation "
                "against wind-tunnel or TMR reference coefficients."
            ),
            "source_context": {
                "nasa_tmr_naca0012": CONTEXT_SOURCES["nasa_tmr_naca0012"],
                "nasa_tmr_onera_m6": CONTEXT_SOURCES["nasa_tmr_onera_m6"],
            },
        },
    )


def write_analysis_report(entry: Entry) -> None:
    write_json(
        entry.analysis_report_path,
        {
            "sample_id": entry.source_id,
            "source_id": entry.source_id,
            "naca_code": entry.code,
            "split": entry.split,
            "design_family": entry.design_family,
            "geometry_paths": {
                "stl_path": str(entry.stl_path.resolve()),
                "voxel_path": str(entry.voxel_path.resolve()),
                "profile_path": str(entry.profile_path.resolve()),
            },
            "mesh_metrics": {
                "chord_m": entry.chord_m,
                "span_m": entry.span_m,
                "thickness_ratio": entry.thickness_ratio,
                "camber_ratio": entry.camber_ratio,
                "mesh_volume_m3": entry.mesh_volume_m3,
                "mesh_area_m2": entry.mesh_area_m2,
                "occupancy_ratio": entry.occupancy_ratio,
            },
            "solver_config": {
                "grid_size": GRID_SIZE,
                "steps": SIMULATION_STEPS,
                "solver_type": "D3Q27",
                "device": "cpu",
            },
            "local_cfd": entry.cfd,
            "response_metric_inputs": {
                "payload_proxy": entry.payload_proxy,
                "thrust_proxy": entry.thrust_proxy,
                "maneuverability_proxy": entry.maneuverability_proxy,
                "structural_proxy": entry.structural_proxy,
            },
            "hashes": {
                "geometry_sha256": entry.geometry_sha256,
                "voxel_sha256": entry.voxel_sha256,
                "profile_sha256": entry.profile_sha256,
            },
            "claim_boundary": (
                "Local geometry-plus-CFD proxy report for manifest grounding. "
                "Not a whole-aircraft or publication-grade aerodynamic validation artifact."
            ),
        },
    )


def build_report(
    *,
    report_path: Path,
    manifest_path: Path,
    provenance_path: Path,
    entries: Sequence[Entry],
    refinement_path: Path,
) -> None:
    counts: Dict[str, int] = {}
    families: Dict[str, int] = {}
    methods: Dict[str, int] = {}
    for entry in entries:
        counts[entry.split] = counts.get(entry.split, 0) + 1
        families[entry.design_family] = families.get(entry.design_family, 0) + 1
        methods[entry.manufacturing_method] = methods.get(entry.manufacturing_method, 0) + 1

    lines = [
        "# Grounded Aircraft Corpus Report",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Provenance ledger: `{provenance_path}`",
        f"- Record count: `{len(entries)}`",
        "",
        "## Sources",
        "",
        "- Primary geometry source: public `Extrality/NACA_simulation` NACA generator code under `ODbL-1.0`.",
        f"- AirfRANS dataset context: {CONTEXT_SOURCES['airfrans_dataset_docs']}",
        f"- AirfRANS library page: {CONTEXT_SOURCES['airfrans_github']}",
        f"- NASA CRM benchmark context: {CONTEXT_SOURCES['nasa_crm_dpw6']}",
        f"- NASA TMR benchmark context: {CONTEXT_SOURCES['nasa_tmr_naca0012']} and {CONTEXT_SOURCES['nasa_tmr_onera_m6']}",
        "",
        "## Included Corpus",
        "",
        "- This manifest is airfoil-section-heavy rather than a full-aircraft corpus.",
        "- All 20 records are watertight 3D extrusions of public NACA 4-digit or 5-digit section profiles.",
        f"- Split counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Design-family counts: `{json.dumps(families, sort_keys=True)}`",
        f"- Manufacturing-method counts: `{json.dumps(methods, sort_keys=True)}`",
        "",
        "## Exclusions",
        "",
        "- No opaque local STL smoke fixtures were promoted into this manifest.",
        "- NASA CRM STEP assets were kept as validation context only; they were not required for the 20-record claim-bearing manifest because local STEP triangulation was too expensive on this host for this turn.",
        "- No flight-test, wind-tunnel payload, or propulsion claims were imported. Design-spec bounds beyond geometry scale are explicitly inferred and marked as such in provenance.",
        "",
        "## Preprocessing",
        "",
        "- Unit-chord NACA profiles were generated from the public source code, scaled to meters, and extruded into watertight STL solids.",
        "- STL files were voxelized with the repo's `AircraftDesignDataset._voxelize_stl` path at `32^3` resolution.",
        "- Local analysis reports use the repo's internal `D3Q27` solver on CPU with fixed settings.",
        "",
        "## Validation",
        "",
        "- Per-record local geometry/CFD reports were generated and used to populate `response_metrics`.",
        f"- Representative refinement study: `{refinement_path}`",
        "- Response metrics are grounded local proxies, not published aerodynamic coefficients or structural certification data.",
        "",
        "## Gate Support",
        "",
        "- Supports `validate_manifest.py --level claim-bearing`.",
        "- Supports `run_condition_benchmark.py` at the current manifest-grounded contract because all records contain explicit grounded response metrics.",
        "- Supports `condition_feasibility.py` because every record has complete manufacturing fields.",
        "- Does not by itself unlock whole-aircraft validity claims; airfoil-section extrusions are expected to fail `CLI/aircraft_validity.py` heuristics that assume fuselage-wing-tail structure.",
        "- Does not unlock publication-grade aerodynamic optimization or external solver validation; the local reports are bounded internal-solver evidence only.",
        "",
        "## Limits",
        "",
        "- Whole-aircraft evidence: absent from the manifest; still needed for aircraft-structure and planform-claim upgrades.",
        "- Airfoil-only evidence: present and reproducible.",
        "- Solver-validation evidence: limited to internal consistency and refinement trends, with NASA CRM/TMR pages recorded as benchmark context rather than reproduced coefficient agreement.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a grounded aircraft-like corpus and provenance ledger.")
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "docs" / "dataset" / "grounded_aircraft_manifest.jsonl"),
        help="Output manifest path.",
    )
    parser.add_argument(
        "--provenance",
        default=str(REPO_ROOT / "docs" / "dataset" / "grounded_aircraft_provenance.json"),
        help="Output provenance ledger path.",
    )
    parser.add_argument(
        "--report",
        default=str(REPO_ROOT / "docs" / "dataset" / "GROUNDED_AIRCRAFT_CORPUS_REPORT.md"),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    provenance_path = Path(args.provenance).resolve()
    report_path = Path(args.report).resolve()

    dataset_root = manifest_path.parent / "grounded_aircraft"
    raw_root = dataset_root / "raw"
    profiles_root = dataset_root / "profiles"
    stl_root = dataset_root / "stl"
    voxels_root = dataset_root / "voxels"
    analysis_root = dataset_root / "reports" / "analysis"
    refinement_root = dataset_root / "reports" / "refinement"

    for path in (raw_root, profiles_root, stl_root, voxels_root, analysis_root, refinement_root):
        path.mkdir(parents=True, exist_ok=True)

    raw_sources = ensure_raw_sources(raw_root)
    naca_module = load_naca_generator(Path(raw_sources["naca_generator_py"]["path"]))

    preprocessing_payload = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "grid_size": GRID_SIZE,
        "simulation_steps": SIMULATION_STEPS,
        "naca_generator_sha256": raw_sources["naca_generator_py"]["sha256"],
        "voxelizer": "AircraftDesignDataset._voxelize_stl",
        "solver": "AdvancedCFDSimulator(D3Q27)",
    }
    preprocessing_hash = hashlib.sha256(
        json.dumps(preprocessing_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    voxelizer = AircraftDesignDataset(num_samples=0, grid_size=GRID_SIZE)
    entries: List[Entry] = []

    for idx, (code, split) in enumerate(AIRFOIL_CODES):
        params, design_family = parse_code(code)
        thickness_ratio, camber_ratio = compute_thickness_and_camber(naca_module, params)
        chord_m = round(0.72 + 0.035 * idx, 4)
        span_m = round(0.42 + 0.025 * idx, 4)
        manufacturing_method = choose_manufacturing_method(thickness_ratio, code)

        profile_points = np.asarray(
            naca_module.naca_generator(params, nb_samples=300, scale=1.0, origin=(0, 0), verbose=False),
            dtype=np.float64,
        )

        source_id = f"naca_{code}"
        profile_path = profiles_root / f"{source_id}.npy"
        stl_path = stl_root / f"{source_id}.stl"
        voxel_path = voxels_root / f"{source_id}.npy"
        analysis_report_path = analysis_root / f"{source_id}.json"

        np.save(profile_path, profile_points)
        mesh = build_mesh(profile_points, chord_m=chord_m, span_m=span_m)
        mesh.export(stl_path)

        analysis = run_local_analysis(
            stl_path=stl_path,
            voxelizer=voxelizer,
            chord_m=chord_m,
            span_m=span_m,
            thickness_ratio=thickness_ratio,
            camber_ratio=camber_ratio,
            grid_size=GRID_SIZE,
            steps=SIMULATION_STEPS,
        )
        np.save(voxel_path, analysis["voxels"].numpy())

        entry = Entry(
            code=code,
            split=split,
            source_id=source_id,
            design_family=design_family,
            chord_m=chord_m,
            span_m=span_m,
            thickness_ratio=thickness_ratio,
            camber_ratio=camber_ratio,
            manufacturing_method=manufacturing_method,
            stl_path=stl_path,
            voxel_path=voxel_path,
            profile_path=profile_path,
            analysis_report_path=analysis_report_path,
            geometry_sha256=sha256_file(stl_path),
            voxel_sha256=sha256_file(voxel_path),
            profile_sha256=sha256_file(profile_path),
            payload_proxy=analysis["payload_proxy"],
            thrust_proxy=analysis["thrust_proxy"],
            maneuverability_proxy=analysis["maneuverability_proxy"],
            structural_proxy=analysis["structural_proxy"],
            mesh_volume_m3=float(mesh.volume),
            mesh_area_m2=float(mesh.area),
            occupancy_ratio=analysis["occupancy_ratio"],
            cfd=analysis["cfd"],
        )
        entries.append(entry)

    assign_design_specs(entries)
    for entry in entries:
        write_analysis_report(entry)

    refinement_path = refinement_root / "grid_refinement.json"
    run_refinement_study(codes=REFINEMENT_CASES, stl_dir=stl_root, report_path=refinement_path)

    manifest_records = build_manifest_records(
        entries,
        manifest_root=manifest_path,
        preprocessing_hash=preprocessing_hash,
        raw_source=raw_sources["naca_generator_py"],
    )
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest_records),
        encoding="utf-8",
    )

    provenance_records = {}
    for entry in entries:
        provenance_records[entry.source_id] = {
            "source_url": raw_sources["naca_generator_py"]["url"],
            "source_license": raw_sources["naca_generator_py"]["license"],
            "date_accessed": ACCESS_DATE,
            "original_units": "unit_chord_normalized_profile",
            "manifest_split": entry.split,
            "design_family": entry.design_family,
            "reason_for_inclusion": (
                "Section profile chosen to provide reproducible variation across NACA 4-digit "
                "and 5-digit families for grounded conditioning metadata and local response proxies."
            ),
            "geometry_artifacts": {
                "profile_path": str(entry.profile_path.resolve()),
                "stl_path": str(entry.stl_path.resolve()),
                "voxel_path": str(entry.voxel_path.resolve()),
            },
            "hashes": {
                "profile_sha256": entry.profile_sha256,
                "stl_sha256": entry.geometry_sha256,
                "voxel_sha256": entry.voxel_sha256,
            },
            "preprocessing_hash": preprocessing_hash,
            "analysis_report_path": str(entry.analysis_report_path.resolve()),
            "inference_boundary": {
                "published_geometry": "public NACA generator source",
                "aircraft_design_spec": "inferred from local geometry and CFD proxy ranks",
                "response_metrics": "local geometry and CFD analysis only",
            },
        }

    provenance = {
        "schema_version": 1,
        "date_accessed": ACCESS_DATE,
        "claim_boundary": (
            "This ledger grounds an airfoil-section corpus with explicit local proxy metrics. "
            "It is not a whole-aircraft or flight-validated dataset."
        ),
        "build_environment": {
            "python": sys.version,
            "torch": getattr(torch, "__version__", ""),
            "numpy": getattr(np, "__version__", ""),
            "trimesh": getattr(_trimesh_module(), "__version__", ""),
        },
        "source_catalog": {
            **raw_sources,
            **{
                key: {
                    "url": url,
                    "license": "public_web_reference",
                    "date_accessed": ACCESS_DATE,
                    "usage_terms": "Used as citation or validation context only; not redistributed as corpus geometry.",
                }
                for key, url in CONTEXT_SOURCES.items()
            },
        },
        "preprocessing": preprocessing_payload | {"preprocessing_hash": preprocessing_hash},
        "records": provenance_records,
    }
    write_json(provenance_path, provenance)
    build_report(
        report_path=report_path,
        manifest_path=manifest_path,
        provenance_path=provenance_path,
        entries=entries,
        refinement_path=refinement_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
