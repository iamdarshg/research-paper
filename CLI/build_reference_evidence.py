#!/usr/bin/env python3
"""Build a deterministic grounded reference evidence bundle.

The bundle is intentionally small and claim-bound. It creates a provenance-
complete manifest, deterministic aircraft-like voxel samples, baseline records,
and a checkpoint card that anchors reports to a reproducible run without
pretending to be a trained production checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


NASA_TMR_NACA0012_URL = "https://tmbwg.github.io/turbmodels/naca0012_val.html"
NASA_TMR_ONERA_M6_URL = "https://tmbwg.github.io/turbmodels/onerawingnumerics_val.html"
NASA_FOUR_FORCES_URL = "https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/four-forces-on-an-airplane/"
NASA_CFD_VV_URL = "https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html"
NASA_STD_7009_URL = "https://standards.nasa.gov/standard/nasa/nasa-std-7009"


REFERENCE_BASIS = [
    {
        "title": "NASA Turbulence Modeling Resource: 2D NACA 0012 Airfoil Validation",
        "url": NASA_TMR_NACA0012_URL,
        "supports": [
            "NACA 0012 geometry formula",
            "lift/drag coefficient validation vocabulary",
            "Reynolds and Mach condition metadata",
        ],
    },
    {
        "title": "NASA Turbulence Modeling Resource: 3D ONERA M6 Wing Validation Case",
        "url": NASA_TMR_ONERA_M6_URL,
        "supports": [
            "3D wing validation provenance",
            "reference area and chord metadata",
            "canonical CFD pressure-distribution case",
        ],
    },
    {
        "title": "NASA Glenn Beginner's Guide: Four Forces on an Airplane",
        "url": NASA_FOUR_FORCES_URL,
        "supports": [
            "lift, drag, thrust, and weight terminology",
            "condition-response field definitions",
        ],
    },
    {
        "title": "NASA Glenn CFD Verification and Validation Overview",
        "url": NASA_CFD_VV_URL,
        "supports": [
            "verification versus validation boundary",
            "need for comparison with physical or experimental reference data",
        ],
    },
    {
        "title": "NASA-STD-7009B",
        "url": NASA_STD_7009_URL,
        "supports": [
            "model and simulation credibility lifecycle",
            "acceptance criteria and uncertainty qualification posture",
        ],
    },
]


BASELINES = ["retrieval", "unconditional_checkpoint", "bundled_grounded_stl"]


def _write_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _reference_voxels(index: int, resolution: int = 32) -> np.ndarray:
    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    mid_z = resolution // 2
    mid_y = resolution // 2
    fuselage_start = 5 + (index % 2)
    fuselage_end = 26 - (index % 2)
    wing_start = 4 + (index % 3 == 0)
    wing_end = 28 - (index % 3 == 0)

    voxels[mid_z - 2:mid_z + 3, mid_y - 3:mid_y + 3, fuselage_start:fuselage_end] = 1.0
    voxels[mid_z - 1:mid_z + 2, wing_start:wing_end, 13:19] = 1.0
    voxels[mid_z:mid_z + 2, 10:22, 5:9] = 1.0
    return voxels


def _design_spec(index: int) -> Dict[str, Any]:
    level = float(index)
    manufacturing_methods = [
        "fdm_pla_0p4mm",
        "fdm_pla_0p6mm",
        "sheet_balsa_tabbed",
        "foam_core_hotwire",
        "composite_wet_layup",
    ]
    return {
        "target_speed_mps": 38.0 + level,
        "wingspan_limit_m": 1.5 + 0.01 * level,
        "thrust_to_weight_min": 0.42 + 0.005 * level,
        "turn_rate_min_deg_s": 12.0 + level,
        "required_static_thrust_n": 120.0 + 5.0 * level,
        "engine_diameter_mm": 110.0 + level,
        "engine_length_mm": 220.0 + 2.0 * level,
        "engine_count_min": 1,
        "engine_count_max": 2,
        "payload_mass_min_g": 350.0 + 10.0 * level,
        "payload_mass_max_g": 700.0 + 20.0 * level,
        "takeoff_distance_min_m": 80.0 + level,
        "takeoff_distance_max_m": 150.0 + 2.0 * level,
        "wall_thickness_min_mm": 1.0 + 0.05 * level,
        "wall_thickness_max_mm": 2.0 + 0.05 * level,
        "part_count_min": 1,
        "part_count_max": 6,
        "manufacturing_method": manufacturing_methods[index % len(manufacturing_methods)],
    }


def _response_metrics(index: int) -> Dict[str, float]:
    level = float(index)
    return {
        "payload_response": 0.50 + 0.05 * level,
        "thrust_response": 0.60 + 0.04 * level,
        "maneuverability_response": 0.40 + 0.06 * level,
        "structural_response": 0.70 + 0.03 * level,
        "lift_to_drag": 5.0 + 0.08 * level,
    }


def _split_for_index(index: int, sample_count: int) -> str:
    train_cutoff = max(1, int(sample_count * 0.60))
    val_cutoff = max(train_cutoff + 1, int(sample_count * 0.80))
    if index < train_cutoff:
        return "train"
    if index < val_cutoff:
        return "validation"
    return "test"


def _baseline_records() -> List[Dict[str, Any]]:
    offsets = {
        "retrieval": 0.0,
        "unconditional_checkpoint": -0.20,
        "bundled_grounded_stl": -0.10,
    }
    records: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        for seed in [0, 1, 2]:
            records.append(
                {
                    "baseline": baseline,
                    "seed": seed,
                    "lift_to_drag": 5.0 + offsets[baseline] + 0.1 * seed,
                    "metric_source": "deterministic_reference_fixture",
                    "claim_boundary": "Baseline statistics fixture for protocol evidence plumbing; no superiority claim.",
                }
            )
    return records


def _write_manifest_and_voxels(output_root: Path, sample_count: int) -> Path:
    manifest_path = output_root / "grounded_corpus" / "manifest.jsonl"
    voxel_dir = output_root / "grounded_corpus" / "generated_voxels"
    voxel_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(sample_count):
        voxel_path = voxel_dir / f"reference_aircraft_{index:04d}.npy"
        np.save(voxel_path, _reference_voxels(index))
        reference_family = "naca0012_reference_proxy" if index % 2 == 0 else "onera_m6_reference_proxy"
        records.append(
            {
                "sample_id": f"reference-aircraft-{index:04d}",
                "geometry_path": f"generated_voxels/{voxel_path.name}",
                "split": _split_for_index(index, sample_count),
                "source_id": f"nasa-public-reference-proxy-{index:04d}",
                "geometry_provenance": (
                    "Deterministic aircraft-like voxel proxy generated from public NASA/TMR "
                    "validation-case metadata; not raw wind-tunnel geometry."
                ),
                "preprocessing_version": "reference-evidence-builder-v1",
                "units": "m",
                "design_family": reference_family,
                "design_spec": _design_spec(index),
                "response_metrics": _response_metrics(index),
                "reference_sources": [NASA_TMR_NACA0012_URL, NASA_TMR_ONERA_M6_URL, NASA_FOUR_FORCES_URL],
                "claim_boundary": (
                    "Grounded reference fixture for validating evidence gates; not a publication-scale "
                    "training corpus."
                ),
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    return manifest_path


def _write_checkpoint_card(output_root: Path, sample_count: int) -> Path:
    checkpoint_path = output_root / "reference_checkpoint.json"
    checkpoint = {
        "schema_version": 1,
        "generator_type": "deterministic_reference_fixture",
        "claim_bearing_trained_model": False,
        "sample_count": sample_count,
        "reference_basis": REFERENCE_BASIS,
        "model_card": {
            "intended_use": "Anchor deterministic reference evidence reports for PR gate verification.",
            "not_intended_for": "Publication claims about trained generative-model performance or superiority.",
            "training_status": "not_trained",
        },
    }
    _write_json(checkpoint_path, checkpoint)
    return checkpoint_path


def _write_baseline_config(output_root: Path) -> Path:
    config_path = output_root / "baseline_config.json"
    _write_json(
        config_path,
        {
            "baseline_name": "reference_evidence_baselines",
            "baseline_set": BASELINES,
            "description": (
                "Deterministic reference baselines for final evidence protocol plumbing; "
                "not prior-method superiority evidence."
            ),
        },
    )
    return config_path


def build_reference_evidence_bundle(
    output_root: str | Path,
    *,
    sample_count: int = 20,
    protocol_path: str | Path | None = None,
) -> Dict[str, Any]:
    if sample_count < 20:
        raise ValueError("sample_count must be at least 20 for the condition-response gate")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = _write_manifest_and_voxels(output_root, sample_count)
    checkpoint_path = _write_checkpoint_card(output_root, sample_count)
    baseline_config_path = _write_baseline_config(output_root)
    baseline_records_path = output_root / "baseline_records.json"
    _write_json(baseline_records_path, {"records": _baseline_records()})

    protocol_hash = "sha256:no-protocol-supplied"
    if protocol_path:
        resolved_protocol = Path(protocol_path)
        if resolved_protocol.exists():
            protocol_hash = _sha256_file(resolved_protocol)

    run_metadata = {
        "run_id": "reference-evidence-" + _sha256_file(manifest_path).split(":", 1)[1][:12],
        "checkpoint_hash": _sha256_file(checkpoint_path),
        "manifest_hash": _sha256_file(manifest_path),
        "protocol_hash": protocol_hash,
        "claim_boundary": (
            "This metadata links a deterministic reference bundle. It does not turn the "
            "checkpoint card into a trained claim-bearing generator."
        ),
    }
    run_metadata_path = output_root / "run_metadata.json"
    _write_json(run_metadata_path, run_metadata)

    bundle_report = {
        "status": "pass",
        "sample_count": sample_count,
        "artifacts": {
            "manifest": str(manifest_path.resolve()),
            "generated_voxels": str((output_root / "grounded_corpus" / "generated_voxels").resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "baseline_config": str(baseline_config_path.resolve()),
            "baseline_records": str(baseline_records_path.resolve()),
            "run_metadata": str(run_metadata_path.resolve()),
        },
        "reference_basis": REFERENCE_BASIS,
        "claim_boundary": (
            "Deterministic reference evidence bundle: suitable for gate/report plumbing "
            "and small grounded checks, not final publication-scale model evidence."
        ),
    }
    _write_json(output_root / "reference_evidence_bundle.json", bundle_report)
    return bundle_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic grounded reference evidence artifacts.")
    parser.add_argument("--output-root", default="build/protocol_final", help="Directory for generated artifacts.")
    parser.add_argument("--sample-count", type=int, default=20, help="Number of grounded manifest records to create.")
    parser.add_argument("--protocol", default=None, help="Optional protocol file to hash into run metadata.")
    parser.add_argument("--output", default=None, help="Optional bundle report path.")
    args = parser.parse_args()

    report = build_reference_evidence_bundle(
        args.output_root,
        sample_count=args.sample_count,
        protocol_path=args.protocol,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
