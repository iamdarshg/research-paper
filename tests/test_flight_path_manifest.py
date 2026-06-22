import json
import os
import sys

import numpy as np


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from build_aircraft_flight_path_manifest import build_flight_path_manifest


def test_build_flight_path_manifest_rewrites_paths_and_adds_profiles(tmp_path):
    source_dir = tmp_path / "source"
    voxel_dir = source_dir / "voxels"
    voxel_dir.mkdir(parents=True)
    np.save(voxel_dir / "sample.npy", np.zeros((4, 4, 4), dtype=np.float32))
    source_manifest = source_dir / "manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "source_id": "source-sample",
                "geometry_path": "voxels/sample.npy",
                "split": "train",
                "design_spec": {
                    "target_speed_mps": 84.0,
                    "takeoff_distance_max_m": 1600,
                    "turn_rate_min_deg_s": 3.5,
                    "payload_mass_max_g": 1000,
                    "manufacturing_method": "composite_wet_layup",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_manifest = tmp_path / "combined" / "manifest.jsonl"
    report_path = tmp_path / "combined" / "report.json"
    report = build_flight_path_manifest(
        [source_manifest],
        output_manifest=output_manifest,
        report_path=report_path,
        run_id="unit-test",
    )

    record = json.loads(output_manifest.read_text(encoding="utf-8").strip())
    resolved_geometry = (output_manifest.parent / record["geometry_path"]).resolve()

    assert report["record_count"] == 1
    assert report_path.exists()
    assert resolved_geometry == (voxel_dir / "sample.npy").resolve()
    assert record["flight_path"]["profile_id"] == "sample"
    assert len(record["flight_path"]["segments"]) >= 4
    assert record["flight_path"]["provenance"].startswith("deterministic conditioning")
