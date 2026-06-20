import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from filter_manifest_by_aircraft_validity import filter_manifest_by_aircraft_validity


def _minimally_plausible_aircraft(res=32):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    mid_y = res // 2
    voxels[mid_z - 2:mid_z + 3, mid_y - 3:mid_y + 3, 6:26] = 1.0
    voxels[mid_z - 1:mid_z + 2, 5:27, 13:19] = 1.0
    voxels[mid_z:mid_z + 2, 10:22, 5:9] = 1.0
    return voxels.numpy()


def _asymmetric_blob(res=32):
    voxels = torch.zeros((res, res, res))
    voxels[12:17, 2:7, 5:10] = 1.0
    voxels[12:17, 2:5, 17:24] = 1.0
    return voxels.numpy()


def test_filter_manifest_by_aircraft_validity_writes_passing_records_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        voxel_dir = root / "voxels"
        voxel_dir.mkdir()
        np.save(voxel_dir / "valid.npy", _minimally_plausible_aircraft())
        np.save(voxel_dir / "invalid.npy", _asymmetric_blob())
        records = [
            {"sample_id": "valid", "source_id": "s1", "geometry_path": "voxels/valid.npy", "split": "train"},
            {"sample_id": "invalid", "source_id": "s2", "geometry_path": "voxels/invalid.npy", "split": "test"},
        ]
        manifest = root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

        report = filter_manifest_by_aircraft_validity(
            manifest,
            root / "filtered" / "manifest.jsonl",
            root / "filtered" / "report.json",
        )

        filtered_records = [
            json.loads(line)
            for line in (root / "filtered" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert report["source_record_count"] == 2
        assert report["kept_record_count"] == 1
        assert report["rejected_record_count"] == 1
        assert filtered_records == [
            {
                **records[0],
                "geometry_path": "../voxels/valid.npy",
            }
        ]
        assert report["samples"][1]["failed_checks"]
