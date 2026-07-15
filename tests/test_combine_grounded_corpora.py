import json
import os
import sys
from pathlib import Path

import numpy as np


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from combine_grounded_corpora import combine_manifests


def _record(sample_id: str, geometry_path: str):
    return {
        "sample_id": sample_id,
        "source_id": sample_id,
        "geometry_path": geometry_path,
        "split": "train",
        "geometry_provenance": "public CAD",
        "preprocessing_version": "test",
        "units": "normalized",
        "design_family": "test",
        "canonicalization": {"permutation": [0, 1, 2]},
        "design_spec": {},
    }


def test_combiner_deduplicates_canonical_content_and_rewrites_geometry_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("combine_grounded_corpora.validate_manifest_file", lambda *args, **kwargs: {"status": "pass"})
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    voxels = np.zeros((16, 16, 16), dtype=np.uint8)
    voxels[4:12, 7:9, 3:13] = 1
    np.save(first / "a.npy", voxels)
    np.save(second / "b.npy", voxels)
    other = voxels.copy()
    other[2:4, 2:4, 2:4] = 1
    np.save(second / "c.npy", other)
    (first / "manifest.jsonl").write_text(json.dumps(_record("a", "a.npy")) + "\n", encoding="utf-8")
    (second / "manifest.jsonl").write_text(
        "\n".join((json.dumps(_record("b", "b.npy")), json.dumps(_record("c", "c.npy")))) + "\n",
        encoding="utf-8",
    )

    report = combine_manifests(
        [first / "manifest.jsonl", second / "manifest.jsonl"],
        output_manifest=tmp_path / "combined" / "manifest.jsonl",
        output_report=tmp_path / "combined" / "report.json",
        unique_geometry_target=2,
    )

    records = [json.loads(line) for line in (tmp_path / "combined" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert report["record_count"] == 2
    assert report["dropped_counts"] == {"duplicate_canonical_geometry": 1}
    assert len({record["canonical_content_sha256"] for record in records}) == 2
    assert all(record["geometry_path"].startswith("..") for record in records)
    assert all(record["conditioning_mode"] == "unconditioned_source_metadata_only" for record in records)
    assert all(value is None for value in records[0]["design_spec"].values())
