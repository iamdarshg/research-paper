from pathlib import Path

from CLI.validate_manifest import validate_manifest_records


def _records(tmp_path: Path, count: int, *, unique: int):
    geometry_path = tmp_path / "geometry.npy"
    geometry_path.write_bytes(b"geometry")
    records = []
    for index in range(count):
        identity = index % unique
        records.append(
            {
                "geometry_path": geometry_path.name,
                "split": "train",
                "source_id": f"source-{index}",
                "geometry_variant_id": f"variant-{identity}",
                "geometry_sha256": f"hash-{identity}",
                "geometry_provenance": "public exact CAD",
                "preprocessing_version": "voxelizer-v1",
                "units": "normalized",
                "design_family": f"family-{identity % 10}",
                "design_spec": {},
            }
        )
    return records


def test_manifest_gate_requires_600_unique_geometries(tmp_path, monkeypatch):
    monkeypatch.setattr("CLI.validate_manifest._required_design_spec_fields", lambda: [])

    report = validate_manifest_records(
        _records(tmp_path, 600, unique=599),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        level="claim-bearing",
    )

    assert report["status"] == "fail"
    assert report["record_count"] == 600
    assert report["unique_geometry_count"] == 599
    assert report["duplicate_geometry_record_count"] == 1
    assert report["unique_geometry_target"] == 600
    assert report["unique_geometry_target_met"] is False


def test_claim_bearing_manifest_passes_at_unique_geometry_target(tmp_path, monkeypatch):
    monkeypatch.setattr("CLI.validate_manifest._required_design_spec_fields", lambda: [])

    report = validate_manifest_records(
        _records(tmp_path, 600, unique=600),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        level="claim-bearing",
    )

    assert report["status"] == "pass"
    assert report["unique_geometry_target_met"] is True


def test_basic_validation_reports_target_without_enforcing_it(tmp_path):
    report = validate_manifest_records(
        _records(tmp_path, 2, unique=1),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        level="basic",
    )

    assert report["status"] == "pass"
    assert report["record_count"] == 2
    assert report["unique_geometry_count"] == 1
    assert report["duplicate_geometry_record_count"] == 1
    assert report["unique_geometry_target"] == 600
    assert report["unique_geometry_target_met"] is False
