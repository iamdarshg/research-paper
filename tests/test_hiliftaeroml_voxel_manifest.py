import numpy as np

from CLI.build_hiliftaeroml_voxel_manifest import (
    artifact_stem_for_record,
    build_manifest_record,
    main,
    select_unique_hilift_variants,
)


def _catalog_record(index: int, aoa: int = 4, *, geometry_sha256: str | None = None):
    variant = f"geo_LHC{index:03d}"
    run_id = f"{variant}_AoA_{aoa}"
    record = {
        "source_id": f"hiliftaeroml_surface_{run_id}",
        "source_collection": "hiliftaeroml_crm_hl_surface_runs",
        "source_page": "https://huggingface.co/datasets/nvidia/HiLiftAeroML",
        "exact_cad_url": f"https://example.test/{run_id}/{run_id}.stl",
        "file_format": "stl",
        "force_moment_url": f"https://example.test/{run_id}/force_mom_{run_id}.csv",
        "geometry_values_url": f"https://example.test/{run_id}/geo_values_{run_id}.csv",
        "geometry_variant_id": variant,
        "angle_of_attack_deg": aoa,
        "geometry_uniqueness": "repeated_geometry_variant_across_aoa",
        "source_license": "CC-BY-4.0",
        "license_training_status": "permissive_attribution_required",
        "geometry_kind": "crm_hl_parametric_variant_surface_run",
    }
    if geometry_sha256 is not None:
        record["geometry_sha256"] = geometry_sha256
    return record


def test_selection_uses_one_surface_per_new_geometry_variant():
    catalog = [
        _catalog_record(1, aoa=8),
        _catalog_record(1, aoa=4),
        _catalog_record(2, aoa=8),
        _catalog_record(2, aoa=4),
        _catalog_record(3, aoa=4),
    ]

    selected = select_unique_hilift_variants(
        catalog_records=catalog,
        existing_variant_ids={"geo_LHC001"},
        target_unique_geometries=3,
    )

    assert [row["geometry_variant_id"] for row in selected] == ["geo_LHC002", "geo_LHC003"]
    assert [row["angle_of_attack_deg"] for row in selected] == [4, 4]
    assert len({row["geometry_variant_id"] for row in selected}) == len(selected)


def test_selection_skips_existing_and_catalog_duplicate_content():
    selected = select_unique_hilift_variants(
        catalog_records=[
            _catalog_record(1, geometry_sha256="existing-content"),
            _catalog_record(2, geometry_sha256="duplicate-content"),
            _catalog_record(3, geometry_sha256="duplicate-content"),
            _catalog_record(4, geometry_sha256="new-content"),
        ],
        existing_variant_ids={"geo_LHC001"},
        existing_content_hashes={"existing-content"},
        target_unique_geometries=3,
    )

    assert [row["geometry_variant_id"] for row in selected] == ["geo_LHC002", "geo_LHC004"]


def test_target_unique_geometries_cli_default_is_600(monkeypatch):
    captured = {}

    def fake_build(args):
        captured["target"] = args.target_unique_geometries
        return {"record_count": 1, "failure_count": 0}

    monkeypatch.setattr(
        "CLI.build_hiliftaeroml_voxel_manifest.build_hilift_manifest",
        fake_build,
    )

    assert main([]) == 0
    assert captured["target"] == 600


def test_build_manifest_record_uses_geometry_variant_for_design_spec_seed(tmp_path):
    voxel_path = tmp_path / "voxels" / "sample.npy"
    voxel_path.parent.mkdir()
    np.save(voxel_path, np.ones((4, 4, 4), dtype=np.float32))
    manifest_path = tmp_path / "manifest.jsonl"
    metrics = {
        "source_extents": [1.0, 5.0, 10.0],
        "occupancy_ratio": 0.01,
        "occupied_voxels": 64,
        "source_vertices": 8,
        "source_faces": 12,
        "source_is_watertight": True,
        "source_bounds": [[0, 0, 0], [1, 5, 10]],
    }

    first = build_manifest_record(
        _catalog_record(1, aoa=4),
        manifest_path=manifest_path,
        voxel_path=voxel_path,
        voxel_sha256="voxel",
        geometry_sha256="raw",
        metrics=metrics,
        force_moment={"cd": 0.1, "cl": 1.2, "cm": -0.02},
    )
    second = build_manifest_record(
        _catalog_record(1, aoa=8),
        manifest_path=manifest_path,
        voxel_path=voxel_path,
        voxel_sha256="voxel",
        geometry_sha256="raw",
        metrics=metrics,
        force_moment={"cd": 0.2, "cl": 1.5, "cm": -0.04},
    )

    assert first["geometry_path"] == "voxels/sample.npy"
    assert first["design_spec"] == second["design_spec"]
    assert first["response_metrics"]["drag_coefficient"] == 0.1
    assert second["response_metrics"]["lift_coefficient"] == 1.5
    assert first["geometry_uniqueness"] == "repeated_geometry_variant_across_aoa"
    assert first["source_collection"] == "hiliftaeroml_crm_hl_surface_runs"
    assert first["source_license"] == "CC-BY-4.0"
    assert first["license_training_status"] == "permissive_attribution_required"
    assert first["geometry_kind"] == "crm_hl_parametric_variant_surface_run"
    assert first["geometry_sha256"] == "raw"
    assert first["design_family"] == "hiliftaeroml_crm_hl"
    assert first["split"] == second["split"]


def test_artifact_stem_can_cache_repeated_aoa_records_by_geometry_variant():
    record = _catalog_record(7, aoa=12)

    assert artifact_stem_for_record(record, cache_by_geometry_variant=True) == "geo_LHC007"
    assert (
        artifact_stem_for_record(record, cache_by_geometry_variant=False)
        == "hiliftaeroml_surface_geo_LHC007_AoA_12"
    )
