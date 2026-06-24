import numpy as np

from CLI.build_hiliftaeroml_voxel_manifest import (
    artifact_stem_for_record,
    build_manifest_record,
    select_hilift_surface_records,
)


def _catalog_record(index: int, aoa: int = 4):
    variant = f"geo_LHC{index:03d}"
    run_id = f"{variant}_AoA_{aoa}"
    return {
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
    }


def test_select_hilift_surface_records_picks_needed_count_to_cross_target():
    catalog = [_catalog_record(index) for index in range(1, 500)]

    selected = select_hilift_surface_records(
        catalog,
        existing_manifest_count=370,
        target_total_records=752,
        max_records=500,
    )

    assert len(selected) == 382
    assert selected[0]["source_id"] == "hiliftaeroml_surface_geo_LHC001_AoA_4"


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


def test_artifact_stem_can_cache_repeated_aoa_records_by_geometry_variant():
    record = _catalog_record(7, aoa=12)

    assert artifact_stem_for_record(record, cache_by_geometry_variant=True) == "geo_LHC007"
    assert (
        artifact_stem_for_record(record, cache_by_geometry_variant=False)
        == "hiliftaeroml_surface_geo_LHC007_AoA_12"
    )
