import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import build_aircraftverse_corpus as aircraftverse
from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    DesignSpec,
    ModelConfig,
    build_structured_latent_code,
    infer_conditioning_dim,
)


def _valid_performance():
    return {
        "Interferences": 0,
        "Mass": 2.0,
        "Max_Distance": 10.0,
        "Hover_Time": 5.0,
        "Max_Speed": 20.0,
        "Power_MFD": 1.0,
        "Power_MxSpd": 2.0,
        "Speed_MFD": 3.0,
        "Battery_Current_Ratio": 0.8,
    }


def _plausible_aircraft(res=32):
    voxels = torch.zeros((res, res, res))
    mid_z = res // 2
    mid_y = res // 2
    voxels[mid_z - 2:mid_z + 3, mid_y - 3:mid_y + 3, 6:26] = 1.0
    voxels[mid_z - 1:mid_z + 2, 5:27, 13:19] = 1.0
    voxels[mid_z:mid_z + 2, 10:22, 5:9] = 1.0
    return voxels.numpy()


def test_source_performance_rejects_interference_and_zero_filled_output():
    interference = _valid_performance()
    interference["Interferences"] = 1
    with pytest.raises(aircraftverse.CorpusBuildError, match="interferences") as captured:
        aircraftverse.validate_source_performance(interference)
    assert captured.value.code == "source_interference"

    zero_filled = _valid_performance()
    zero_filled["Max_Speed"] = 0.0
    with pytest.raises(aircraftverse.CorpusBuildError) as captured:
        aircraftverse.validate_source_performance(zero_filled)
    assert captured.value.code == "performance_invalid"


def test_source_performance_accepts_only_finite_feasible_values():
    validated = aircraftverse.validate_source_performance(_valid_performance())

    assert validated["Mass"] == 2.0
    assert validated["Battery_Current_Ratio"] == 0.8


def test_geometry_only_admission_never_turns_invalid_performance_into_a_label():
    incomplete = _valid_performance()
    incomplete["Max_Distance"] = 0.0

    validated, report = aircraftverse.assess_source_performance_for_geometry_admission(
        incomplete,
        allow_unavailable=True,
    )

    assert validated is None
    assert report["status"] == "unavailable"
    assert "excluded from labels" in report["claim_boundary"]

    interference = _valid_performance()
    interference["Interferences"] = 1
    with pytest.raises(aircraftverse.CorpusBuildError) as captured:
        aircraftverse.assess_source_performance_for_geometry_admission(
            interference,
            allow_unavailable=True,
        )
    assert captured.value.code == "source_interference"


def test_checksum_and_selection_are_fail_closed_and_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "shard.zip"
        archive.write_bytes(b"known-content")
        with pytest.raises(aircraftverse.CorpusBuildError) as captured:
            aircraftverse.verify_archive_checksum(
                archive,
                expected_md5="00000000000000000000000000000000",
                expected_size=len(b"known-content"),
            )
        assert captured.value.code == "archive_checksum_mismatch"

    ids = ["design_1", "design_2", "design_3"]
    first = aircraftverse.deterministic_design_order(ids, seed=3, shard_key="AircraftVerse_1.zip")
    second = aircraftverse.deterministic_design_order(reversed(ids), seed=3, shard_key="AircraftVerse_1.zip")
    assert first == second


def test_aircraftverse_parameter_records_expose_declared_linear_dimensions():
    low_level = {
        "parameters": [
            {"parameter_name": "FUSELAGE_LENGTH", "value": "145.0"},
            {"parameter_name": "BODY_ROT_ANGLE", "value": "90.0"},
            {"parameter_name": "Arm_1_Length", "value": "326.5"},
        ]
    }

    dimensions = list(aircraftverse._declared_linear_dimensions(low_level))

    assert dimensions == [145.0, 326.5]


def test_design_record_preserves_native_metadata_and_persists_canonical_voxels(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive_path = root / "AircraftVerse_1.zip"
        mesh_bytes = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(file_type="stl")
        prefix = "AircraftVerse_1/design_42"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"{prefix}/cadfile.stl", mesh_bytes)
            archive.writestr(f"{prefix}/Geom.stp", "ISO-10303-21;")
            archive.writestr(f"{prefix}/design_tree.json", json.dumps({"component": "source-native"}))
            archive.writestr(f"{prefix}/design_low_level.json", json.dumps({"fuselage_length": 1.0}))
            archive.writestr(f"{prefix}/design_seq.json", json.dumps(["source-native"]))
            archive.writestr(f"{prefix}/output.json", json.dumps(_valid_performance()))

        monkeypatch.setattr(aircraftverse, "voxelize_mesh", lambda mesh, grid_size: np.transpose(_plausible_aircraft(), (1, 2, 0)))
        metadata = {
            "key": archive_path.name,
            "checksum": "md5:" + hashlib.md5(archive_path.read_bytes()).hexdigest(),
            "size": archive_path.stat().st_size,
            "url": "local-test",
        }
        with zipfile.ZipFile(archive_path) as archive:
            record = aircraftverse._record_from_design(
                archive=archive,
                archive_path=archive_path,
                archive_metadata=metadata,
                prefix=prefix,
                design_id="design_42",
                output_dir=root / "out",
                grid_size=32,
            )

        persisted = np.load(root / "out" / record["geometry_path"])
        assert record["source_native_performance"] == _valid_performance()
        assert record["validated_source_performance"]["Max_Speed"] == 20.0
        assert record["source_performance_validation"]["status"] == "validated"
        assert all(value is None for value in record["design_spec"].values())
        assert not any(record["design_spec_availability"].values())
        assert record["conditioning_mode"] == "unconditioned_source_metadata_only"
        assert record["canonicalization"]["permutation"] != [0, 1, 2]
        assert persisted.shape == (32, 32, 32)
        assert len(record["voxel_sha256"]) == 64

        manifest = root / "out" / "manifest.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        dataset = AircraftDesignDataset(
            grid_size=32,
            latent_dim=16,
            manifest_path=str(manifest),
            seed=0,
        )
        assert tuple(dataset.condition_vectors.shape) == (1, infer_conditioning_dim())
        assert not bool(dataset.condition_vectors.any())
        assert bool(torch.all(dataset.latent_codes >= 0.0))
        assert bool(torch.all(dataset.latent_codes <= 1.0))

        scaled = ModelConfig.scaled_for_corpus(600, 96)
        scaled_dataset = AircraftDesignDataset(
            grid_size=32,
            latent_dim=scaled.latent_dim,
            manifest_path=str(manifest),
            seed=0,
        )
        assert tuple(scaled_dataset.latent_codes.shape) == (1, scaled.latent_dim)


def test_multiscale_latent_distinguishes_unconditioned_aircraft_shapes():
    first = torch.from_numpy(_plausible_aircraft(32)).float()
    second = first.clone()
    second[:, 4:10, 12:20] = 1.0
    condition = torch.zeros(infer_conditioning_dim())
    spec = DesignSpec()

    first_latent = build_structured_latent_code(spec, first, condition, 192)
    second_latent = build_structured_latent_code(spec, second, condition, 192)

    assert first_latent.shape == (192,)
    assert int(torch.count_nonzero(first_latent != second_latent)) >= 20
    assert bool(torch.all((0.0 <= first_latent) & (first_latent <= 1.0)))


def test_mesh_loader_removes_zero_area_tessellation_faces_without_changing_valid_geometry():
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        faces=np.array([[0, 1, 2], [0, 0, 1], [0, 1, 3]]),
        process=False,
    )

    cleaned = aircraftverse._load_stl_mesh(mesh.export(file_type="stl"))

    assert len(cleaned.faces) == 2
    assert np.all(trimesh.triangles.area(cleaned.triangles) > 0.0)
