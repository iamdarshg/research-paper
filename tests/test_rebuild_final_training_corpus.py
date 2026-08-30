import hashlib
import importlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


DESIGN_SPEC_FIELDS = (
    "target_speed_mps",
    "wingspan_limit_m",
    "thrust_to_weight_min",
    "turn_rate_min_deg_s",
    "required_static_thrust_n",
    "engine_diameter_mm",
    "engine_length_mm",
    "engine_count_min",
    "engine_count_max",
    "payload_mass_min_g",
    "payload_mass_max_g",
    "takeoff_distance_min_m",
    "takeoff_distance_max_m",
    "wall_thickness_min_mm",
    "wall_thickness_max_mm",
    "part_count_min",
    "part_count_max",
    "manufacturing_method",
)


def _builder_module():
    try:
        return importlib.import_module("CLI.rebuild_final_training_corpus")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Task 2 rebuild module is missing: {exc}")


def _semantic_hash(voxels: np.ndarray) -> str:
    canonical = (voxels > 0.5).astype(np.uint8)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _aircraft_like(index: int = 0) -> np.ndarray:
    """Create a connected, centered 96-cubed fixture with a deterministic variant."""
    voxels = np.zeros((96, 96, 96), dtype=np.uint8)
    x_offset = index % 3
    voxels[42:54, 30:66, 20 + x_offset : 76 + x_offset] = 1
    voxels[36:60, 45:51, 35 + x_offset : 61 + x_offset] = 1
    if index:
        voxels[39:42, 48:52, 28 + x_offset : 34 + x_offset] = 1
    return voxels


def _null_metadata():
    return {
        "design_spec": {field: None for field in DESIGN_SPEC_FIELDS},
        "design_spec_availability": {field: False for field in DESIGN_SPEC_FIELDS},
        "design_spec_provenance": {
            field: "unavailable; not inherited or inferred for generated geometry"
            for field in DESIGN_SPEC_FIELDS
        },
    }


def _write_source_manifest(root: Path, arrays, splits=None) -> Path:
    source_root = root / "source"
    voxel_root = source_root / "voxels"
    voxel_root.mkdir(parents=True)
    splits = splits or ["train"] * len(arrays)
    records = []
    for index, (array, split) in enumerate(zip(arrays, splits)):
        filename = f"source-{index}.npy"
        np.save(voxel_root / filename, array, allow_pickle=False)
        content_hash = _semantic_hash(array)
        voxel_file_hash = hashlib.sha256((voxel_root / filename).read_bytes()).hexdigest()
        record = {
            "source_id": f"source-{index}",
            "sample_id": f"sample-{index}",
            "source_type": "original",
            "geometry_path": f"voxels/{filename}",
            "canonical_content_sha256": content_hash,
            "voxel_sha256": voxel_file_hash,
            "split": split,
            "geometry_provenance": "controlled source fixture",
            "preprocessing_version": "controlled-source-v1",
            "units": "normalized voxel lattice; occupancy is dimensionless",
            "design_family": "controlled_source",
            "source_manifest_path": str((source_root / "manifest.jsonl").resolve()),
            "archive_url": str((source_root / "archives" / "source.zip").resolve()),
        }
        record.update(_null_metadata())
        records.append(record)

    manifest = source_root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def _read_records(manifest: Path):
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_small_build(source_manifest: Path, output_dir: Path, **overrides):
    builder = _builder_module()
    defaults = {
        "perturbation_batches": (),
        "procedural_count": 0,
        "expected_original_count": 1,
        "expected_perturbation_count": 0,
        "expected_procedural_count": 0,
        "expected_total_count": 1,
    }
    defaults.update(overrides)
    return builder.rebuild_final_training_corpus(source_manifest, output_dir, **defaults)


def test_build_perturbation_record_has_truthful_claim_metadata():
    builder = _builder_module()
    parent = {"source_id": "parent-001", "split": "holdout"}
    record = builder.build_perturbation_record(
        parent,
        transform="wing_dihedral_up",
        parent_record_index=17,
        parent_hash="a" * 64,
        child_hash="b" * 64,
        geometry_path="voxels/" + "b" * 64 + ".npy",
    )

    assert record["source_id"] == "perturb:wing_dihedral_up:parent-001"
    assert record["source_type"] == "perturbation_expanded"
    assert record["parent_source_id"] == "parent-001"
    assert record["parent_record_index"] == 17
    assert record["parent_canonical_content_sha256"] == "a" * 64
    assert record["transform"] == "wing_dihedral_up"
    assert record["geometry_path"] == "voxels/" + "b" * 64 + ".npy"
    assert record["canonical_content_sha256"] == "b" * 64
    assert record["voxel_sha256"] == "b" * 64
    assert record["split"] == "holdout"
    assert record["conditioning_mode"] == "unconditioned_source_metadata_only"
    assert record["design_family"] == "generated_perturbation"
    assert record["canonicalization"]["permutation"] == [0, 1, 2]
    assert record["split"] == "holdout"
    assert "not independent CAD" in record["geometry_provenance"]
    assert record["preprocessing_version"] == "final-training-corpus-v1-perturbation-v1"
    assert record["units"] == "normalized voxel lattice; occupancy is dimensionless"


def test_build_procedural_record_has_truthful_claim_metadata():
    builder = _builder_module()
    record = builder.build_procedural_record(
        aircraft_type="glider",
        accepted_index=23,
        attempt=91,
        seed=42,
        child_hash="c" * 64,
        geometry_path="voxels/" + "c" * 64 + ".npy",
    )

    assert record["source_id"] == "proc:glider:23"
    assert record["source_type"] == "procedural"
    assert record["aircraft_type"] == "glider"
    assert record["accepted_index"] == 23
    assert record["attempt"] == 91
    assert record["generator_seed"] == 42
    assert record["geometry_path"] == "voxels/" + "c" * 64 + ".npy"
    assert record["canonical_content_sha256"] == "c" * 64
    assert record["voxel_sha256"] == "c" * 64
    assert record["split"] == "train"
    assert record["conditioning_mode"] == "unconditioned_source_metadata_only"
    assert record["design_family"] == "generated_procedural_glider"
    assert record["canonicalization"]["permutation"] == [0, 1, 2]
    assert "NOT real CAD" in record["geometry_provenance"]
    assert record["preprocessing_version"] == "final-training-corpus-v1-procedural-v1"
    assert record["units"] == "normalized voxel lattice; occupancy is dimensionless"


def test_generated_records_have_complete_null_unavailable_conditioning():
    builder = _builder_module()
    records = [
        builder.build_perturbation_record(
            {"source_id": "parent", "split": "train"},
            transform="nose_thin",
            parent_record_index=0,
            parent_hash="a" * 64,
            child_hash="b" * 64,
            geometry_path="voxels/" + "b" * 64 + ".npy",
        ),
        builder.build_procedural_record(
            aircraft_type="delta_wing",
            accepted_index=0,
            attempt=1,
            seed=42,
            child_hash="c" * 64,
            geometry_path="voxels/" + "c" * 64 + ".npy",
        ),
    ]

    for record in records:
        assert set(record["design_spec"]) == set(DESIGN_SPEC_FIELDS)
        assert all(value is None for value in record["design_spec"].values())
        assert record["design_spec_availability"] == {field: False for field in DESIGN_SPEC_FIELDS}
        assert set(record["design_spec_provenance"]) == set(DESIGN_SPEC_FIELDS)
        assert all(record["design_spec_provenance"].values())
        assert all(not value for value in record["design_spec_availability"].values())


def test_missing_original_geometry_fails_closed_without_publishing(tmp_path):
    builder = _builder_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    missing_hash = "d" * 64
    record = {
        "source_id": "missing-source",
        "geometry_path": "voxels/missing.npy",
        "canonical_content_sha256": missing_hash,
        "split": "train",
    }
    source_manifest = source_root / "manifest.jsonl"
    source_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output_dir = tmp_path / "published"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        _run_small_build(source_manifest, output_dir)

    assert not output_dir.exists()
    assert not any("staging" in path.name for path in tmp_path.iterdir())


def test_raw_source_count_is_checked_before_duplicate_content_filtering(tmp_path):
    array = _aircraft_like()
    source_manifest = _write_source_manifest(tmp_path, [array, array.copy()])
    output_dir = tmp_path / "published"

    with pytest.raises(ValueError, match="raw source record count"):
        _run_small_build(source_manifest, output_dir)

    assert not output_dir.exists()


def test_duplicate_canonical_source_content_fails_closed(tmp_path):
    array = _aircraft_like()
    source_manifest = _write_source_manifest(tmp_path, [array, array.copy()])
    output_dir = tmp_path / "published"

    with pytest.raises(ValueError, match="duplicate.*canonical"):
        _run_small_build(
            source_manifest,
            output_dir,
            expected_original_count=2,
            expected_total_count=2,
        )

    assert not output_dir.exists()


def test_duplicate_source_id_with_different_content_fails_closed(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like(0), _aircraft_like(1)])
    records = _read_records(source_manifest)
    records[1]["source_id"] = records[0]["source_id"]
    source_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "published"

    with pytest.raises(ValueError, match="duplicate source_id"):
        _run_small_build(
            source_manifest,
            output_dir,
            expected_original_count=2,
            expected_total_count=2,
        )

    assert not output_dir.exists()


def test_source_semantic_and_file_hashes_are_validated_independently(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    valid_output = tmp_path / "valid-published"
    _run_small_build(source_manifest, valid_output)
    valid_record = _read_records(valid_output / "combined_training_manifest.jsonl")[0]
    assert valid_record["canonical_content_sha256"] == _semantic_hash(_aircraft_like())
    assert valid_record["voxel_sha256"] == hashlib.sha256(
        (valid_output / valid_record["geometry_path"]).read_bytes()
    ).hexdigest()

    records = _read_records(source_manifest)
    records[0]["voxel_sha256"] = "f" * 64
    source_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "invalid-published"

    with pytest.raises(ValueError, match="hash"):
        _run_small_build(source_manifest, output_dir)
    assert not output_dir.exists()


def test_rebuild_resamples_and_replays_declared_target_grid(tmp_path):
    builder = _builder_module()
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published-128"

    report = _run_small_build(source_manifest, output_dir, target_grid_size=128)

    record = _read_records(output_dir / "combined_training_manifest.jsonl")[0]
    geometry = np.load(output_dir / record["geometry_path"], allow_pickle=False)
    build_spec = json.loads((output_dir / "build_spec.json").read_text(encoding="utf-8"))
    assert geometry.shape == (128, 128, 128)
    assert geometry.dtype == np.uint8
    assert build_spec["source_grid_shape"] == [96, 96, 96]
    assert build_spec["target_grid_size"] == 128
    assert build_spec["grid_shape"] == [128, 128, 128]
    assert report["shape_counts"] == {"128x128x128": 1}
    assert builder.replay_published_corpus(output_dir, source_manifest)["status"] == "pass"


def test_perturbation_children_inherit_parent_splits_and_report_grouped_counts(tmp_path):
    source_manifest = _write_source_manifest(
        tmp_path,
        [_aircraft_like(0), _aircraft_like(1), _aircraft_like(2)],
        splits=["train", "val", "test"],
    )
    output_dir = tmp_path / "published"

    report = _run_small_build(
        source_manifest,
        output_dir,
        perturbation_batches=(("wing_dihedral_up",),),
        expected_original_count=3,
        expected_perturbation_count=3,
        expected_total_count=6,
    )

    records = _read_records(output_dir / "combined_training_manifest.jsonl")
    originals = {record["source_id"]: record for record in records[:3]}
    children = records[3:]
    assert all(child["split"] == originals[child["parent_source_id"]]["split"] for child in children)
    assert report["parent_split_counts"]["descendants_by_parent_split"] == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert report["parent_split_counts"]["cross_split_violations"] == 0

def test_rebuild_uses_two_explicit_perturbation_batches_and_seed_42(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published"
    batches = (("wing_dihedral_up", "tail_widen_30"), ("wing_dihedral_down",))

    report = _run_small_build(
        source_manifest,
        output_dir,
        perturbation_batches=batches,
        expected_perturbation_count=3,
        expected_total_count=4,
        procedural_seed=42,
    )

    build_spec = json.loads((output_dir / "build_spec.json").read_text(encoding="utf-8"))
    records = _read_records(output_dir / "combined_training_manifest.jsonl")
    assert build_spec["perturbation_batches"] == [list(batch) for batch in batches]
    assert build_spec["procedural_seed"] == 42
    assert report["batch_counts"]["perturbation_accepted"] == 3
    assert [record["transform"] for record in records[1:]] == [
        "wing_dihedral_up",
        "tail_widen_30",
        "wing_dihedral_down",
    ]


def test_rebuild_preserves_original_split_counts_and_marks_generated_train(tmp_path):
    source_manifest = _write_source_manifest(
        tmp_path,
        [_aircraft_like(0), _aircraft_like(1), _aircraft_like(2)],
        splits=["train", "val", "test"],
    )
    output_dir = tmp_path / "published"

    _run_small_build(
        source_manifest,
        output_dir,
        procedural_count=1,
        expected_original_count=3,
        expected_procedural_count=1,
        expected_total_count=4,
    )

    records = _read_records(output_dir / "combined_training_manifest.jsonl")
    assert Counter(record["split"] for record in records[:3]) == Counter({"train": 1, "val": 1, "test": 1})
    assert all(record["split"] == "train" for record in records[3:])
    assert all(record["source_type"] == "original" for record in records[:3])
    assert records[3]["source_type"] == "procedural"


def test_rebuild_writes_only_relative_content_addressed_geometry_paths(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)

    manifest_text = (output_dir / "combined_training_manifest.jsonl").read_text(encoding="utf-8")
    records = _read_records(output_dir / "combined_training_manifest.jsonl")
    for record in records:
        geometry_path = Path(record["geometry_path"])
        assert not geometry_path.is_absolute()
        assert geometry_path.parts[0] == "voxels"
        assert re.fullmatch(r"[0-9a-f]{64}\.npy", geometry_path.name)
        assert (output_dir / geometry_path).is_file()
    assert str(source_manifest.resolve()) not in manifest_text


def test_rebuild_sanitizes_nested_file_urls_and_embedded_local_paths(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    records = _read_records(source_manifest)
    records[0]["nested_metadata"] = {
        "local_file_url": "file:///C:/Users/Darsh Gupta/AppData/Local/raw.stl",
        "note": r"derived from D:\CodeProjects\research-paper\raw.stl",
    }
    source_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)

    manifest_text = (output_dir / "combined_training_manifest.jsonl").read_text(encoding="utf-8")
    assert "file:///C:/Users" not in manifest_text
    assert "D:\\CodeProjects\\research-paper" not in manifest_text


def test_sparse_npy_writer_preserves_an_occupied_final_voxel(tmp_path):
    array = _aircraft_like()
    array[-1, -1, -1] = 1
    source_manifest = _write_source_manifest(tmp_path, [array])
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)

    record = _read_records(output_dir / "combined_training_manifest.jsonl")[0]
    loaded = np.load(output_dir / record["geometry_path"], allow_pickle=False)
    assert loaded[-1, -1, -1] == 1
    assert _semantic_hash(loaded) == record["canonical_content_sha256"]


def test_cleanup_rejects_empty_or_current_directory_targets(tmp_path, monkeypatch):
    builder = _builder_module()
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    for candidate in (Path(), Path(".")):
        with pytest.raises(ValueError, match="staging|target|directory"):
            builder._safe_cleanup_staging(candidate, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_cleanup_rejects_reparse_staging_target_without_following_it(tmp_path):
    builder = _builder_module()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    staging_link = tmp_path / ".published.staging-attacker"
    try:
        staging_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse|symlink|staging"):
        builder._safe_cleanup_staging(staging_link, tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_builder_rejects_symlinked_output_target_before_publication(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_target = tmp_path / "outside-target"
    output_link = tmp_path / "published-link"
    try:
        output_link.symlink_to(output_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse|symlink|unsafe|output"):
        _run_small_build(source_manifest, output_link)

    assert not output_target.exists()


def test_rebuild_loads_after_output_directory_relocation(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)

    relocated = tmp_path / "relocated" / "corpus"
    relocated.parent.mkdir()
    shutil.move(str(output_dir), str(relocated))
    manifest = relocated / "combined_training_manifest.jsonl"

    from CLI.aircraft_diffusion_cfd import AircraftDesignDataset
    from CLI.validate_manifest import validate_manifest_file

    basic = validate_manifest_file(str(manifest), level="basic", unique_geometry_target=1)
    claim = validate_manifest_file(str(manifest), level="claim-bearing", unique_geometry_target=1)
    assert basic["status"] == "pass"
    assert claim["status"] == "pass"

    representative = relocated / "representative.jsonl"
    representative.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    dataset = AircraftDesignDataset(manifest_path=str(representative), grid_size=96, latent_dim=8)
    assert len(dataset) == 1
    assert tuple(dataset[0]["geometry"].shape) == (96, 96, 96)


def test_full_replay_recomputes_published_identity_and_build_identity(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)

    builder = _builder_module()
    replay = builder.replay_published_corpus(
        output_dir,
        source_manifest,
        unique_geometry_target=1,
        expected_total_count=1,
    )
    assert replay["status"] == "pass"
    assert replay["record_count"] == 1
    assert replay["recomputed_geometry_count"] == 1
    build_spec = json.loads((output_dir / "build_spec.json").read_text(encoding="utf-8"))
    assert build_spec["builder_commit"]
    assert build_spec["dependency_versions"]["numpy"]
    assert build_spec["storage"]["filesystem_compression"] == "not controlled by builder"


@pytest.mark.parametrize("bad_kind", ["shape", "hash"])
def test_rebuild_rejects_bad_shape_or_hash_before_publish(tmp_path, bad_kind):
    source_root = tmp_path / "source"
    voxel_root = source_root / "voxels"
    voxel_root.mkdir(parents=True)
    array = _aircraft_like()
    if bad_kind == "shape":
        array = np.zeros((95, 96, 96), dtype=np.uint8)
    np.save(voxel_root / "source.npy", array, allow_pickle=False)
    declared_hash = "e" * 64 if bad_kind == "hash" else _semantic_hash(array)
    record = {
        "source_id": "source-0",
        "geometry_path": "voxels/source.npy",
        "canonical_content_sha256": declared_hash,
        "split": "train",
    }
    source_manifest = source_root / "manifest.jsonl"
    source_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output_dir = tmp_path / "published"

    with pytest.raises(ValueError, match="(shape|hash|96)"):
        _run_small_build(source_manifest, output_dir)

    assert not output_dir.exists()


def test_rebuild_manifest_and_reports_are_byte_deterministic(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _run_small_build(source_manifest, first)
    _run_small_build(source_manifest, second)

    for filename in ("combined_training_manifest.jsonl", "build_spec.json", "report.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    first_records = _read_records(first / "combined_training_manifest.jsonl")
    second_records = _read_records(second / "combined_training_manifest.jsonl")
    assert first_records[0]["geometry_path"] == second_records[0]["geometry_path"]
    assert (first / first_records[0]["geometry_path"]).read_bytes() == (
        second / second_records[0]["geometry_path"]
    ).read_bytes()


def test_full_target_contract_is_declared_by_default():
    builder = _builder_module()
    assert builder.DEFAULT_EXPECTED_ORIGINAL_COUNT == 1069
    assert builder.DEFAULT_EXPECTED_PERTURBATION_COUNT == 4958
    assert builder.DEFAULT_EXPECTED_PROCEDURAL_COUNT == 2000
    assert builder.DEFAULT_EXPECTED_TOTAL_COUNT == 8027
    assert builder.DEFAULT_PERTURBATION_BATCHES == (
        ("wing_dihedral_up", "tail_widen_30"),
        ("wing_dihedral_down", "tail_widen_50", "nose_thin", "airfoil_thicken"),
    )


def test_seed_42_procedural_stream_has_each_declared_family(tmp_path):
    del tmp_path
    generator = importlib.import_module("CLI.procedural_aircraft_generator")
    samples, stats = generator.generate_procedural_samples(2000, 42)

    assert len(samples) == 2000
    assert stats["accepted"] == 2000
    assert all(stats["per_type"][aircraft_type] >= 1 for aircraft_type in generator.AIRCRAFT_TYPES)


def test_standalone_perturbation_cli_emits_claim_bearing_records(tmp_path, monkeypatch):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "perturb-output"
    perturb = importlib.import_module("CLI.perturb_corpus")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perturb_corpus",
            "--manifest",
            str(source_manifest),
            "--output-dir",
            str(output_dir),
            "--transforms",
            "wing_dihedral_up",
        ],
    )

    assert perturb.main() == 0
    records = _read_records(output_dir / "manifest.jsonl")
    assert records
    record = records[0]
    assert record["geometry_path"] == "voxels/" + record["canonical_content_sha256"] + ".npy"
    assert record["geometry_provenance"]
    assert record["preprocessing_version"] == "final-training-corpus-v1-perturbation-v1"
    assert record["units"] == "normalized voxel lattice; occupancy is dimensionless"
    assert record["design_family"] == "generated_perturbation"
    assert set(record["design_spec"]) == set(DESIGN_SPEC_FIELDS)
    assert all(value is None for value in record["design_spec"].values())
    assert record["design_spec_availability"] == {field: False for field in DESIGN_SPEC_FIELDS}
    assert set(record["design_spec_provenance"]) == set(DESIGN_SPEC_FIELDS)


def test_standalone_perturbation_cli_fails_closed_on_missing_geometry(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_manifest = source_root / "manifest.jsonl"
    source_manifest.write_text(
        json.dumps(
            {
                "source_id": "missing",
                "geometry_path": "voxels/missing.npy",
                "canonical_content_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "perturb-output"
    perturb = importlib.import_module("CLI.perturb_corpus")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perturb_corpus",
            "--manifest",
            str(source_manifest),
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        perturb.main()
    assert not output_dir.exists()


def test_standalone_procedural_cli_emits_claim_bearing_records(tmp_path, monkeypatch):
    output_dir = tmp_path / "procedural-output"
    procedural = importlib.import_module("CLI.procedural_aircraft_generator")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "procedural_aircraft_generator",
            "--count",
            "1",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert procedural.main() == 0
    records = _read_records(output_dir / "manifest.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["geometry_path"] == "voxels/" + record["canonical_content_sha256"] + ".npy"
    assert record["geometry_provenance"]
    assert record["preprocessing_version"] == "final-training-corpus-v1-procedural-v1"
    assert record["units"] == "normalized voxel lattice; occupancy is dimensionless"
    assert record["design_family"].startswith("generated_procedural_")
    assert set(record["design_spec"]) == set(DESIGN_SPEC_FIELDS)
    assert all(value is None for value in record["design_spec"].values())
    assert record["design_spec_availability"] == {field: False for field in DESIGN_SPEC_FIELDS}
    assert set(record["design_spec_provenance"]) == set(DESIGN_SPEC_FIELDS)


def test_published_corpus_validation_rejects_deleted_geometry(tmp_path):
    source_manifest = _write_source_manifest(tmp_path, [_aircraft_like()])
    output_dir = tmp_path / "published"
    _run_small_build(source_manifest, output_dir)
    record = _read_records(output_dir / "combined_training_manifest.jsonl")[0]
    (output_dir / record["geometry_path"]).unlink()

    builder = _builder_module()
    with pytest.raises((FileNotFoundError, ValueError), match="(exist|missing|geometry)"):
        builder.validate_published_corpus(output_dir, unique_geometry_target=1)
