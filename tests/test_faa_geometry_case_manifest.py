import json
from pathlib import Path

from build_faa_geometry_case_manifest import build_geometry_case_manifest


def _write_jsonl(path: Path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _case_record(source_id: str, speed: float = 140.0):
    return {
        "source_id": source_id,
        "aircraft_type": {
            "icao_code": "A320",
            "manufacturer": "AIRBUS",
            "model": "Airbus A320",
            "engine_class": "jet",
            "num_engines": 2,
        },
        "flight_regime": {
            "provenance_level": "observed_flight_case",
            "route_average_speed_mps": speed,
        },
        "flight_path": {"segments": [{"name": "cruise", "target_speed_mps": speed}]},
        "design_spec": {
            "target_speed_mps": speed,
            "wingspan_limit_m": 35.0,
            "thrust_to_weight_min": 0.4,
            "turn_rate_min_deg_s": 3.0,
            "required_static_thrust_n": 200000.0,
            "engine_diameter_mm": 900,
            "engine_length_mm": 2400,
            "engine_count_min": 2,
            "engine_count_max": 2,
            "payload_mass_min_g": 1000000.0,
            "payload_mass_max_g": 3000000.0,
            "takeoff_distance_min_m": 900.0,
            "takeoff_distance_max_m": 2200.0,
            "wall_thickness_min_mm": 1.0,
            "wall_thickness_max_mm": 2.0,
            "part_count_min": 1,
            "part_count_max": 8,
            "manufacturing_method": "composite_wet_layup",
        },
    }


def _geometry_record(path: str, source_id: str, family: str, speed: float = 130.0, span: float = 34.0):
    return {
        "source_id": source_id,
        "sample_id": source_id,
        "geometry_path": path,
        "geometry_provenance": f"{source_id} public whole-aircraft geometry",
        "preprocessing_version": "test-geometry-v1",
        "units": "m",
        "design_family": family,
        "split": "train",
        "design_spec": {
            "target_speed_mps": speed,
            "wingspan_limit_m": span,
        },
    }


def test_geometry_case_manifest_filters_airfoil_sources_and_attaches_existing_geometry(tmp_path: Path):
    geometry_dir = tmp_path / "geom"
    geometry_dir.mkdir()
    whole = geometry_dir / "whole.npy"
    airfoil = geometry_dir / "airfoil.npy"
    whole.write_bytes(b"whole")
    airfoil.write_bytes(b"airfoil")

    cases = tmp_path / "cases.jsonl"
    geometries = tmp_path / "geometries.jsonl"
    output = tmp_path / "out" / "manifest.jsonl"
    report = tmp_path / "out" / "report.json"
    _write_jsonl(cases, [_case_record("case-1"), _case_record("case-2")])
    _write_jsonl(
        geometries,
        [
            _geometry_record("geom/airfoil.npy", "naca-0012", "airfoil_section_4digit"),
            _geometry_record("geom/whole.npy", "whole-aircraft-1", "transport_whole_aircraft"),
        ],
    )

    summary = build_geometry_case_manifest(
        flight_case_manifest=cases,
        geometry_manifest=geometries,
        output_manifest=output,
        report_path=report,
        target_records=2,
        run_id="unit-test",
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["record_count"] == 2
    assert summary["eligible_geometry_count"] == 1
    assert summary["excluded_geometry_count"] == 1
    assert {record["geometry_association"]["base_geometry_source_id"] for record in records} == {"whole-aircraft-1"}
    assert all(record["geometry_path"].endswith("geom/whole.npy") for record in records)
    assert all(record["design_family"] == "faa_opensky_geometry_proxy_whole_aircraft" for record in records)
    assert all(record["geometry_association"]["method"] == "deterministic_diversified_proxy" for record in records)


def test_geometry_case_manifest_assigns_stable_splits_and_limits_record_count(tmp_path: Path):
    geometry_dir = tmp_path / "geom"
    geometry_dir.mkdir()
    for name in ["fast.npy", "slow.npy"]:
        (geometry_dir / name).write_bytes(name.encode("utf-8"))

    cases = tmp_path / "cases.jsonl"
    geometries = tmp_path / "geometries.jsonl"
    output = tmp_path / "out" / "manifest.jsonl"
    report = tmp_path / "out" / "report.json"
    _write_jsonl(cases, [_case_record(f"case-{idx}", speed=90.0 + idx) for idx in range(12)])
    _write_jsonl(
        geometries,
        [
            _geometry_record("geom/slow.npy", "slow", "whole_aircraft", speed=95.0),
            _geometry_record("geom/fast.npy", "fast", "whole_aircraft", speed=160.0),
        ],
    )

    summary = build_geometry_case_manifest(
        flight_case_manifest=cases,
        geometry_manifest=geometries,
        output_manifest=output,
        report_path=report,
        target_records=10,
        run_id="unit-test",
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 10
    assert summary["record_count"] == 10
    assert summary["split_counts"] == {"holdout": 1, "train": 8, "validation": 1}
    assert records[0]["split"] == "holdout"
    assert records[1]["split"] == "validation"
    assert records[2]["split"] == "train"
    assert json.loads(report.read_text(encoding="utf-8"))["run_id"] == "unit-test"


def test_geometry_case_manifest_diversifies_across_all_eligible_geometry(tmp_path: Path):
    geometry_dir = tmp_path / "geom"
    geometry_dir.mkdir()
    for name in ["g1.npy", "g2.npy", "g3.npy"]:
        (geometry_dir / name).write_bytes(name.encode("utf-8"))

    cases = tmp_path / "cases.jsonl"
    geometries = tmp_path / "geometries.jsonl"
    output = tmp_path / "out" / "manifest.jsonl"
    report = tmp_path / "out" / "report.json"
    _write_jsonl(cases, [_case_record(f"case-{idx}", speed=120.0) for idx in range(6)])
    _write_jsonl(
        geometries,
        [
            _geometry_record("geom/g1.npy", "g1", "whole_aircraft", speed=120.0),
            _geometry_record("geom/g2.npy", "g2", "whole_aircraft", speed=120.0),
            _geometry_record("geom/g3.npy", "g3", "whole_aircraft", speed=120.0),
        ],
    )

    summary = build_geometry_case_manifest(
        flight_case_manifest=cases,
        geometry_manifest=geometries,
        output_manifest=output,
        report_path=report,
        target_records=6,
        run_id="unit-test",
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    associated = {record["geometry_association"]["base_geometry_source_id"] for record in records}
    assert associated == {"g1", "g2", "g3"}
    assert summary["unique_geometry_associations"] == 3
