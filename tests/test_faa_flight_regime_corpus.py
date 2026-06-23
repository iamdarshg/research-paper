import json
from pathlib import Path

import pytest

from build_faa_flight_regime_corpus import (
    build_observed_flight_case_records,
    build_regime_records,
    summarize_observed_flights,
    write_regime_corpus,
)


def _faa_record(**overrides):
    record = {
        "icao_code": "A320",
        "faa_designator": "A320",
        "manufacturer": "AIRBUS",
        "model": "Airbus A320",
        "physical_class_engine": "Jet",
        "num_engines": 2,
        "approach_speed_knot": 136.0,
        "wingspan_ft": 111.9,
        "length_ft": 123.3,
        "mtow_lb": 171961.0,
        "faa_registry": "Yes",
        "registration_count": 698,
        "tmfs_operations_fy24": 1397404,
    }
    record.update(overrides)
    return record


def test_faa_only_record_builds_bounded_regime_without_claiming_observed_flights():
    records = build_regime_records([_faa_record()], observed_stats={}, spec_overrides={})

    assert len(records) == 1
    record = records[0]
    assert record["aircraft_type"]["icao_code"] == "A320"
    assert record["flight_regime"]["provenance_level"] == "faa_characteristics_only"
    assert record["flight_regime"]["observed_flight_count"] == 0
    assert record["flight_regime"]["approach_speed_mps"] == pytest.approx(69.964, abs=0.001)
    assert record["design_spec"]["target_speed_mps"] > record["flight_regime"]["approach_speed_mps"]
    assert record["regime_provenance"]["faa_source_url"].startswith("https://www.faa.gov/")
    assert "not observed telemetry" in record["regime_provenance"]["limitations"]


def test_observed_adsb_rows_override_cruise_envelope_and_record_sample_count():
    observed_stats = summarize_observed_flights(
        [
            {"icao_code": "A320", "groundspeed_mps": 210.0, "altitude_m": 9500.0, "vertical_rate_mps": 0.0},
            {"icao_code": "A320", "groundspeed_mps": 230.0, "altitude_m": 10300.0, "vertical_rate_mps": 0.2},
            {"icao_code": "A320", "groundspeed_mps": 250.0, "altitude_m": 11100.0, "vertical_rate_mps": -0.1},
        ],
        min_observed_flights=2,
    )

    records = build_regime_records([_faa_record()], observed_stats=observed_stats, spec_overrides={})

    regime = records[0]["flight_regime"]
    assert regime["provenance_level"] == "observed_adsb"
    assert regime["observed_flight_count"] == 3
    assert regime["cruise_speed_mps"]["p50"] == pytest.approx(230.0)
    assert regime["cruise_altitude_m"]["p50"] == pytest.approx(10300.0)
    assert records[0]["design_spec"]["target_speed_mps"] == pytest.approx(230.0)


def test_opensky_flightlist_rows_infer_observed_speed_distance_and_duration():
    observed_stats = summarize_observed_flights(
        [
            {
                "typecode": "A320",
                "firstseen": "2020-04-01 00:00:00+00:00",
                "lastseen": "2020-04-01 02:00:00+00:00",
                "latitude_1": 40.6413,
                "longitude_1": -73.7781,
                "altitude_1": 1000.0,
                "latitude_2": 33.9416,
                "longitude_2": -118.4085,
                "altitude_2": 1200.0,
            },
            {
                "typecode": "A320",
                "firstseen": "2020-04-02 00:00:00+00:00",
                "lastseen": "2020-04-02 02:10:00+00:00",
                "latitude_1": 40.6413,
                "longitude_1": -73.7781,
                "altitude_1": 1200.0,
                "latitude_2": 33.9416,
                "longitude_2": -118.4085,
                "altitude_2": 1400.0,
            },
        ],
        min_observed_flights=2,
    )

    a320 = observed_stats["A320"]
    assert a320["sample_count"] == 2
    assert a320["duration_s"]["p50"] == pytest.approx(7500.0)
    assert a320["route_distance_km"]["p50"] > 3900.0
    assert a320["route_average_speed_mps"]["p50"] > 500.0
    assert "cruise_speed_mps" not in a320
    assert "cruise_altitude_m" not in a320


def test_published_military_spec_sets_caps_without_calling_them_observed_tracks():
    spec_overrides = {
        "F16": {
            "source_label": "USAF Museum F-16A fact sheet",
            "source_url": "https://www.nationalmuseum.af.mil/Visit/Museum-Exhibits/Fact-Sheets/Display/Article/196735/general-dynamics-f-16a-fighting-falcon/",
            "max_speed_mps": 601.2,
            "cruise_speed_mps": 257.9,
            "service_ceiling_m": 16764.0,
            "range_km": 2264.0,
        }
    }

    records = build_regime_records(
        [
            _faa_record(
                icao_code="F16",
                faa_designator="F16",
                manufacturer="LOCKHEED-GENERAL DYNAMICS",
                model="Lockheed F-16 Fighting Falcon",
                approach_speed_knot=150.0,
                wingspan_ft=32.8,
                length_ft=49.5,
                mtow_lb=42000.0,
            )
        ],
        observed_stats={},
        spec_overrides=spec_overrides,
    )

    record = records[0]
    regime = record["flight_regime"]
    assert regime["provenance_level"] == "published_spec"
    assert regime["observed_flight_count"] == 0
    assert regime["max_speed_mps"] == pytest.approx(601.2)
    assert regime["service_ceiling_m"] == pytest.approx(16764.0)
    assert record["design_spec"]["target_speed_mps"] == pytest.approx(257.9)
    assert record["regime_provenance"]["published_spec_source_url"] == spec_overrides["F16"]["source_url"]


def test_military_spec_and_observed_rows_keep_combined_provenance():
    spec_overrides = {
        "F16": {
            "source_label": "USAF Museum F-16A fact sheet",
            "source_url": "https://www.nationalmuseum.af.mil/Visit/Museum-Exhibits/Fact-Sheets/Display/Article/196735/general-dynamics-f-16a-fighting-falcon/",
            "max_speed_mps": 601.2,
            "cruise_speed_mps": 257.9,
            "service_ceiling_m": 16764.0,
            "range_km": 2264.0,
        }
    }
    observed_stats = {
        "F16": {
            "sample_count": 40,
            "route_average_speed_mps": {"p05": 100.0, "p50": 180.0, "p95": 260.0},
            "duration_s": {"p05": 1200.0, "p50": 2400.0, "p95": 3600.0},
        }
    }

    records = build_regime_records(
        [
            _faa_record(
                icao_code="F16",
                model="Lockheed F-16 Fighting Falcon",
                approach_speed_knot=150.0,
                wingspan_ft=32.8,
                length_ft=49.5,
                mtow_lb=42000.0,
            )
        ],
        observed_stats=observed_stats,
        spec_overrides=spec_overrides,
    )

    regime = records[0]["flight_regime"]
    assert regime["provenance_level"] == "observed_adsb_and_published_spec"
    assert regime["observed_flight_count"] == 40
    assert regime["max_speed_mps"] == pytest.approx(601.2)
    assert regime["route_average_speed_mps"]["p50"] == pytest.approx(180.0)


def test_observed_flight_case_records_are_per_flight_training_cases():
    cases = build_observed_flight_case_records(
        [_faa_record()],
        [
            {
                "typecode": "A320",
                "firstseen": "2020-04-01 00:00:00+00:00",
                "lastseen": "2020-04-01 02:00:00+00:00",
                "latitude_1": 40.6413,
                "longitude_1": -73.7781,
                "latitude_2": 33.9416,
                "longitude_2": -118.4085,
            },
            {
                "typecode": "ZZZZ",
                "firstseen": "2020-04-01 00:00:00+00:00",
                "lastseen": "2020-04-01 02:00:00+00:00",
                "latitude_1": 40.0,
                "longitude_1": -73.0,
                "latitude_2": 33.0,
                "longitude_2": -118.0,
            },
        ],
        spec_overrides={},
        max_cases=10,
    )

    assert len(cases) == 1
    case = cases[0]
    assert case["source_id"] == "opensky-flight-case-A320-000000"
    assert case["aircraft_type"]["icao_code"] == "A320"
    assert case["flight_regime"]["provenance_level"] == "observed_flight_case"
    assert case["flight_regime"]["route_average_speed_mps"] > 500.0
    assert case["design_spec"]["target_speed_mps"] == pytest.approx(case["flight_regime"]["route_average_speed_mps"])


def test_write_regime_corpus_outputs_jsonl_and_report(tmp_path: Path):
    records = build_regime_records([_faa_record()], observed_stats={}, spec_overrides={})
    manifest = tmp_path / "regimes.jsonl"
    report = tmp_path / "report.json"

    summary = write_regime_corpus(records, manifest, report, run_id="unit-test")

    assert summary["record_count"] == 1
    assert summary["provenance_counts"] == {"faa_characteristics_only": 1}
    assert json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])["aircraft_type"]["icao_code"] == "A320"
    assert json.loads(report.read_text(encoding="utf-8"))["run_id"] == "unit-test"
