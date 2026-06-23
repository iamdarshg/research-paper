#!/usr/bin/env python3
"""Build a provenance-aware FAA aircraft flight-regime corpus.

This corpus describes expected operating regimes by aircraft type. It is not a
geometry manifest and intentionally does not fabricate geometry paths.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


FAA_AIRCRAFT_DATA_URL = "https://www.faa.gov/airports/engineering/aircraft_char_database/aircraft_data"
OPENSKY_AIRCRAFT_DATABASE_URL = "https://opensky-network.org/datasets/metadata/aircraftDatabase.csv"

KNOT_TO_MPS = 0.514444
FT_TO_M = 0.3048
LB_TO_KG = 0.45359237
MPH_TO_MPS = 0.44704
NM_TO_KM = 1.852


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, str) and not value.strip():
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _to_int(value: Any, default: int = 0) -> int:
    number = _to_float(value)
    if number is None:
        return default
    return int(round(number))


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() == "nan":
        return default
    return text


def _read_table(path_or_url: str, *, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = Path(path_or_url.split("?", 1)[0]).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path_or_url)
    return pd.read_excel(path_or_url, sheet_name=sheet_name or "ACD_Data")


def download_source(url: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    if not output_path.exists() or output_path.stat().st_size == 0:
        urlretrieve(url, output_path)
    return output_path


def _row_value(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            value = row[name]
            if not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def normalize_faa_frame(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        icao_code = _clean_text(_row_value(raw, "ICAO_Code", "icao_code")).upper()
        if not icao_code:
            continue
        wingspan_ft = _to_float(
            _row_value(
                raw,
                "Wingspan_ft_with_winglets_sharklets",
                "Wingspan_ft_without_winglets_sharklets",
                "wingspan_ft",
            )
        )
        if wingspan_ft is None:
            wingspan_ft = _to_float(
                _row_value(raw, "Wingspan_ft_without_winglets_sharklets", "wingspan_ft")
            )
        records.append(
            {
                "icao_code": icao_code,
                "faa_designator": _clean_text(_row_value(raw, "FAA_Designator", "faa_designator")).upper(),
                "manufacturer": _clean_text(_row_value(raw, "Manufacturer", "manufacturer")),
                "model": _clean_text(_row_value(raw, "Model_FAA", "model")),
                "model_bada": _clean_text(_row_value(raw, "Model_BADA", "model_bada")),
                "physical_class_engine": _clean_text(
                    _row_value(raw, "Physical_Class_Engine", "physical_class_engine")
                ),
                "num_engines": _to_int(_row_value(raw, "Num_Engines", "num_engines")),
                "approach_speed_knot": _to_float(
                    _row_value(raw, "Approach_Speed_knot", "approach_speed_knot")
                ),
                "wingspan_ft": wingspan_ft,
                "length_ft": _to_float(_row_value(raw, "Length_ft", "length_ft")),
                "tail_height_ft": _to_float(_row_value(raw, "Tail_Height_at_OEW_ft", "tail_height_ft")),
                "mtow_lb": _to_float(_row_value(raw, "MTOW_lb", "mtow_lb")),
                "malw_lb": _to_float(_row_value(raw, "MALW_lb", "malw_lb")),
                "icao_wtc": _clean_text(_row_value(raw, "ICAO_WTC", "icao_wtc")),
                "faa_weight": _clean_text(_row_value(raw, "FAA_Weight", "faa_weight")),
                "faa_registry": _clean_text(_row_value(raw, "FAA_Registry", "faa_registry")),
                "registration_count": _to_int(_row_value(raw, "Registration_Count", "registration_count")),
                "tmfs_operations_fy24": _to_int(
                    _row_value(raw, "TMFS_Operations_FY24", "tmfs_operations_fy24")
                ),
                "last_update": _clean_text(_row_value(raw, "LastUpdate", "last_update")),
            }
        )
    return records


def load_faa_aircraft_characteristics(path_or_url: str = FAA_AIRCRAFT_DATA_URL) -> List[Dict[str, Any]]:
    return normalize_faa_frame(_read_table(path_or_url, sheet_name="ACD_Data"))


def _canonical_icao_code(row: Mapping[str, Any]) -> str:
    for key in ("icao_code", "ICAO_Code", "icao_type", "typecode", "aircraft_type", "model_icao"):
        value = _clean_text(row.get(key)).upper()
        if value:
            return value
    return ""


def _metric_value(row: Mapping[str, Any], canonical: str, alternatives: Sequence[str], *, unit: str) -> float | None:
    value = _to_float(row.get(canonical))
    if value is not None:
        return value
    for name in alternatives:
        value = _to_float(row.get(name))
        if value is None:
            continue
        lowered = name.lower()
        if unit == "speed_mps" and ("knot" in lowered or lowered.endswith("_kt") or lowered.endswith("_kts")):
            return value * KNOT_TO_MPS
        if unit == "altitude_m" and ("_ft" in lowered or "feet" in lowered):
            return value * FT_TO_M
        return value
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    numeric = _to_float(text)
    if numeric is not None and numeric > 100000:
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S%z")
        except ValueError:
            return None


def _duration_from_timestamps(row: Mapping[str, Any]) -> float | None:
    first = _parse_timestamp(row.get("firstseen") or row.get("first_seen"))
    last = _parse_timestamp(row.get("lastseen") or row.get("last_seen"))
    if not first or not last:
        return None
    seconds = (last - first).total_seconds()
    if seconds <= 0:
        return None
    return seconds


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def _route_distance_from_endpoints(row: Mapping[str, Any]) -> float | None:
    lat1 = _to_float(row.get("latitude_1") or row.get("start_latitude"))
    lon1 = _to_float(row.get("longitude_1") or row.get("start_longitude"))
    lat2 = _to_float(row.get("latitude_2") or row.get("end_latitude"))
    lon2 = _to_float(row.get("longitude_2") or row.get("end_longitude"))
    if None in {lat1, lon1, lat2, lon2}:
        return None
    return _haversine_km(float(lat1), float(lon1), float(lat2), float(lon2))


def _altitude_from_endpoints(row: Mapping[str, Any]) -> float | None:
    altitude_1 = _to_float(row.get("altitude_1"))
    altitude_2 = _to_float(row.get("altitude_2"))
    values = [value for value in (altitude_1, altitude_2) if value is not None]
    if not values:
        return None
    return max(values)


def _extract_observed_metrics(row: Mapping[str, Any]) -> Dict[str, Any] | None:
    icao_code = _canonical_icao_code(row)
    if not icao_code:
        return None
    speed = _metric_value(
        row,
        "groundspeed_mps",
        ("groundspeed_kt", "groundspeed_knot", "velocity", "velocity_mps", "speed_mps"),
        unit="speed_mps",
    )
    altitude = _metric_value(
        row,
        "altitude_m",
        ("baro_altitude", "geo_altitude", "altitude_ft", "cruise_altitude_ft"),
        unit="altitude_m",
    )
    vertical_rate = _metric_value(
        row,
        "vertical_rate_mps",
        ("vertical_rate", "vertrate", "climb_rate_mps"),
        unit="speed_mps",
    )
    duration = _metric_value(row, "duration_s", ("duration_seconds", "flight_duration_s"), unit="plain")
    if duration is None:
        duration = _duration_from_timestamps(row)
    distance = _metric_value(
        row, "route_distance_km", ("distance_km", "great_circle_distance_km"), unit="plain"
    )
    if distance is None:
        distance = _route_distance_from_endpoints(row)
    route_average_speed = None
    if speed is None and duration and distance:
        route_average_speed = distance * 1000.0 / duration
    metrics = {
        "icao_code": icao_code,
        "groundspeed_mps": speed,
        "route_average_speed_mps": route_average_speed,
        "altitude_m": altitude,
        "vertical_rate_mps": vertical_rate,
        "duration_s": duration,
        "route_distance_km": distance,
    }
    if all(metrics[key] is None for key in metrics if key != "icao_code"):
        return None
    return metrics


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "p05": round(float(np.percentile(array, 5)), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
    }


def summarize_observed_flights(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_observed_flights: int = 5,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        metrics = _extract_observed_metrics(row)
        if metrics is None:
            continue
        icao_code = metrics["icao_code"]
        bucket = grouped.setdefault(
            icao_code,
            {
                "groundspeed_mps": [],
                "route_average_speed_mps": [],
                "altitude_m": [],
                "vertical_rate_mps": [],
                "duration_s": [],
                "route_distance_km": [],
            },
        )
        for key in (
            "groundspeed_mps",
            "route_average_speed_mps",
            "altitude_m",
            "vertical_rate_mps",
            "duration_s",
            "route_distance_km",
        ):
            value = metrics.get(key)
            if value is not None:
                bucket[key].append(float(value))

    summaries: Dict[str, Dict[str, Any]] = {}
    for icao_code, values in grouped.items():
        sample_count = len(values["groundspeed_mps"]) or max(len(items) for items in values.values())
        if sample_count < min_observed_flights:
            continue
        summary: Dict[str, Any] = {"sample_count": sample_count}
        if values["groundspeed_mps"]:
            summary["cruise_speed_mps"] = _percentiles(values["groundspeed_mps"])
        if values["route_average_speed_mps"]:
            summary["route_average_speed_mps"] = _percentiles(values["route_average_speed_mps"])
        if values["altitude_m"]:
            summary["cruise_altitude_m"] = _percentiles(values["altitude_m"])
        if values["vertical_rate_mps"]:
            summary["vertical_rate_mps"] = _percentiles(values["vertical_rate_mps"])
        if values["duration_s"]:
            summary["duration_s"] = _percentiles(values["duration_s"])
        if values["route_distance_km"]:
            summary["route_distance_km"] = _percentiles(values["route_distance_km"])
        summaries[icao_code] = summary
    return summaries


def load_observed_flight_csv(path: Path, *, min_observed_flights: int = 5) -> Dict[str, Dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return summarize_observed_flights(csv.DictReader(handle), min_observed_flights=min_observed_flights)


def load_spec_overrides(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("records", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("spec override JSON must be a list or an object with a records list")
    overrides: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("spec override entries must be objects")
        icao_code = _canonical_icao_code(entry)
        if not icao_code:
            raise ValueError(f"spec override entry is missing icao_code: {entry}")
        overrides[icao_code] = dict(entry)
    return overrides


def _engine_class(record: Mapping[str, Any]) -> str:
    text = _clean_text(record.get("physical_class_engine")).lower()
    if "rotor" in text or "helicopter" in text:
        return "rotorcraft"
    if "turbo" in text and "prop" in text:
        return "turboprop"
    if "piston" in text:
        return "piston"
    if "jet" in text or "turbofan" in text:
        return "jet"
    return "unknown"


def _weight_class(record: Mapping[str, Any]) -> str:
    mtow = _to_float(record.get("mtow_lb"), 0.0) or 0.0
    if mtow >= 300000:
        return "heavy"
    if mtow >= 70000:
        return "transport"
    if mtow >= 20000:
        return "medium"
    return "light"


def _baseline_cruise_speed_mps(record: Mapping[str, Any]) -> float:
    approach = (_to_float(record.get("approach_speed_knot"), 100.0) or 100.0) * KNOT_TO_MPS
    engine = _engine_class(record)
    if engine == "jet":
        return round(min(max(approach * 2.1, 135.0), 260.0), 6)
    if engine == "turboprop":
        return round(min(max(approach * 1.65, 85.0), 170.0), 6)
    if engine == "piston":
        return round(min(max(approach * 1.35, 45.0), 115.0), 6)
    if engine == "rotorcraft":
        return round(min(max(approach * 1.15, 35.0), 95.0), 6)
    return round(max(approach * 1.5, 60.0), 6)


def _baseline_altitude_m(record: Mapping[str, Any]) -> Dict[str, float]:
    engine = _engine_class(record)
    weight = _weight_class(record)
    if engine == "jet" and weight in {"heavy", "transport"}:
        return {"p05": 7000.0, "p50": 10668.0, "p95": 12496.8}
    if engine == "jet":
        return {"p05": 4500.0, "p50": 9000.0, "p95": 12192.0}
    if engine == "turboprop":
        return {"p05": 1500.0, "p50": 6100.0, "p95": 8840.0}
    if engine == "piston":
        return {"p05": 300.0, "p50": 2400.0, "p95": 4500.0}
    if engine == "rotorcraft":
        return {"p05": 100.0, "p50": 600.0, "p95": 1800.0}
    return {"p05": 300.0, "p50": 3000.0, "p95": 9000.0}


def _turn_rate_target(record: Mapping[str, Any], spec: Mapping[str, Any] | None) -> float:
    model = f"{record.get('icao_code', '')} {record.get('model', '')}".lower()
    if spec and _to_float(spec.get("max_speed_mps")) and any(token in model for token in ("f-", "fighter", "hornet", "falcon")):
        return 12.0
    weight = _weight_class(record)
    if weight == "heavy":
        return 2.0
    if weight == "transport":
        return 3.0
    if weight == "medium":
        return 5.0
    return 8.0


def _thrust_to_weight(record: Mapping[str, Any], spec: Mapping[str, Any] | None) -> float:
    model = f"{record.get('icao_code', '')} {record.get('model', '')}".lower()
    if spec and any(token in model for token in ("f-", "fighter", "hornet", "falcon")):
        return 0.9
    engine = _engine_class(record)
    if engine == "jet":
        return 0.42 if _weight_class(record) in {"heavy", "transport"} else 0.58
    if engine == "turboprop":
        return 0.32
    if engine == "rotorcraft":
        return 0.6
    return 0.28


def _takeoff_bounds_m(record: Mapping[str, Any]) -> tuple[float, float]:
    approach = _to_float(record.get("approach_speed_knot"), 100.0) or 100.0
    weight = _weight_class(record)
    if weight == "heavy":
        upper = 3200.0
    elif weight == "transport":
        upper = 2600.0
    elif weight == "medium":
        upper = 1800.0
    else:
        upper = 900.0
    upper *= max(0.75, min(1.35, approach / 130.0))
    return round(upper * 0.45, 3), round(upper, 3)


def _schema_compatible_design_spec(
    record: Mapping[str, Any],
    regime: Mapping[str, Any],
    spec: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    mtow_kg = (_to_float(record.get("mtow_lb"), 0.0) or 0.0) * LB_TO_KG
    twr = _thrust_to_weight(record, spec)
    takeoff_min, takeoff_max = _takeoff_bounds_m(record)
    wingspan_m = (_to_float(record.get("wingspan_ft"), 6.0) or 6.0) * FT_TO_M
    payload_upper_kg = max(1.0, mtow_kg * 0.18)
    payload_lower_kg = payload_upper_kg * 0.25
    return {
        "target_speed_mps": round(float(regime["cruise_speed_mps"]["p50"]), 6),
        "wingspan_limit_m": round(wingspan_m, 6),
        "thrust_to_weight_min": round(twr, 6),
        "turn_rate_min_deg_s": round(_turn_rate_target(record, spec), 6),
        "required_static_thrust_n": round(max(1.0, mtow_kg * 9.80665 * twr), 6),
        "engine_diameter_mm": 900 if _engine_class(record) == "jet" else 500,
        "engine_length_mm": 2400 if _engine_class(record) == "jet" else 900,
        "engine_count_min": max(1, _to_int(record.get("num_engines"), 1)),
        "engine_count_max": max(1, _to_int(record.get("num_engines"), 1)),
        "payload_mass_min_g": round(payload_lower_kg * 1000.0, 3),
        "payload_mass_max_g": round(payload_upper_kg * 1000.0, 3),
        "takeoff_distance_min_m": takeoff_min,
        "takeoff_distance_max_m": takeoff_max,
        "wall_thickness_min_mm": 1.0,
        "wall_thickness_max_mm": 2.0,
        "part_count_min": 1,
        "part_count_max": 8,
        "manufacturing_method": "composite_wet_layup",
    }


def _flight_path_from_regime(record: Mapping[str, Any], regime: Mapping[str, Any]) -> Dict[str, Any]:
    cruise = float(regime["cruise_speed_mps"]["p50"])
    altitude = float(regime["cruise_altitude_m"]["p50"])
    approach = float(regime["approach_speed_mps"])
    takeoff_min, takeoff_max = _takeoff_bounds_m(record)
    return {
        "profile_id": f"{record['icao_code']}_expected_regime",
        "segments": [
            {
                "name": "takeoff_roll",
                "start_speed_mps": 0.0,
                "end_speed_mps": round(max(approach * 1.1, cruise * 0.45), 6),
                "distance_m": takeoff_max,
            },
            {
                "name": "climb",
                "target_speed_mps": round(max(approach * 1.25, cruise * 0.62), 6),
                "target_altitude_m": round(min(altitude, 3000.0), 3),
            },
            {
                "name": "cruise_or_mission",
                "target_speed_mps": round(cruise, 6),
                "target_altitude_m": round(altitude, 3),
            },
            {
                "name": "descent",
                "target_speed_mps": round(max(approach * 1.2, cruise * 0.55), 6),
                "target_altitude_m": 1000.0,
            },
            {
                "name": "approach",
                "target_speed_mps": round(approach, 6),
                "distance_m": takeoff_min,
            },
        ],
        "provenance": regime["provenance_level"],
    }


def _build_regime(
    record: Mapping[str, Any],
    observed: Mapping[str, Any] | None,
    spec: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    approach_mps = (_to_float(record.get("approach_speed_knot"), 100.0) or 100.0) * KNOT_TO_MPS
    cruise_speed = {"p05": _baseline_cruise_speed_mps(record) * 0.85, "p50": _baseline_cruise_speed_mps(record), "p95": _baseline_cruise_speed_mps(record) * 1.12}
    cruise_altitude = _baseline_altitude_m(record)
    provenance_level = "faa_characteristics_only"
    observed_count = 0

    if spec:
        spec_cruise = _to_float(spec.get("cruise_speed_mps"))
        spec_max = _to_float(spec.get("max_speed_mps"))
        if spec_cruise is not None:
            cruise_speed = {
                "p05": round(spec_cruise * 0.72, 6),
                "p50": round(spec_cruise, 6),
                "p95": round(min(spec_max or spec_cruise * 1.15, spec_cruise * 1.12), 6),
            }
        ceiling = _to_float(spec.get("service_ceiling_m"))
        if ceiling is not None:
            cruise_altitude = {
                "p05": round(ceiling * 0.35, 6),
                "p50": round(ceiling * 0.65, 6),
                "p95": round(ceiling * 0.9, 6),
            }
        provenance_level = "published_spec"

    if observed:
        if observed.get("cruise_speed_mps"):
            cruise_speed = dict(observed["cruise_speed_mps"])
        if observed.get("cruise_altitude_m"):
            cruise_altitude = dict(observed["cruise_altitude_m"])
        observed_count = int(observed.get("sample_count", 0))
        provenance_level = "observed_adsb_and_published_spec" if spec else "observed_adsb"

    regime: Dict[str, Any] = {
        "provenance_level": provenance_level,
        "observed_flight_count": observed_count,
        "approach_speed_mps": round(approach_mps, 6),
        "cruise_speed_mps": {key: round(float(value), 6) for key, value in cruise_speed.items()},
        "cruise_altitude_m": {key: round(float(value), 6) for key, value in cruise_altitude.items()},
    }
    if observed and observed.get("vertical_rate_mps"):
        regime["vertical_rate_mps"] = dict(observed["vertical_rate_mps"])
    if observed and observed.get("route_average_speed_mps"):
        regime["route_average_speed_mps"] = dict(observed["route_average_speed_mps"])
    if observed and observed.get("duration_s"):
        regime["duration_s"] = dict(observed["duration_s"])
    if observed and observed.get("route_distance_km"):
        regime["route_distance_km"] = dict(observed["route_distance_km"])
    if spec:
        for key in ("max_speed_mps", "service_ceiling_m", "range_km", "combat_radius_km"):
            value = _to_float(spec.get(key))
            if value is not None:
                regime[key] = round(value, 6)
    return regime


def build_regime_records(
    faa_records: Sequence[Mapping[str, Any]],
    *,
    observed_stats: Mapping[str, Mapping[str, Any]],
    spec_overrides: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for source_record in faa_records:
        icao_code = _clean_text(source_record.get("icao_code")).upper()
        if not icao_code:
            continue
        record = dict(source_record)
        record["icao_code"] = icao_code
        observed = observed_stats.get(icao_code)
        spec = spec_overrides.get(icao_code)
        regime = _build_regime(record, observed, spec)
        design_spec = _schema_compatible_design_spec(record, regime, spec)
        output.append(
            {
                "source_id": f"faa-regime-{icao_code}",
                "aircraft_type": {
                    "icao_code": icao_code,
                    "faa_designator": _clean_text(record.get("faa_designator")),
                    "manufacturer": _clean_text(record.get("manufacturer")),
                    "model": _clean_text(record.get("model")),
                    "engine_class": _engine_class(record),
                    "num_engines": _to_int(record.get("num_engines")),
                },
                "faa_characteristics": {
                    "approach_speed_knot": _to_float(record.get("approach_speed_knot")),
                    "wingspan_m": round((_to_float(record.get("wingspan_ft"), 0.0) or 0.0) * FT_TO_M, 6),
                    "length_m": round((_to_float(record.get("length_ft"), 0.0) or 0.0) * FT_TO_M, 6),
                    "mtow_kg": round((_to_float(record.get("mtow_lb"), 0.0) or 0.0) * LB_TO_KG, 6),
                    "registration_count": _to_int(record.get("registration_count")),
                    "tmfs_operations_fy24": _to_int(record.get("tmfs_operations_fy24")),
                },
                "flight_regime": regime,
                "flight_path": _flight_path_from_regime(record, regime),
                "design_spec": design_spec,
                "regime_provenance": {
                    "faa_source_url": FAA_AIRCRAFT_DATA_URL,
                    "opensky_aircraft_database_url": OPENSKY_AIRCRAFT_DATABASE_URL,
                    "published_spec_source_url": _clean_text(spec.get("source_url")) if spec else "",
                    "published_spec_source_label": _clean_text(spec.get("source_label")) if spec else "",
                    "limitations": (
                        "FAA-only regimes are bounded estimates from aircraft-characteristics fields, not observed telemetry. "
                        "observed_adsb regimes require joined ADS-B/Mode-S state rows. published_spec regimes are public envelope caps, "
                        "not measured operational distributions."
                    ),
                },
            }
        )
    return output


def build_observed_flight_case_records(
    faa_records: Sequence[Mapping[str, Any]],
    observed_rows: Iterable[Mapping[str, Any]],
    *,
    spec_overrides: Mapping[str, Mapping[str, Any]],
    max_cases: int | None = None,
) -> List[Dict[str, Any]]:
    faa_by_code = {_clean_text(record.get("icao_code")).upper(): dict(record) for record in faa_records}
    output: List[Dict[str, Any]] = []
    for row in observed_rows:
        metrics = _extract_observed_metrics(row)
        if metrics is None:
            continue
        icao_code = metrics["icao_code"]
        faa_record = faa_by_code.get(icao_code)
        if not faa_record:
            continue
        spec = spec_overrides.get(icao_code)
        base_regime = _build_regime(faa_record, None, spec)
        target_speed = metrics.get("groundspeed_mps") or metrics.get("route_average_speed_mps")
        if target_speed is None:
            continue

        provenance_level = "observed_flight_case_and_published_spec" if spec else "observed_flight_case"
        regime = dict(base_regime)
        regime["provenance_level"] = provenance_level
        regime["observed_flight_count"] = 1
        if metrics.get("groundspeed_mps") is not None:
            regime["observed_groundspeed_mps"] = round(float(metrics["groundspeed_mps"]), 6)
        if metrics.get("route_average_speed_mps") is not None:
            regime["route_average_speed_mps"] = round(float(metrics["route_average_speed_mps"]), 6)
        if metrics.get("altitude_m") is not None:
            regime["observed_altitude_m"] = round(float(metrics["altitude_m"]), 6)
        if metrics.get("vertical_rate_mps") is not None:
            regime["observed_vertical_rate_mps"] = round(float(metrics["vertical_rate_mps"]), 6)
        if metrics.get("duration_s") is not None:
            regime["duration_s"] = round(float(metrics["duration_s"]), 6)
        if metrics.get("route_distance_km") is not None:
            regime["route_distance_km"] = round(float(metrics["route_distance_km"]), 6)

        design_spec = _schema_compatible_design_spec(faa_record, base_regime, spec)
        design_spec["target_speed_mps"] = round(float(target_speed), 6)
        case_index = len(output)
        output.append(
            {
                "source_id": f"opensky-flight-case-{icao_code}-{case_index:06d}",
                "aircraft_type": {
                    "icao_code": icao_code,
                    "faa_designator": _clean_text(faa_record.get("faa_designator")),
                    "manufacturer": _clean_text(faa_record.get("manufacturer")),
                    "model": _clean_text(faa_record.get("model")),
                    "engine_class": _engine_class(faa_record),
                    "num_engines": _to_int(faa_record.get("num_engines")),
                },
                "flight_regime": regime,
                "flight_path": _flight_path_from_regime(faa_record, base_regime),
                "design_spec": design_spec,
                "regime_provenance": {
                    "faa_source_url": FAA_AIRCRAFT_DATA_URL,
                    "opensky_source": "OpenSky/Zenodo flight-list row",
                    "published_spec_source_url": _clean_text(spec.get("source_url")) if spec else "",
                    "limitations": (
                        "Per-flight cases are observed route/track summaries when built from OpenSky flight-list rows. "
                        "Route-average speed is not cruise speed, and endpoint rows do not provide full 3D trajectories."
                    ),
                },
            }
        )
        if max_cases is not None and len(output) >= max_cases:
            break
    return output


def write_regime_corpus(
    records: Sequence[Mapping[str, Any]],
    output_manifest: Path,
    report_path: Path,
    *,
    run_id: str,
) -> Dict[str, Any]:
    output_manifest = Path(output_manifest)
    report_path = Path(report_path)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    provenance_counts: Dict[str, int] = {}
    for record in records:
        level = str(record.get("flight_regime", {}).get("provenance_level", "unknown"))
        provenance_counts[level] = provenance_counts.get(level, 0) + 1
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "provenance_counts": provenance_counts,
        "output_manifest": str(output_manifest.resolve()),
        "claim_boundary": (
            "This is a flight-regime conditioning corpus. It does not provide geometry ground truth. "
            "FAA-only records are estimated envelopes; observed_adsb records are derived from supplied ADS-B/Mode-S rows; "
            "published_spec records are bounded by official public specifications."
        ),
    }
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _iter_observed_rows(paths: Sequence[str]) -> Iterable[Mapping[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        opener = gzip.open if path.suffix.lower() == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)


def _load_observed_sources(paths: Sequence[str], *, min_observed_flights: int) -> Dict[str, Dict[str, Any]]:
    return summarize_observed_flights(
        _iter_observed_rows(paths),
        min_observed_flights=min_observed_flights,
    )


def build_corpus_from_sources(
    *,
    faa_source: str,
    observed_flights_csv: Sequence[str],
    military_specs_json: Path | None,
    output_manifest: Path,
    report_path: Path,
    run_id: str,
    min_observed_flights: int,
    max_records: int | None = None,
    output_case_manifest: Path | None = None,
    case_report_path: Path | None = None,
    max_observed_cases: int | None = None,
) -> Dict[str, Any]:
    faa_records = load_faa_aircraft_characteristics(faa_source)
    if max_records is not None:
        faa_records = faa_records[:max_records]
    observed_stats = _load_observed_sources(
        observed_flights_csv,
        min_observed_flights=min_observed_flights,
    ) if observed_flights_csv else {}
    spec_overrides = load_spec_overrides(military_specs_json)
    records = build_regime_records(
        faa_records,
        observed_stats=observed_stats,
        spec_overrides=spec_overrides,
    )
    summary = write_regime_corpus(records, output_manifest, report_path, run_id=run_id)
    if output_case_manifest and case_report_path and observed_flights_csv:
        case_records = build_observed_flight_case_records(
            faa_records,
            _iter_observed_rows(observed_flights_csv),
            spec_overrides=spec_overrides,
            max_cases=max_observed_cases,
        )
        summary["case_manifest"] = write_regime_corpus(
            case_records,
            output_case_manifest,
            case_report_path,
            run_id=f"{run_id}-observed-cases",
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faa-source", default=FAA_AIRCRAFT_DATA_URL)
    parser.add_argument("--cache-dir", default="build/source_cache")
    parser.add_argument("--observed-flights-csv", action="append", default=[])
    parser.add_argument("--military-specs-json", default=None)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-case-manifest", default=None)
    parser.add_argument("--case-report", default=None)
    parser.add_argument("--run-id", default="faa-flight-regime-corpus")
    parser.add_argument("--min-observed-flights", type=int, default=5)
    parser.add_argument("--max-observed-cases", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    if bool(args.output_case_manifest) != bool(args.case_report):
        parser.error("--output-case-manifest and --case-report must be supplied together")

    faa_source = args.faa_source
    if _is_url(faa_source):
        faa_source = str(download_source(faa_source, Path(args.cache_dir), "faa_aircraft_data.xlsx"))

    summary = build_corpus_from_sources(
        faa_source=faa_source,
        observed_flights_csv=args.observed_flights_csv,
        military_specs_json=Path(args.military_specs_json) if args.military_specs_json else None,
        output_manifest=Path(args.output_manifest),
        report_path=Path(args.report),
        run_id=args.run_id,
        min_observed_flights=args.min_observed_flights,
        max_records=args.max_records,
        output_case_manifest=Path(args.output_case_manifest) if args.output_case_manifest else None,
        case_report_path=Path(args.case_report) if args.case_report else None,
        max_observed_cases=args.max_observed_cases,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
