"""Load and validate the repository-wide experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


GLOBAL_CONFIG_PATH = Path(__file__).resolve().with_name("config.yaml")


def load_global_config(path: Path | str = GLOBAL_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Global config must contain a mapping: {config_path}")

    required_sections = ("model", "scaling", "training", "cfd", "report")
    missing = [name for name in required_sections if not isinstance(payload.get(name), Mapping)]
    if missing:
        raise ValueError(f"Global config is missing mapping sections: {', '.join(missing)}")

    latent_dim = int(payload["model"].get("latent_dim", 0))
    grid_resolution = int(payload["model"].get("grid_resolution", 0))
    unique_count = int(payload["scaling"].get("capacity_basis_unique_geometries", 0))
    if latent_dim <= 0 or grid_resolution <= 0 or unique_count <= 0:
        raise ValueError("latent_dim, grid_resolution, and capacity_basis_unique_geometries must be positive")
    return payload


GLOBAL_CONFIG = load_global_config()


def config_value(section: str, key: str, fallback: Any = None) -> Any:
    """Return a value from the validated global config."""
    return GLOBAL_CONFIG.get(section, {}).get(key, fallback)
