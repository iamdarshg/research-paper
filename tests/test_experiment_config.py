from pathlib import Path

import yaml

from aircraft_diffusion_cfd import ModelConfig, TrainingConfig
from experiment_config import GLOBAL_CONFIG_PATH, load_global_config
from update_model_capacity_report import END_MARKER, START_MARKER, _config_digest, build_report


def test_global_config_drives_default_and_scaled_latent_width():
    config = load_global_config()
    assert GLOBAL_CONFIG_PATH.name == "config.yaml"
    assert int(config["model"]["latent_dim"]) == 512
    assert int(config["training"]["consistency_interval"]) == 10
    assert float(config["training"]["learning_rate"]) == 2.0e-5
    assert float(config["training"]["converter_learning_rate"]) == 2.0e-5
    assert float(config["training"]["consistency_student_learning_rate"]) == 2.0e-5
    assert float(config["training"]["student_direct_gradient_max_norm"]) == 0.25
    assert ModelConfig().latent_dim == 512
    assert ModelConfig.scaled_for_corpus(349, 96).latent_dim == 512
    assert TrainingConfig().consistency_interval == 10
    assert TrainingConfig().learning_rate == 2.0e-5
    assert TrainingConfig().student_direct_gradient_max_norm == 0.25


def test_global_config_rejects_missing_required_sections(tmp_path: Path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"model": {"latent_dim": 192}}), encoding="utf-8")
    try:
        load_global_config(path)
    except ValueError as exc:
        assert "missing mapping sections" in str(exc)
    else:
        raise AssertionError("incomplete config should fail closed")


def test_capacity_report_is_explicit_about_memory_limitations():
    report = build_report()
    assert report.startswith(START_MARKER)
    assert report.endswith(END_MARKER)
    assert "Trainable parameters" in report
    assert "not measured peak VRAM" in report


def test_mainline_128_preset_has_exactly_294719529_trainable_parameters():
    config = load_global_config()
    model_config = ModelConfig.scaled_for_corpus(
        int(config["scaling"]["capacity_basis_unique_geometries"]),
        int(config["model"]["grid_resolution"]),
    )

    assert model_config.grid_resolution == 128
    assert model_config.latent_dim == 512
    assert model_config.coordinate_chunk_size == 8192
    assert model_config.coordinate_decoder_width == 3328
    assert model_config.coordinate_decoder_depth == 12
    assert "Trainable parameters: `294,719,529`" in build_report()


def test_config_digest_is_independent_of_mapping_order():
    first = {"model": {"latent_dim": 192, "grid_resolution": 96}, "training": {"batch_size": 1}}
    reordered = {"training": {"batch_size": 1}, "model": {"grid_resolution": 96, "latent_dim": 192}}

    assert _config_digest(first) == _config_digest(reordered)
