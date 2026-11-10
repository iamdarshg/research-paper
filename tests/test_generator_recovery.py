"""Regression tests for the aircraft-generator recovery pass (2026-11-05).

These tests pin the root causes measured in
`evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`:

F1  A probability field that has collapsed to a near-constant value materialises
    into a solid block (or nothing) and must not be reported as an aircraft.
F2  The procedural corpus generator used its own, laxer validity gate, so the
    "accepted" corpus was mostly invalid under the paper's declared screen.
    There is now one screen, and the procedural generator uses it.
F3  The `train` CLI accepted `--learning-rate` but silently trained the decoder
    and the consistency student at the much smaller values from
    `CLI/config.yaml` (2e-5), which starved the only module that can turn a
    latent into occupied voxels.
F4  Generation used the 4-step consistency student only. A deterministic
    teacher-DDIM sampler is now available and is used when the student has not
    converged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_validity import evaluate_aircraft_validity  # noqa: E402
from procedural_aircraft_generator import (  # noqa: E402
    AIRCRAFT_TYPES,
    generate_aircraft,
    validate,
)
from aircraft_diffusion_cfd import (  # noqa: E402
    NoiseSchedule,
    DiffusionConfig,
    sample_latent_with_ddim,
)


# --- F1: a collapsed probability field is not an aircraft ------------------


def test_near_constant_probability_field_fails_declared_screen():
    """The measured 16^3 smoke-checkpoint output (p in [0.44, 0.57]) fails.

    At a 0.5 materialisation threshold such a field becomes a solid block: its
    extents span the whole lattice, which violates span_sanity,
    normalization_margin and the grounded occupancy envelope.
    """
    field = torch.full((32, 32, 32), 0.5) + 0.01 * torch.rand(32, 32, 32)
    binary = (field > 0.5).float()
    report = evaluate_aircraft_validity(binary.numpy(), canonicalize=True)

    assert report["status"] == "fail"
    assert "span_sanity" in report["failed_checks"]
    assert "normalization_margin" in report["failed_checks"]


def test_empty_field_fails_declared_screen():
    report = evaluate_aircraft_validity(np.zeros((32, 32, 32), dtype=np.float32))
    assert report["status"] == "fail"
    assert "nonempty_occupancy" in report["failed_checks"]


# --- F2: one validity screen, used by the procedural corpus generator -----


def test_procedural_validator_agrees_with_declared_screen():
    rng = np.random.default_rng(0)
    cases = [
        np.zeros((96, 96, 96), dtype=np.float32),
        np.ones((96, 96, 96), dtype=np.float32),
        (rng.random((96, 96, 96)) > 0.5).astype(np.float32),
    ]
    for case in cases:
        expected = evaluate_aircraft_validity(case, canonicalize=False)["status"] == "pass"
        assert validate(case) is bool(expected)


def test_every_accepted_procedural_sample_passes_declared_screen():
    """Every family produces samples that pass the paper's own screen.

    Before the fix the procedural gate accepted 96^3 shapes the declared screen
    rejects (measured: 3/24 across the eight families passed).
    """
    import random

    rng = random.Random(42)
    accepted = 0
    for aircraft_type in AIRCRAFT_TYPES:
        for _ in range(3):
            voxels = generate_aircraft(rng, aircraft_type)
            if not validate(voxels):
                continue
            report = evaluate_aircraft_validity(voxels, canonicalize=True)
            assert report["status"] == "pass", (
                f"{aircraft_type} accepted but failed: {report['failed_checks']}"
            )
            accepted += 1
    assert accepted >= len(AIRCRAFT_TYPES) * 2


def test_procedural_samples_are_bilaterally_symmetric():
    import random

    rng = random.Random(7)
    for aircraft_type in AIRCRAFT_TYPES:
        voxels = generate_aircraft(rng, aircraft_type)
        assert np.array_equal(voxels, voxels[:, ::-1, :]), aircraft_type


# --- F3: decoder / student learning rates are controllable ----------------


def test_train_cli_exposes_decoder_learning_rate_options():
    from aircraft_diffusion_cfd import train as train_command

    options = {param.name for param in train_command.params}
    assert "converter_learning_rate" in options
    assert "consistency_student_learning_rate" in options
    assert "latent_augmentation_std" in options


def test_training_config_accepts_explicit_decoder_rate():
    from aircraft_diffusion_cfd import TrainingConfig

    config = TrainingConfig(
        learning_rate=2e-4,
        converter_learning_rate=2e-4,
        consistency_student_learning_rate=2e-4,
    )
    assert config.converter_learning_rate == 2e-4
    assert config.consistency_student_learning_rate == 2e-4


def test_training_config_default_decoder_rate_is_unchanged():
    from aircraft_diffusion_cfd import TrainingConfig

    config = TrainingConfig(learning_rate=2e-4)
    # Legacy behaviour preserved: the decoder keeps the config.yaml rate.
    assert config.converter_learning_rate == 2e-5


# --- F4: teacher-DDIM sampler --------------------------------------------


class _RecordingDenoiser(torch.nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.calls = 0
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.latent_dim = latent_dim

    def forward(self, x, t, condition=None):
        self.calls += 1
        return torch.zeros_like(x)


def test_ddim_sampler_uses_the_trained_denoiser():
    denoiser = _RecordingDenoiser()
    schedule = NoiseSchedule(DiffusionConfig(timesteps=20, student_steps=4))
    latent = sample_latent_with_ddim(
        denoiser,
        schedule,
        (2, denoiser.latent_dim),
        num_steps=5,
        condition=None,
    )
    assert denoiser.calls == 5
    assert tuple(latent.shape) == (2, denoiser.latent_dim)
    # Every value stays inside the model's declared latent support.
    assert float(latent.min()) >= 0.0
    assert float(latent.max()) <= 1.0


def test_optimized_generator_rejects_unknown_sampler(tmp_path):
    from aircraft_diffusion_cfd import OptimizedAircraftGenerator

    checkpoint_path = _grep_checkpoint_path()
    if checkpoint_path is None:
        import pytest

        pytest.skip("no checkpoint available in the working tree")
    generator = OptimizedAircraftGenerator(checkpoint_path, device=torch.device("cpu"))
    try:
        generator.generate(sampler="not-a-sampler")
    except ValueError as error:
        assert "sampler" in str(error)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("unknown sampler was accepted")
    del tmp_path


def _grep_checkpoint_path():
    root = os.path.dirname(os.path.dirname(__file__))
    for candidate in (
        os.path.join(root, "checkpoints_protocol_smoke", "final_optimized_model.pt"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None
