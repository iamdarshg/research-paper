"""Latent standardization for the diffusion path (2026-11-12 audit).

Measured on the 660-record procedural training split: latents span [0, 0.389]
with per-dimension std 0.0003-0.043 (mean 0.019) while DDPM noise has unit
variance; training noise-MSE sat at 2.4-3.2 (worse than predicting zero).
These tests pin the contract of the fix:

* statistics come from the given latents only, with a positive floor;
* standardize/destandardize are exact inverses, and ``None`` statistics are
  the identity, so legacy checkpoints keep the legacy sampler bit-for-bit;
* the DDIM sampler bounds and returns RAW latents while it iterates in the
  standardized space;
* statistics survive the checkpoint payload round trip.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (  # noqa: E402
    DiffusionConfig,
    NoiseSchedule,
    TrainingConfig,
    compute_latent_standardization,
    destandardize_latent,
    latent_standardization_from_payload,
    latent_standardization_payload,
    sample_latent_with_ddim,
    standardize_latent,
)


def _small_latents(n: int = 64, d: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.linspace(0.05, 0.35, d)
    spread = torch.linspace(0.0005, 0.04, d)
    return base + spread * torch.randn((n, d), generator=g)


def test_statistics_match_training_latents_and_respect_floor():
    latents = _small_latents()
    latents[:, 0] = 0.2  # zero-variance dimension
    mean, scale = compute_latent_standardization(latents, floor=1e-3)
    assert torch.allclose(mean, latents.mean(0), atol=1e-6)
    assert scale[0].item() == pytest.approx(1e-3)
    assert torch.allclose(scale[1:], latents[:, 1:].std(0).clamp_min(1e-3), atol=1e-6)
    standardized = standardize_latent(latents, mean, scale)
    assert standardized[:, 1:].std(0).sub(1.0).abs().max().item() < 1e-4


def test_round_trip_is_exact_and_none_is_identity():
    latents = _small_latents()
    mean, scale = compute_latent_standardization(latents)
    back = destandardize_latent(standardize_latent(latents, mean, scale), mean, scale)
    assert torch.allclose(back, latents, atol=1e-6)
    assert standardize_latent(latents, None, None) is latents
    assert destandardize_latent(latents, None, None) is latents


def test_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        compute_latent_standardization(torch.zeros(1, 4))
    with pytest.raises(ValueError):
        compute_latent_standardization(torch.zeros(4, 4), floor=0.0)
    with pytest.raises(FloatingPointError):
        compute_latent_standardization(torch.full((4, 4), float("nan")))


class _ConstantNoise(torch.nn.Module):
    """Deterministic stand-in denoiser: predicts a fixed fraction of x_t."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, condition=None):  # noqa: D401 - test double
        return 0.3 * x + self.anchor


def test_ddim_with_identity_statistics_matches_legacy_sampler():
    schedule = NoiseSchedule(DiffusionConfig())
    noise = torch.randn((3, 8), generator=torch.Generator().manual_seed(1))
    legacy = sample_latent_with_ddim(
        _ConstantNoise(), schedule, (3, 8), num_steps=10, initial_noise=noise
    )
    explicit_none = sample_latent_with_ddim(
        _ConstantNoise(), schedule, (3, 8), num_steps=10, initial_noise=noise,
        latent_standardization_mean=None, latent_standardization_scale=None,
    )
    identity = sample_latent_with_ddim(
        _ConstantNoise(), schedule, (3, 8), num_steps=10, initial_noise=noise,
        latent_standardization_mean=torch.zeros(8),
        latent_standardization_scale=torch.ones(8),
    )
    assert torch.equal(legacy, explicit_none)
    assert torch.allclose(legacy, identity, atol=1e-6)


def test_ddim_with_statistics_returns_raw_latent_inside_support():
    schedule = NoiseSchedule(DiffusionConfig())
    latents = _small_latents()
    mean, scale = compute_latent_standardization(latents)
    noise = torch.randn((4, 8), generator=torch.Generator().manual_seed(2))
    out = sample_latent_with_ddim(
        _ConstantNoise(), schedule, (4, 8), num_steps=10, initial_noise=noise,
        latent_standardization_mean=mean, latent_standardization_scale=scale,
    )
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
    # A denoiser that predicts a fraction of x_t yields a standardized x0 of
    # moderate magnitude, so the raw output stays near the training cloud
    # instead of saturating at the [0, 1] corners like the unscaled path.
    assert (out - mean).abs().div(scale).max().item() < 50.0


def test_payload_round_trip_and_legacy_absence():
    mean, scale = compute_latent_standardization(_small_latents())
    payload = latent_standardization_payload(mean, scale)
    restored_mean, restored_scale = latent_standardization_from_payload(payload)
    assert torch.allclose(restored_mean, mean) and torch.allclose(restored_scale, scale)
    assert latent_standardization_from_payload(None) == (None, None)
    assert latent_standardization_payload(None, None) is None
    with pytest.raises(ValueError):
        latent_standardization_from_payload({"mean": [0.0, 0.0], "scale": [1.0, 0.0]})


def test_training_config_default_is_legacy_off():
    assert TrainingConfig().latent_standardization is False
    assert TrainingConfig(latent_standardization=True).latent_standardization is True
