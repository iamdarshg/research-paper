"""Classifier-free guidance (2026-11-15 audit).

``DiffusionConfig.guidance_scale`` (7.5) and ``OptimizedAircraftGenerator.generate
(guidance_scale=7.5)`` were accepted but never used: training had no condition
dropout and no sampler combined conditional and unconditional predictions.
Measured on checkpoint C4 (48 holdout conditions): denoising eps-MSE with the
correct condition was only 1-3% below a shuffled condition, and DDIM samples were
no closer to their own condition's latent than to others (retrieval 2.4%, chance
2.1%). These tests pin the contract of the fix:

* a per-sample ``condition_mask`` of 0 reproduces the ``condition=None`` path
  exactly; a mask of 1 is the unmasked conditional path;
* condition dropout in training is off by default (legacy bit-for-bit) and the
  mask helper drops the requested fraction;
* DDIM with ``guidance_scale=1`` is the unguided sampler; other scales combine
  eps_u + w * (eps_c - eps_u);
* the generator refuses guidance > 1 on a checkpoint trained without dropout
  (its unconditional branch was never trained) instead of silently applying it.
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
    LatentDiffusionUNet,
    ModelConfig,
    NoiseSchedule,
    TrainingConfig,
    condition_dropout_mask,
    infer_conditioning_dim,
    resolve_guidance_scale,
    sample_latent_with_ddim,
)

COND_DIM = infer_conditioning_dim()


def _unet() -> LatentDiffusionUNet:
    torch.manual_seed(0)
    config = ModelConfig(
        latent_dim=16, base_grid_resolution=8, encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8], attention_groups=8, attention_kv_groups=4,
        num_attention_layers=0, conditioning_dim=COND_DIM,
    )
    return LatentDiffusionUNet(config, DiffusionConfig(timesteps=8)).eval()


def test_mask_zero_matches_unconditional_and_mask_one_matches_conditional():
    model = _unet()
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(1))
    t = torch.tensor([0, 3, 7])
    cond = torch.rand(3, COND_DIM, generator=torch.Generator().manual_seed(2)) * 5
    with torch.no_grad():
        conditional = model(x, t, condition=cond)
        unconditional = model(x, t, condition=None)
        mixed = model(x, t, condition=cond, condition_mask=torch.tensor([1.0, 0.0, 1.0]))
        all_on = model(x, t, condition=cond, condition_mask=torch.ones(3))
    assert torch.allclose(all_on, conditional, atol=1e-6)
    assert torch.allclose(mixed[1], unconditional[1], atol=1e-6)
    assert torch.allclose(mixed[0], conditional[0], atol=1e-6)
    assert torch.allclose(mixed[2], conditional[2], atol=1e-6)
    assert not torch.allclose(conditional[1], unconditional[1], atol=1e-4)


def test_condition_dropout_default_off_and_mask_rate():
    assert TrainingConfig().condition_dropout == 0.0
    assert condition_dropout_mask(8, 0.0, torch.device("cpu")) is None
    g = torch.Generator().manual_seed(0)
    mask = condition_dropout_mask(20000, 0.1, torch.device("cpu"), generator=g)
    assert mask.shape == (20000,)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert abs((1 - mask.mean().item()) - 0.1) < 0.01
    with pytest.raises(ValueError):
        condition_dropout_mask(4, 1.5, torch.device("cpu"))


class _CondAwareNoise(torch.nn.Module):
    """Test double: eps = 0.1 x  (+0.2 when a condition is present and unmasked)."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, x, t, condition=None, condition_mask=None):
        self.calls.append(None if condition is None else (condition_mask if condition_mask is not None else "on"))
        out = 0.1 * x + self.anchor
        if condition is not None:
            on = torch.ones(x.shape[0]) if condition_mask is None else condition_mask
            out = out + 0.2 * on[:, None]
        return out


def test_guidance_scale_one_is_the_unguided_sampler():
    schedule = NoiseSchedule(DiffusionConfig())
    noise = torch.randn((2, 8), generator=torch.Generator().manual_seed(3))
    cond = torch.ones(2, COND_DIM)
    base = sample_latent_with_ddim(_CondAwareNoise(), schedule, (2, 8), num_steps=6,
                                   condition=cond, initial_noise=noise)
    model = _CondAwareNoise()
    one = sample_latent_with_ddim(model, schedule, (2, 8), num_steps=6, condition=cond,
                                  initial_noise=noise, guidance_scale=1.0)
    assert torch.equal(base, one)
    assert all(c == "on" for c in model.calls)  # no unconditional evaluations


def test_guided_step_combines_conditional_and_unconditional():
    schedule = NoiseSchedule(DiffusionConfig())
    noise = torch.randn((2, 8), generator=torch.Generator().manual_seed(4))
    cond = torch.ones(2, COND_DIM)

    class _Guided(_CondAwareNoise):
        pass

    w = 3.0
    guided = sample_latent_with_ddim(_Guided(), schedule, (2, 8), num_steps=6, condition=cond,
                                     initial_noise=noise, guidance_scale=w,
                                     latent_value_min=-1e9, latent_value_max=1e9)

    class _Expected(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, t, condition=None):
            # eps_u + w (eps_c - eps_u) = 0.1 x + w * 0.2
            return 0.1 * x + w * 0.2

    expected = sample_latent_with_ddim(_Expected(), schedule, (2, 8), num_steps=6, condition=cond,
                                       initial_noise=noise, latent_value_min=-1e9, latent_value_max=1e9)
    assert torch.allclose(guided, expected, atol=1e-5)


def test_guidance_refused_without_condition_dropout_training():
    # legacy checkpoint (no dropout): default resolves to 1, explicit >1 is refused
    assert resolve_guidance_scale(None, trained_condition_dropout=0.0, configured=7.5) == 1.0
    assert resolve_guidance_scale(1.0, trained_condition_dropout=0.0, configured=7.5) == 1.0
    with pytest.raises(ValueError):
        resolve_guidance_scale(3.0, trained_condition_dropout=0.0, configured=7.5)
    # trained with dropout: default is the configured scale, explicit wins
    assert resolve_guidance_scale(None, trained_condition_dropout=0.1, configured=7.5) == 7.5
    assert resolve_guidance_scale(2.0, trained_condition_dropout=0.1, configured=7.5) == 2.0
    with pytest.raises(ValueError):
        resolve_guidance_scale(0.0, trained_condition_dropout=0.1, configured=7.5)
