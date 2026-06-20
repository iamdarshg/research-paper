import os
import sys

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (
    ConsistencyModel,
    DiffusionConfig,
    ModelConfig,
    TrainingConfig,
    restore_resume_learning_rate_if_zero,
)


def test_consistency_add_noise_preserves_latent_shape():
    config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        conditioning_dim=0,
        use_torch_compile=False,
    )
    model = ConsistencyModel(config, DiffusionConfig())
    latent = torch.zeros((2, 4))
    noise = torch.ones_like(latent)
    timesteps = torch.tensor([0, 10])

    noised = model._add_noise(latent, timesteps, noise)

    assert noised.shape == latent.shape


def test_training_config_weights_generation_reconstruction_by_default():
    config = TrainingConfig()

    assert config.generation_reconstruction_weight == 1.0


def test_restore_resume_learning_rate_if_zero_resets_completed_checkpoint_lr():
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.0)

    restored = restore_resume_learning_rate_if_zero(optimizer, 2e-4)

    assert restored is True
    assert [group["lr"] for group in optimizer.param_groups] == [2e-4]


def test_restore_resume_learning_rate_if_zero_preserves_active_lr():
    layer = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    restored = restore_resume_learning_rate_if_zero(optimizer, 2e-4)

    assert restored is False
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4]
