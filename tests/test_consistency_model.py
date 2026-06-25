import os
import sys

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (
    CFDConfig,
    ConsistencyModel,
    DiffusionConfig,
    ModelConfig,
    OptimizedDiffusionTrainer,
    TrainingConfig,
    balanced_voxel_bce_with_logits,
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


def test_balanced_voxel_bce_penalizes_sparse_empty_prior():
    target = torch.zeros((1, 1000), dtype=torch.float32)
    target[0, :5] = 1.0
    empty_prior_logits = torch.full_like(target, -5.3)
    perfect_logits = torch.where(target > 0.5, torch.full_like(target, 8.0), torch.full_like(target, -8.0))

    empty_prior_loss = balanced_voxel_bce_with_logits(empty_prior_logits, target)
    perfect_loss = balanced_voxel_bce_with_logits(perfect_logits, target)

    assert float(empty_prior_loss) > 2.0
    assert float(perfect_loss) < 0.001


def test_trainer_syncs_consistency_teacher_from_diffusion_model():
    model_config = ModelConfig(
        latent_dim=4,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        base_grid_resolution=4,
        grid_resolution=4,
        conditioning_dim=0,
        use_torch_compile=False,
    )
    trainer = OptimizedDiffusionTrainer(
        model_config,
        DiffusionConfig(timesteps=8, teacher_steps=8, student_steps=2),
        TrainingConfig(num_epochs=1),
        CFDConfig(base_grid_resolution=4),
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        for index, parameter in enumerate(trainer.diffusion_model.parameters(), start=1):
            parameter.fill_(index * 0.01)
        for parameter in trainer.consistency_model.teacher_model.parameters():
            parameter.zero_()

    trainer._sync_consistency_teacher()

    for teacher_parameter, diffusion_parameter in zip(
        trainer.consistency_model.teacher_model.parameters(),
        trainer.diffusion_model.parameters(),
    ):
        assert torch.equal(teacher_parameter, diffusion_parameter)


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
