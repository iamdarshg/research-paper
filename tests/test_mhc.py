import sys
from pathlib import Path

import torch


CLI_DIR = Path(__file__).resolve().parents[1] / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

from aircraft_diffusion_cfd import (  # noqa: E402
    ConsistencyModel,
    DiffusionConfig,
    LatentDiffusionUNet,
    LatentTo3DConverter,
    ModelConfig,
)
from mhc import (  # noqa: E402
    ManifoldHyperConnection,
    load_state_dict_mhc_compatible,
)


def test_mhc_is_identity_like_and_doubly_stochastic():
    module = ManifoldHyperConnection(16, streams=8, sinkhorn_iterations=8)
    routing = module.routing()

    assert torch.allclose(routing.sum(dim=-1), torch.ones(8), atol=1e-4)
    assert torch.allclose(routing.sum(dim=-2), torch.ones(8), atol=1e-4)
    assert torch.diagonal(routing).mean() > routing[~torch.eye(8, dtype=torch.bool)].mean()

    update = torch.randn(3, 16)
    mixed = module(update)
    assert mixed.shape == update.shape
    assert torch.max(torch.abs(mixed - update)) < 0.01


def test_mhc_supports_convolution_layout_and_gradients():
    module = ManifoldHyperConnection(16, streams=4, sinkhorn_iterations=4)
    update = torch.randn(2, 16, 2, 2, 2, requires_grad=True)
    output = module(update)
    output.square().mean().backward()

    assert output.shape == update.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(update.grad).all()
    assert torch.isfinite(module.routing_logits.grad).all()


def test_production_diffusion_unet_propagates_mhc_to_all_residual_blocks():
    config = ModelConfig(
        latent_dim=16,
        base_grid_resolution=8,
        encoder_channels=[8, 8, 8],
        decoder_channels=[8, 8, 8],
        attention_groups=8,
        attention_kv_groups=4,
        num_attention_layers=0,
        mhc_enabled=True,
        mhc_streams=8,
    )
    model = LatentDiffusionUNet(config, DiffusionConfig(timesteps=8))
    blocks = list(model.down_blocks) + [model.mid_block] + list(model.up_blocks)
    assert all(block.mhc is not None for block in blocks)

    output = model(torch.randn(2, 16), torch.tensor([0, 3]))
    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()

    consistency = ConsistencyModel(config, DiffusionConfig(timesteps=8))
    assert any(
        "mhc" in name.lower()
        for name, _ in consistency.teacher_model.named_parameters()
    )
    assert any(
        "mhc" in name.lower()
        for name, _ in consistency.student_model.named_parameters()
    )


def test_production_coordinate_decoder_uses_mhc_updates():
    converter = LatentTo3DConverter(
        latent_dim=8,
        grid_resolution=8,
        coordinate_decoder_threshold=1,
        coordinate_decoder_width=8,
        coordinate_decoder_depth=2,
        coordinate_chunk_size=64,
        mhc_enabled=True,
        mhc_streams=8,
    )
    assert len(converter.mhc_coordinate_blocks) == 2
    output = converter(torch.randn(2, 8))
    assert output.shape == (2, 8, 8, 8)
    assert torch.isfinite(output).all()


def test_production_dense_decoder_uses_mhc_updates():
    converter = LatentTo3DConverter(
        latent_dim=8,
        grid_resolution=8,
        coordinate_decoder_threshold=96,
        mhc_enabled=True,
        mhc_streams=8,
    )
    assert converter.decoder_mode == "dense"
    assert len(converter.mhc_dense_hidden) == 2
    output = converter(torch.randn(2, 8))
    assert output.shape == (2, 8, 8, 8)
    assert torch.isfinite(output).all()


def test_old_checkpoint_can_warm_start_enabled_mhc_module():
    old = LatentTo3DConverter(
        latent_dim=8,
        grid_resolution=8,
        coordinate_decoder_threshold=1,
        coordinate_decoder_width=8,
        coordinate_decoder_depth=1,
    )
    enabled = LatentTo3DConverter(
        latent_dim=8,
        grid_resolution=8,
        coordinate_decoder_threshold=1,
        coordinate_decoder_width=8,
        coordinate_decoder_depth=1,
        mhc_enabled=True,
        mhc_streams=8,
    )
    result = load_state_dict_mhc_compatible(enabled, old.state_dict())
    assert result.missing_keys
    assert all("mhc" in key.lower() for key in result.missing_keys)
