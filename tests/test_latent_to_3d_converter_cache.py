"""Task 3 parity tests: cached Fourier-encoded coordinate grid.

Verifies that caching the full-grid Fourier encoding in ``LatentTo3DConverter``
produces bit-identical outputs (``torch.equal``) to re-encoding per call, for
both ``forward_flat_indices`` (subset via ``index_select``) and ``forward``
(full grid), and that the cache invalidates on ``coordinate_fourier_bands`` /
dtype change and is never populated on the identity path (bands <= 0).
"""

import os
import sys

import pytest
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import LatentTo3DConverter

# Task 6-1: the fused cat+chunk decode is mathematically identical to three
# separate decodes, but torch's batched matmul may select a different BLAS
# kernel for a [3B*N, D] input than for [B*N, D], producing last-bit float32
# round-off from the different reduction order. Measured max abs diff <= 3e-8
# on CPU/Windows; 1e-6 gives a ~30x margin while still failing on any real
# divergence.
FUSED_DECODE_ATOL = 1e-6


def _make_converter(bands, grid_resolution=8, chunk_size=64):
    return LatentTo3DConverter(
        latent_dim=4,
        grid_resolution=grid_resolution,
        coordinate_decoder_threshold=8,
        coordinate_chunk_size=chunk_size,
        coordinate_decoder_width=16,
        coordinate_decoder_depth=1,
        coordinate_fourier_bands=bands,
        enable_coordinate_gradient_checkpointing=False,
    )


def _uncached_forward_flat_indices(converter, latent, flat_indices):
    """Exact pre-cache code path: index_select the raw grid, then re-encode."""
    flat_indices = flat_indices.to(device=latent.device, dtype=torch.long)
    coords = converter._coordinates(latent.device, latent.dtype).index_select(0, flat_indices)
    encoded = converter._encode_coordinates(coords)
    chunks = []
    chunk_size = converter._effective_coordinate_chunk_size(latent.device)
    for start in range(0, encoded.shape[0], chunk_size):
        chunks.append(
            converter._checkpointed_coordinate_chunk(
                latent, encoded[start:start + chunk_size]
            )
        )
    return torch.cat(chunks, dim=1)


def _uncached_forward(converter, latent):
    """Exact pre-cache code path: re-encode the full grid every call."""
    coords = converter._encode_coordinates(
        converter._coordinates(latent.device, latent.dtype)
    )
    chunks = []
    chunk_size = converter._effective_coordinate_chunk_size(latent.device)
    for start in range(0, coords.shape[0], chunk_size):
        chunks.append(
            converter._checkpointed_coordinate_chunk(latent, coords[start:start + chunk_size])
        )
    voxels = torch.cat(chunks, dim=1)
    return voxels.view(latent.shape[0], *converter.output_shape)


@pytest.mark.parametrize("fourier_bands", [8, 2, 0])
def test_encoded_grid_cache_is_bit_identical_to_reencode(fourier_bands):
    converter = _make_converter(fourier_bands)
    device = torch.device("cpu")
    dtype = torch.float32
    with torch.no_grad():
        reference = converter._encode_coordinates(
            converter._coordinates(device, dtype)
        )
        cached = converter._encode_full_coordinate_grid(device, dtype)
    assert torch.equal(cached, reference)

    if fourier_bands > 0:
        assert converter._encoded_coordinate_grid.numel() == reference.numel()
        assert converter._cached_coordinate_fourier_bands == fourier_bands
        indices = torch.tensor([0, 1, 7, 100, 255, 256, 511], dtype=torch.long)
        subset_cached = converter._encoded_coordinate_grid.index_select(0, indices)
        subset_reference = converter._encode_coordinates(
            converter._coordinates(device, dtype).index_select(0, indices)
        )
        assert torch.equal(subset_cached, subset_reference)
    else:
        # Identity path is never cached.
        assert converter._encoded_coordinate_grid.numel() == 0
        assert converter._cached_coordinate_fourier_bands == -1


def test_cache_is_reused_when_unchanged_and_invalidated_on_bands_change():
    converter = _make_converter(bands=8)
    device = torch.device("cpu")
    dtype = torch.float32
    with torch.no_grad():
        first = converter._encode_full_coordinate_grid(device, dtype)
        first_ptr = converter._encoded_coordinate_grid.data_ptr()
        second = converter._encode_full_coordinate_grid(device, dtype)
    assert torch.equal(first, second)
    # Buffer object is reused on a cache hit (no re-encode).
    assert converter._encoded_coordinate_grid.data_ptr() == first_ptr
    assert converter._cached_coordinate_fourier_bands == 8

    converter.coordinate_fourier_bands = 4
    with torch.no_grad():
        third = converter._encode_full_coordinate_grid(device, dtype)
    assert third.shape[-1] == 3 * (1 + 2 * 4)
    assert converter._cached_coordinate_fourier_bands == 4
    assert converter._encoded_coordinate_grid.numel() == (8 ** 3) * 3 * (1 + 2 * 4)
    reference = converter._encode_coordinates(converter._coordinates(device, dtype))
    assert torch.equal(third, reference)


def test_cache_invalidates_on_dtype_change():
    converter = _make_converter(bands=8)
    device = torch.device("cpu")
    with torch.no_grad():
        f32 = converter._encode_full_coordinate_grid(device, torch.float32)
        f64 = converter._encode_full_coordinate_grid(device, torch.float64)
    assert f32.dtype == torch.float32
    assert f64.dtype == torch.float64
    assert converter._encoded_coordinate_grid.dtype == torch.float64
    reference_f64 = converter._encode_coordinates(
        converter._coordinates(device, torch.float64)
    )
    assert torch.equal(f64, reference_f64)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("fourier_bands", [8, 0])
def test_forward_flat_indices_cold_vs_warm_and_uncached_bit_identical(
    batch_size, fourier_bands
):
    torch.manual_seed(1234)
    converter = _make_converter(fourier_bands)
    converter.eval()
    latent = torch.randn(batch_size, 4, dtype=torch.float32)
    indices = torch.tensor([0, 3, 9, 100, 255, 256, 511], dtype=torch.long)

    with torch.no_grad():
        cold = converter.forward_flat_indices(latent, indices)
        warm = converter.forward_flat_indices(latent, indices)
        reference = _uncached_forward_flat_indices(converter, latent, indices)

    assert cold.shape == (batch_size, indices.numel())
    assert torch.equal(cold, warm)
    assert torch.equal(warm, reference)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("fourier_bands", [8, 0])
def test_forward_cold_vs_warm_and_uncached_bit_identical(batch_size, fourier_bands):
    torch.manual_seed(4321)
    converter = _make_converter(fourier_bands)
    converter.eval()
    latent = torch.randn(batch_size, 4, dtype=torch.float32)

    with torch.no_grad():
        cold = converter(latent)
        warm = converter(latent)
        reference = _uncached_forward(converter, latent)

    assert cold.shape == (batch_size, *converter.output_shape)
    assert torch.equal(cold, warm)
    assert torch.equal(warm, reference)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("fourier_bands", [8, 0])
def test_fused_stacked_decode_matches_separate_decodes(batch_size, fourier_bands):
    """Task 6-1 regression: the three-way decoder fuse matches separate decodes.

    The production geometry-loss path (OptimizedDiffusionTrainer) fuses the
    three ``forward_flat_indices`` decodes of ``latent`` / ``x0_pred`` /
    ``generation_latent`` into ONE stacked call (``torch.cat(..., dim=0)``)
    followed by ``torch.chunk(..., 3, dim=0)``. Cat + chunk are mathematically
    identical to three separate decodes, so the fused output must match within
    float32 last-bit round-off (``FUSED_DECODE_ATOL``), not approximately.
    """
    torch.manual_seed(20260815)
    converter = _make_converter(fourier_bands)
    converter.eval()
    latent = torch.randn(batch_size, 4, dtype=torch.float32)
    x0_pred = torch.randn(batch_size, 4, dtype=torch.float32)
    generation_latent = torch.randn(batch_size, 4, dtype=torch.float32)
    flat_indices = torch.tensor([0, 3, 9, 100, 255, 256, 511], dtype=torch.long)

    with torch.no_grad():
        separate = tuple(
            converter.forward_flat_indices(code, flat_indices)
            for code in (latent, x0_pred, generation_latent)
        )
        stacked = converter.forward_flat_indices(
            torch.cat((latent, x0_pred, generation_latent), dim=0),
            flat_indices,
        )
        fused = tuple(torch.chunk(stacked, 3, dim=0))

    for fused_part, separate_part in zip(fused, separate):
        assert fused_part.shape == separate_part.shape
        assert torch.allclose(
            fused_part,
            separate_part,
            atol=FUSED_DECODE_ATOL,
            rtol=FUSED_DECODE_ATOL,
        ), (
            f"fused vs separate decode diverged: "
            f"max_abs_diff={float((fused_part - separate_part).abs().max().item()):.3e}"
        )
