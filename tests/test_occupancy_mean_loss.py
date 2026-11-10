"""Regression tests for the data-dependent occupancy-matching training loss.

Context (2026-11-10 occupancy pass).  The frozen validity screen fails every
generated sample on ``grounded_occupancy_envelope`` because DDIM generation
lands at occupancy ratio ~0.051 against a 0.04 bound, while the source
training geometry sits at ~0.0265.  Diagnosis on the shipped checkpoint
(``scripts/diagnose_occupancy_path.py``, run1) measured the cause as a
model-side mass defect, not a threshold artifact:

  * clean-latent decode        occupancy 1041 voxels (1.21x target),
    probability mass 0.0331;
  * DDIM-50 free-running decode occupancy 1678 voxels (1.99x target),
    probability mass 0.0504;
  * target occupancy          864 voxels, mass 0.0264;
  * generated latent sits 3.93 L2 from its nearest *training* latent while the
    training latent cloud's own nearest-neighbour spacing is 0.069, and the
    decoder answers an off-manifold probe (uniform random latent) with 0.0482
    occupancy -- i.e. the sampler emits a latent the decoder has never seen and
    the decoder's generic answer is ~2x too dense.

The candidate fix under test is a *data-dependent* occupancy loss: for each
sample in the batch, penalise the difference between the predicted mean
occupancy and THAT SAMPLE'S ground-truth occupancy fraction.  No fixed output
occupancy target, no per-sample post-hoc threshold, no top-k rescaling.  The
materialization threshold (0.5) and the 12-check screen stay frozen.

These tests were written before the implementation and failed with
``ImportError`` on first run (recorded in the pass evidence file).
"""

from __future__ import annotations

import pytest
import torch

from aircraft_diffusion_cfd import (
    TrainingConfig,
    combine_training_loss_terms,
    geometry_occupancy_fraction,
    occupancy_mean_matching_loss,
    occupancy_soft_fraction,
    validate_collapse_watchdog_config,
)


def _paired_geometry(occupied: int, total: int = 32 * 32 * 32) -> torch.Tensor:
    """A (1, 32, 32, 32) binary target with exactly ``occupied`` voxels set."""
    flat = torch.zeros(total, dtype=torch.float32)
    flat[:occupied] = 1.0
    return flat.reshape(1, 32, 32, 32)


def test_geometry_occupancy_fraction_is_measured_per_sample():
    """Target occupancy must come from each source sample, not a constant."""
    batch = torch.stack(
        [
            flat
            for flat in (
                _paired_geometry(400).reshape(-1),
                _paired_geometry(1200).reshape(-1),
            )
        ]
    )

    fractions = geometry_occupancy_fraction(batch)

    assert fractions.shape == (2,)
    assert float(fractions[0]) == pytest.approx(400 / 32768)
    assert float(fractions[1]) == pytest.approx(1200 / 32768)
    assert float(fractions[0]) < float(fractions[1])


def test_occupancy_soft_fraction_tracks_the_frozen_threshold():
    """The soft occupancy surrogate must approximate the 0.5-threshold count."""
    logits = torch.full((1, 32, 32, 32), -8.0)
    logits.reshape(1, -1)[0, :1000] = 8.0

    soft = occupancy_soft_fraction(logits, temperature=0.05)

    assert float(soft) == pytest.approx(1000 / 32768, abs=2.0e-3)

    # Gradient-safe default: the mean probability of the same field.
    mask = torch.zeros(1, 32, 32, 32, dtype=torch.bool)
    mask.reshape(1, -1)[0, :1000] = True
    mean_probability = torch.sigmoid(logits).masked_select(mask).mean()
    assert float(occupancy_soft_fraction(logits)) == pytest.approx(
        float(mean_probability) / 32768 * 1000 + float(torch.sigmoid(logits).masked_select(~mask).mean()) / 32768 * 31768,
        rel=1.0e-5,
    )


def test_occupancy_mean_matching_loss_vanishes_at_the_source_occupancy():
    """A field whose occupancy equals the source sample's scores ~0."""
    target_geometry = _paired_geometry(868)
    logits = torch.full((1, 32, 32, 32), -12.0)
    logits.reshape(1, -1)[0, :868] = 12.0

    loss = occupancy_mean_matching_loss(
        logits, geometry_occupancy_fraction(target_geometry)
    )

    assert float(loss) < 1.0e-3


def test_occupancy_mean_matching_loss_penalizes_over_occupancy():
    """Over-occupied predictions cost more than matched ones, and one gradient
    step moves the predicted occupancy toward the source occupancy."""
    target_geometry = _paired_geometry(868)
    target_occupancy = geometry_occupancy_fraction(target_geometry)

    matched = torch.full((1, 32, 32, 32), -3.0)
    matched.reshape(1, -1)[0, :868] = 3.0
    matched = matched.requires_grad_(True)

    over = torch.full((1, 32, 32, 32), -3.0)
    over.reshape(1, -1)[0, :1700] = 3.0
    over = over.requires_grad_(True)

    matched_loss = occupancy_mean_matching_loss(matched, target_occupancy)
    over_loss = occupancy_mean_matching_loss(over, target_occupancy)

    assert float(over_loss) > float(matched_loss) + 0.5

    over_loss.backward()
    assert over.grad is not None
    excess_gradient = over.grad.reshape(1, -1)[0, 868:1700]
    # Excess occupied voxels must be pushed *down*: d loss / d logit > 0, and a
    # gradient-descent step of -lr * grad therefore lowers their logits.
    assert bool((excess_gradient > 0).all())

    with torch.no_grad():
        stepped = over - 0.5 * over.grad
        stepped_loss = occupancy_mean_matching_loss(stepped, target_occupancy)
        stepped_occupancy = occupancy_soft_fraction(stepped)
    predicted_before = occupancy_soft_fraction(over.detach())
    assert float(stepped_loss) < float(over_loss)
    assert float(predicted_before - target_occupancy) > float(
        stepped_occupancy - target_occupancy
    )


def test_occupancy_mean_matching_loss_uses_per_sample_targets():
    """Two rows sharing one prediction but different source occupancies must
    produce two different per-row errors (a constant target could not)."""
    sparse_target = geometry_occupancy_fraction(_paired_geometry(400))
    dense_target = geometry_occupancy_fraction(_paired_geometry(1310))
    assert float(sparse_target) != float(dense_target)

    shared = torch.full((1, 32, 32, 32), -12.0)
    shared.reshape(1, -1)[0, :900] = 12.0

    sparse_loss = occupancy_mean_matching_loss(shared, sparse_target)
    dense_loss = occupancy_mean_matching_loss(shared, dense_target)

    assert float(sparse_loss) > 0.0
    assert float(dense_loss) > 0.0
    assert float(sparse_loss) > float(dense_loss)

    batched = occupancy_mean_matching_loss(
        shared.expand(2, -1, -1, -1).contiguous(),
        torch.cat([sparse_target, dense_target]),
    )
    assert float(batched) == pytest.approx(
        (float(sparse_loss) + float(dense_loss)) / 2.0, rel=1.0e-5
    )


def test_training_config_occupancy_weight_defaults_off_and_is_validated():
    """Default 0.0 keeps legacy behaviour bit-identical; negatives must fail."""
    assert TrainingConfig().geometry_occupancy_mean_weight == 0.0

    with pytest.raises(ValueError):
        validate_collapse_watchdog_config(
            TrainingConfig(geometry_occupancy_mean_weight=-1.0)
        )

    validate_collapse_watchdog_config(
        TrainingConfig(geometry_occupancy_mean_weight=1.0)
    )


def test_combine_training_loss_terms_adds_the_weighted_occupancy_term():
    """The occupancy term must enter the single optimizer objective, scaled by
    its configured weight, and contribute nothing at weight 0."""
    logits = torch.zeros((1, 4, 4, 4))
    zero = logits.new_zeros(())
    seeded = {
        "mse_loss_val": torch.tensor(0.5),
        "geometry_loss_val": torch.tensor(0.25),
        "generation_geometry_loss_val": torch.tensor(0.125),
        "consistency_loss": torch.tensor(0.0625),
        "latent_reconstruction_loss_val": torch.tensor(0.03125),
    }
    occupancy_loss = torch.tensor(0.75)

    baseline = combine_training_loss_terms(
        seeded["mse_loss_val"],
        seeded["geometry_loss_val"],
        seeded["generation_geometry_loss_val"],
        seeded["consistency_loss"],
        TrainingConfig(
            geometry_occupancy_mean_weight=0.0,
            latent_reconstruction_weight=1.0,
        ),
        latent_reconstruction_loss_val=seeded["latent_reconstruction_loss_val"],
        occupancy_mean_loss_val=occupancy_loss,
    )
    weighted = combine_training_loss_terms(
        seeded["mse_loss_val"],
        seeded["geometry_loss_val"],
        seeded["generation_geometry_loss_val"],
        seeded["consistency_loss"],
        TrainingConfig(
            geometry_occupancy_mean_weight=2.0,
            latent_reconstruction_weight=1.0,
        ),
        latent_reconstruction_loss_val=seeded["latent_reconstruction_loss_val"],
        occupancy_mean_loss_val=occupancy_loss,
    )

    assert torch.allclose(baseline, weighted.new_tensor(0.96875), atol=1.0e-6)
    assert float(weighted) > float(baseline)
    assert float(weighted - baseline) == pytest.approx(2.0 * 0.75, rel=1.0e-5)
    assert zero.numel() == 1
