import json
import os
import sys

import numpy as np
import pytest
import torch


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from aircraft_diffusion_cfd import (  # noqa: E402
    AircraftDesignDataset,
    aircraft_collate_fn,
    build_train_loader,
    transfer_training_batch_to_device,
)
from geometry_store import CompactGeometryStore  # noqa: E402


def test_store_deduplicates_content_and_keeps_canonical_uint8_tensor():
    geometry = torch.zeros((8, 8, 8))
    geometry[2:6, 3:5, 1:7] = 1
    equal_geometry = geometry.clone()
    store = CompactGeometryStore()

    first = store.add("a", geometry)
    second = store.add("b", equal_geometry)

    assert torch.equal(geometry, equal_geometry)
    assert first == second
    assert store.unique_count == 1
    assert store.materialize(first).dtype == torch.uint8
    assert store.materialize(first).device.type == "cpu"
    assert store.materialize(first).is_contiguous()
    first_read = store.materialize(first)
    second_read = store.materialize(second)
    assert first_read.data_ptr() != second_read.data_ptr()
    first_read.zero_()
    assert torch.equal(store.get(second), (geometry > 0.5).to(torch.uint8))


def test_store_rejects_supplied_hash_hit_with_different_content():
    store = CompactGeometryStore()
    store.add("a", torch.zeros((4, 4, 4)), content_hash="declared")

    with pytest.raises(ValueError, match="declared"):
        store.add("b", torch.ones((4, 4, 4)), content_hash="declared")


def test_store_rejects_supplied_hash_hit_with_different_shape():
    store = CompactGeometryStore()
    store.add("a", torch.zeros((2, 2, 2)), content_hash="declared")

    with pytest.raises(ValueError, match="declared"):
        store.add("b", torch.zeros((1, 2, 4)), content_hash="declared")


def test_manifest_records_reference_shared_geometry_and_preserve_getitem(tmp_path):
    geometry = np.zeros((4, 4, 4), dtype=np.float32)
    geometry[1:3, :, 2] = 1
    np.save(tmp_path / "geometry.npy", geometry)
    np.save(tmp_path / "latent-a.npy", np.zeros(8, dtype=np.float32))
    np.save(tmp_path / "latent-b.npy", np.ones(8, dtype=np.float32))
    records = [
        {
            "source_id": "a",
            "geometry_path": "geometry.npy",
            "latent_path": "latent-a.npy",
        },
        {
            "source_id": "b",
            "geometry_path": "geometry.npy",
            "latent_path": "latent-b.npy",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    dataset = AircraftDesignDataset(manifest_path=str(manifest_path), latent_dim=8)
    first = dataset[0]
    second = dataset[1]

    assert dataset.geometry_indices == [0, 0]
    assert dataset.geometry_store.unique_count == 1
    assert first["geometry"].data_ptr() != second["geometry"].data_ptr()
    assert first["geometry"].dtype == torch.uint8
    assert first["geometry"].shape == (4, 4, 4)
    assert first["latent"].shape == (8,)
    assert first["condition_vector"].ndim == 1
    first["geometry"].zero_()
    assert int(dataset[1]["geometry"].sum()) == int(np.count_nonzero(geometry))


def test_manifest_npy_loader_does_not_eagerly_expand_uint8_geometry(tmp_path):
    geometry = np.zeros((4, 4, 4), dtype=np.uint8)
    np.save(tmp_path / "geometry.npy", geometry)
    dataset = object.__new__(AircraftDesignDataset)

    loaded = dataset._load_manifest_geometry(
        {"geometry_path": "geometry.npy"},
        tmp_path,
    )

    assert loaded.dtype == torch.uint8


def test_collate_keeps_geometry_uint8():
    batch = [
        {"geometry": torch.zeros((4, 4, 4), dtype=torch.uint8)},
        {"geometry": torch.ones((4, 4, 4), dtype=torch.uint8)},
    ]

    collated = aircraft_collate_fn(batch)

    assert collated["geometry"].dtype == torch.uint8
    assert collated["geometry"].shape == (2, 4, 4, 4)


def test_training_loader_uses_main_process_without_corpus_pinning():
    dataset = [
        {
            "geometry": torch.zeros((4, 4, 4), dtype=torch.uint8),
        }
    ]

    loader = build_train_loader(dataset, batch_size=1)

    assert loader.num_workers == 0
    assert loader.pin_memory is False
    assert loader.collate_fn is aircraft_collate_fn


class _TransferSpy:
    def __init__(self):
        self.calls = []

    def to(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self


def test_training_transfer_converts_geometry_once_and_is_non_blocking():
    latent = _TransferSpy()
    geometry = _TransferSpy()
    condition = _TransferSpy()
    batch = {
        "latent": latent,
        "geometry": geometry,
        "condition_vector": condition,
        "design_spec": ["metadata"],
    }
    device = torch.device("cuda")

    transferred = transfer_training_batch_to_device(batch, device, torch.float16)

    assert transferred["geometry"] is geometry
    assert geometry.calls == [
        (
            (),
            {
                "device": device,
                "dtype": torch.float16,
                "non_blocking": True,
            },
        )
    ]
    assert len(latent.calls) == 1
    assert len(condition.calls) == 1
    assert transferred["design_spec"] == ["metadata"]
