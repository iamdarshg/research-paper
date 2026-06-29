import json
import os
import sys

import numpy as np
import torch


sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI"))

from aircraft_diffusion_cfd import (  # noqa: E402
    AircraftDesignDataset,
    aircraft_collate_fn,
    transfer_training_batch_to_device,
)
from geometry_store import CompactGeometryStore  # noqa: E402


def test_store_deduplicates_content_and_keeps_canonical_uint8_tensor():
    geometry = torch.zeros((8, 8, 8))
    geometry[2:6, 3:5, 1:7] = 1
    noncontiguous = geometry.transpose(0, 1)
    store = CompactGeometryStore()

    first = store.add("a", noncontiguous, content_hash="same")
    second = store.add("b", geometry.clone(), content_hash="same")

    assert first == second
    assert store.unique_count == 1
    assert store.materialize(first).dtype == torch.uint8
    assert store.materialize(first).device.type == "cpu"
    assert store.materialize(first).is_contiguous()
    assert store.materialize(first).data_ptr() == store.materialize(second).data_ptr()


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
            "voxel_sha256": "shared-content",
        },
        {
            "source_id": "b",
            "geometry_path": "geometry.npy",
            "latent_path": "latent-b.npy",
            "voxel_sha256": "shared-content",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    dataset = AircraftDesignDataset(manifest_path=str(manifest_path), latent_dim=8)
    first = dataset[0]
    second = dataset[1]

    assert dataset.geometry_indices == [0, 0]
    assert dataset.geometry_store.unique_count == 1
    assert first["geometry"].data_ptr() == second["geometry"].data_ptr()
    assert first["geometry"].dtype == torch.uint8
    assert first["geometry"].shape == (4, 4, 4)
    assert first["latent"].shape == (8,)
    assert first["condition_vector"].ndim == 1


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
