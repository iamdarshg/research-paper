"""Compact, content-addressed storage for voxel geometries."""

import hashlib
from typing import Optional

import torch


class CompactGeometryStore:
    """Keep one canonical CPU uint8 tensor for each geometry content hash."""

    def __init__(self) -> None:
        self._geometries: list[torch.Tensor] = []
        self._hash_to_index: dict[str, int] = {}

    @staticmethod
    def content_hash(geometry: torch.Tensor) -> str:
        compact = (geometry.detach().cpu() > 0.5).to(torch.uint8).contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(compact.shape)).encode("ascii"))
        digest.update(compact.numpy().tobytes())
        return digest.hexdigest()

    def add(
        self,
        source_id: str,
        geometry: torch.Tensor,
        *,
        content_hash: Optional[str] = None,
    ) -> int:
        del source_id
        compact = (geometry.detach().cpu() > 0.5).to(torch.uint8).contiguous()
        stable_hash = content_hash or self.content_hash(compact)
        existing = self._hash_to_index.get(stable_hash)
        if existing is not None:
            return existing

        index = len(self._geometries)
        self._geometries.append(compact)
        self._hash_to_index[stable_hash] = index
        return index

    def materialize(self, index: int) -> torch.Tensor:
        return self._geometries[index]

    def get(self, index: int) -> torch.Tensor:
        return self.materialize(index)

    @property
    def unique_count(self) -> int:
        return len(self._geometries)
