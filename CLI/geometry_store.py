"""Compact, content-addressed storage for voxel geometries."""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch


class CompactGeometryStore:
    """Keep one canonical CPU uint8 tensor for each geometry content hash."""

    def __init__(self) -> None:
        self._geometries: list[torch.Tensor | Path] = []
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
            canonical = self._geometries[existing]
            if canonical.shape != compact.shape or not torch.equal(canonical, compact):
                raise ValueError(
                    f"Content hash {stable_hash!r} does not match the canonical compact geometry"
                )
            return existing

        index = len(self._geometries)
        self._geometries.append(compact)
        self._hash_to_index[stable_hash] = index
        return index

    def add_file(
        self,
        source_id: str,
        path: str | Path,
        *,
        content_hash: str,
    ) -> int:
        """Register a NumPy geometry without retaining its voxel payload in RAM."""
        del source_id
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Geometry file does not exist: {resolved}")
        try:
            geometry = np.load(resolved, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Unable to load geometry file {resolved}: {exc}") from exc
        if not isinstance(geometry, np.ndarray) or geometry.ndim != 3:
            raise ValueError(f"Geometry file must contain a 3D array: {resolved}")

        existing = self._hash_to_index.get(content_hash)
        if existing is not None:
            canonical = self.materialize(existing)
            candidate = torch.from_numpy(np.array(geometry, copy=True))
            candidate = (candidate > 0.5).to(torch.uint8).contiguous()
            if canonical.shape != candidate.shape or not torch.equal(canonical, candidate):
                raise ValueError(
                    f"Content hash {content_hash!r} does not match the canonical compact geometry"
                )
            return existing

        index = len(self._geometries)
        self._geometries.append(resolved)
        self._hash_to_index[content_hash] = index
        return index

    def materialize(self, index: int) -> torch.Tensor:
        geometry = self._geometries[index]
        if isinstance(geometry, Path):
            loaded = np.load(geometry, mmap_mode="r", allow_pickle=False)
            tensor = torch.from_numpy(np.array(loaded, copy=True))
            return (tensor > 0.5).to(torch.uint8).contiguous()
        return geometry.clone()

    def get(self, index: int) -> torch.Tensor:
        return self.materialize(index)

    @property
    def unique_count(self) -> int:
        return len(self._geometries)
