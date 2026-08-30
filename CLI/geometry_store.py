"""Compact, content-addressed storage for voxel geometries."""

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch


_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _release_streamed_file_cache(handle) -> None:
    """Keep corpus integrity scans from pinning clean pages in a RAM cgroup."""
    fadvise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or advice is None:
        return
    try:
        fadvise(handle.fileno(), 0, 0, advice)
    except OSError:
        pass


def file_sha256(path: str | Path, *, force: bool = False) -> str:
    """Hash a file without retaining its payload, caching only unchanged stats."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _FILE_SHA256_CACHE.get(key)
    if cached is not None and not force:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        try:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        finally:
            _release_streamed_file_cache(handle)
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[key] = value
    return value


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
            canonical = self.materialize(existing)
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
        stable_hash = content_hash
        if len(content_hash) == 64 and all(
            character.lower() in "0123456789abcdef" for character in content_hash
        ):
            stable_hash = content_hash.lower()
            actual_hash = file_sha256(resolved)
            if actual_hash != stable_hash:
                raise ValueError(
                    f"Declared content hash {content_hash!r} does not match "
                    f"geometry file {resolved} ({actual_hash})"
                )

        existing = self._hash_to_index.get(stable_hash)
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
        self._hash_to_index[stable_hash] = index
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
