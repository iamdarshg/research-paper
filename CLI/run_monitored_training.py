#!/usr/bin/env python3
"""Run a monitored training job with convergence and oscillation checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
import queue
import random
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from aircraft_diffusion_cfd import (
    AircraftDesignDataset,
    CFDConfig,
    DiffusionConfig,
    ModelConfig,
    OptimizedDiffusionTrainer,
    TrainingConfig,
    _load_checkpoint_metadata,
    aircraft_collate_fn,
    capture_rng_state,
    infer_conditioning_dim,
    restore_rng_state,
    resolve_grounded_grid_size,
    resolve_run_state_path,
)
from experiment_config import GLOBAL_CONFIG_PATH, config_value
from training_stability import (
    compute_core_loss,
    evaluate_directional_promotion_gate,
    summarize_stability,
)
from sdf_utils import prepare_edt_workspace


_JSONL_RECORD_COUNTS: Dict[str, int] = {}


def _count_jsonl_records(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def _append_jsonl(path: Path, record: Dict[str, Any]) -> Dict[str, Any]:
    """Append one JSONL record (append + flush only, no fsync or full-file scan).

    Returns the durable append offset and global_step used to build the
    run-state reconciliation metadata. The full-file sha256 prefix digest is NOT
    computed per append; the run-state save path computes it once over the
    recorded offset (see OptimizedDiffusionTrainer.build_run_state). The record
    count is kept as an in-memory running counter (initialized once per path to
    cover resume) instead of re-counting the whole file on every append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = str(path)
    if key not in _JSONL_RECORD_COUNTS:
        _JSONL_RECORD_COUNTS[key] = _count_jsonl_records(path)
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    encoded = (line + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        end = handle.tell()
    _JSONL_RECORD_COUNTS[key] += 1
    return {
        "offset": int(end),
        "global_step": int(record.get("global_step", 0)),
        "record_count": int(_JSONL_RECORD_COUNTS[key]),
    }


class _AsyncRecordWriter:
    """Single-owner background writer for JSONL records and tensorboard scalars.

    All file and SummaryWriter I/O runs on one daemon thread fed by a bounded
    FIFO queue. The producer assigns monotonic integer sequence numbers; work
    items are ``('jsonl', seq, path, record)`` and ``('tb', seq, step, tags)``.
    The worker performs the EXACT byte sequence the synchronous ``_append_jsonl``
    used today -- same ``json.dumps(record, sort_keys=True, allow_nan=False)``
    + newline encode, same ``open("ab")``/seek/write/flush/tell, same
    ``_JSONL_RECORD_COUNTS`` increments -- one record at a time, so file bytes,
    offsets, and counter state are byte-identical to the synchronous writer.

    ``flush_barrier(seq)`` blocks until every item with ``seq' <= seq`` is
    durably finished and returns the metadata of the latest completed record,
    which is what run-state saves use for the updates-log reconciliation prefix.
    """

    def __init__(self, summary_writer: Any, *, maxsize: int = 64) -> None:
        self._summary_writer = summary_writer
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._notify = threading.Condition(self._lock)
        self._next_seq = 0
        self._last_jsonl_seq: Optional[int] = None
        # The first JSONL record gets sequence 0; start the completed watermark
        # at -1 so flush_barrier(0) actually blocks until record 0 is durable
        # instead of returning before it (0 < 0 is false with a 0 watermark).
        self._completed_seq = -1
        self._last_meta: Dict[str, Any] = {}
        self._closed = False
        # First worker failure is stashed here and re-raised to the producer at
        # the next flush_barrier/enqueue so a dead writer can never hang the
        # training loop (see _run/_flush_barrier_fail_fast).
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            name="async-records-writer",
            daemon=True,
        )
        self._thread.start()

    def enqueue_jsonl(self, path: Path, record: Dict[str, Any]) -> int:
        """Enqueue one JSONL append; returns its monotonic sequence number."""
        self._raise_stashed_error()
        seq = self._next_sequence()
        self._queue.put(("jsonl", seq, path, record))
        with self._lock:
            self._last_jsonl_seq = seq
        return seq

    def enqueue_tb_batch(self, step: int, tags: Dict[str, float]) -> int:
        """Enqueue a batch of scalar tags to be written at ``step``."""
        self._raise_stashed_error()
        seq = self._next_sequence()
        self._queue.put(("tb", seq, int(step), dict(tags)))
        return seq

    def flush_barrier(self, seq: Optional[int] = None) -> Dict[str, Any]:
        """Block until all items with ``seq' <= seq`` are durable.

        Returns the metadata of the latest completed JSONL record (``{}`` when
        no record has been written yet). When ``seq`` is ``None`` the barrier
        targets the most recently enqueued JSONL sequence.

        If the worker thread dies (any I/O or serialization error), the stashed
        error is re-raised here instead of hanging the caller forever.
        """
        with self._lock:
            target = self._last_jsonl_seq if seq is None else int(seq)
            if target is None:
                self._raise_stashed_error()
                return dict(self._last_meta)
            while self._completed_seq < target and not self._closed:
                if self._error is not None or not self._thread.is_alive():
                    break
                self._notify.wait(timeout=30.0)
            self._raise_stashed_error()
            return dict(self._last_meta)

    def close(self, timeout: float = 30.0) -> None:
        """Drain pending work (after a final barrier) and stop the worker.

        The SummaryWriter is flushed and closed only after the queue drains, so
        no trailing record is lost on normal run end or handled exceptions. A
        stashed worker failure is re-raised AFTER cleanup so it is never masked
        by this finally-block close.
        """
        with self._lock:
            target = self._last_jsonl_seq
        try:
            if target is not None:
                self.flush_barrier(target)
        finally:
            with self._lock:
                self._closed = True
                self._notify.notify_all()
            try:
                # Sentinel wakes the worker to exit; queued items behind it are
                # still processed first (FIFO), so trailing telemetry is drained.
                # Bounded so a dead worker with a full queue cannot hang close().
                self._queue.put(None, timeout=1.0)
            except Exception:
                pass
            self._thread.join(timeout=timeout)
            writer = self._summary_writer
            if writer is not None:
                try:
                    writer.flush()
                except Exception:
                    pass
                try:
                    writer.close()
                except Exception:
                    pass

    def _next_sequence(self) -> int:
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            return seq

    def _raise_stashed_error(self) -> None:
        """Re-raise the worker's first failure to the producer, if any."""
        error = self._error
        if error is not None:
            raise error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    break
                kind = item[0]
                if kind == "jsonl":
                    _, seq, path, record = item
                    meta = _append_jsonl(path, record)
                    with self._lock:
                        self._completed_seq = max(self._completed_seq, seq)
                        self._last_meta = dict(meta)
                        self._notify.notify_all()
                elif kind == "tb":
                    _, seq, step, tags = item
                    writer = self._summary_writer
                    if writer is not None:
                        for tag, value in tags.items():
                            writer.add_scalar(str(tag), float(value), step)
                    with self._lock:
                        self._completed_seq = max(self._completed_seq, seq)
                        self._notify.notify_all()
            except Exception as exc:
                # Stash the first worker failure so the producer re-raises it at
                # the next flush_barrier/enqueue instead of hanging on a dead
                # thread; the loop then terminates so close() joins promptly.
                with self._lock:
                    if self._error is None:
                        self._error = exc
                    self._notify.notify_all()
                break
            finally:
                self._queue.task_done()


def _updates_log_reconciliation_metadata(
    path: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Stamp the durable-prefix sha256 onto updates-log metadata.

    Mirrors the digest computation the run-state save path performs over the
    recorded offset (OptimizedDiffusionTrainer.build_run_state) so callers that
    hold only the per-append metadata can produce a reconcile-ready checkpoint.
    """
    stamped = dict(metadata)
    offset = int(metadata.get("offset", -1))
    if offset < 0:
        return stamped
    with path.open("rb") as handle:
        prefix = handle.read(offset)
    stamped["sha256"] = hashlib.sha256(prefix).hexdigest()
    return stamped


def _prepare_geometry_threshold_for_run(
    trainer: OptimizedDiffusionTrainer,
    calibration_loader: Any,
    *,
    resume_run_state: Optional[Path],
) -> Dict[str, Any]:
    """Restore an exact-run threshold before compatibility is constructed."""
    if resume_run_state is None:
        calibration = trainer.calibrate_geometry_materialization_threshold(
            calibration_loader
        )
    else:
        resolved_path = resolve_run_state_path(resume_run_state)
        # I1: route the user-supplied run-state through the shared trust-gated
        # loader (weights_only=True first; weights_only=False fallback ONLY for
        # a trusted local artifact under build/). Untrusted paths re-raise.
        state = _load_checkpoint_metadata(resolved_path)
        if "geometry_probability_threshold" not in state:
            raise ValueError("Exact resume state is missing its saved geometry threshold")
        trainer._set_geometry_probability_threshold(
            state["geometry_probability_threshold"],
            calibrated=bool(state.get("geometry_threshold_calibrated", True)),
            calibration=state.get("geometry_threshold_calibration"),
        )
        calibration = dict(state.get("geometry_threshold_calibration", {}))

    if not trainer.training_config.calibrate_geometry_materialization_threshold:
        # Fixed-threshold mode: the config value is authoritative and overrides
        # any checkpoint/saved threshold (e.g. the failed run's calibrated
        # 0.9752, which sits in the distribution tail and is the root fragility).
        fixed_threshold = float(
            trainer.training_config.geometry_materialization_threshold
        )
        trainer._set_geometry_probability_threshold(
            fixed_threshold,
            calibrated=True,
            calibration={
                **calibration,
                "source": "config_fixed",
                "frozen_for_run": True,
                "threshold": fixed_threshold,
            },
        )
        calibration = dict(trainer.geometry_threshold_calibration)
    return calibration


def _dataclass_fingerprint(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


_EXPERIMENT_FLAG_NAMES = (
    "graph_decode_mlp",
    "batch_guard_dot_reads",
    "deferred_solver_reads",
    "tf32_gemm_math",
)


def _experiment_flags_fingerprint() -> Dict[str, bool]:
    """Resolve every live experiment flag from the loaded YAML.

    R4 (PR 41 review, item 4): these flags were not recorded in the exact-resume
    fingerprint, so a resume could silently flip numerics (``tf32_gemm_math``)
    or execution behavior (``graph_decode_mlp`` / ``batch_guard_dot_reads`` /
    ``deferred_solver_reads``) without any incompatibility being raised. Reads go
    through the same ``config_value`` accessors the trainer uses, so this is the
    effective configuration, not the YAML defaults.
    """
    return {
        name: bool(config_value("experiment", name, False))
        for name in _EXPERIMENT_FLAG_NAMES
    }


def _build_objective_configuration_fingerprint(
    *,
    args: Any,
    training_config: TrainingConfig,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    cfd_config: CFDConfig,
    geometry_probability_threshold: float,
    sample_order: List[int],
    promotion_sample_order: List[int],
) -> Dict[str, Any]:
    """Describe every live model, objective, and optimizer behavior."""
    return {
        "num_epochs": int(args.num_epochs),
        "planned_optimizer_updates": int(args.planned_optimizer_updates),
        "batch_size": int(args.batch_size),
        "subset_seed": int(args.subset_seed),
        "sample_order": list(sample_order),
        "promotion_split": str(args.promotion_split),
        "promotion_sample_order": list(promotion_sample_order),
        "promotion_evaluation_samples": int(args.promotion_evaluation_samples),
        "promotion_generation_seeds": int(args.promotion_generation_seeds),
        "solver": str(args.solver),
        "lbm_stream_bfl_backend": str(args.lbm_stream_bfl_backend),
        "geometry_materialization_threshold": float(geometry_probability_threshold),
        "training_config": _dataclass_fingerprint(training_config),
        "model_config": _dataclass_fingerprint(model_config),
        "diffusion_config": _dataclass_fingerprint(diffusion_config),
        "cfd_config": _dataclass_fingerprint(cfd_config),
        # R4 (PR 41 review, item 4): numerics (tf32_gemm_math) and execution
        # (graph_decode_mlp / batch_guard_dot_reads / deferred_solver_reads)
        # flags. A resume with these flipped silently changes arithmetic or
        # kernel behavior; the fingerprint now records them.
        "experiment_flags": _experiment_flags_fingerprint(),
    }


def _reset_epoch_checkpoint_segment(
    resume_state_info: Dict[str, Any],
    *,
    next_epoch: int,
) -> None:
    """Start checkpoint cadence from zero after a completed epoch."""
    resume_state_info["epoch_index"] = int(next_epoch)
    resume_state_info["completed_in_epoch"] = 0


def _reconcile_updates_log(path: Path, checkpoint_log: Dict[str, Any]) -> Dict[str, Any]:
    """Reconcile append-only updates to the last durable run-state boundary."""
    expected_offset = int(checkpoint_log.get("offset", -1))
    expected_digest = str(checkpoint_log.get("sha256", ""))
    if expected_offset < 0 or not expected_digest:
        raise ValueError("run-state is missing durable updates-log metadata")
    if not path.exists():
        if expected_offset > 0:
            raise ValueError("checkpoint is ahead of missing updates log")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    payload = path.read_bytes()
    if len(payload) < expected_offset:
        raise ValueError(
            "checkpoint is ahead of durable updates log: "
            f"{expected_offset} > {len(payload)} bytes"
        )
    prefix = payload[:expected_offset]
    if hashlib.sha256(prefix).hexdigest() != expected_digest:
        raise ValueError("updates log disagrees with durable run-state prefix")
    if prefix and not prefix.endswith(b"\n"):
        raise ValueError("durable updates-log boundary is not a complete JSONL record")
    trailing_records = 0
    for line in payload[expected_offset:].splitlines():
        if line.strip():
            json.loads(line)
            trailing_records += 1
    if len(payload) > expected_offset:
        with path.open("r+b") as handle:
            handle.truncate(expected_offset)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "truncated_records": trailing_records,
        "offset": expected_offset,
        "sha256": expected_digest,
        "global_step": int(checkpoint_log.get("global_step", 0)),
    }


def _iter_loader_without_rng_advance(loader: DataLoader):
    """Create a deterministic loader iterator without consuming training RNG."""
    state = capture_rng_state()
    iterator = iter(loader)
    restore_rng_state(state)
    return iterator


def _manifest_identity(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_sample_order(dataset: Dataset) -> list[int]:
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return list(range(len(dataset)))
    return [int(index) for index in indices]


def _run_state_checkpoint_due(
    completed_in_epoch: int,
    segment_start_batch: int,
    checkpoint_every_updates: int,
) -> bool:
    """Use a bounded cadence relative to the current invocation's start."""
    cadence = int(checkpoint_every_updates)
    completed_since_start = int(completed_in_epoch) - int(segment_start_batch)
    return (
        cadence > 0
        and completed_since_start > 0
        and completed_since_start % cadence == 0
    )


def _resume_epoch_position(
    epoch_index: int,
    completed_in_epoch: int,
    updates_in_epoch: int,
) -> tuple[int, int]:
    if int(completed_in_epoch) >= int(updates_in_epoch):
        return int(epoch_index) + 1, 0
    return int(epoch_index), int(completed_in_epoch)


def restore_promotion_baseline(
    resume_state_info: Dict[str, Any],
    *,
    promotion_split: str,
    promotion_sample_order: list[int],
    evaluation_samples: int,
    generation_seeds: int,
) -> Dict[str, Any]:
    metadata = dict(resume_state_info.get("run_state_metadata", {}))
    baseline = dict(metadata.get("promotion_baseline", {}))
    if not baseline:
        raise ValueError(
            "Exact resume requires the original promotion baseline in run-state"
        )
    identity = dict(metadata.get("promotion_baseline_identity", {}))
    expected = {
        "split": str(promotion_split),
        "sample_order": list(promotion_sample_order),
        "evaluation_samples": int(evaluation_samples),
        "generation_seeds": int(generation_seeds),
    }
    mismatches = [
        key for key, value in expected.items() if identity.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Exact resume promotion baseline identity mismatch: "
            + ", ".join(mismatches)
        )
    return baseline


class RunLocalCosineScheduler:
    """Cosine decay over successful optimizer updates with a nonzero floor."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        total_updates: int,
        min_lr_ratio: float,
    ) -> None:
        if int(total_updates) <= 0:
            raise ValueError("total_updates must be greater than 0")
        if not 0.0 < float(min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        self.optimizer = optimizer
        self.total_updates = int(total_updates)
        self.min_lr_ratio = float(min_lr_ratio)
        self.completed_updates = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self._apply_learning_rates()

    def _factor(self) -> float:
        progress = min(
            1.0,
            max(0.0, self.completed_updates / max(self.total_updates, 1)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply_learning_rates(self) -> None:
        factor = self._factor()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr) * factor

    def step(self) -> None:
        self.completed_updates = min(
            self.total_updates,
            self.completed_updates + 1,
        )
        self._apply_learning_rates()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_type": "run_local_cosine_updates_v1",
            "total_updates": self.total_updates,
            "min_lr_ratio": self.min_lr_ratio,
            "completed_updates": self.completed_updates,
            "base_lrs": list(self.base_lrs),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict.get("scheduler_type") != "run_local_cosine_updates_v1":
            raise ValueError("Incompatible scheduler state")
        if int(state_dict["total_updates"]) != self.total_updates:
            raise ValueError("Scheduler update horizon does not match this run")
        if float(state_dict["min_lr_ratio"]) != self.min_lr_ratio:
            raise ValueError("Scheduler minimum LR ratio does not match this run")
        base_lrs = [float(value) for value in state_dict["base_lrs"]]
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("Scheduler optimizer group count does not match")
        self.base_lrs = base_lrs
        self.completed_updates = min(
            self.total_updates,
            max(0, int(state_dict["completed_updates"])),
        )
        self._apply_learning_rates()


class ResumableEpochSampler(Sampler[int]):
    """Deterministic per-epoch shuffle that survives an exact resume.

    R7 (PR 41 review, item 7): the monitored train loader previously used
    ``shuffle=False``, so every epoch visited the same sample order. This
    sampler draws a fresh permutation per epoch seeded by ``(subset_seed,
    epoch)``, so the order is reproducible from the seed alone. A resumed
    process calls ``set_epoch`` with the run-state's epoch_index, regenerates
    the identical permutation, and ``train_epoch``'s ``start_batch`` skip
    continues at the exact ``completed_in_epoch`` offset.

    The subset COMPOSITION (which samples are in the epoch) is unchanged and is
    still fingerprinted / resume-validated via ``_dataset_sample_order``; only
    the per-epoch iteration order becomes a fresh deterministic shuffle.
    """

    def __init__(self, size: int, *, subset_seed: int) -> None:
        self._size = int(size)
        self._subset_seed = int(subset_seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = list(range(self._size))
        # random.seed(str) uses SHA-512 of the key, so the permutation is stable
        # across interpreter runs (independent of PYTHONHASHSEED).
        rng = random.Random(f"{self._subset_seed}:{self._epoch}")
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self._size


def _build_epoch_dataset(
    dataset: Dataset,
    *,
    max_samples_per_epoch: int,
    subset_seed: int,
    split: str = "train",
) -> Dataset:
    metadata = getattr(dataset, "metadata", {}) or {}
    assignments = metadata.get("split_assignments")
    if isinstance(assignments, list) and len(assignments) == len(dataset):
        indices = [
            index
            for index, assignment in enumerate(assignments)
            if str(assignment) == str(split)
        ]
        if not indices:
            raise ValueError(f"Dataset has no records in requested split {split!r}")
    else:
        indices = list(range(len(dataset)))

    if max_samples_per_epoch <= 0 or max_samples_per_epoch >= len(indices):
        if len(indices) == len(dataset):
            return dataset
        return Subset(dataset, indices)

    rng = random.Random(subset_seed)
    rng.shuffle(indices)
    return Subset(dataset, indices[:max_samples_per_epoch])


def _build_split_dataset(dataset: Dataset, split: str) -> Dataset:
    return _build_epoch_dataset(
        dataset,
        max_samples_per_epoch=0,
        subset_seed=0,
        split=split,
    )


def _geometry_promotion_metrics(
    promotion: Dict[str, Any],
) -> tuple[Dict[str, float], tuple[float, ...]]:
    reconstruction_recall = float(
        promotion.get(
            "reconstruction_recall",
            promotion.get("reconstruction_topk_recall", 0.0),
        )
    )
    generated_recall = float(
        promotion.get(
            "generated_recall",
            promotion.get("generated_topk_recall", 0.0),
        )
    )
    generated_worst_recall = float(
        promotion.get(
            "generated_worst_recall",
            promotion.get("generated_worst_topk_recall", 0.0),
        )
    )
    generated_occupancy_error = abs(
        float(promotion.get("generated_mean_occupied_fraction", 0.0))
        - float(promotion.get("target_mean_occupied_fraction", 0.0))
    )
    metrics = {
        "promotion_reconstruction_recall": reconstruction_recall,
        "promotion_generated_recall": generated_recall,
        "promotion_generated_worst_recall": generated_worst_recall,
        # Compatibility aliases for existing monitors and history reports.
        "promotion_reconstruction_topk_recall": reconstruction_recall,
        "promotion_generated_topk_recall": generated_recall,
        "promotion_generated_worst_topk_recall": generated_worst_recall,
        "promotion_generated_aircraft_valid_fraction": float(
            promotion.get("generated_aircraft_valid_fraction", 0.0)
        ),
        "promotion_generated_unique_fraction": float(
            promotion.get("generated_unique_fraction", 0.0)
        ),
        "promotion_generated_mean_largest_component_fraction": float(
            promotion.get("generated_mean_largest_component_fraction", 0.0)
        ),
        "promotion_generated_mean_normalization_boundary_fraction": float(
            promotion.get(
                "generated_mean_normalization_boundary_fraction",
                1.0,
            )
        ),
        "promotion_generated_occupancy_error": generated_occupancy_error,
        "promotion_gate_passed": float(promotion.get("status") == "pass"),
    }
    rank = (
        metrics["promotion_generated_aircraft_valid_fraction"],
        -metrics["promotion_generated_occupancy_error"],
        metrics["promotion_generated_unique_fraction"],
        metrics["promotion_generated_mean_largest_component_fraction"],
        -metrics["promotion_generated_mean_normalization_boundary_fraction"],
        metrics["promotion_generated_worst_recall"],
        metrics["promotion_generated_recall"],
        metrics["promotion_reconstruction_recall"],
    )
    metrics["geometry_selection_metric"] = (
        metrics["promotion_generated_occupancy_error"]
        + 1.0
        - metrics["promotion_generated_recall"]
    )
    return metrics, rank


def _restore_best_promotion_rank(
    metadata: Mapping[str, Any],
) -> tuple[float, ...]:
    """R5 (PR 41 review, item 5): restore the persisted best promotion rank.

    Falls back to the fail-safe ``(-1.0,) * 8`` when the run-state predates
    this field or holds a non-numeric value, so the lexicographic rank gate
    never errors on a resume.
    """
    value = metadata.get("best_promotion_rank")
    if value is None:
        return (-1.0,) * 8
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return (-1.0,) * 8


def _sync_best_checkpoint_state(
    trainer: OptimizedDiffusionTrainer,
    *,
    best_promotion_rank: tuple[float, ...],
    best_geometry_metric: float,
    best_checkpoint_path: str | None,
) -> None:
    """R5 (PR 41 review, item 5): mirror the best-checkpoint selection into the
    trainer's run_state_metadata so the next run-state save persists it for an
    exact resume (the metadata dict round-trips verbatim through build_run_state
    / load_run_state).
    """
    trainer.run_state_metadata.update(
        {
            "best_promotion_rank": list(best_promotion_rank),
            "best_geometry_metric": float(best_geometry_metric),
            "best_checkpoint_path": best_checkpoint_path,
        }
    )


def _geometry_non_regression(
    candidate: Dict[str, Any],
    baseline: Dict[str, Any],
) -> Dict[str, Any]:
    tolerances = {
        "generated_aircraft_valid_fraction": float(
            config_value("training", "promotion_valid_fraction_tolerance", 0.0)
        ),
        "generated_unique_fraction": float(
            config_value("training", "promotion_unique_fraction_tolerance", 0.05)
        ),
        "generated_mean_largest_component_fraction": float(
            config_value("training", "promotion_component_fraction_tolerance", 0.02)
        ),
        "generated_mean_normalization_boundary_fraction": float(
            config_value("training", "promotion_boundary_fraction_tolerance", 0.01)
        ),
        "generated_worst_recall": float(
            config_value("training", "promotion_worst_recall_tolerance", 0.01)
        ),
        "generated_occupancy_error": float(
            config_value(
                "training",
                "promotion_occupancy_error_tolerance",
                0.005,
            )
        ),
    }
    checks = {
        "generated_aircraft_valid_fraction": (
            float(candidate.get("generated_aircraft_valid_fraction", 0.0))
            >= float(baseline.get("generated_aircraft_valid_fraction", 0.0))
            - tolerances["generated_aircraft_valid_fraction"]
        ),
        "generated_unique_fraction": (
            float(candidate.get("generated_unique_fraction", 0.0))
            >= float(baseline.get("generated_unique_fraction", 0.0))
            - tolerances["generated_unique_fraction"]
        ),
        "generated_mean_largest_component_fraction": (
            float(candidate.get("generated_mean_largest_component_fraction", 0.0))
            >= float(
                baseline.get("generated_mean_largest_component_fraction", 0.0)
            )
            - tolerances["generated_mean_largest_component_fraction"]
        ),
        "generated_mean_normalization_boundary_fraction": (
            float(
                candidate.get(
                    "generated_mean_normalization_boundary_fraction",
                    1.0,
                )
            )
            <= float(
                baseline.get(
                    "generated_mean_normalization_boundary_fraction",
                    1.0,
                )
            )
            + tolerances["generated_mean_normalization_boundary_fraction"]
        ),
        "generated_worst_recall": (
            float(
                candidate.get(
                    "generated_worst_recall",
                    candidate.get("generated_worst_topk_recall", 0.0),
                )
            )
            >= float(
                baseline.get(
                    "generated_worst_recall",
                    baseline.get("generated_worst_topk_recall", 0.0),
                )
            )
            - tolerances["generated_worst_recall"]
        ),
        "generated_occupancy_error": (
            abs(
                float(
                    candidate.get("generated_mean_occupied_fraction", 0.0)
                )
                - float(candidate.get("target_mean_occupied_fraction", 0.0))
            )
            <= abs(
                float(
                    baseline.get("generated_mean_occupied_fraction", 0.0)
                )
                - float(baseline.get("target_mean_occupied_fraction", 0.0))
            )
            + tolerances["generated_occupancy_error"]
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "status": "pass" if not failed_checks else "fail",
        "checks": checks,
        "failed_checks": failed_checks,
        "tolerances": tolerances,
    }


def _build_history_payload(
    *,
    args: argparse.Namespace,
    device: torch.device,
    history: List[Dict[str, Any]],
    stability: Dict[str, Any],
    checkpoint_path: str | None,
    model_config: ModelConfig,
    best_checkpoint_path: str | None = None,
    best_geometry_metric: float | None = None,
    initial_geometry_promotion: Dict[str, Any] | None = None,
    initial_geometry_promotion_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "config": {
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "resume_from": str(Path(args.resume_from).resolve()) if args.resume_from else None,
            "resume_run_state": (
                str(Path(args.resume_run_state).resolve())
                if args.resume_run_state
                else None
            ),
            "warm_start_from": (
                str(Path(args.warm_start_from).resolve())
                if args.warm_start_from
                else None
            ),
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "global_config": str(GLOBAL_CONFIG_PATH),
            "latent_dim": model_config.latent_dim,
            "grid_size_requested": args.grid_size,
            "grid_size_resolved": args.resolved_grid_size,
            "learning_rate": args.learning_rate,
            "lr_schedule": "run_local_cosine_updates_v1",
            "lr_min_ratio": args.lr_min_ratio,
            "planned_optimizer_updates": args.planned_optimizer_updates,
            "updates_output": str(Path(args.updates_output).resolve()),
            "converter_learning_rate": config_value("training", "converter_learning_rate", 2e-5),
            "consistency_student_learning_rate": config_value(
                "training", "consistency_student_learning_rate", 2e-5
            ),
            "solver": args.solver,
            "lbm_stream_bfl_backend": args.lbm_stream_bfl_backend,
            "cpu_threads": args.cpu_threads,
            "max_samples_per_epoch": args.max_samples_per_epoch,
            "subset_seed": args.subset_seed,
            "training_split": args.training_split,
            "promotion_split": args.promotion_split,
            "training_sample_count": args.training_sample_count,
            "promotion_sample_count": args.promotion_sample_count,
            "promotion_evaluation_samples": args.promotion_evaluation_samples,
            "promotion_generation_seeds": args.promotion_generation_seeds,
            "stop_on_promotion_pass": args.stop_on_promotion_pass,
            "stability_metric": args.stability_metric,
            "convergence_window": args.convergence_window,
            "convergence_target": args.convergence_target,
            "convergence_cv_threshold": args.convergence_cv_threshold,
            "convergence_drift_threshold": args.convergence_drift_threshold,
            "required_geometry_loss_max": args.required_geometry_loss_max,
            "oscillation_cv_threshold": args.oscillation_cv_threshold,
            "early_stop_on_convergence": args.early_stop_on_convergence,
            "save_every": args.save_every,
            "checkpoint_every_updates": args.checkpoint_every_updates,
            "stop_after_updates": args.stop_after_updates,
            "save_final_checkpoint": args.save_final_checkpoint,
            "direct_solver_loss_weight": args.direct_solver_loss_weight,
            "direct_solver_steps": args.direct_solver_steps,
            "direct_solver_directions": args.direct_solver_directions,
            "direct_connectivity_weight": args.direct_connectivity_weight,
            "direct_aircraft_validity_weight": args.direct_aircraft_validity_weight,
            "direct_solver_perturbation": args.direct_solver_perturbation,
            "direct_solver_perturbation_grid_size": args.direct_solver_perturbation_grid_size,
            "direct_solver_gradient_clip": config_value(
                "training", "direct_solver_gradient_clip", 1.0
            ),
            "geometry_probability_threshold": getattr(
                args,
                "geometry_probability_threshold",
                None,
            ),
            "geometry_threshold_calibration": getattr(
                args,
                "geometry_threshold_calibration",
                None,
            ),
        },
        "device": str(device),
        "checkpoint_path": (
            str(Path(checkpoint_path).resolve()) if checkpoint_path else None
        ),
        "best_checkpoint_path": (
            str(Path(best_checkpoint_path).resolve()) if best_checkpoint_path else None
        ),
        "best_geometry_metric": best_geometry_metric,
        "initial_geometry_promotion": initial_geometry_promotion,
        "initial_geometry_promotion_report": initial_geometry_promotion_report,
        "history": history,
        "stability": stability,
    }


def _load_monitored_history(path: Path) -> List[Dict[str, Any]]:
    """R6 (PR 41 review, item 6): read the persisted monitored-history payload
    and return its epoch records.

    The monitored loop rewrites the history file every epoch with the full
    payload, so the file is always the latest complete history. Seeding the
    in-memory list from it on resume keeps the stability/early-stop window and
    the history JSONL continuity intact instead of restarting cold (which would
    both delay convergence detection and drop pre-resume rows).
    """
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    records = payload.get("history")
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GPU-monitored training sweep with stability checks.")
    parser.add_argument("--manifest", required=True, help="Grounded manifest used for training.")
    parser.add_argument("--num-epochs", type=int, default=int(config_value("training", "num_epochs", 200)))
    parser.add_argument("--batch-size", type=int, default=int(config_value("training", "batch_size", 1)))
    parser.add_argument("--latent-dim", type=int, default=int(config_value("model", "latent_dim", 192)))
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=float(config_value("training", "learning_rate", 2e-5)))
    parser.add_argument(
        "--lr-min-ratio",
        type=float,
        default=float(config_value("training", "lr_min_ratio", 0.10)),
        help="Nonzero fraction of each optimizer-group base LR at the end of this run.",
    )
    parser.add_argument("--solver", default=str(config_value("cfd", "solver", "D3Q27")))
    parser.add_argument(
        "--lbm-stream-bfl-backend",
        choices=("pytorch_reference", "fused_stream_bfl"),
        default="pytorch_reference",
        help=(
            "Streaming/BFL backend for the direct D3Q27 solver. fused_stream_bfl "
            "requires the parity gates in tests/test_d3q27_kernel_parity.py and "
            "tests/test_direct_solver_fused_parity.py to pass on this machine."
        ),
    )
    parser.add_argument("--save-dir", default="./checkpoints_monitored")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument(
        "--resume-run-state",
        default=None,
        help="Resume an interrupted run exactly at its next unprocessed update.",
    )
    parser.add_argument("--warm-start-from", default=None)
    parser.add_argument("--history-output", default="./build/monitored_training/history.json")
    parser.add_argument(
        "--updates-output",
        default=None,
        help="Append-only per-optimizer-update JSONL; defaults beside history.json.",
    )
    parser.add_argument("--save-every", type=int, default=int(config_value("training", "save_interval", 25)))
    parser.add_argument(
        "--checkpoint-every-updates",
        type=int,
        default=0,
        help="Atomically save latest_run_state.pt every N optimizer updates; 0 disables it.",
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        default=0,
        help="Bounded interruption hook for smoke tests; 0 runs the configured horizon.",
    )
    parser.add_argument(
        "--save-final-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the final optimizer state even when it does not pass promotion.",
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-samples-per-epoch", type=int, default=0)
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--training-split", default="train")
    parser.add_argument("--promotion-split", default="val")
    parser.add_argument(
        "--promotion-evaluation-samples",
        type=int,
        default=int(config_value("training", "overfit_geometry_gate_samples", 16)),
    )
    parser.add_argument(
        "--promotion-generation-seeds",
        type=int,
        default=int(config_value("training", "promotion_generation_seeds", 6)),
    )
    parser.add_argument(
        "--stop-on-promotion-pass",
        action=argparse.BooleanOptionalAction,
        default=bool(config_value("training", "stop_on_promotion_pass", True)),
    )
    parser.add_argument("--stability-metric", default="optimization_loss")
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--convergence-target", type=float, default=20.0)
    parser.add_argument("--convergence-cv-threshold", type=float, default=0.08)
    parser.add_argument("--convergence-drift-threshold", type=float, default=0.35)
    parser.add_argument(
        "--required-geometry-loss-max",
        type=float,
        default=float(config_value("training", "required_geometry_loss_max", 0.20)),
    )
    parser.add_argument("--oscillation-cv-threshold", type=float, default=0.30)
    parser.add_argument("--early-stop-on-convergence", action="store_true")
    parser.add_argument("--direct-solver-loss-weight", type=float, default=float(config_value("training", "direct_solver_loss_weight", 1.0)))
    parser.add_argument("--direct-solver-steps", type=int, default=int(config_value("training", "direct_solver_steps", 5)))
    parser.add_argument("--direct-solver-directions", type=int, default=int(config_value("training", "direct_solver_directions", 16)))
    parser.add_argument("--direct-connectivity-weight", type=float, default=float(config_value("training", "direct_connectivity_weight", 1.0)))
    parser.add_argument("--direct-aircraft-validity-weight", type=float, default=float(config_value("training", "direct_aircraft_validity_weight", 1.0)))
    parser.add_argument("--direct-solver-perturbation", type=float, default=float(config_value("training", "direct_solver_perturbation", 0.15)))
    parser.add_argument("--direct-solver-perturbation-grid-size", type=int, default=int(config_value("training", "direct_solver_perturbation_grid_size", 12)))
    args = parser.parse_args()
    if sum(bool(value) for value in (args.resume_from, args.resume_run_state, args.warm_start_from)) > 1:
        parser.error("--resume-run-state, --resume-from, and --warm-start-from are mutually exclusive")
    if not 0.0 < float(args.lr_min_ratio) <= 1.0:
        parser.error("--lr-min-ratio must be in (0, 1]")
    if args.promotion_evaluation_samples <= 0:
        parser.error("--promotion-evaluation-samples must be greater than 0")
    if args.promotion_generation_seeds <= 0:
        parser.error("--promotion-generation-seeds must be greater than 0")
    if args.checkpoint_every_updates < 0:
        parser.error("--checkpoint-every-updates must be nonnegative")
    if args.stop_after_updates < 0:
        parser.error("--stop-after-updates must be nonnegative")

    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, args.cpu_threads)))
    except RuntimeError:
        pass

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=args.grid_size,
        latent_dim=args.latent_dim,
        manifest_path=args.manifest,
    )
    resolved_grid_size = resolve_grounded_grid_size(
        args.grid_size,
        detected_grid_size=dataset.grid_size,
        solver=args.solver,
        source_label=args.manifest,
    )
    args.resolved_grid_size = resolved_grid_size
    prepare_edt_workspace((resolved_grid_size,) * 3)

    observed_unique_geometry_count = int(
        dataset.metadata.get("unique_geometry_count", len(dataset))
    )
    capacity_geometry_count = max(
        observed_unique_geometry_count,
        int(config_value("scaling", "capacity_basis_unique_geometries", observed_unique_geometry_count)),
    )
    model_config = ModelConfig.scaled_for_corpus(
        capacity_geometry_count,
        resolved_grid_size,
        conditioning_dim=infer_conditioning_dim(),
        latent_dim=args.latent_dim,
    )
    if args.resume_from:
        checkpoint_metadata = _load_checkpoint_metadata(
            args.resume_from,
            authorized_paths=(Path(args.resume_from).resolve(),),
        )
        checkpoint_model_config = ModelConfig(**checkpoint_metadata["model_config"])
        if int(checkpoint_model_config.grid_resolution) != int(resolved_grid_size):
            raise ValueError(
                "Resume checkpoint grid resolution does not match the grounded dataset: "
                f"{checkpoint_model_config.grid_resolution} != {resolved_grid_size}"
            )
        if int(checkpoint_model_config.latent_dim) != int(args.latent_dim):
            raise ValueError(
                "Resume checkpoint latent width does not match --latent-dim: "
                f"{checkpoint_model_config.latent_dim} != {args.latent_dim}"
            )
        model_config = checkpoint_model_config
    if int(dataset.latent_dim) != int(model_config.latent_dim):
        dataset = AircraftDesignDataset(
            num_samples=0,
            grid_size=resolved_grid_size,
            latent_dim=model_config.latent_dim,
            manifest_path=args.manifest,
        )
    epoch_dataset = _build_epoch_dataset(
        dataset,
        max_samples_per_epoch=args.max_samples_per_epoch,
        subset_seed=args.subset_seed,
        split=args.training_split,
    )
    promotion_dataset = _build_split_dataset(dataset, args.promotion_split)
    args.training_sample_count = len(epoch_dataset)
    args.promotion_sample_count = len(promotion_dataset)
    sample_order = _dataset_sample_order(epoch_dataset)

    print(f"Using device: {device}")
    print(f"CPU threads capped at: {args.cpu_threads}")
    print(f"Using grounded lattice resolution: {resolved_grid_size}^3")
    print(f"Training samples per epoch: {len(epoch_dataset)}/{len(dataset)}")
    print(f"Promotion samples: {len(promotion_dataset)}/{len(dataset)} ({args.promotion_split})")

    diffusion_config = DiffusionConfig(teacher_steps=1000, student_steps=4)
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        disconnection_penalty=30.0,
        precision="bfloat16",
        enable_pipeline_parallelism=False,
        direct_solver_loss_weight=args.direct_solver_loss_weight,
        direct_solver_interval=int(config_value("training", "direct_solver_interval", 4)),
        direct_solver_steps=args.direct_solver_steps,
        direct_solver_directions=args.direct_solver_directions,
        direct_solver_perturbation=args.direct_solver_perturbation,
        direct_solver_perturbation_grid_size=args.direct_solver_perturbation_grid_size,
        direct_connectivity_weight=args.direct_connectivity_weight,
        direct_aircraft_validity_weight=args.direct_aircraft_validity_weight,
        overfit_geometry_gate_samples=args.promotion_evaluation_samples,
        promotion_generation_seeds=args.promotion_generation_seeds,
        require_direct_solver_every_iteration=False,
    )
    cfd_config = CFDConfig(
        base_grid_resolution=resolved_grid_size,
        solver_type=args.solver,
        use_fused_stream_bfl=(args.lbm_stream_bfl_backend == "fused_stream_bfl"),
    )

    # R7 (PR 41 review, item 7): per-epoch deterministic shuffle. The sampler
    # regenerates the current epoch's permutation from (subset_seed, epoch), so
    # a resumed process continues at the exact completed_in_epoch offset. The
    # subset composition (sample_order) is unchanged and stays fingerprinted.
    train_sampler = ResumableEpochSampler(
        len(epoch_dataset),
        subset_seed=args.subset_seed,
    )
    train_loader = DataLoader(
        epoch_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=aircraft_collate_fn,
    )
    args.planned_optimizer_updates = len(train_loader) * max(1, int(args.num_epochs))
    promotion_loader = DataLoader(
        promotion_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=aircraft_collate_fn,
    )
    calibration_dataset = _build_split_dataset(dataset, args.training_split)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=aircraft_collate_fn,
    )

    trainer = OptimizedDiffusionTrainer(model_config, diffusion_config, training_config, cfd_config, device=device)
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    elif args.warm_start_from:
        trainer.warm_start_checkpoint(args.warm_start_from)
    threshold_calibration = _prepare_geometry_threshold_for_run(
        trainer,
        calibration_loader,
        resume_run_state=(Path(args.resume_run_state) if args.resume_run_state else None),
    )
    args.geometry_probability_threshold = float(
        trainer.geometry_probability_threshold
    )
    args.geometry_threshold_calibration = threshold_calibration
    trainer.scheduler = RunLocalCosineScheduler(
        trainer.optimizer,
        total_updates=args.planned_optimizer_updates,
        min_lr_ratio=args.lr_min_ratio,
    )
    trainer.scheduler_step_per_update = True

    run_state_target = Path(
        args.resume_run_state or (Path(args.save_dir) / "latest_run_state.pt")
    ).resolve()
    run_compatibility = {
        "manifest_identity": _manifest_identity(args.manifest),
        "grid_size": int(resolved_grid_size),
        "latent_dim": int(model_config.latent_dim),
        "split": str(args.training_split),
        "sample_count": int(len(epoch_dataset)),
        "configuration": _build_objective_configuration_fingerprint(
            args=args,
            training_config=training_config,
            model_config=model_config,
            diffusion_config=diffusion_config,
            cfd_config=cfd_config,
            geometry_probability_threshold=trainer.geometry_probability_threshold,
            sample_order=sample_order,
            promotion_sample_order=_dataset_sample_order(promotion_dataset),
        ),
    }

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    history_output = Path(args.history_output).resolve()
    history_output.parent.mkdir(parents=True, exist_ok=True)
    updates_output = (
        Path(args.updates_output).resolve()
        if args.updates_output
        else history_output.with_name("updates.jsonl")
    )
    updates_output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume_run_state:
        updates_output.write_text("", encoding="utf-8")
    elif not updates_output.exists():
        raise FileNotFoundError(
            f"Exact resume requires the existing updates JSONL: {updates_output}"
        )
    args.updates_output = str(updates_output)
    trainer.run_state_checkpoint_path = str(run_state_target)
    trainer.run_state_updates_log_path = str(updates_output)
    trainer.update_metrics_callback = lambda record: _append_jsonl(
        updates_output,
        record,
    )

    resume_state_info = {
        "epoch_index": 0,
        "completed_in_epoch": 0,
        "sample_order": list(sample_order),
    }
    if args.resume_run_state:
        resume_state_info = trainer.load_run_state(
            args.resume_run_state,
            expected_compatibility=run_compatibility,
        )
        _reconcile_updates_log(
            updates_output,
            resume_state_info.get("log_reconciliation", {}),
        )
        if resume_state_info["sample_order"] != sample_order:
            raise ValueError(
                "Incompatible run-state resume: sample_order differs from the current epoch"
            )
        resume_state_info["epoch_index"], resume_state_info["completed_in_epoch"] = (
            _resume_epoch_position(
                resume_state_info["epoch_index"],
                resume_state_info.get("completed_in_epoch", 0),
                len(train_loader),
            )
        )

    writer = _AsyncRecordWriter(trainer.writer)
    trainer.records_writer = writer
    pending_jsonl_seq: List[Optional[int]] = [None]

    def _drain_updates_log_for_save() -> None:
        """Flush the async writer before any run-state save (load-bearing).

        The run-state resume path sha256s the updates JSONL prefix up to the
        recorded byte-offset; the barrier guarantees the writer has durably
        produced every record through ``pending_jsonl_seq[0]`` (and its offset)
        before save_run_state snapshots it.
        """
        trainer.run_state_log_metadata = writer.flush_barrier(pending_jsonl_seq[0])

    def record_update(record: Dict[str, Any]) -> None:
        # Enqueue-and-return: the writer thread performs the append and computes
        # the offset off the CPU update path. run_state_log_metadata is
        # populated by _drain_updates_log_for_save at save time, never here.
        pending_jsonl_seq[0] = writer.enqueue_jsonl(updates_output, record)

    trainer.update_metrics_callback = record_update
    current_epoch_index = int(resume_state_info.get("epoch_index", 0))

    def maybe_save_run_state(
        completed_in_epoch: int,
        total_in_epoch: int,
        force: bool = False,
    ) -> Optional[str]:
        # A forced save (bounded stop_after_updates interruption) bypasses the
        # cadence gate; it is idempotent with the cadence save (same state,
        # atomic ``.previous`` fallback).
        if not force and not _run_state_checkpoint_due(
            completed_in_epoch,
            int(resume_state_info["completed_in_epoch"]),
            args.checkpoint_every_updates,
        ):
            return None
        _drain_updates_log_for_save()
        trainer.save_run_state(
            run_state_target,
            epoch_index=current_epoch_index,
            completed_in_epoch=int(completed_in_epoch),
            sample_order=sample_order,
            compatibility=run_compatibility,
        )
        return str(run_state_target)

    trainer.run_state_checkpoint_callback = maybe_save_run_state
    trainer.resumed_from_update = int(trainer.global_step) if args.resume_run_state else 0
    trainer.stop_after_updates = (
        int(trainer.global_step) + int(args.stop_after_updates)
        if args.stop_after_updates > 0
        else None
    )

    try:

        history: List[Dict[str, Any]] = []
        final_checkpoint_path = str((save_dir / "final_monitored_model.pt").resolve())
        candidate_best_checkpoint_path = str(
            (save_dir / "best_geometry_model.pt").resolve()
        )
        best_checkpoint_path: str | None = None
        best_geometry_metric = float("inf")
        best_promotion_rank = (-1.0,) * 8
        selection_interval = max(1, int(training_config.promotion_interval_epochs))
        initial_geometry_promotion = None
        initial_geometry_promotion_report = None
        promotion_baseline: Dict[str, Any] = {}

        if args.resume_run_state:
            resumed_metadata = dict(resume_state_info.get("run_state_metadata", {}))
            promotion_baseline = restore_promotion_baseline(
                resume_state_info,
                promotion_split=args.promotion_split,
                promotion_sample_order=_dataset_sample_order(promotion_dataset),
                evaluation_samples=args.promotion_evaluation_samples,
                generation_seeds=args.promotion_generation_seeds,
            )
            initial_geometry_promotion_report = (
                dict(resumed_metadata.get("promotion_baseline_report", {}))
                or dict(promotion_baseline)
            )
            initial_geometry_promotion = dict(
                resumed_metadata.get("promotion_baseline_metrics", {})
            ) or None
            trainer.run_state_metadata = resumed_metadata
            # R5 (PR 41 review, item 5): restore the best-checkpoint selection
            # state so a resume keeps the lexicographic rank gate and reported
            # best path/metric instead of resetting to (-1,)*8 / inf / None and
            # clobbering best_geometry_model.pt with a worse promotion.
            best_promotion_rank = _restore_best_promotion_rank(resumed_metadata)
            best_geometry_metric = resumed_metadata.get(
                "best_geometry_metric", float("inf")
            )
            best_checkpoint_path = resumed_metadata.get(
                "best_checkpoint_path", best_checkpoint_path
            )
            # R6 (PR 41 review, item 6): seed the stability/early-stop history
            # from the persisted history payload so the convergence window and
            # the history JSONL survive an exact resume instead of restarting
            # cold (which would delay early-stop and drop pre-resume rows).
            restored_history = _load_monitored_history(history_output)
            if restored_history:
                history = restored_history

        if not args.resume_run_state and (
            args.resume_from or args.warm_start_from or not promotion_baseline
        ):
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            baseline_promotion = trainer.evaluate_geometry_promotion_gate(promotion_loader)
            promotion_baseline = dict(baseline_promotion)
            initial_geometry_promotion_report = dict(baseline_promotion)
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.set_rng_state(torch_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            baseline_metrics, best_promotion_rank = _geometry_promotion_metrics(
                baseline_promotion
            )
            best_geometry_metric = baseline_metrics["geometry_selection_metric"]
            initial_geometry_promotion = {
                **baseline_metrics,
                "status": str(baseline_promotion.get("status", "fail")),
                "source_checkpoint": (
                    str(Path(args.resume_from or args.warm_start_from).resolve())
                    if (args.resume_from or args.warm_start_from)
                    else "fresh_run_initial_state"
                ),
            }
            trainer.run_state_metadata = {
                "promotion_baseline": dict(promotion_baseline),
                "promotion_baseline_report": dict(initial_geometry_promotion_report),
                "promotion_baseline_metrics": dict(baseline_metrics),
                "promotion_baseline_identity": {
                    "split": str(args.promotion_split),
                    "sample_order": _dataset_sample_order(promotion_dataset),
                    "evaluation_samples": int(args.promotion_evaluation_samples),
                    "generation_seeds": int(args.promotion_generation_seeds),
                    "materialization_mode": baseline_promotion.get(
                        "materialization_mode"
                    ),
                    "geometry_probability_threshold": baseline_promotion.get(
                        "geometry_probability_threshold"
                    ),
                },
            }
            best_checkpoint_path = (
                str(Path(args.resume_from or args.warm_start_from).resolve())
                if (args.resume_from or args.warm_start_from)
                else None
            )
            # R5: seed the persisted best-checkpoint selection state so the
            # first run-state save captures it for an exact resume.
            _sync_best_checkpoint_state(
                trainer,
                best_promotion_rank=best_promotion_rank,
                best_geometry_metric=best_geometry_metric,
                best_checkpoint_path=best_checkpoint_path,
            )
            initial_promotion_path = history_output.with_name(
                "initial_geometry_promotion.json"
            )
            initial_promotion_path.write_text(
                json.dumps(initial_geometry_promotion_report, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                "Initial geometry promotion baseline: "
                "valid_fraction="
                f"{baseline_metrics['promotion_generated_aircraft_valid_fraction']:.6g}, "
                "occupancy_error="
                f"{baseline_metrics['promotion_generated_occupancy_error']:.6g}, "
                "unique_fraction="
                f"{baseline_metrics['promotion_generated_unique_fraction']:.6g}, "
                "worst_recall="
                f"{baseline_metrics['promotion_generated_worst_recall']:.6g}, "
                f"mean_recall={baseline_metrics['promotion_generated_recall']:.6g}"
            )

        start_epoch = int(resume_state_info.get("epoch_index", 0))
        for epoch in range(start_epoch, args.num_epochs):
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            current_epoch_index = epoch
            start_batch = 0
            if epoch == int(resume_state_info.get("epoch_index", 0)):
                start_batch = int(resume_state_info.get("completed_in_epoch", 0))
            # R7: regenerate this epoch's deterministic sample order. On resume
            # start_epoch is the run-state epoch_index, so the permutation is
            # re-derived identically and start_batch continues at the recorded
            # completed_in_epoch offset.
            train_sampler.set_epoch(epoch)
            metrics = trainer.train_epoch(
                train_loader,
                grid_size=resolved_grid_size,
                start_batch=start_batch,
            )
            if (
                trainer.stop_after_updates is not None
                and trainer.global_step >= trainer.stop_after_updates
            ):
                print(
                    "Stopped at the requested bounded interruption point after writing "
                    f"{run_state_target}"
                )
                break
            _reset_epoch_checkpoint_segment(
                resume_state_info,
                next_epoch=epoch + 1,
            )
            if args.checkpoint_every_updates > 0:
                _drain_updates_log_for_save()
                trainer.save_run_state(
                    run_state_target,
                    epoch_index=epoch + 1,
                    completed_in_epoch=0,
                    sample_order=sample_order,
                    compatibility=run_compatibility,
                )
            metrics = {
                "epoch": epoch + 1,
                **{key: float(value) for key, value in metrics.items()},
            }
            if not getattr(trainer, "scheduler_step_per_update", False):
                trainer.scheduler.step()
            for group in trainer.optimizer.param_groups:
                group_name = str(group.get("name", "unnamed"))
                metrics[f"learning_rate_{group_name}"] = float(group.get("lr", 0.0))
            metrics["core_loss"] = compute_core_loss(metrics)
            metrics["selected_as_best_geometry_checkpoint"] = 0.0
            metrics["geometry_selection_evaluated"] = 0.0
            promotion_passed = False
            if (epoch + 1) % selection_interval == 0:
                promotion = trainer.evaluate_geometry_promotion_gate(promotion_loader)
                directional_gate = (
                    evaluate_directional_promotion_gate(promotion, promotion_baseline)
                    if promotion_baseline
                    else {"status": "pass", "failed_conditions": [], "conditions": {}}
                )
                non_regression = {
                    **directional_gate,
                    "failed_checks": list(
                        directional_gate.get("failed_conditions", [])
                    ),
                }
                promotion_passed = (
                    promotion.get("status") == "pass"
                    and non_regression.get("status") == "pass"
                )
                metrics["geometry_selection_evaluated"] = 1.0
                promotion_metrics, promotion_rank = _geometry_promotion_metrics(promotion)
                metrics.update(promotion_metrics)
                metrics["promotion_gate_passed"] = float(promotion_passed)
                metrics["promotion_non_regression_passed"] = float(
                    non_regression.get("status") == "pass"
                )
                metrics["promotion_non_regression_failed_count"] = float(
                    len(non_regression.get("failed_checks", []))
                )
                metrics["promotion_report"] = dict(promotion)
                metrics["promotion_non_regression_report"] = dict(non_regression)
                metrics["promotion_directional_gate"] = dict(directional_gate)
                # Only gate the best-checkpoint save on the lexicographic
                # promotion rank improving over the best seen so far; every
                # passing promotion otherwise overwrites the best checkpoint
                # regardless of rank. Metric recording below is unchanged.
                candidate_improved = (
                    promotion_passed and promotion_rank > best_promotion_rank
                )
                if candidate_improved:
                    best_promotion_rank = promotion_rank
                    best_geometry_metric = metrics["geometry_selection_metric"]
                    trainer.save_checkpoint(candidate_best_checkpoint_path)
                    best_checkpoint_path = candidate_best_checkpoint_path
                    metrics["selected_as_best_geometry_checkpoint"] = 1.0
                    # R5: mirror into run_state_metadata so the next run-state
                    # save persists the selection gate across an exact resume.
                    _sync_best_checkpoint_state(
                        trainer,
                        best_promotion_rank=best_promotion_rank,
                        best_geometry_metric=best_geometry_metric,
                        best_checkpoint_path=best_checkpoint_path,
                    )
            else:
                metrics["geometry_selection_metric"] = float("nan")
            history.append(metrics)

            stability = summarize_stability(
                history,
                metric=args.stability_metric,
                window=args.convergence_window,
                convergence_target=args.convergence_target,
                convergence_cv_threshold=args.convergence_cv_threshold,
                convergence_drift_threshold=args.convergence_drift_threshold,
                oscillation_cv_threshold=args.oscillation_cv_threshold,
                required_geometry_loss_max=args.required_geometry_loss_max,
            )

            metric_stats = stability.get("metric_stats", {})
            latest_metric = metrics.get(args.stability_metric, 0.0)
            print(
                "Stability "
                f"status={stability['status']} "
                f"metric={args.stability_metric} "
                f"mean={metric_stats.get('mean', latest_metric):.4f} "
                f"cv={metric_stats.get('cv', 0.0):.4f}"
            )
            if stability.get("suspected_root_cause"):
                print(f"Suspected instability root cause: {stability['suspected_root_cause']}")

            payload = _build_history_payload(
                args=args,
                device=device,
                history=history,
                stability=stability,
                checkpoint_path=(
                    final_checkpoint_path if args.save_final_checkpoint else None
                ),
                model_config=model_config,
                best_checkpoint_path=best_checkpoint_path,
                best_geometry_metric=(
                    best_geometry_metric if np.isfinite(best_geometry_metric) else None
                ),
                initial_geometry_promotion=initial_geometry_promotion,
                initial_geometry_promotion_report=initial_geometry_promotion_report,
            )
            history_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                checkpoint_path = save_dir / f"checkpoint_monitored_ep{epoch + 1}.pt"
                trainer.save_checkpoint(str(checkpoint_path))

            if args.stop_on_promotion_pass and promotion_passed:
                print(
                    f"Stopping after epoch {epoch + 1}: validation geometry promotion gate passed."
                )
                break

            if args.early_stop_on_convergence and stability["converged"]:
                print(f"Early stopping at epoch {epoch + 1}: convergence criteria met.")
                break

        if args.save_final_checkpoint:
            trainer.save_checkpoint(final_checkpoint_path)
            print(f"Final monitored checkpoint saved to {final_checkpoint_path}")
        else:
            print("Final checkpoint save disabled for this smoke run.")
        print(f"History written to {history_output}")
    finally:

        writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



