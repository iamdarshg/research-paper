#!/usr/bin/env python3
"""NONBLOCK-1 async-records-writer byte-parity regression test (CPU-only).

Preserves the NONBLOCK-1 byte-parity gate (brief Step 4) as a committed test.
Compares, over the same record sequence:
  1. synchronous _append_jsonl vs the async _AsyncRecordWriter writer thread
     -> byte-identical JSONL files, identical offsets, identical
        _JSONL_RECORD_COUNTS state, and identical _reconcile_updates_log output
        on both files (resume-time reconciliation semantics, incl. the
        trailing-record truncation case).
  2. tensorboard: the 13 synchronous add_scalar calls vs the 2 batched enqueues
     expanded by the writer thread -> identical tag -> (step, value) sets read
     back through tensorboard's EventAccumulator (13 tags; the conditional
     5-tag direct_* batch fires only when direct_count > 0).

Run standalone from the repo root:
    python tests/test_nonblock1_async_writer_parity.py
or via pytest:
    python -m pytest tests/test_nonblock1_async_writer_parity.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "CLI"))

from run_monitored_training import (  # noqa: E402
    _JSONL_RECORD_COUNTS,
    _append_jsonl,
    _reconcile_updates_log,
    _updates_log_reconciliation_metadata,
    _AsyncRecordWriter,
)
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)


# --------------------------------------------------------------------------
# Fixtures shared by both writers
# --------------------------------------------------------------------------

def _make_records(count: int) -> List[Dict[str, Any]]:
    records = []
    for step in range(1, count + 1):
        records.append(
            {
                "kind": "optimizer_update",
                "global_step": step,
                "completed_in_epoch": step,
                "total_in_epoch": count,
                "run_state_checkpoint_path": None,
                "resumed_from_update": None,
                "remaining_in_epoch": count - step,
                "losses": {
                    "optimization": 0.1 * step,
                    "mse": 0.05 * step,
                    "clean_geometry": 0.2,
                    "geometry": 0.3,
                    "generation_geometry": 0.4,
                    "consistency": 0.5,
                    "latent_reconstruction": 0.6,
                    "direct_solver": 0.7,
                    "threshold_positive_margin_loss": 0.0,
                    "threshold_negative_margin_loss": 0.0,
                },
                "student_gradients": {
                    "data": {
                        "raw_norm": 1.0,
                        "applied_norm": 0.8,
                        "scale": 0.9,
                        "present": True,
                        "nonzero": True,
                        "anchor_cosine_before": 0.1,
                        "anchor_cosine_after": 0.2,
                        "conflict_projected": False,
                        "projection_norm": 0.3,
                    }
                },
                "learning_rates": {"diffusion": 2e-5, "converter": 1e-5},
            }
        )
    return records


def _make_tb_values(epoch: int) -> Dict[str, Any]:
    """Mirror the values the production per-epoch tensorboard block computes."""
    denominator = max(3, 1)  # pretend 3 processed updates
    avg = float(epoch) / denominator
    base = {
        "Loss/total": avg,
        "Loss/optimization": avg,
        "Loss/mse": 0.1 * epoch / denominator,
        "Loss/clean_geometry_reconstruction": 0.2 * epoch / denominator,
        "Loss/geometry_reconstruction": 0.3 * epoch / denominator,
        "Loss/generation_reconstruction": 0.4 * epoch / denominator,
        "Loss/consistency": 0.5 * epoch / denominator,
        "Loss/direct_solver": 0.7 * epoch / denominator,
    }
    direct = {
        "Loss/direct_solver_eval": 0.05 * epoch,
        "Loss/direct_occupancy": 0.06 * epoch,
        "Loss/direct_aero": 0.07 * epoch,
        "Loss/direct_connectivity": 0.08 * epoch,
        "Loss/direct_aircraft_validity": 0.09 * epoch,
    }
    return base, direct


# --------------------------------------------------------------------------
# 1) JSONL byte-parity
# --------------------------------------------------------------------------

def test_jsonl_parity() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="nonblock1_jsonl_"))
    sync_path = tmp / "sync_updates.jsonl"
    async_path = tmp / "async_updates.jsonl"
    records = _make_records(5)

    # Synchronous baseline (today's writer).
    sync_metas: List[Dict[str, Any]] = []
    for record in records:
        sync_metas.append(_append_jsonl(sync_path, record))

    # Async writer (NONBLOCK-1).
    writer = _AsyncRecordWriter(None)
    seqs = []
    for record in records:
        seqs.append(writer.enqueue_jsonl(async_path, record))
    async_last_meta = writer.flush_barrier(seqs[-1])
    writer.close()

    sync_bytes = sync_path.read_bytes()
    async_bytes = async_path.read_bytes()
    assert sync_bytes == async_bytes, "JSONL files differ byte-for-byte"
    assert sync_metas[-1] == async_last_meta, (
        "async last metadata != sync last metadata: "
        f"{sync_metas[-1]} vs {async_last_meta}"
    )
    assert _JSONL_RECORD_COUNTS[str(sync_path)] == len(records)
    assert _JSONL_RECORD_COUNTS[str(async_path)] == len(records)
    assert _JSONL_RECORD_COUNTS[str(sync_path)] == _JSONL_RECORD_COUNTS[str(async_path)]

    # Resume-time reconciliation on both files -> identical result.
    sync_stamped = _updates_log_reconciliation_metadata(sync_path, sync_metas[-1])
    async_stamped = _updates_log_reconciliation_metadata(async_path, async_last_meta)
    assert sync_stamped["sha256"] == async_stamped["sha256"]
    sync_recon = _reconcile_updates_log(sync_path, sync_stamped)
    async_recon = _reconcile_updates_log(async_path, async_stamped)
    assert sync_recon == async_recon, (
        "reconciliation differs: " f"{sync_recon} vs {async_recon}"
    )
    assert sync_recon["truncated_records"] == 0
    assert sync_recon["offset"] == sync_metas[-1]["offset"]

    # Now append one extra (trailing) record and re-reconcile -> both truncate it.
    _append_jsonl(sync_path, records[0])
    writer2 = _AsyncRecordWriter(None)
    writer2.enqueue_jsonl(async_path, records[0])
    writer2.flush_barrier(None)
    writer2.close()
    sync_recon2 = _reconcile_updates_log(sync_path, sync_stamped)
    async_recon2 = _reconcile_updates_log(async_path, async_stamped)
    assert sync_recon2 == async_recon2 == {**sync_recon, "truncated_records": 1}


# --------------------------------------------------------------------------
# 2) Tensorboard scalar parity (13 add_scalar vs 2 batched enqueues)
# --------------------------------------------------------------------------

def _write_sync_scalars(writer: SummaryWriter, base, direct, step: int,
                        direct_count: int) -> None:
    # Exact original order (train_epoch tensorboard block).
    writer.add_scalar('Loss/total', base['Loss/total'], step)
    writer.add_scalar('Loss/optimization', base['Loss/optimization'], step)
    writer.add_scalar('Loss/mse', base['Loss/mse'], step)
    writer.add_scalar('Loss/clean_geometry_reconstruction',
                      base['Loss/clean_geometry_reconstruction'], step)
    writer.add_scalar('Loss/geometry_reconstruction',
                      base['Loss/geometry_reconstruction'], step)
    writer.add_scalar('Loss/generation_reconstruction',
                      base['Loss/generation_reconstruction'], step)
    writer.add_scalar('Loss/consistency', base['Loss/consistency'], step)
    writer.add_scalar('Loss/direct_solver', base['Loss/direct_solver'], step)
    if direct_count > 0:
        writer.add_scalar('Loss/direct_solver_eval', direct['Loss/direct_solver_eval'], step)
        writer.add_scalar('Loss/direct_occupancy', direct['Loss/direct_occupancy'], step)
        writer.add_scalar('Loss/direct_aero', direct['Loss/direct_aero'], step)
        writer.add_scalar('Loss/direct_connectivity', direct['Loss/direct_connectivity'], step)
        writer.add_scalar('Loss/direct_aircraft_validity',
                          direct['Loss/direct_aircraft_validity'], step)


def test_tensorboard_parity() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="nonblock1_tb_"))
    sync_dir = tmp / "runs_sync"
    async_dir = tmp / "runs_async"

    sync_writer = SummaryWriter(log_dir=str(sync_dir))
    async_writer = SummaryWriter(log_dir=str(async_dir))
    async_writer.flush()  # ensure file writer exists before thread starts
    rec_writer = _AsyncRecordWriter(async_writer)

    epochs = [1, 2, 3]
    for epoch in epochs:
        base, direct = _make_tb_values(epoch)
        step = epoch * 10
        direct_count = epoch if (epoch % 2 == 0) else 0  # epochs 2 only
        _write_sync_scalars(sync_writer, base, direct, step, direct_count)
        rec_writer.enqueue_tb_batch(step, base)
        if direct_count > 0:
            rec_writer.enqueue_tb_batch(step, direct)

    sync_writer.flush()
    rec_writer.close()  # drains queue, flushes and closes async_writer
    sync_writer.close()

    def read_scalars(logdir: Path) -> Dict[str, List[tuple]]:
        ea = EventAccumulator(str(logdir))
        ea.Reload()
        out: Dict[str, List[tuple]] = {}
        for tag in sorted(ea.Tags().get("scalars", [])):
            out[tag] = [(float(ev.step), float(ev.value)) for ev in ea.Scalars(tag)]
        return out

    sync_scalars = read_scalars(sync_dir)
    async_scalars = read_scalars(async_dir)

    assert set(sync_scalars) == set(async_scalars), (
        "tag sets differ: sync-only=%r async-only=%r"
        % (set(sync_scalars) - set(async_scalars), set(async_scalars) - set(sync_scalars))
    )
    for tag in sync_scalars:
        assert sync_scalars[tag] == async_scalars[tag], (
            f"tag {tag!r} differs: {sync_scalars[tag]} vs {async_scalars[tag]}"
        )
    assert len(sync_scalars) == 13, f"expected 13 scalar tags, got {len(sync_scalars)}"


def main() -> int:
    test_jsonl_parity()
    test_tensorboard_parity()
    print("ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
