#!/usr/bin/env python3
"""Task-3 regression: _AsyncRecordWriter flush_barrier must wait for seq 0.

Finding #3 (PR-41 GPT-5.6 review, red): ``_AsyncRecordWriter.__init__``
initialized ``_completed_seq = 0``. The first JSONL record is assigned sequence
0 (``_next_seq = 0``), so ``flush_barrier(0)`` computes ``target = 0`` and the
drain loop condition ``self._completed_seq < target`` is ``0 < 0`` = False.
The barrier therefore returned IMMEDIATELY, before record 0 was durable, and
``self._last_meta`` stayed ``{}``. Load-bearing: run-state saves reconcile the
updates-log prefix against that metadata (``run_state_log_metadata``); with
``--checkpoint-every-updates 1`` the first save would reconcile against empty
metadata.

Fix: initialize ``self._completed_seq = -1`` (record seqs still start at 0;
``-1 < 0`` is True so the barrier waits for seq 0 to drain).

This module targets that seam directly. The worker thread is gated inside
``_append_jsonl`` so the red path is deterministic: the barrier must still be
blocked while the record is in flight. The fake summary writer is ``None``,
matching the JSONL-only writer pattern used by the existing run-monitored tests
(``test_nonblock1_async_writer_parity.py`` and the FakeTrainer in
``test_constrained_recovery_review.py``); the tensorboard path is covered by
``test_nonblock1_async_writer_parity.py``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = REPO_ROOT / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import run_monitored_training as monitored_training  # noqa: E402
from run_monitored_training import (  # noqa: E402
    _AsyncRecordWriter,
    _reconcile_updates_log,
    _updates_log_reconciliation_metadata,
)


def _make_single_record() -> Dict[str, Any]:
    return {
        "kind": "optimizer_update",
        "global_step": 1,
        "completed_in_epoch": 1,
        "total_in_epoch": 1,
        "remaining_in_epoch": 0,
        "losses": {"optimization": 0.1, "mse": 0.05},
    }


def test_flush_barrier_waits_for_first_record_seq0_durability(tmp_path):
    """flush_barrier(0) must block until record 0 is durable.

    With the buggy ``_completed_seq = 0`` init the barrier returns before the
    worker has made record 0 durable, so ``_last_meta`` is empty and the file
    is empty at barrier-return time. The worker is gated inside
    ``_append_jsonl`` so this is deterministic on the red path: the barrier
    MUST still be blocked while record 0 is in flight.
    """
    gate = threading.Event()
    entered = threading.Event()
    real_append = monitored_training._append_jsonl

    def gated_append(path: Path, record: Dict[str, Any]) -> Dict[str, Any]:
        entered.set()
        assert gate.wait(timeout=30.0), "worker blocked on gate timed out"
        return real_append(path, record)

    monitored_training._append_jsonl = gated_append
    writer = None
    barrier_holder: Dict[str, Any] = {}
    try:
        jsonl_path = tmp_path / "updates.jsonl"
        # Fake summary writer: None exercises the JSONL-only path (matches the
        # existing run-monitored tests; the tensorboard path is covered by
        # test_nonblock1_async_writer_parity.py).
        writer = _AsyncRecordWriter(None)
        record = _make_single_record()

        # Exactly ONE record; its sequence is the writer's first (0).
        seq = writer.enqueue_jsonl(jsonl_path, record)
        assert seq == 0, f"expected first record seq 0, got {seq}"

        def call_barrier() -> None:
            barrier_holder["meta"] = writer.flush_barrier(seq)

        barrier_thread = threading.Thread(
            target=call_barrier,
            name="flush-barrier-caller",
        )
        barrier_thread.start()

        # The worker has entered _append_jsonl and is blocked on the gate, so
        # record 0 is NOT yet durable. The barrier must still be waiting.
        assert entered.wait(timeout=30.0), "worker never reached _append_jsonl"
        assert barrier_thread.is_alive(), (
            "flush_barrier(0) returned before record 0 was durable: the "
            "_completed_seq=0 init made the drain loop return immediately"
        )

        # Release the worker; the barrier may now complete.
        gate.set()
        barrier_thread.join(timeout=30.0)
        assert not barrier_thread.is_alive(), "flush_barrier(0) never returned"

        # (a) The returned metadata is non-empty and carries the record's
        # identity (the reconciliation prefix is built from it).
        meta = barrier_holder.get("meta")
        assert meta, "flush_barrier(0) returned empty metadata"
        assert meta["global_step"] == record["global_step"]
        assert meta["record_count"] == 1
        assert meta["offset"] > 0

        # (b) The JSONL file actually contains the record bytes at barrier
        # return time.
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, f"expected 1 record line, got {len(lines)}"
        assert json.loads(lines[0]) == record

        # (c) The resume-reconciliation prefix hash is non-empty and matches
        # the durable file (a save at this point reconciles cleanly).
        stamped = _updates_log_reconciliation_metadata(jsonl_path, meta)
        assert stamped["sha256"], "reconciliation prefix hash is empty"
        assert stamped["sha256"] == hashlib.sha256(
            jsonl_path.read_bytes()
        ).hexdigest()
        reconciliation = _reconcile_updates_log(jsonl_path, stamped)
        assert reconciliation["truncated_records"] == 0
        assert reconciliation["offset"] == meta["offset"]
    finally:
        gate.set()
        if writer is not None:
            writer.close()
        monitored_training._append_jsonl = real_append


def test_flush_barrier_with_no_records_returns_empty(tmp_path):
    """A barrier with nothing enqueued still returns ``{}`` (no hang)."""
    writer = _AsyncRecordWriter(None)
    try:
        # ``seq=None`` targets the last-enqueued JSONL seq (None here), so the
        # barrier returns ``{}`` without waiting.
        assert writer.flush_barrier() == {}
    finally:
        writer.close()
    # After close the drain loop is disabled, so an explicit seq-0 barrier also
    # returns ``{}`` instead of waiting forever for a record that never came.
    assert writer.flush_barrier(0) == {}
