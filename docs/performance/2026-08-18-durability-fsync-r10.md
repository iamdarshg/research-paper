# R10 — Checkpoint / log durability (PR 41 review, item 10)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery` (worktree `pr41-review`)

## The reviewer finding

> Durable artifacts (checkpoints, run-state) are not written crash-safely: a
> power loss mid-save can leave a torn file at the public path, silently
> corrupting a resume.

## The audit

Five durable-write paths exist. Each was checked against the gold standard
`atomic_save_run_state` (write-temp → fsync → `os.replace`, plus a `.previous`
copy so an interrupted replacement cannot destroy the last good state):

| Path | Was | Now |
|---|---|---|
| `atomic_save_run_state` (aircraft_diffusion_cfd.py:212) | temp + fsync + atomic + `.previous` | unchanged (already correct) |
| Trainer `save_checkpoint` (trainer.py:277) | `torch.save` **direct to the final path** — a crash mid-save tears the public checkpoint | routes through new `atomic_write_checkpoint` |
| Run-state checkpoint save (aircraft_diffusion_cfd.py:~8573) | temp + `os.replace`, **no fsync** — the temp sat only in the OS page cache | fsyncs the temp before the replace |
| JSONL telemetry appends (run_monitored_training.py) | flush-only, by documented design | unchanged — deliberately *not* fsynced per record |

### JSONL decision

The telemetry JSONL stays flush-only. Per-record fsync would serialize every
update at the storage device, which is the single most frequent write in the
run. The design already reconciles the log at resume via file offset + sha256
(`_load_jsonl_tail`/`truncate` path), so a torn tail is detected and truncated
rather than trusted. This is a documented perf-for-durability tradeoff; the
claim-bearing run's *checkpoint* artifacts — the ones a resume depends on — are
all fsynced.

## A real bug the tests caught

The first implementation opened the temp file **read-only** to fsync it:

```python
with temporary.open("rb") as handle:
    os.fsync(handle.fileno())
```

On Windows this raises `OSError: [Errno 9] Bad file descriptor` — `os.fsync`
requires a handle with write access. The new durability tests failed on the
first run and the pattern was corrected to match `atomic_save_run_state`:

```python
with temporary.open("wb") as handle:
    torch.save(checkpoint, handle)
    handle.flush()
    os.fsync(handle.fileno())
```

Every `os.fsync` call site now uses a write-capable handle (`"wb"` or `"r+b"`;
grep-verified across CLI/).

## Test evidence

- `test_atomic_write_checkpoint_serializes_and_cleans_tmp` — a successful write
  leaves a loadable checkpoint (parent dir auto-created) and **no `.tmp`
  sibling**.
- `test_atomic_write_checkpoint_failure_preserves_existing_target` — a
  serialization failure (unpicklable object) cleans up the temp and leaves a
  pre-existing target **byte-identical** (no torn write).
- `test_smoke_pipeline.py::test_save_checkpoint_includes_cfd_config_payload`
  updated for the new write-handle argument (same `.tmp`-first contract, plus
  cwd-litter cleanup since `torch.save`/`os.replace` are mocked).
- Full suite: **512 passed / 2 skipped** (baseline 510 / 2 + 2 new tests).

## Paper implication

"Checkpoints and run-state are crash-safe (write-temp / fsync / atomic
replace)" is now a defensible reproducibility statement for the claim-bearing
run. The `atomic_save_run_state` `.previous`-copy semantics also mean an
interrupted replacement can always recover the last good state.
