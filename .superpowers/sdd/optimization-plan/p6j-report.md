# P6j — Checkpoint unpack (7,576 unpack_hook, 4.9 s)

**Status: DONE_WITH_CONCERNS**
**Commit:** `3cf5463a7abc25dc47bd168e683718dd8addbde2` on branch `worktree-agent-a7e0ce40ccc84908a` (worktree of `codex/constrained-aircraft-recovery`; base fast-forwarded to `4a99491` before editing).
**Validation:** `python -m py_compile CLI/aircraft_diffusion_cfd.py` OK; CPU round-trip `save_checkpoint -> load_checkpoint -> OptimizedAircraftGenerator` passes (state dicts byte-equal, optimizer Adam state carried); `_load_checkpoint_metadata` loads the real `step1305.pt` via `weights_only=True` (no fallback); `pytest tests/test_conditioning.py -q` 11 passed.

---

## 1. Measurement (the explicit MEASURE-first requirement)

Checkpoint under test: `build/recovery_ladder_20260814/step1305.pt` — **463.6 MB, 1,356 tensors**, 17 top-level keys
(diffusion_model 126/62.8 MB, consistency_model 252/79.5 MB, converter 24/33.0 MB, ema_model 126/62.8 MB,
optimizer 828/225.0 MB, scalars/configs 0). Torch 2.9.1+cu130, CPU-only (GPU shared — no GPU measurements run).

| loader (CPU, median of 3) | time |
|---|---|
| `torch.load(weights_only=False)` — current big-load default | **0.59–0.64 s** |
| `torch.load(weights_only=True)` | 0.85–0.92 s (≈0.25 s slower) |
| `torch.load(weights_only=True, mmap=True)` | 0.51–0.53 s |
| fresh-process cold load (`weights_only=False`) | 0.57–0.85 s |

**Per-chunk (split-file) checkpointing — the queue item's question — does NOT pay at 96³:**
3-file split (model 528 tensors/238 MB, optimizer 828 tensors/225 MB, meta 0):

| scenario | time |
|---|---|
| single-blob load | 0.64 s |
| sequential 3-chunk load | 0.54 s |
| **parallel 3-chunk load** | **0.49 s** (Δ ≈ 0.15 s warm) |
| single-blob save | 1.06 s |
| 3-chunk save | 1.08 s |

The best per-chunk win is ~0.15 s on a **one-time** load, while the load itself is sub-second. The 4.9 s figure
in the queue is not the file load.

### Where the 7,576 `unpack_hook` / 4.9 s actually comes from

The repo's own de-black-box (`docs/performance/residual-blackbox-report.md`, table "Pass B — cProfile") attributes
the row to **"decoder activation-checkpoint bookkeeping"**, not the checkpoint file. The cProfile row is
`checkpoint unpack_hook | 7,576 | 0.14 s self | 4.85 s cumtime | decoder activation-checkpoint bookkeeping`.

That is `LatentTo3DConverter._checkpointed_coordinate_chunk` (CLI/aircraft_diffusion_cfd.py:3108), which calls
`activation_checkpoint(..., use_reentrant=False)` on each 16,384-voxel coordinate chunk
(`coordinate_chunk_size: 16384`, `coordinate_gradient_checkpointing: true` in CLI/config.yaml). Non-reentrant
activation checkpointing registers `pack_hook`/`unpack_hook` via `torch.autograd.graph.saved_tensors_hooks`;
the 7,576 `unpack_hook` calls are the per-saved-tensor unpacks + recompute during the backward pass across the
~54 coordinate chunks per full update. This is per-update (every update), which is why it showed as "per update".

CPU micro-benchmark of the wrapper (width-192, depth-6 MLP, forward+backward, min of 5) confirms the checkpoint
wrapper costs ~1.3–1.8× the direct call at production chunk sizes:

| chunk | direct fwd+bwd | ckpt fwd+bwd | overhead | ratio |
|---|---|---|---|---|
| 4096 | 0.101 s | 0.145 s | +0.043 s | 1.43× |
| 8192 | 0.215 s | 0.389 s | +0.174 s | 1.81× |
| 16384 | 0.647 s | 0.963 s | +0.316 s | 1.49× |
| 32768 | 0.488 s | 0.623 s | +0.135 s | 1.28× |

**Conclusion:** at 96³ the checkpoint *file* unpack is ~0.6 s one-time (not 4.9 s, not per-update), and per-chunk
file checkpointing saves ~0.15 s warm — not worth the format/file-count complexity. The real 4.9 s/update lever is
the coordinate-decoder activation-checkpoint bookkeeping/recompute, which lives in the converter MLP forward path
**outside this item's assigned scope** (see Concerns).

## 2. What changed and expected saving

Routed the three bare `torch.load(path, map_location=device)` big-checkpoint load sites through the existing
`_load_checkpoint_metadata` helper (`weights_only=True` first, trusted-path/authorized-path `weights_only=False`
fallback), preserving `map_location` and the exact checkpoint file format:

- `OptimizedDiffusionTrainer.load_checkpoint` (was CLI/aircraft_diffusion_cfd.py:7722)
- `OptimizedAircraftGenerator.__init__` (was :7849)
- `train()` resume-from (was :8768)

`authorized_paths=(path,)` is passed so operator-specified `--resume-from` / `--warm-start-from` / `--checkpoint`
paths outside `build/` keep the existing fallback semantics. No format change; old artifacts still load — the real
`step1305.pt` was verified `weights_only`-clean (loads without fallback), and the round-trip test covers a fresh
save→load through the same helper.

**Expected saving:** ≈ 0 s/update on the per-update wall (file loads are one-time, per process/resume). The change
closes the `weights_only=False` security gap on the big-checkpoint loaders — the queue's explicit "preserve the
weights_only security contract" — at a small one-time cost (weights_only measured ~0.25 s slower per load). It is a
hygiene/security alignment, not the 4.9 s win the queue anticipated.

## 3. Concerns

1. **The 4.9 s target is out of scope.** The queue's headline number is decoder activation-checkpoint bookkeeping
   (`_checkpointed_coordinate_chunk` :3108), which is inside the converter MLP forward path. This item's scope
   ("checkpoint save/load region; do NOT touch the converter MLP") excludes it, and the converter is reserved for
   P6c fusion work. Recommend the controller dispatch a follow-up to evaluate `coordinate_gradient_checkpointing`
   for the coordinate decoder at 96³ (the de-black-box's own suggestion, "likely disable for small chunks"): the
   measured wrapper overhead is ~1.3–1.8× at 16,384-chunk, so a VRAM-checked disable (or fewer/larger chunks to
   amortize pack/unpack) is the real per-update lever. Needs GPU for the VRAM envelope.
2. `weights_only=True` is ~0.25 s slower per big-load (one-time). This item is a security-contract alignment, not a
   speed win.
3. GPU-side figures (load-to-CUDA H2D cost; VRAM headroom for any activation-checkpoint change) were not measured —
   GPU shared per instructions.
4. `CLI/profile_training_update.py` keeps its own copy of `_load_checkpoint_metadata`; it was not touched (out of
   scope) but already uses `weights_only=True` for its metadata load.
