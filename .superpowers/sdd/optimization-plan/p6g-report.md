# P6g — Merge the 54 margin-chunk backwards into one (engine calls 60 → ~7)

**Status: DONE**
**Commit:** `(see below)` on branch `worktree-agent-ad5198d0c4b79b778` (worktree of
`codex/constrained-aircraft-recovery`; base moved to `39dc30e` before editing).
**Validation (CPU-only, per instructions):** `python -m py_compile CLI/aircraft_diffusion_cfd.py` OK;
`pytest tests/test_constrained_recovery_review.py -q` → **35 passed** (includes the pinned
`test_coordinate_threshold_margin_backpropagates_all_chunks_after_data_backward`, the dense-branch
margin tests, and the `train_epoch` margin-path tests); `pytest tests/test_multiobjective_gradients.py -q`
→ **17 passed** (21 in a combined targeted run). GPU parity + benchmark + peak-VRAM are the controller's
to run after this lands.

---

## 1. Status gate / base reconciliation (important context)

The brief's binding gate is: if Cap-0's `.item()`-hoist / de-sync edits are NOT present in the base,
STOP and report BLOCKED. Cap-0 (`8f05365`, merge of `1182d1c`) **IS** present at the true base
`39dc30e` (tip of `codex/constrained-aircraft-recovery`), and `_backward_full_grounded_threshold_margin`
exists there at line 6114.

However, the worktree handed to this implementer was checked out at `3a29b43` (tip of `main`, an
ancestor of the base) — a **stale/wrong ref**: that snapshot predates all P6 merges and its
`CLI/aircraft_diffusion_cfd.py` has no `_backward_full_grounded_threshold_margin` at all. `git status`
was clean. I moved the worktree to the stated base (`git reset --hard 39dc30e`), re-verified
`8f05365` is an ancestor and the target function is present, then implemented there. This is a
worktree-setup correction, not a Cap-0 absence — the gate condition (Cap-0 edits present) is satisfied
at the base the brief names.

## 2. The change (Candidate A, exact)

In `_backward_full_grounded_threshold_margin` (coordinate branch, `CLI/aircraft_diffusion_cfd.py`),
mirroring the already-merged `_backward_full_grounded_coordinate_loss` (single in-graph
`total_chunk_loss`, one `.backward()`):

1. **Before the loop:** `total_chunk_loss = flat_target.new_zeros(())` (added next to the existing
   `positive_sum` / `negative_sum` init).
2. **In the loop:** after building `chunk_loss` (per-chunk arithmetic byte-unchanged), accumulate
   `total_chunk_loss = total_chunk_loss + chunk_loss`.
3. **After the loop:** replaced the per-chunk
   `(scale * chunk_loss).backward(retain_graph=stop < total_voxels)` with a single
   `(scale * total_chunk_loss).backward()` (no `retain_graph`).

Net diff: 9 insertions / 5 deletions, all inside this one function.

### Candidate A step 1 was already done by Cap-0 (no-op)

The brief's step 1 (hoist the two in-loop `.item()` reads at ~6127/6136) does **not** apply to the
merged base: Cap-0's implementation commit `1182d1c` already replaced the margin function's per-chunk
`float(...item()) > 0.0` guards with device-side masks (`has_positive`/`has_negative`, clamped divisor
counts). `grep` over the function body confirms zero `.item()` reads remain in the margin loop — nothing
to hoist. The only `.item()` reads near this code are the telemetry-only
`int(positive_count.item())`/`int(negative_count.item())` reads in the coordinate-loss reporting dict
(2 syncs/update, left in place by Cap-0 by design). The functional intent of step 1 — zero device syncs
in the margin loop — is fully satisfied.

## 3. What was preserved byte-identical

- **Function name and signature** — pinned by
  `tests/test_constrained_recovery_review.py::test_coordinate_threshold_margin_backpropagates_all_chunks_after_data_backward`.
- **Data backward** `data_optimization_loss_val.backward(retain_graph=bool(self.geometry_threshold_calibrated))`
  (now line 6561–6563; the brief's "~6509" predates Cap-0's line shift). Untouched.
- **No `.grad` clear on margin-function entry** — it still accumulates onto `.grad` populated by the
  data backward, so `margin_gradient_delta = after − before` telemetry (~6589–6602, 6863) still isolates
  the margin contribution unchanged.
- **Detached `positive_sum` / `negative_sum` accumulation** and the **`detached_loss` return** — code
  bytes unchanged (one surrounding comment refreshed to reflect the single backward; no behavior change).
- **Dense branch** (`decoder_mode == "dense"`, `loss.backward()` once) — untouched.

## 4. Expected saving / engine-call count

The coordinate margin branch previously issued one `run_backward` engine call per coordinate chunk —
54 at 96³ production chunking — now **1**. That is the whole P6g prize: ~53 engine calls/update removed
from this function alone; combined with the already-merged collapses (coordinate-loss single backward,
data backward) this delivers the brief's "engine calls 60 → ~7" (cProfile `run_backward`), est. 6–10 s/u.
Static `.backward()` call sites in the file remain 10 (this was already a single static site; the win is
dynamic per-update).

Gradient equivalence: `(scale * total_chunk_loss).backward()` = `scale * Σ_chunks d(chunk_loss_i)/dθ`,
same sum the interleaved per-chunk backwards produced. The mirror function already uses this exact
pattern and is parity-green, so gradients are expected last-ulp (~1e-7 relative) different — well inside
the parity envelope (GRAD_ATOL 5e-4 / max(1e-3·grad_scale), LOSS_ATOL 5e-5, guard_dot ≥ −1e-8).

## 5. Peak VRAM

**Not measured** — CPU-only per instructions (GPU shared; controller runs the VRAM check). Memory
reasoning: the old code already kept the shared upstream latent graph alive across all 54 chunks via
`retain_graph=True`, and the new code holds the same graph to the single final backward, so the envelope
is comparable to the parity-green coordinate-loss function (which holds an identical in-graph total).
`coordinate_gradient_checkpointing` (enabled in training) keeps each chunk's decoder graph a small
checkpoint reference recomputed on backward, exactly as in the mirror.

## 6. Concerns

1. **Worktree was on the wrong ref.** It started at `3a29b43` (main tip, ancestor of the base), which
   predates Cap-0 and lacks the target function; I reset it to the brief's stated base `39dc30e` and
   verified Cap-0 is an ancestor before implementing. Worth a check on how the dispatch created this
   worktree's branch so future implementers don't silently land on a stale snapshot.
2. **Candidate A step 1 was a no-op** in the merged base (Cap-0 already removed the in-loop `.item()`
   reads). The dispatch note "two in-loop `.item()` reads at ~6127/6136 are still present" does not match
   the base; the intent (no device syncs in the loop) is satisfied.
3. GPU parity, the 96³ before/after gradient harness, the profile benchmark vs 58.34 s/u, engine-call
   count (60 → 7), and peak-VRAM (< 8 GB) are deferred to the controller, per the brief.
4. Candidate D (drop the measured-direct `.backward()` at ~6580) was **not** attempted — optional and
   explicitly low priority; its region was not needed for this change.
