# PR 41 — review reconciliation (13 items → commits → evidence)

Date: 2026-08-18
Branch: `codex/constrained-aircraft-recovery`
PR: https://github.com/iamdarshg/research-paper/pull/41
Review HEAD: `987a16c`

This is the evidence surface for the 13-item PR 41 review round. Each item maps
to a commit, the tests that verify it, and any dedicated evidence doc. Every
item was verified at review HEAD with the full suite at **512 passed /
2 skipped** and the solver/BFL parity gates green.

| # | Review item | Commit | Verification | Evidence doc |
|---|---|---|---|---|
| 1 | **TF32 scope** — pin the D3Q27 solver to IEEE fp32 (NN-only TF32) | `2e9293b` | `_ieee_fp32_math` decorator on the solver hot paths (advanced_lbm_solver.py:65, applied at :595/:1215); NN-side `allow_tf32` set in aircraft_diffusion_cfd.py:6004; solver bit-identity holds through the parity suite (a solver precision drift would break every parity gate) | — |
| 2 | **Effective Reynolds** — report realized `tau_actual` / Reynolds | `d675de0` | `tests/test_solver.py` (`test_*reynolds*`/`tau_actual`) | `docs/performance/2026-08-17-effective-reynolds-r2.md` |
| 3 | **FP64 gradient flattening** — per-tensor fp64 reductions, drop full-model flatten | `ed9dfb5` | `tests/test_multiobjective_gradients.py` (constrained-projection determinism, low-precision residuals, non-finite guards) | — |
| 4 | **Numerics flags in exact-resume fingerprint** | `3905072` | `tests/test_consistency_model.py`: `test_run_state_compatibility_configuration_uses_intersection_semantics`, `test_run_state_compatibility_reports_all_immutable_mismatches` (tf32 flag flips → incompatibility) | — |
| 5 | **Best-checkpoint persistence** through exact resume | `e5f369c` | `tests/test_training_stability.py`: `test_sync_best_checkpoint_state_mirrors_into_run_state_metadata`, `test_restore_best_promotion_rank_round_trips_and_falls_back` | — |
| 6 | **Stability/early-stop history** persisted through resume | `67d9b53` | `tests/test_training_stability.py`: `test_load_monitored_history_round_trips_and_is_defensive` | — |
| 7 | **Resumable deterministic shuffling** in place of `shuffle=False` | `99d0aeb` | `tests/test_training_stability.py`: `test_resumable_epoch_sampler_is_deterministic_per_epoch_and_resumable` | — |
| 8 | **Mission-adaptive CFD semantics** — solver objective is mission-independent | `6e46f8c` | `tests/test_cfd_solver_contract.py`; `tests/test_constrained_recovery_review.py` mission/design-spec guards | `docs/performance/2026-08-18-mission-adaptive-cfd-semantics-r8.md` |
| 9 | **Config source-of-truth** — CFDConfig honors `config.yaml` | `0276d4c` | `tests/test_config.py`; `tests/test_constrained_recovery_review.py::test_both_cfdconfig_classes_read_mach_number_from_config_yaml` | `docs/performance/2026-08-18-config-source-of-truth-r9.md` |
| 10 | **Checkpoint/log durability** — write-temp/fsync/atomic-replace | `f2ab840` | `tests/test_consistency_model.py::test_atomic_write_checkpoint_serializes_and_cleans_tmp`, `::test_atomic_write_checkpoint_failure_preserves_existing_target` (caught a real Windows read-handle fsync bug); `tests/test_smoke_pipeline.py::test_save_checkpoint_includes_cfd_config_payload` | `docs/performance/2026-08-18-durability-fsync-r10.md` |
| 11 | **Production-faithful benchmark** — s/u + VRAM at production config | `eb62055` | harness JSONs (gitignored `build/perf/baseline/profile_result_c{1,2,4}.json`); C=1 **27.5 s/u mean, 7.46 GiB reserved** | `docs/performance/2026-08-18-production-faithful-benchmark-r11.md` |
| 12 | **Cheap solver-memory** — per-direction q-algebra + cached BFL `max_count` | `987a16c` | 5 parity-gate files (37 passed / 2 skipped); isolated q-algebra peak 627→143 MiB; integrated peak 7,460→7,334 MiB; `torch.equal(old, new)` at 96³ | `docs/performance/2026-08-18-solver-memory-r12.md` |
| 13 | **Clean PR evidence surface** | this doc | full-suite 512/2 at `987a16c`; scratch under `build/` is gitignored; worktree-local RAM-check scripts and the 08-16 merge report are gitignored (never committed) | `docs/performance/2026-08-18-pr41-review-reconciliation.md` |

## Claim-bearing numbers locked by this review round

Measured on the production-faithful harness (real `step1305.pt` checkpoint, real
1069-geometry corpus, `--no-instrument`, TF32-NN-on, solver IEEE fp32) on
RTX 4060 Laptop 8 GB:

- **27.5 s/u** mean full optimizer update at 96³ / batch 1 (p90 ≈ 30 s/u),
  ~2.1× faster than the pre-review 62.66 s/u baseline.
- **7.33 GiB peak reserved** after R12 (down from 7.46 GiB pre-R12); the C=1
  configuration stays under the 8 GiB card with ~0.67 GiB headroom; C=4
  reproduces the WDDM spill boundary (8.4 GiB) that motivates the
  `_DIRECT_SOLVER_BATCH_CHUNK = 1` standing constraint.
- Solver numerics are **bit-identical** across R10/R12 durability and memory
  changes — no reproducibility claim in the paper changes from this round.

## Evidence-surface hygiene

- `build/` (probe scripts, harness JSONs) is gitignored — measurement artifacts
  are reproducible from the committed docs and harness, not committed binaries.
- Worktree-local scratch (`.ramcheck.ps1`, `.ramwatch.sh`, `MERGE-REPORT.md`)
  is gitignored at repo root so `git add -A` cannot sweep it into the PR; the
  merge it documents (`a58a2ec`, `f55f8c6`, `51e2ace`) is already in history.
