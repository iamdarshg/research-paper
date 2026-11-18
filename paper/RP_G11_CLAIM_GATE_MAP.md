# RP-G11 — abstract / introduction / conclusion claims mapped to gates (2026-11-18)

RP-G11 requires every claim in the abstract, introduction and conclusion to map to a **passed** gate, with blocked claims listed. This map was written against the paper text on `audit/rp-g10-paper-claims` (PR #52).

Gate verdicts come from the audit's gate table (PR #51 thread):
- RP-G0: PASS.
- RP-G1: not claimable (<200 held-out samples on the paper's model).
- RP-G2: FAIL.
- RP-G3: literal PASS on the audit's D3 arm only.
- RP-G4: FAIL; the oracle ceiling is below the bar.
- RP-G5: FAIL (model).
- RP-G6: literal PASS on audit arms, but a constant also passes.
- RP-G7: in progress (independent canonical OpenFOAM cases).
- RP-G8: open.
- RP-G9: open.
- RP-G10: PARTIAL.

"Re-derived" below means a command was run on 2026-11-18 from committed files only.

| # | Claim (where) | Evidence in repo | Re-derived? | Gate | Status |
|---|---|---|---|---|---|
| A1 | Five root causes of the earlier validity failure were isolated by measurement (abstract, intro 1, conclusion) | `paper/CLAIMS_EVIDENCE_MATRIX.md` C13, prose only; it says the 16³/32³/96³ checkpoints "are not present in the tree and were not re-run" | no | none (diagnostic) | **self-reported, no gate** |
| A2 | Procedural corpus gate admitted 47, 1 passed the screen (abstract) | C13 prose | no | none | **self-reported** |
| A3 | Retrained 32³ checkpoint: 9/9 held-out reconstructions, 45/45 generated samples pass (abstract, intro 2, conclusion) | checkpoint and arrays not committed (labelled in PR #52) | no | RP-G1 needs ≥200 held-out samples (45 < 200); RP-G6-type claim on this model unverifiable | **blocked: not a passed gate** |
| A4 | The 45 draws are near-identical (2 or 5 distinct arrays) (abstract, conclusion) | four companion files say 2, the paper said 5 | no | RP-G3 (≥5 distinct) | **unverifiable; direction (collapse) consistent with RP-G2/G5 FAIL** |
| A5 | Every emitted voxel is present in the training corpus (abstract, intro 2, conclusion) | no committed computation for the paper's model; audit arms give novelty 0.0 | direction only | RP-G2 FAIL | **consistent with a failed gate (claim is negative)** |
| A6 | A model-free constant array passes the same screen (abstract, intro 2, conclusion) | `scripts/recovery_corpus_claims_check.py` (PR #52) | **yes** | RP-G5 controls | **supported** |
| A7 | Novel, conditioned and diverse generation not demonstrated (abstract, conclusion) | RP-G2/G4 FAIL; RP-G4 oracle ceiling ρ 0.15 against a 0.5 bar | yes (audit) | RP-G2, RP-G4 | **supported (negative claim)** |
| A8 | Reference-area defect: a passing 0.25 % row is really 14.2 % (abstract, intro 3, conclusion) | `tests/test_reference_area_regression.py` | **yes, 9/9 pass** | RP-G7 area convention | **supported** |
| A9 | Corrected frozen corpus meets the 5 % gate on 1/16 sonicFoam, 0/14 simpleFoam rows (abstract, conclusion) | `scripts/recompute_corpus_consistent_reference_area.py` over committed `docs/benchmarks/*_20260613/` | **yes**: 2/16 → 1/16 and 0/14 → 0/14, once the script's hard-coded Windows paths were made repo-relative (this commit) | RP-G7 (≥2 of ≥3 canonical cases within 5 %) is a different test | **supported as a measurement; RP-G7 itself not passed by it** |
| A10 | The n = 12 condition-response study is non-significant (abstract, intro 3, conclusion) | `scripts/condition_response_stats_n12.py` reads `build/protocol_smoke/condition_validation_n12.json`, which is **not committed on any branch** | **no** | RP-G8 (n ≥ 12, CIs, nulls, pre-declared exclusions) open | **unverifiable** |
| A11 | Documented D3Q27 instability and same-seed run-to-run variation (conclusion) | not traced in this pass | no | RP-G9 | **unverified** |
| A12 | Aerodynamic, structural, manufacturing and performance claims blocked (abstract, conclusion) | FINAL_RUN_GATES | n/a | — | **supported (negative claim)** |
| A13 | Implementation exists: structured-conditioning plumbing, direct internal-solver term with SPSA gradients, sequential candidate optimizer (conclusion) | `CLI/advanced_lbm_solver.py` (SPSA probe path), `CLI/sequential_diagnostic_optimizer.py`, conditioning in `CLI/aircraft_diffusion_cfd.py` | code exists | none (implementation claim) | **supported as existence only** |
| A14 | Each root cause isolated on a single 8 GiB host with a recorded command (intro 1) | commands not committed for the missing checkpoints | no | RP-G9 | **self-reported** |
| A15 | Both the four-step and the teacher-DDIM sampler give 45/45 (conclusion) | same as A3 | no | — | **self-reported** |
| A16 | A claim-gating protocol separates measured from fixture-only results (intro 4) | `paper/FINAL_RUN_GATES.md`, `CLAIMS_EVIDENCE_MATRIX.md` | yes (files exist) | — | **supported** |

## Verdict: RP-G11 FAIL (as defined)

The paper's central positive result (A1–A4, A14, A15) maps to **no passed gate**. Since PR #52 those claims are honestly labelled self-reported. That makes the paper truthful, but it does not satisfy RP-G11. The claims that do stand (A6, A7, A8, A9, A12, A13, A16) are either negative/bounding claims or measurement-integrity findings.

**Blocked claims, and what would unblock each:**

| Claims | What would unblock them |
|---|---|
| A3 / A4 / A15 | Commit the recovery checkpoint plus generated arrays (`build/recovery_2026-11-05/`). Then RP-G1 needs ≥200 held-out samples, and RP-G3 needs ≥5 distinct arrays. |
| A1 / A2 / A14 | Commit the diagnostic commands and the outputs they produced. |
| A10 | Commit `condition_validation_n12.json` and pre-declared exclusions (RP-G8). |
| A11 | Commit the instability logs (RP-G9). |

Reproducibility fixes in this commit: `scripts/recompute_corpus_consistent_reference_area.py` and `scripts/condition_response_stats_n12.py` no longer hard-code `C:\Users\...` paths. Both default to repo-relative paths with environment overrides. The second now fails with a clear message naming the uncommitted input.
