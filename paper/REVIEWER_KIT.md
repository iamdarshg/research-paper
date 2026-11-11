# REVIEWER_KIT — What a Reviewer Can Run, and Where Each Claim Lives

**Paper**: *Aircraft-Like Voxel-Geometry Generation with a CFD-Informed Evaluation Path: Measured Validity on a Declared Held-Out Split* (Gupta)
**Generator recovery**: 2026-11-05/06 · report: `evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`

This kit maps manuscript claims to evidence and lists the commands used to reproduce
them. The recovery and manuscript updates are working-tree changes based on repository
commit `6eaa290`; no git mutations were performed.

## 0. Environment (as measured on the authors' host)

- Windows, Python 3.12.2, torch 2.9.1+cu130, NVIDIA RTX 4060 Laptop 8 GB
- OpenFOAM: **not installed**. All OpenFOAM agreement numbers are frozen artifacts from
  `repos/research-paper-validation/docs/benchmarks/…20260613/` and are labeled frozen.
- No TeX toolchain installed; the PDF must be rebuilt with the commands in §5.

## 1. Reproduce the test suite (fast, CPU/GPU agnostic)

```bash
python -m pytest tests -q
```
Expected on the final recovery working tree: **675 passed, 4 skipped** (includes the 10
new tests in `tests/test_generator_recovery.py`; see
`evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md` §5.4).

## 2. Reproduce the corrected low-Mach agreement table (the headline science correction)

The paper's Table (frozen low-Mach corpus agreement) claims:
- sonicFoam: **1/16 rows < 5%** Cd error (only Mach 0.3 / of-32 / lbm-32 at 1.14%)
- simpleFoam: **0/14 rows < 5%**

To reproduce from the frozen corpus without OpenFOAM:

```bash
python scripts/recompute_corpus_consistent_reference_area.py
```
which reads the frozen `comparison_rows.csv` / `suite_rows.json` in
`repos/research-paper-validation/docs/benchmarks/`, renormalizes the LBM coefficient onto
the sweep reference area (`C_corr = C_raw × Aref_LBM / Aref_OF`, Aref_OF = 0.27734375),
and writes the corrected tables to `evidence/research-paper-publish/corrected_corpus/`.
Every row carries `lbm_reference_area`, `openfoam_reference_area`, and the convention
label. The old 0.25% row evaluates to **14.2%** after the fix.

## 3. Reproduce the condition-response statistics (the honest null result)

The paper's Table (condition-response correlations, smoke checkpoint, grid 32, 100
steps) claims n=12 non-significance:

```bash
python CLI/aircraft_diffusion_cfd.py validate-conditions \
  --checkpoint checkpoints_protocol_smoke/final_optimized_model.pt \
  --num-seeds 12 --grid-size 32 \
  --output build/protocol_smoke/condition_validation_n12.json
python scripts/condition_response_stats_n12.py
```
The digest reports r, p, and Fisher-z 95% CIs for both the raw n=12 set and the n=10 set
excluding two documented D3Q27 blow-up seeds, plus the run-to-run variation check against
the October n=4 artifact. Key expected values (as run 2026-11-01): target_speed↔drag
r=−0.198 (n=10, p=0.583); wingspan↔lift r=0.603 (p=0.065) — all non-significant.

## 4. Reproduce the validity screen

```bash
python scripts/agent_d_validity_harness.py
```
Expected: NASA CRM whole-aircraft voxels **15/15 pass**; bundled root STLs **0/4 fail**
with the failure modes listed in the paper.

## 5. Reproduce the held-out generator recovery

The new claim is narrow: the 32³ checkpoint emits screen-passing voxel geometry on a
pre-registered held-out CRM split, but collapses to one geometry across conditions and
seeds. The complete protocol and all artifacts are documented in
`evidence/research-paper-publish/GENERATOR_RECOVERY_2026-11-05.md`.

```bash
python scripts/recovery_generator_eval.py \
  --checkpoint build/recovery_2026-11-05/recovered_32cubed_dense/final_optimized_model.pt \
  --manifest build/recovery_2026-11-05/manifest_holdout.jsonl \
  --output build/recovery_2026-11-05/eval_after_dense.json \
  --threshold 0.5 --seeds 5
python scripts/recovery_materialize_samples.py \
  --checkpoint build/recovery_2026-11-05/recovered_32cubed_dense/final_optimized_model.pt \
  --manifest build/recovery_2026-11-05/manifest_holdout.jsonl \
  --output-dir build/recovery_2026-11-05/generated_voxels --threshold 0.5 --seeds 5
python CLI/aircraft_validity.py \
  --input-dir build/recovery_2026-11-05/generated_voxels \
  --output build/recovery_2026-11-05/aircraft_validity_generated.json
```

Expected: reconstruction 9/9, consistency generation 45/45, teacher-DDIM generation
45/45; the independent validity CLI re-screens 45/45. **Do not interpret 45/45 as
conditioned generation**: there are only 2 distinct arrays among 45, mean cross-condition
IoU 1.000.

## 6. Rebuild the PDF

On a TeX-enabled machine (not available in this environment):

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
The `.tex` files are the authoritative manuscript; the checked-in `research-paper.pdf` is
stale and must be regenerated.

## 7. Claim → evidence map

| Claim in manuscript | Evidence location | Reproduce with |
| --- | --- | --- |
| Proof-of-concept pipeline; code paths execute end-to-end | C01–C03, C09; 665-pass suite | §1 command |
| Reference-area-corrected CFD agreement (sonicFoam 1/16, simpleFoam 0/14 rows < 5%) | C11; corrected tables | §2 command |
| Condition-response: null at n=12, reported with CIs | C12; n=12 artifact + digest | §3 commands |
| Regime limits: low-Mach sanity only, no high-Mach/no-speedup/no-pipeline-loss claims | §regime_limits in results-and-discussion.tex; C11 | §2 + read the .tex |
| Validity screen is a heuristic; CRM 15/15, STLs 0/4 | C09; harness output | §4 command |
| Claims matrix and gate statuses current and fail-closed | `paper/CLAIMS_EVIDENCE_MATRIX.md`, `paper/FINAL_RUN_GATES.md`, `paper/SCIENTIFIC_GATE_STATUS.md` | read the files (all have 2026-11-01 addenda) |
| Issue disposition (#47/#39/#35/#32) | `evidence/research-paper-publish/PUBLISH_READY.md` §4 | read the file |

## 8. What a reviewer should NOT expect

- No freshly generated OpenFOAM results (not installed; frozen corpus only).
- No 128^3 / L4 evidence (issue #47; high-compute cloud job, out of scope here).
- No significant condition-response effect (the honest result is null at the smoke
  checkpoint).
- No claim of diverse or meaningfully conditioned generation: the recovery checkpoint
  emits one geometry across conditions/seeds, despite passing the geometric screen.
- No structural viability, aerodynamic superiority, or validated compressible/high-Mach
  behavior — those gates remain blocked per `FINAL_RUN_GATES.md`.