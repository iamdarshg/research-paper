# Paper, Documentation, and LLM Wrap-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three open paper-blocking GitHub issues, document the real training/runtime limits of the repo, and either produce an honest final smoke run or downgrade the project to a cheapest-viable next-step plan.

**Architecture:** The work splits into three tracks: paper claim control, scripting/runtime stabilization, and training feasibility. The paper track adds a citation audit, a final-run gate document, and rewrites the introduction/related work/methodology/results/conclusion so claims match evidence. The runtime track fixes local execution hazards and aligns the README/docs with what the code can actually do. The training track decides whether a final run on the available 8 GB GPU is meaningful; if not, it records the cheapest viable conditioned-generator path instead of overstating capability.

**Tech Stack:** Python 3.12, PyTorch, Click CLI, pytest, LaTeX (Tectonic), GitHub issues/docs, Markdown.

---

### Task 1: Map open issues to exact repo artifacts

**Files:**
- Create: `docs/superpowers/plans/2026-05-17-paper-docs-and-llm-wrapup.md`
- Modify: `paper/LATEX_TODO.md`
- Modify: `README.md`

- [ ] **Step 1: Record the open-issue scope**

Use the GitHub issue list for:
- `#24 Literature: make novelty claims precise and defensible`
- `#25 Literature: perform claim-by-claim citation audit before final submission`
- `#26 Literature: gate final runs against claims and prior work`

Write a short local mapping in the README and paper TODOs that these issues are the paper finish gate.

- [ ] **Step 2: Verify the current repo surface**

Run:
```powershell
git status --short --branch
rg --files
rg -n "TODO|future|optimized|structurally viable|CFD|manufactur|flight profile|conditioning" README.md CLI paper
```

Expected:
- We can point to exact overclaim locations before editing.

- [ ] **Step 3: Update the paper TODO checklist**

Replace generic TODO items with issue-driven items:
- citation audit,
- positioning/novelty subsection,
- final-run gate checklist,
- Windows-safe CLI verification,
- conditioned-generator feasibility note.

### Task 2: Create the paper audit and claim gate docs

**Files:**
- Create: `paper/CITATION_AUDIT.md`
- Create: `paper/FINAL_RUN_GATES.md`
- Modify: `paper/VALIDATION_TODO.md`

- [ ] **Step 1: Create a paragraph-by-paragraph citation audit**

For each paragraph in:
- `paper/sections/introduction.tex`
- `paper/sections/related-work.tex`
- `paper/sections/methodology.tex`
- `paper/sections/results-and-discussion.tex`

record one of:
- `OK`
- `prior-work citation needed`
- `repo evidence needed`
- `overclaim / soften`
- `baseline needed`
- `definition needed`

- [ ] **Step 2: Create the final-run gating checklist**

Document, at minimum:
- the claim being tested,
- required baseline,
- required metric,
- minimum evidence threshold,
- downgrade wording if the gate fails.

Include gates for:
- aircraft-specific validity,
- aerodynamic optimization,
- structural viability,
- CFD-guided training,
- prior-approach comparison,
- publication-quality validation.

- [ ] **Step 3: Convert speculative validation TODOs into scoped future work**

Move any unrealistic near-term validation promises into clearly labeled future work in `paper/VALIDATION_TODO.md`.

### Task 3: Rewrite the paper so claims match evidence

**Files:**
- Modify: `paper/main.tex`
- Modify: `paper/sections/introduction.tex`
- Modify: `paper/sections/related-work.tex`
- Modify: `paper/sections/methodology.tex`
- Modify: `paper/sections/results-and-discussion.tex`
- Modify: `paper/sections/conclusion.tex`
- Modify: `paper/references.bib`

- [ ] **Step 1: Rewrite the introduction contribution list**

Replace broad claims with testable statements such as:
- implementation of a proof-of-concept latent generative pipeline for voxelized freeform aircraft-like geometry,
- integration point for CFD-derived scoring in the codebase,
- reproducible sanity benchmark and validation workflow.

- [ ] **Step 2: Add a `Positioning and Novelty` subsection**

Place it near the end of `related-work.tex` and make it explicit that:
- the repo reuses standard diffusion/UNet/LBM ideas,
- the present novelty is the assembled proof-of-concept pipeline and validation discipline,
- aircraft-specific conditioned generation and publication-grade validation remain future work.

- [ ] **Step 3: Remove unsupported “already done” wording**

Fix phrases that imply:
- strong aerodynamic optimization,
- structural viability,
- differentiable CFD training already demonstrated,
- full aircraft generation from mission/manufacturing constraints.

- [ ] **Step 4: Expand the bibliography only where needed**

Add only the citations required to support the revised text and make sure keys compile.

### Task 4: Stabilize the scripting/runtime path

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `README.md`
- Modify: `CLI/README.md`
- Modify: `CLI/ARCHITECTURE.md`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Fix Windows console execution hazards**

Run:
```powershell
python CLI\aircraft_diffusion_cfd.py info
```

If it fails due to console encoding or similar startup issues, patch the CLI entry path so:
- `info`,
- `train`,
- `generate`

start cleanly on Windows with the default terminal encoding.

- [ ] **Step 2: Align docs with actual conditioning support**

Document that the current generator only uses:
- `target_speed`
- simple scalar weights

and does **not** yet support:
- full flight profile conditioning,
- manufacturing-method conditioning,
- real aircraft corpus training.

- [ ] **Step 3: Keep the architecture doc honest**

Move still-unfinished optimizations from “implemented” language to “roadmap” language where needed.

### Task 5: Open and seed new training issues

**Files:**
- Create: GitHub issues in `iamdarshg/research-paper`
- Create: `docs/cheapest-viable-conditioned-generator/README.md`

- [ ] **Step 1: Search for duplicates**

Search open GitHub issues for:
- conditioning,
- manufacturing,
- dataset,
- 8 GB GPU,
- training protocol.

- [ ] **Step 2: Open missing blocker issues**

If missing, open issues for:
- conditioned generation schema and model support,
- real aircraft dataset and supervision plan,
- 8 GB final-run boundary / smoke-run protocol.

- [ ] **Step 3: Start the work locally**

Seed the cheapest viable plan README with:
- the minimum data needed,
- the minimum model changes needed,
- the cheapest training path that can run on an 8 GB laptop GPU,
- what cannot be claimed until that work exists.

### Task 6: Verify, then decide whether a final training run is meaningful

**Files:**
- Modify: `docs/cheapest-viable-conditioned-generator/README.md`
- Modify: `README.md`

- [ ] **Step 1: Run the reproducibility checks**

Run:
```powershell
$env:PYTHONPATH='CLI'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
pytest -q tests test_fix.py
```

Expected:
- tests pass,
- any warnings are documented and triaged.

- [ ] **Step 2: Compile the paper**

Run the bundled Tectonic binary against `paper/main.tex` with an explicit output directory.

Expected:
- PDF builds,
- undefined citations or section-reference failures are fixed before close-out.

- [ ] **Step 3: Decide on the final training run**

Only run a final training command if all of the following are true:
- CLI startup works,
- the run is honest for current claims,
- the hardware can complete it in a bounded time,
- the result is meaningful enough to report.

If those conditions fail, do **not** fake a “final model” result. Instead update the cheapest-viable plan README and report the exact blocker.

- [ ] **Step 4: If training is still run, keep it bounded**

Use a smoke-run command with explicit small settings such as:
```powershell
$env:PYTHONUTF8='1'
python CLI\aircraft_diffusion_cfd.py train --num-epochs 1 --batch-size 1 --num-samples 2 --save-dir .\checkpoints_smoke
```

Report it only as:
- a code-path validation run,
- not a publication-grade or conditioned-aircraft result.

### Task 7: Finish the branch honestly

**Files:**
- Modify: working tree as produced by prior tasks

- [ ] **Step 1: Re-run verification before close-out**

Run:
```powershell
$env:PYTHONPATH='CLI'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
pytest -q tests test_fix.py
```

and recompile the paper.

- [ ] **Step 2: Summarize by decision, not by churn**

The final report should state:
- which GitHub issues are now addressed by local edits,
- whether new training issues were opened,
- whether a final training run was attempted,
- whether the repo can currently support a fully AI-driven conditioned airplane generator,
- the cheapest viable next plan if not.
