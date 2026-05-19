# PR 33 Hardening Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish PR #33 by resolving mechanical blockers, tightening stale docs/tests, adding honest scientific gates, and hardening the conditioning/densification smoke workflow without overstating scientific completion.

**Architecture:** Keep the existing conditioned smoke pipeline intact, but add explicit validation, metadata, run-class guardrails, and smoke-only condition-response reporting around it. Treat paper/docs gate files as the source of truth for what is implemented versus what remains scientifically blocked.

**Tech Stack:** Python, Click CLI, PyTorch, NumPy, pytest, Markdown documentation, git

---

### Task 1: Branch Sync And Mergeability

**Files:**
- Modify: branch history and any merge-conflicted files
- Test: `git status --short --branch`

- [ ] Sync `origin/main` into `codex/paper-docs-smoke-fixes`
- [ ] Resolve conflicts without dropping PR work or recent `main` changes
- [ ] Verify branch is clean and mergeable before continuing

### Task 2: Conditioning Validation And CLI Exposure

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Modify: `CLI/conditioning_schema.yaml`
- Modify: `tests/test_conditioning.py`
- Modify: `tests/test_cli.py`

- [ ] Add failing tests for invalid min/max bounds, non-positive physical values, and unknown manufacturing methods
- [ ] Expose more condition fields through `generate` and `batch-generate`
- [ ] Add config/CLI parsing paths that reject invalid combinations clearly
- [ ] Verify condition changes affect at least the procedural smoke path and metadata

### Task 3: Dataset Artifact, Densification, And Run-Class Guardrails

**Files:**
- Modify: `CLI/offline_densify.py`
- Modify: `CLI/rlvr_dataset_bootstrap.py`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `docs/dataset/README.md`
- Modify: `tests/test_rlvr_bootstrap.py`
- Modify: `tests/test_smoke_pipeline.py`

- [ ] Add artifact schema/version/provenance/split metadata and deterministic split generation
- [ ] Make densification reproducible end-to-end
- [ ] Implement bounded CFD reranking or fail loudly when unavailable
- [ ] Handle empty accepted artifacts explicitly and reject them in training/final-claim flows
- [ ] Add smoke/final run classes and refusal gates for claim-bearing runs

### Task 4: Docs, Paper Gates, And Overclaim Cleanup

**Files:**
- Modify: `docs/cheapest-viable-conditioned-generator/README.md`
- Modify: `paper/FINAL_RUN_GATES.md`
- Modify: `paper/CLAIMS_EVIDENCE_MATRIX.md`
- Modify: `paper/main.tex`
- Modify: `paper/sections/*.tex` as needed
- Modify: `README.md`
- Modify: `CLI/README.md`
- Modify: `test_fix.py`

- [ ] Replace stale phrasing about public conditioning support
- [ ] Align paper/gate files on “partial smoke-plumbing, not validated conditioning”
- [ ] Soften runtime/docs benchmark claims unless actually measured in-repo
- [ ] Ensure smoke workflows are never described as final evaluation

### Task 5: Verification And PR Summary

**Files:**
- Modify: any remaining touched files from previous tasks
- Test: `pytest -q tests test_fix.py`
- Test: `python -m pytest tests/test_conditioning.py tests/test_rlvr_bootstrap.py tests/test_smoke_pipeline.py tests/test_cli.py tests/test_benchmark.py -q`
- Test: `python CLI/aircraft_diffusion_cfd.py --help`
- Test: `python CLI/aircraft_diffusion_cfd.py train --num-epochs 1 --batch-size 1 --num-samples 4 --save-dir ./checkpoints_smoke`
- Test: `python CLI/aircraft_diffusion_cfd.py densify-dataset --output-artifact ./smoke_outputs/accepted.pt --num-samples 8 --grid-size 16 --latent-dim 16 --seed 0`

- [ ] Run the required verification commands fresh
- [ ] If a checkpoint exists, run conditioned generation with at least two materially different payloads
- [ ] Summarize what is mechanically complete versus still scientifically blocked
