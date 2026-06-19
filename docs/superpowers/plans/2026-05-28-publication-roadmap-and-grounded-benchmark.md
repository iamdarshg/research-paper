# Publication Roadmap And Grounded Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the repo from an honest conditioned-aircraft proof of concept toward claim-bearing grounded evaluation by building a real corpus contract, executable manifest validation, grounded condition-response benchmarks, stronger validity gates, and a final evidence package.

**Architecture:** Keep the current synthetic/smoke workflow intact as the debug path, but add a parallel grounded path with explicit contracts and fail-closed evaluation. The plan treats the current CLI, protocol runner, and paper gate documents as the stable shell, then fills the missing scientific layers in order: data contract, validation tooling, condition-response benchmarks, aircraft validity checks, baselines, and final-run evidence.

**Tech Stack:** Python, Click CLI, PyTorch, NumPy, pytest, YAML/JSONL manifests, Markdown documentation, git, GitHub PR workflow

---

### Task 1: Lock The Grounded Corpus Contract

**Files:**
- Create: `docs/dataset/GROUNDED_CORPUS_SPEC.md`
- Create: `docs/dataset/manifest_schema.example.json`
- Modify: `docs/dataset/README.md`
- Modify: `CLI/conditioning_schema.yaml`
- Test: `tests/test_manifest_contract.py`

- [ ] Define the minimum claim-bearing manifest contract: geometry provenance, preprocessing version, split assignment, units, and required mission/manufacturing fields.
- [ ] Write one failing test that rejects missing provenance, missing split, or missing required condition fields.
- [ ] Add the smallest implementation needed to validate that contract against manifest records.
- [ ] Update dataset docs so the minimal checked-in manifest is explicitly separated from the future claim-bearing corpus contract.
- [ ] Commit with a corpus-contract message.

### Task 2: Add Executable Manifest Validation

**Files:**
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `CLI/validate_manifest.py`
- Modify: `CLI/run_protocol.py`
- Modify: `CLI/README.md`
- Test: `tests/test_manifest_validator.py`

- [ ] Add a standalone manifest-validation entry point that checks schema, path resolution, splits, and required condition coverage.
- [ ] Add one failing test for a manifest that mixes missing files, invalid split names, and malformed condition vectors.
- [ ] Make the validator produce a machine-readable report plus a human-readable summary.
- [ ] Wire the final protocol path so grounded final workflows can call the validator before claim-bearing runs.
- [ ] Commit with a manifest-validator message.

### Task 3: Build A Grounded Condition-Response Benchmark Harness

**Files:**
- Create: `docs/benchmarks/condition_response_benchmark.md`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `CLI/run_condition_benchmark.py`
- Modify: `CLI/run_protocols/final_cloud.yaml`
- Test: `tests/test_condition_benchmark.py`

- [ ] Define the benchmark contract: fixed A/B condition sweeps, expected directionality checks, seeds, metrics, and output schema.
- [ ] Write a failing test for benchmark report generation with deterministic seeds and required output keys.
- [ ] Implement a benchmark runner that consumes a checkpoint plus grounded manifest metadata and writes a structured report.
- [ ] Keep the benchmark honest by labeling it `blocked` whenever grounded samples or metrics are insufficient.
- [ ] Commit with a condition-benchmark message.

### Task 4: Add Aircraft-Specific Validity Gates

**Files:**
- Create: `docs/benchmarks/aircraft_validity_suite.md`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `CLI/aircraft_validity.py`
- Modify: `paper/FINAL_RUN_GATES.md`
- Test: `tests/test_aircraft_validity.py`

- [ ] Define the first aircraft-specific checks that go beyond connectivity: symmetry, span sanity, wing/body occupancy balance, and tail/body plausibility proxies.
- [ ] Write a failing test with one obviously invalid geometry and one minimally plausible synthetic control.
- [ ] Implement the validity checker as a reusable module and surface it through the CLI.
- [ ] Update final-run gates so any future aircraft-generation claim names this validity suite explicitly.
- [ ] Commit with an aircraft-validity message.

### Task 5: Strengthen Structural And Manufacturing Gatekeeping

**Files:**
- Modify: `CLI/conditioning_schema.yaml`
- Modify: `CLI/aircraft_diffusion_cfd.py`
- Create: `docs/benchmarks/manufacturing_constraints.md`
- Test: `tests/test_manufacturing_constraints.py`

- [ ] Formalize manufacturability checks that are stronger than raw connectivity: wall-thickness bounds, part-count bounds, and engine-count/package consistency.
- [ ] Write failing tests for impossible or self-contradictory condition combinations.
- [ ] Add fail-fast validation so impossible condition payloads are rejected before generation or benchmark runs.
- [ ] Document exactly which manufacturing/structural gates remain heuristic versus which are executable.
- [ ] Commit with a manufacturing-gates message.

### Task 6: Baselines And Statistics

**Files:**
- Modify: `CLI/baseline_config.yaml`
- Modify: `CLI/multi_seed_eval.py`
- Create: `docs/benchmarks/baseline_policy.md`
- Test: `tests/test_baseline_policy.py`

- [ ] Define the minimum baseline set for claim-bearing comparisons: bundled grounded STLs, retrieval baseline, unconditional checkpoint, and any procedural baseline the repo can execute.
- [ ] Write a failing test for a report that omits baseline names or seed counts.
- [ ] Extend report generation so multi-seed outputs include mean, std, and explicit `not enough seeds` blocking states.
- [ ] Update baseline docs so smoke reports and claim-bearing reports cannot be confused.
- [ ] Commit with a baseline-policy message.

### Task 7: Final Evidence Package And Paper Closure

**Files:**
- Modify: `paper/FINAL_RUN_GATES.md`
- Modify: `paper/CITATION_AUDIT.md`
- Modify: `paper/CLAIMS_EVIDENCE_MATRIX.md`
- Modify: `README.md`
- Modify: `CLI/README.md`
- Test: `python -m pytest -q`

- [ ] Add one final checklist artifact that maps each scientific claim to the exact report, benchmark, and corpus requirement needed to support it.
- [ ] Make every claim surface fail closed when any required report is missing.
- [ ] Re-run the full suite plus dry-run protocols after all previous tasks.
- [ ] Update the paper/docs wording only after the executable reports exist.
- [ ] Commit with a final-evidence-package message.

### Task 8: Immediate Next Slice

**Files:**
- Modify first: `docs/dataset/README.md`
- Create first: `docs/dataset/GROUNDED_CORPUS_SPEC.md`
- Create first: `tests/test_manifest_contract.py`
- Modify first: `CLI/aircraft_diffusion_cfd.py` or `CLI/validate_manifest.py`

- [ ] Start with Task 1 and Task 2 before touching new training logic.
- [ ] Do not broaden the model architecture until the grounded data contract and validation path are executable.
- [ ] Treat any additional training runs before those tasks complete as smoke-only development runs.

## Execution Order

1. Corpus contract
2. Manifest validation
3. Grounded condition benchmark
4. Aircraft validity suite
5. Manufacturing/structural gatekeeping
6. Baselines and statistics
7. Final evidence package

## Success Condition

The roadmap succeeds when the repo can distinguish, in executable code and versioned artifacts, between:

- smoke-only synthetic/debug workflows,
- grounded but non-claim-bearing evaluation,
- and the still-future claim-bearing conditioned-aircraft result path.
