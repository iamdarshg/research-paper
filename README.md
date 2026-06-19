# Aircraft Diffusion CFD - Research Repo

> Current status: a proof-of-concept research codebase for synthetic voxel generation, CFD-informed scoring, reproducible smoke checks, and structured conditioning plumbing. It is not yet a validated aircraft-design system or a scientifically supported mission/manufacturing-conditioned airplane generator.

## Overview

This repository combines a latent generative model, voxel decoding, internal lattice-Boltzmann-style scoring, and an OpenFOAM export path. The current experiments are intentionally narrow: they use synthetic training data and reduced sanity runs to validate the code path, not to establish publication-grade aerodynamic or structural performance.

## What Is True Today

- The model consumes structured conditions end to end through the dataset, model, and generator paths.
- The public CLI/config surface exposes the documented conditioning fields for propulsion, maneuverability, payload, takeoff, manufacturing, and geometry bounds.
- Condition-response and claim-gate tooling exists through `validate-conditions`, `condition-response-smoke`, `run_condition_benchmark.py`, `aircraft_validity.py`, `final_evidence.py`, `multi_seed_eval.py`, and the checked-in protocol runner.
- The repo does not yet provide scientific validation of conditioned aircraft generation on grounded aircraft-like data.

## Current Scope

- Proof-of-concept latent generation of freeform or aircraft-like voxel geometries
- Internal D3Q27/OpenFOAM benchmark path for solver cross-checks
- STL export and reproducible local validation tooling
- Small-scale training smoke runs on commodity hardware
- Checked-in smoke/final protocol scaffolding and a minimal manifest-backed grounded wiring path

## Not Yet Implemented At Claimable Quality

- Grounded condition-response evidence on an aircraft-like corpus
- A passing final evidence package that combines manifest, validity, condition-response, manufacturing, and baseline-statistics reports
- Real aircraft dataset training
- Structural validation beyond connectivity heuristics
- Publication-grade aerodynamic optimization claims

## Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd research-paper/CLI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify The CLI Loads

```bash
python aircraft_diffusion_cfd.py --help
python aircraft_diffusion_cfd.py info
```

### 3. Run A Small Training Smoke Test

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --save-dir ./checkpoints_smoke
```

### 4. Generate One STL

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output ./artifacts/smoke_design.stl \
  --target-speed 7.0 \
  --num-steps 4
```

### 5. Run The Internal Benchmark

```bash
python run_internal_benchmark.py
```

Treat that benchmark as a solver cross-check and implementation smoke test, not as a publication-grade aerodynamic benchmark.

### 6. Run Tests

For a normal Python environment:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

For this repo on Windows, if a Unix-style `.venv` was checked out from WSL or Linux:

```powershell
.\run_tests.ps1 -q
```

If you want the script to create a Windows venv first:

```powershell
.\run_tests.ps1 -BootstrapVenv -q
```

### 7. Run A Checked-In Protocol

Smoke path:

```bash
cd CLI
python run_protocol.py --config run_protocols/smoke_8gb.yaml
```

Guarded final path preview:

```bash
cd CLI
python run_protocol.py --config run_protocols/final_cloud.yaml --dry-run
```

Final claim-bearing wording remains blocked until the final evidence package passes:

```bash
python CLI/final_evidence.py
```

## Commands Reference

### `train`

Key arguments:

- `--num-epochs` default `100`
- `--batch-size` default `4`
- `--learning-rate` default `2e-4`
- `--latent-dim` default `16`
- `--precision` default `float32`
- `--disconnection-penalty` default `30.0`
- `--num-samples` default `500`
- `--dataset-artifact` / `--dataset-manifest` optional grounded or densified dataset inputs
- `--resume-from` optional checkpoint path
- `--save-dir` default `./checkpoints`
- `--run-class` `smoke` or `final`
- `--baseline-config`, `--claim-gates` required for final runs

### `generate`

Key arguments:

- `--checkpoint` required
- `--output` default `aircraft_optimized.stl`
- `--target-speed` default `7.0`
- `--thrust-to-weight-min`, `--turn-rate-min-deg-s`, `--required-static-thrust-n`
- `--engine-diameter-mm`, `--engine-length-mm`, `--engine-count-min`, `--engine-count-max`
- `--wingspan-limit-m`, payload bounds, takeoff bounds, wall-thickness bounds, part-count bounds, and `--manufacturing-method`
- `--num-steps` default `4`
- `--use-marching-cubes` exposed and defaults on in the current CLI

The generator path consumes the documented condition vector from [`CLI/conditioning_schema.yaml`](CLI/conditioning_schema.yaml). What is still missing is grounded scientific validation that those controls reliably steer aircraft-like outputs in the intended direction.

### `batch-generate`

Key arguments:

- `--checkpoint` required
- `--output-dir` output directory
- `--num-designs` number of STL files to emit
- `--seed` deterministic seed for manifest metadata
- `--vary-conditions` samples deterministic `DesignSpec` variation and records it in `batch_manifest.json`

### `evaluate-baselines`

Voxelizes and evaluates the bundled grounded STL examples. This is runnable repo-level baseline tooling, not publication-grade baseline evidence.

### `validate-conditions`

Runs a multi-seed condition-response sweep and writes correlation summaries for the current checkpoint. Treat the result as checkpoint-level evidence only, not grounded aircraft validation.

## Data Status

The current repo does not ship a publication-grade aircraft corpus. It has:

- a procedural/synthetic training path
- checked-in densification artifacts for smoke workflows
- a minimal manifest-backed grounded wiring artifact at [`docs/dataset/minimal_grounded_manifest.jsonl`](docs/dataset/minimal_grounded_manifest.jsonl)

That minimal manifest exists to validate the dataset wiring and protocol guardrails. It is not a scientifically adequate aircraft corpus.

## Reproducibility Files

- [`CLI/conditioning_schema.yaml`](CLI/conditioning_schema.yaml)
- [`CLI/baseline_config.yaml`](CLI/baseline_config.yaml)
- [`CLI/run_protocol.py`](CLI/run_protocol.py)
- [`CLI/run_protocols/smoke_8gb.yaml`](CLI/run_protocols/smoke_8gb.yaml)
- [`CLI/run_protocols/final_cloud.yaml`](CLI/run_protocols/final_cloud.yaml)
- [`paper/FINAL_RUN_GATES.md`](paper/FINAL_RUN_GATES.md)
- [`paper/CITATION_AUDIT.md`](paper/CITATION_AUDIT.md)
- [`paper/CLAIMS_EVIDENCE_MATRIX.md`](paper/CLAIMS_EVIDENCE_MATRIX.md)

## System Requirements

- GPU: NVIDIA CUDA-capable GPU with 8GB+ VRAM
- CPU: multi-core processor
- RAM: 16GB+ system RAM recommended
- Python: 3.9+

Key dependencies include PyTorch, NumPy, SciPy, scikit-image, TensorBoard, and `trimesh`.

## Bottom Line

This repo is release-ready as an honest proof of concept. It is not publication-ready evidence of conditioned aircraft generation.
