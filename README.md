# Aircraft Diffusion CFD - Research Repo

> Current status: a proof-of-concept research codebase for synthetic voxel generation, CFD-informed scoring, reproducible smoke checks, and structured conditioning plumbing. It is not yet a validated aircraft-design system or a scientifically supported mission/manufacturing-conditioned airplane generator.

## Overview

This repository combines a latent generative model, voxel decoding, internal lattice-Boltzmann-style scoring, and an OpenFOAM export path. The current experiments are intentionally narrow: they use synthetic training data and reduced sanity runs to validate the code path, not to establish publication-grade aerodynamic or structural performance.

## What Is True Today

- The model consumes structured conditions end to end through the dataset, model, and generator paths.
- The public CLI/config surface exposes the documented conditioning fields for propulsion, maneuverability, payload, takeoff, manufacturing, and geometry bounds.
- Condition-response and claim-gate tooling exists through `validate-conditions`, `condition-response-smoke`, `run_condition_benchmark.py`, `aircraft_validity.py`, `final_evidence.py`, `multi_seed_eval.py`, and the checked-in protocol runner.
- The repo includes grounded-data wiring and aircraft-corpus artifacts for reproducibility checks, but it does not yet provide scientific validation of conditioned aircraft generation on a publication-grade aircraft corpus.

## Current Scope

- Proof-of-concept latent generation of freeform or aircraft-like voxel geometries
- Internal D3Q27/OpenFOAM benchmark path for solver cross-checks
- STL export and reproducible local validation tooling
- Small-scale training smoke runs on commodity hardware
- Checked-in smoke/final protocol scaffolding and manifest-backed grounded wiring paths

## Not Yet Implemented At Claimable Quality

- Passing final evidence package across all gates
- Real aircraft dataset training at publication scale
- Structural validation beyond connectivity and manufacturing heuristics
- Publication-grade aerodynamic optimization claims
- Validated evidence that each exposed condition reliably steers aircraft-like outputs in the intended direction

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

Expected output shows the PyTorch version, CUDA availability, GPU memory when CUDA is present, and a smoke-status summary. It is not a measured benchmark.

### 3. Run A Small Training Smoke Test

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --save-dir ./checkpoints_smoke
```

Use larger training settings only after the smoke path works on your machine.

### 4. Generate One STL

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output ./artifacts/smoke_design.stl \
  --target-speed 7.0 \
  --num-steps 4
```

### 5. Batch Generate Optional Designs

```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output-dir ./artifacts/designs \
  --num-designs 5
```

### 6. Print Runtime Status

```bash
python aircraft_diffusion_cfd.py performance-benchmark
```

This command reports compiled-in smoke-run status. Do not treat it as a measured speed, memory, accuracy, or aerodynamic benchmark.

### 7. Run Tests

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

### 8. Run A Checked-In Protocol

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

The checked-in protocol configs are the canonical, repeatable entry points for smoke and final runs. They keep smoke artifacts (`checkpoints_protocol_smoke`, `build/protocol_smoke`) and final-eval artifacts (`checkpoints_protocol_final`, `build/protocol_final`) in separate paths.

Final claim-bearing wording remains blocked until the final evidence package passes:

```bash
python CLI/final_evidence.py
```

## Commands Reference

### `train`

Key arguments:

- `--num-epochs` default `100`
- `--batch-size` default `4`
- `--learning-rate` default `2e-5`
- `--latent-dim` default `16`
- `--precision` default `float32`
- `--disconnection-penalty` default `30.0`
- `--num-samples` default `500`
- `--dataset-artifact` / `--dataset-manifest` optional grounded or densified dataset inputs
- `--resume-from` optional checkpoint path
- `--save-dir` default `./checkpoints`
- `--run-class` `smoke` or `final`
- `--baseline-config`, `--claim-gates` required for final runs
- `--enable-consistency` / `--disable-consistency`
- `--enable-pipeline` / `--disable-pipeline`
- `--enable-checkpointing` / `--disable-checkpointing`
- `--enable-compile`

### `generate`

Key arguments:

- `--checkpoint` required
- `--output` default `aircraft_optimized.stl`
- `--target-speed` default `7.0`
- `--thrust-to-weight-min`, `--turn-rate-min-deg-s`, `--required-static-thrust-n`
- `--engine-diameter-mm`, `--engine-length-mm`, `--engine-count-min`, `--engine-count-max`
- `--wingspan-limit-m`, payload bounds, takeoff bounds, wall-thickness bounds, part-count bounds, and `--manufacturing-method`
- `--num-steps` default `4`
- `--use-marching-cubes` / `--no-marching-cubes`

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

### `performance-benchmark`
Prints the smoke-run feature status summary. It is intentionally phrased as status output, not a benchmark claim.

<!-- MODEL_CAPACITY_START -->
### Configured Model Capacity

Generated by `python CLI/update_model_capacity_report.py` from [`CLI/config.yaml`](CLI/config.yaml).

- Config digest: `53aedcfc58c9`
- Capacity basis: `8,027` configured unique geometries (planning input, not a packaged-corpus claim)
- Lattice / latent width: `128^3` / `512`
- Learning rates (diffusion / converter / consistency student): `2e-05` / `2e-05` / `2e-05`
- Scaled channels: `[128, 176, 224]`; coordinate decoder: `3328` wide x `12` residual blocks
- Trainable parameters: `294,719,529` (diffusion `20,819,824`, consistency student `6,165,432`, converter `267,734,273`)
- Trainable FP32 weight storage: `1,124.3 MiB`
- Resident training-model FP32 parameter storage, including teacher and EMA: `1,283.1 MiB`
- Static FP32 training-memory lower bound with gradients and AdamW states: `4,655.9 MiB`
- One dense scalar grid / coordinate grid / one D3Q27 population field: `8.0 MiB` / `24.0 MiB` / `216.0 MiB`

The memory values are deterministic storage calculations, not measured peak VRAM. Activations, allocator overhead, solver scratch arrays, and runtime libraries increase peak usage; measured runs remain the authority for hardware sizing.
<!-- MODEL_CAPACITY_END -->

## Data Status

The current repo does not ship a publication-grade aircraft training corpus. It has:

- a procedural/synthetic training path
- checked-in densification artifacts for smoke workflows
- a minimal manifest-backed grounded wiring artifact at [`docs/dataset/minimal_grounded_manifest.jsonl`](docs/dataset/minimal_grounded_manifest.jsonl)
- public-source whole-aircraft corpus builders and generated Airshow/NASA geometry manifests
- a generated 5,000-record FAA/OpenSky geometry-case manifest used by the guarded final protocol

Those artifacts validate dataset wiring, provenance, and protocol guardrails. They do not by themselves prove aircraft-design performance.

## Reproducibility Files

- [`CLI/conditioning_schema.yaml`](CLI/conditioning_schema.yaml)
- [`CLI/baseline_config.yaml`](CLI/baseline_config.yaml)
- [`CLI/run_protocol.py`](CLI/run_protocol.py)
- [`CLI/run_protocols/smoke_8gb.yaml`](CLI/run_protocols/smoke_8gb.yaml)
- [`CLI/run_protocols/final_cloud.yaml`](CLI/run_protocols/final_cloud.yaml)
- [`docs/dataset/minimal_grounded_manifest.jsonl`](docs/dataset/minimal_grounded_manifest.jsonl)
- [`docs/dataset/nasa_crm_whole_aircraft_manifest.jsonl`](docs/dataset/nasa_crm_whole_aircraft_manifest.jsonl)
- [`docs/dataset/faa_opensky_flight_regime_corpus_20260623.md`](docs/dataset/faa_opensky_flight_regime_corpus_20260623.md)
- [`paper/FINAL_RUN_GATES.md`](paper/FINAL_RUN_GATES.md)
- [`paper/CITATION_AUDIT.md`](paper/CITATION_AUDIT.md)
- [`paper/CLAIMS_EVIDENCE_MATRIX.md`](paper/CLAIMS_EVIDENCE_MATRIX.md)

## Historical Schedule Notes

Earlier experiments used progressive 16^3, 24^3, and 32^3 grids and informal RTX 3090 timing estimates. Those notes are historical context only. The current CLI distinguishes smoke runs from guarded final runs, and the current trainer executes one configured grid size rather than an automatic progressive schedule.

## System Requirements

- GPU: NVIDIA CUDA-capable GPU with 8GB+ VRAM recommended
- CPU: multi-core processor
- RAM: 16GB+ system RAM recommended
- Python: 3.9+

Key dependencies include PyTorch, NumPy, SciPy, scikit-image, TensorBoard, and `trimesh`.

## Bottom Line

This repo is release-ready as an honest proof of concept. It is not publication-ready evidence of conditioned aircraft generation.
