# CLI Documentation Index

This directory exposes a proof-of-concept command-line interface for synthetic aircraft-like voxel generation and CFD-informed scoring. The docs in this folder now track the current public CLI surface in `aircraft_diffusion_cfd.py`.

## Start Here

- `README.md`: full CLI reference, current behavior notes, and proof-of-concept limits
- `QUICKSTART.md`: fastest path to a smoke run
- `examples.py`: runnable command recipes plus clearly labeled pseudocode where the CLI does not expose a workflow directly

## Current Commands

```bash
python aircraft_diffusion_cfd.py --help
```

The current commands are:

- `train`
- `generate`
- `batch-generate`
- `condition-response-smoke`
- `densify-dataset`
- `performance-benchmark`
- `info`

## What These Docs Deliberately Do Not Claim

- No production-readiness claim
- No validated aircraft-design claim
- No staged `16 -> 24 -> 32` training claim
- No promise that the internal Python classes are a stable SDK

## Practical File Map

- `aircraft_diffusion_cfd.py`: main CLI and implementation
- `ARCHITECTURE.md`: deeper implementation notes
- `requirements.txt`: Python dependencies
- `config.yaml`: configuration artifact kept in the repo, but not the main public entry point for the current CLI

## Quick Reference

Smoke-check the environment:

```bash
python aircraft_diffusion_cfd.py info
python aircraft_diffusion_cfd.py performance-benchmark
```

Train a tiny checkpoint:

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --save-dir ./checkpoints_smoke
```

Generate from the final checkpoint:

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output ./artifacts/smoke_design.stl
```

## Checkpoint Naming

Use `final_optimized_model.pt` when referring to the final artifact created by the current `train` command.
