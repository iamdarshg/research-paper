# Aircraft Structural Design CLI

This directory contains a proof-of-concept CLI for synthetic aircraft-like voxel generation plus CFD-informed scoring. The public surface today is the Click CLI in `aircraft_diffusion_cfd.py`; internal classes are still evolving and should not be treated as a stable SDK.

## Scope and Limits

- The training dataset is synthetic.
- The workflow is not validated for real aircraft design, certification, or manufacturing decisions.
- The CLI exposes only lightweight conditioning. `generate` accepts `--target-speed`, but the other `DesignSpec` weights are fixed inside the command.
- `batch-generate` uses a fixed `DesignSpec(target_speed=50.0)` and fixed `num_steps=4`.
- Several flags are exposed as on-switches only in the current implementation: `--enable-consistency`, `--enable-pipeline`, `--enable-checkpointing`, and `--use-marching-cubes`.
- `info` and `performance-benchmark` report compiled-in feature/status messages; they are not full runtime validation of every optimization path.

## Install

```bash
pip install -r requirements.txt
```

Python 3.9+ is expected. CUDA is optional, but practical training and generation are much slower on CPU.

## Current CLI Surface

```bash
python aircraft_diffusion_cfd.py --help
```

Commands exposed by the current file:

- `train`
- `generate`
- `batch-generate`
- `performance-benchmark`
- `info`

## Training

Basic run:

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --save-dir ./checkpoints_smoke
```

Current training options:

- `--num-epochs` default `100`
- `--batch-size` default `4`
- `--learning-rate` default `2e-4`
- `--latent-dim` default `16`
- `--precision` default `float32`
- `--disconnection-penalty` default `30.0`
- `--num-samples` default `500`
- `--resume-from` optional checkpoint path
- `--save-dir` default `./checkpoints`
- `--enable-consistency` exposed, currently defaults on
- `--enable-pipeline` exposed, currently defaults on
- `--enable-checkpointing` exposed, currently defaults on
- `--enable-compile` defaults off
- `--solver` default `D3Q27`

Checkpoint behavior in the current code:

- The final checkpoint is saved to `<save-dir>/final_optimized_model.pt`.
- Periodic checkpoints are currently written by the trainer as `checkpoint_optimized_grid<grid>_ep<epoch>.pt` relative to the working directory, not under `--save-dir`.

Resolution behavior in the current code:

- `D3Q27` training uses a base grid resolution of `16`.
- Any other solver string falls back to `32`.
- The current trainer runs one configured grid size; it does not execute the older staged `16 -> 24 -> 32` schedule described in stale docs.

## Generation

Generate one STL artifact from a checkpoint:

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints/final_optimized_model.pt \
  --output aircraft_optimized.stl \
  --target-speed 7.0 \
  --num-steps 4 \
  --solver D3Q27
```

Current generation options:

- `--checkpoint` required
- `--output` default `aircraft_optimized.stl`
- `--target-speed` default `7.0`
- `--num-steps` default `4`
- `--use-marching-cubes` exposed, currently defaults on
- `--solver` default `D3Q27`

What the command does today:

- Loads `OptimizedAircraftGenerator`
- Builds a `DesignSpec` with fixed weights `0.33 / 0.33 / 0.34`
- Generates a voxel grid
- Writes an STL
- Runs a final CFD analysis pass and prints drag/lift coefficients

## Batch Generation

```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint ./checkpoints/final_optimized_model.pt \
  --output-dir ./generations_optimized \
  --num-designs 3
```

Current batch options:

- `--checkpoint` required
- `--output-dir` default `./generations_optimized`
- `--num-designs` default `5`

Current limitations:

- Uses fixed `target_speed=50.0`
- Uses fixed `num_steps=4`
- Always writes filenames shaped like `aircraft_optimized_001.stl`

## Inspection Commands

Print environment and status information:

```bash
python aircraft_diffusion_cfd.py info
```

Print the benchmark/status summary:

```bash
python aircraft_diffusion_cfd.py performance-benchmark
```

These commands are useful as smoke checks because they are fast and do not require a checkpoint.

## Python API Note

The file still defines internal classes such as `DesignSpec`, `OptimizedDiffusionTrainer`, and `OptimizedAircraftGenerator`. They can be useful for local experiments, but the stable, documented interface for this directory is the CLI. If you want guided command recipes instead of direct imports, see `examples.py`.

## Troubleshooting

Checkpoint path missing:

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints/final_optimized_model.pt
```

If that path does not exist yet, train first or point `--checkpoint` at a real file.

Quick environment check:

```bash
python aircraft_diffusion_cfd.py info
python aircraft_diffusion_cfd.py --help
```

## Status

This is a research proof of concept with runnable CLI entry points, not a production aircraft-design tool.
