# Quick Start

This guide focuses on the CLI exactly as it exists today. It is meant for smoke runs and orientation, not for claiming real aircraft-design quality.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Confirm the CLI Loads

```bash
python aircraft_diffusion_cfd.py --help
python aircraft_diffusion_cfd.py info
```

If those commands work, the current Click entry points and the main imports are available in your environment.

## 3. Optional Fast Status Check

```bash
python aircraft_diffusion_cfd.py performance-benchmark
```

This prints the proof-of-concept benchmark/status summary. It does not train a model.

## 4. Run a Small Training Smoke Test

For repeatable smoke/final workflows, prefer the protocol runner (`run_protocol.py`) with the checked-in YAML configs. The manual steps below are for quick experimentation only.

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --save-dir ./checkpoints_smoke
```

What to expect from the current implementation:

- The final checkpoint should be written to `./checkpoints_smoke/final_optimized_model.pt`.
- The trainer may also emit periodic files named like `checkpoint_optimized_grid16_ep5.pt` relative to the working directory when the epoch count reaches the save interval.
- With the default solver `D3Q27`, the current CLI chooses a base grid resolution of `16`.

## 5. Generate One STL from That Checkpoint

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output ./artifacts/smoke_design.stl \
  --target-speed 7.0 \
  --num-steps 4
```

What this command does today:

- Loads the optimized checkpoint
- Generates a voxel design with the consistency-model path
- Exports an STL
- Runs a final CFD analysis pass and prints drag/lift coefficients

The public CLI now exposes a partial structured conditioning subset here, including speed, thrust-to-weight, turn-rate, static thrust, engine geometry, engine count, wingspan, payload bounds, takeoff bounds, wall-thickness bounds, part-count bounds, and manufacturing method. It still does not make the workflow scientifically validated conditioned generation.

## 6. Generate a Small Batch

```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output-dir ./batch_outputs \
  --num-designs 2
```

Current behavior to know up front:

- `batch-generate` uses a fixed `num_steps=4`
- `batch-generate` can reuse one condition payload or deterministically vary conditions with `--vary-conditions`
- `batch-generate` writes `batch_manifest.json` with the exact condition payload for every STL
- Output files are named like `aircraft_optimized_001.stl`

## 7. Optional Condition-Response Smoke Check

```bash
python aircraft_diffusion_cfd.py condition-response-smoke \
  --output ./build/condition_response_smoke.json
```

This writes a small JSON summary for the procedural conditioning path. It is useful for smoke-level directional checks only.

## 8. Useful Option Reference

`train`

- `--num-epochs` default `100`
- `--batch-size` default `4`
- `--learning-rate` default `2e-4`
- `--latent-dim` default `16`
- `--precision` default `float32`
- `--disconnection-penalty` default `30.0`
- `--num-samples` default `500`
- `--resume-from` optional
- `--save-dir` default `./checkpoints`
- `--enable-compile` default off
- `--solver` default `D3Q27`

`generate`

- `--checkpoint` required
- `--output` default `aircraft_optimized.stl`
- `--target-speed` default `7.0`
- `--thrust-to-weight-min` default `0.45`
- `--turn-rate-min-deg-s` default `18.0`
- propulsion, payload, takeoff, wall-thickness, part-count, and manufacturing options are also exposed
- `--num-steps` default `4`
- `--solver` default `D3Q27`

`batch-generate`

- `--checkpoint` required
- `--output-dir` default `./generations_optimized`
- `--num-designs` default `5`
- `--seed` default `0`
- `--vary-conditions` optional deterministic variation

## 9. Current Caveats

- This is a synthetic-data research workflow.
- The docs intentionally do not promise production readiness.
- The current trainer runs one configured grid size; it does not run the older staged `16 -> 24 -> 32` schedule.
- The CLI exposes paired flags: `--enable-consistency/--disable-consistency`, `--enable-pipeline/--disable-pipeline`, `--enable-checkpointing/--disable-checkpointing`, and `--use-marching-cubes/--no-marching-cubes`.
- Smoke outputs and smoke condition-response summaries are not claim-bearing evaluation artifacts.

## 10. If Something Fails

Check the public help text again:

```bash
python aircraft_diffusion_cfd.py train --help
python aircraft_diffusion_cfd.py generate --help
python aircraft_diffusion_cfd.py batch-generate --help
```

If generation says the checkpoint is missing, point `--checkpoint` at a real `final_optimized_model.pt` or train a new smoke checkpoint first.
