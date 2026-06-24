# Aircraft Structural Design CLI

This directory contains a proof-of-concept CLI for synthetic aircraft-like voxel generation plus CFD-informed scoring. The public surface today is the Click CLI in `aircraft_diffusion_cfd.py`; internal classes are still evolving and should not be treated as a stable SDK.

## Scope and Limits

- The training dataset is synthetic.
- The workflow is not validated for real aircraft design, certification, or manufacturing decisions.
- The repo now has structured conditioning plumbing: dataset generation, the diffusion model, and the generator consume a documented 22-slot condition vector.
- The public CLI/config surface exposes the current documented conditioning fields. What is still missing is grounded condition-response evidence on aircraft-like data.
- `batch-generate` now records a manifest with the exact `DesignSpec` payload and condition vector for each STL. It still uses a fixed `num_steps=4`.
- The consistency/pipeline/checkpointing flags and marching-cubes toggle are paired enable/disable switches (for example `--enable-consistency/--disable-consistency` and `--use-marching-cubes/--no-marching-cubes`).
- `info` and `performance-benchmark` report compiled-in feature/status messages; they are smoke-oriented status checks, not full runtime validation of every optimization path.

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
- `evaluate-baselines`
- `validate-conditions`
- `condition-response-smoke`
- `densify-dataset`
- `performance-benchmark`
- `info`

Standalone helpers:

- `run_protocol.py`
- `validate_manifest.py`
- `run_condition_benchmark.py`
- `aircraft_validity.py`
- `final_evidence.py`
- `run_protocols/smoke_8gb.yaml`
- `run_protocols/final_cloud.yaml`
- `baseline_config.yaml`

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
- `--grid-size` optional explicit training/CFD voxel resolution override
- `--precision` default `float32`
- `--disconnection-penalty` default `30.0`
- `--num-samples` default `500`
- `--dataset-artifact` optional densified dataset artifact
- `--dataset-manifest` optional grounded dataset manifest (`.json`, `.jsonl`, `.yaml`)
- `--resume-from` optional checkpoint path
- `--save-dir` default `./checkpoints` (use smoke/final-specific paths to keep artifacts separated)
- `--run-class` `smoke` or `final`
- `--baseline-config` required for final runs
- `--claim-gates` required for final runs
- `--enable-consistency` / `--disable-consistency` (defaults on)
- `--enable-pipeline` / `--disable-pipeline` (defaults off; sequential evaluator execution is the safe default)
- `--enable-checkpointing` / `--disable-checkpointing` (defaults on)
- `--enable-compile` defaults off
- `--solver` default `D3Q27`

Checkpoint behavior in the current code:

- The final checkpoint is saved to `<save-dir>/final_optimized_model.pt`.
- Periodic checkpoints are saved under `<save-dir>/checkpoint_optimized_grid<grid>_ep<epoch>.pt`.

Resolution behavior in the current code:

- If `--grid-size` is provided, training uses that explicit resolution.
- Otherwise `D3Q27` training uses a base grid resolution of `16`.
- Any other solver string falls back to `32`.
- The current trainer runs one configured grid size; it does not execute the older staged `16 -> 24 -> 32` schedule described in stale docs.

Grounded-manifest behavior:

- `--dataset-manifest` lets training ingest a checked-in or external grounded corpus manifest.
- Each manifest record may provide `geometry_path` or `stl_path`.
- Each manifest record may optionally provide `design_spec`, `condition_vector`, `latent_path`, and `split`.
- Relative paths are resolved from the manifest file's directory.
- The checked-in minimal example is [`docs/dataset/minimal_grounded_manifest.jsonl`](../docs/dataset/minimal_grounded_manifest.jsonl).
- That checked-in manifest is wiring validation only, not a publication-grade aircraft corpus.

## Generation

Generate one STL artifact from a checkpoint:

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output aircraft_optimized.stl \
  --target-speed 7.0 \
  --num-steps 4 \
  --solver D3Q27
```

Current generation options:

- `--checkpoint` required
- `--output` default `aircraft_optimized.stl`
- `--target-speed` default `7.0`
- `--thrust-to-weight-min` default `0.45`
- `--turn-rate-min-deg-s` default `18.0`
- `--required-static-thrust-n` default `180.0`
- `--engine-diameter-mm` / `--engine-length-mm`
- `--engine-count-min` / `--engine-count-max`
- `--wingspan-limit-m`
- `--payload-mass-min-g` / `--payload-mass-max-g`
- `--takeoff-distance-min-m` / `--takeoff-distance-max-m`
- `--wall-thickness-min-mm` / `--wall-thickness-max-mm`
- `--part-count-min` / `--part-count-max`
- `--manufacturing-method`
- `--num-steps` default `4`
- `--use-marching-cubes` / `--no-marching-cubes` (defaults on)
- `--solver` default `D3Q27`

What the command does today:

- Loads `OptimizedAircraftGenerator`
- Builds a `DesignSpec`, converts it into the internal condition-vector format, and passes that vector into the generator path
- Generates a voxel grid
- Writes an STL
- Runs a final CFD analysis pass and prints drag/lift coefficients

Important nuance:

- The condition-vector seam is real code plumbing, not just a TODO. See `conditioning_schema.yaml` and `config.yaml`.
- The CLI/config surface exposes the documented condition fields, but the repo is not yet validated as a mission-conditioned or manufacturing-conditioned aircraft generator.
- `condition-response-smoke` writes a JSON report for directional smoke evidence only; it is not a scientific benchmark.
- No grounded condition-response benchmark currently demonstrates that changing payload, takeoff, wingspan, wall-thickness, part-count, or manufacturing inputs reliably changes generated outputs in the intended direction.

## Batch Generation

```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --output-dir ./generations_optimized \
  --num-designs 3
```

Current batch options:

- `--checkpoint` required
- `--output-dir` default `./generations_optimized`
- `--num-designs` default `5`
- `--seed` default `0`
- `--vary-conditions` to sample deterministic `DesignSpec` variation
- the same documented public conditioning fields exposed by `generate`

Current limitations:

- Uses fixed `num_steps=4`
- Writes filenames shaped like `aircraft_optimized_001.stl`
- Needs claim-bearing evaluation before batch outputs can be used as scientific evidence

## Baseline Evaluation

Establish performance baselines using grounded aircraft STLs:

```bash
python aircraft_diffusion_cfd.py evaluate-baselines \
  --grid-size 32 \
  --steps 500 \
  --output ./baseline_report.json
```

## Condition-Response Validation

Run a multi-seed condition-response sweep and compute Pearson correlations:

```bash
python aircraft_diffusion_cfd.py validate-conditions \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --num-seeds 20 \
  --output ./condition_validation.json
```

## Multi-Seed Evaluation Study

Automate aggregated performance studies across multiple seeds using the standalone script:

```bash
python multi_seed_eval.py \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt \
  --num-seeds 10 \
  --baseline-config ./baseline_config.yaml \
  --baseline-report ./baseline_report.json \
  --validation-report ./condition_validation.json \
  --output-dir ./eval_results \
  --output-report ./baseline_statistics.json
```

The script now emits the claim-gate statistics bundle consumed by `final_evidence.py`.

## Protocol Runner

Run the checked-in 8 GB smoke protocol:

```bash
python run_protocol.py --config run_protocols/smoke_8gb.yaml
```

Preview the guarded final-eval protocol without executing it:

```bash
python run_protocol.py --config run_protocols/final_cloud.yaml --dry-run
```

The final protocol is intentionally conservative: it starts with grounded-manifest validation, trains against explicitly named dataset/baseline inputs, emits `baseline_statistics.json`, and then re-validates the manifest before assembling the final evidence package so report lineage fields can agree on one checkpoint.

Validate a manifest directly:

```bash
python validate_manifest.py --manifest ../docs/dataset/minimal_grounded_manifest.jsonl --level basic
python validate_manifest.py --manifest ../docs/dataset/minimal_grounded_manifest.jsonl --level claim-bearing
python validate_manifest.py --manifest ../build/faa_geometry_case_corpus_20260624/geometry_case_manifest_5k.jsonl --level claim-bearing
```

Run the grounded condition-response gate directly:

```bash
python run_condition_benchmark.py \
  --checkpoint ../checkpoints_protocol_final/final_optimized_model.pt \
  --manifest ../build/faa_geometry_case_corpus_20260624/geometry_case_manifest_5k.jsonl \
  --output ../build/protocol_final/condition_benchmark.json
```

Evaluate the final evidence package:

```bash
python final_evidence.py \
  --manifest-validation ../build/protocol_final/manifest_validation.json \
  --aircraft-validity ../build/protocol_final/aircraft_validity.json \
  --condition-benchmark ../build/protocol_final/condition_benchmark.json \
  --manufacturing-constraints ../build/protocol_final/manufacturing_constraints.json \
  --baseline-statistics ../build/protocol_final/baseline_statistics.json
```

Missing, failed, or blocked reports keep claim-bearing wording blocked.

The protocol runner is the canonical, repeatable entry point for smoke and final runs. The checked-in configs keep smoke outputs (`checkpoints_protocol_smoke`, `build/protocol_smoke`) and final-eval outputs (`checkpoints_protocol_final`, `build/protocol_final`) in clearly separated paths.

## Condition-Response Smoke Benchmark

```bash
python aircraft_diffusion_cfd.py condition-response-smoke \
  --output ./build/condition_response_smoke.json
```

This command writes a small JSON report from the procedural conditioning path. Use it to confirm that materially different condition payloads produce measurable directional deltas in the smoke geometry proxies. Do not treat it as aircraft-level conditional validation.

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
They should not be cited as claim-bearing evidence for aerodynamic quality, production readiness, or conditioned generation performance.

## Python API Note

The file still defines internal classes such as `DesignSpec`, `OptimizedDiffusionTrainer`, and `OptimizedAircraftGenerator`. They can be useful for local experiments, but the stable, documented interface for this directory is the CLI. If you want guided command recipes instead of direct imports, see `examples.py`.

## Troubleshooting

Checkpoint path missing:

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints_smoke/final_optimized_model.pt
```

If that path does not exist yet, train first or point `--checkpoint` at a real file.

Quick environment check:

```bash
python aircraft_diffusion_cfd.py info
python aircraft_diffusion_cfd.py --help
```

## Status

This is a research proof of concept with runnable CLI entry points, not a production aircraft-design tool or a claim-bearing benchmark harness.
