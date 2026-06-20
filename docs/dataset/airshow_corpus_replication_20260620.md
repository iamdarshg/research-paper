# Airshow Corpus Replication Document, 2026-06-20

This document records the commands, expected outputs, hashes, and claim
boundaries for reproducing the public VSP Airshow corpus smoke run used in the
paper. Commands are written for PowerShell from the repository root:

`D:\CodeProjects\research-paper`

## Requirements

- Network access to the public VSP Airshow web app and public storage URLs.
- Python environment with the repository dependencies installed.
- Optional: enough local time to rerun the full test suite after the corpus
  artifacts are rebuilt.

The public web source may change over time. If Airshow records or public
storage URLs change after 2026-06-20, exact counts may differ. The reported
paper run is anchored by the manifest hash below.

## Step 1: Build the Corpus

```powershell
python CLI\build_airshow_corpus.py --output-dir build\airshow_grounded_corpus_20260620 --grid-size 16 --allowed-licenses 1 2 3
```

Expected paper-run summary:

- public documents observed: 381
- eligible after license and geometry filtering: 357
- converted records: 355
- failures: 2 stale public storage 404s
- split counts: train 250, val 31, test 38, holdout 36
- licenses: 208 CC0, 18 CC BY, 129 CC BY-SA

Expected manifest:

`build\airshow_grounded_corpus_20260620\manifest.jsonl`

Expected manifest SHA-256:

`7bb59bab9cc8ed3a836377a35d3c38d5c0086a56b617b2695e131486451885a6`

## Step 2: Validate the Manifest

```powershell
python CLI\validate_manifest.py --manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --level claim-bearing
```

Expected paper-run result:

- validation status: pass
- record count: 355
- manifest hash:
  `7bb59bab9cc8ed3a836377a35d3c38d5c0086a56b617b2695e131486451885a6`

## Step 3: Train the Smoke Checkpoint

```powershell
python CLI\aircraft_diffusion_cfd.py train --num-epochs 3 --batch-size 8 --dataset-manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --grid-size 16 --latent-dim 16 --save-dir build\airshow_training_20260620\checkpoints --disable-pipeline --disable-checkpointing --solver D3Q27
```

Expected checkpoint:

`build\airshow_training_20260620\checkpoints\final_optimized_model.pt`

Expected checkpoint SHA-256:

`71e808aa3c35142f145da267bb4eb7050300adc383e3a070d571eec36413d4f6`

Expected final epoch metrics:

- loss: 21.5905
- MSE: 0.7997
- geometry-reconstruction loss: 0.0778
- consistency loss: 0.00109
- connectivity diagnostic: 0.00149
- aerodynamic diagnostic: 20.7104
- global step: 135

These are smoke-run diagnostics. A later loss-semantics audit showed that the
connectivity and aerodynamic values are detached diagnostics, not differentiable
solver training signals. They should not be interpreted as convergence,
generalization, or CFD-guided gradient evidence.

## Step 4: Run Three Flight-Path Smoke Checks

```powershell
python CLI\run_airshow_flight_path_tests.py --checkpoint build\airshow_training_20260620\checkpoints\final_optimized_model.pt --manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --output-dir build\airshow_training_20260620\flight_path_tests --grid-size 16 --num-steps 4 --cfd-steps 100
```

Expected report:

`build\airshow_training_20260620\flight_path_tests\flight_path_results.json`

Expected case outcomes:

| Case | Occupancy | D3Q27 Cd | D3Q27 L/D | Validity |
| --- | ---: | ---: | ---: | --- |
| `short_takeoff_payload` | 0.014160 | 1.204657 | 0.004383 | fail: `span_sanity` |
| `high_speed_sprint` | 0.012939 | 0.912229 | 0.030508 | fail: `span_sanity` |
| `endurance_turning` | 0.012451 | 0.982371 | 0.046629 | fail: `span_sanity` |

Expected claim labels:

- `claim_bearing_cfd=false`
- `label_tier=lbm_raw`
- `lbm_converged=false`

## Step 5: Render Manuscript Figures

```powershell
python CLI\render_airshow_figures.py --corpus-report build\airshow_grounded_corpus_20260620\corpus_report.json --flight-report build\airshow_training_20260620\flight_path_tests\flight_path_results.json --output-dir paper\figures
```

Expected outputs:

- `paper\figures\airshow_corpus_summary.png`
- `paper\figures\airshow_training_losses.png`
- `paper\figures\airshow_flight_path_metrics.png`
- `paper\figures\airshow_generated_geometry.png`

## Step 6: Focused Verification

```powershell
python -m py_compile CLI\build_airshow_corpus.py CLI\run_airshow_flight_path_tests.py CLI\render_airshow_figures.py
python -m pytest tests\test_airshow_corpus.py -q
python -m pytest -q
```

The paper-run focused test result was:

- `tests\test_airshow_corpus.py`: 2 passed
- full suite: 210 passed

## Step 7: Compile the Paper

```powershell
python C:\Users\Darsh Gupta\.codex\plugins\cache\openai-bundled\latex\0.2.2\scripts\compile_latex.py D:\CodeProjects\research-paper\paper\main.tex --compiler tectonic --json
```

Expected output:

`paper\main.pdf`

The paper-run compile completed successfully. The remaining LaTeX messages were
non-fatal layout/font warnings.

## Replication Boundary

The replication target is the complete smoke path:

1. public Airshow document discovery,
2. license and preview-geometry filtering,
3. mesh parsing and voxelization,
4. manifest validation,
5. bounded checkpoint training,
6. three conditioned generation checks,
7. raw internal D3Q27 scoring,
8. figure rendering,
9. and paper compilation.

The replication target is not validated aircraft design. A successful rerun
does not support claims that generated samples are aircraft-valid, that raw
D3Q27 labels are publication-grade CFD, or that the method outperforms
established aircraft-design workflows.
