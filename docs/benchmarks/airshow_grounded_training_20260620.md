# Airshow Grounded Training Smoke Evidence, 2026-06-20

This report replaces the earlier synthetic-only scale-up evidence with a
public-geometry run built from VSP Airshow. It is still smoke evidence: it
checks that the corpus builder, manifest validator, training path, generator,
STL export, aircraft-validity screen, and D3Q27 solver execute on hundreds of
public aircraft-like geometries. It does not establish validated aircraft
performance.

## Public Sources

- VSP Airshow: <https://airshow.openvsp.org/>
- OpenVSP project site: <https://openvsp.org/>
- OpenVSP source repository: <https://github.com/OpenVSP/OpenVSP>
- OpenVSP license page: <https://openvsp.org/license.shtml>

OpenVSP describes itself as a parametric aircraft geometry tool, and the
OpenVSP site links VSP Airshow. The Airshow records used here are public model
documents and public storage URLs discovered from the Airshow web app. Airshow
manufacturer/name fields are treated as source metadata, not as manufacturer
certification. The generated `design_spec` fields in the manifest are
deterministic conditioning inferences from normalized geometry and repository
defaults, not facts asserted by NASA, Lockheed, or any other named entity.

## Commands

```powershell
python CLI\build_airshow_corpus.py --output-dir build\airshow_grounded_corpus_20260620 --grid-size 16 --allowed-licenses 1 2 3
python CLI\validate_manifest.py --manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --level claim-bearing
python CLI\aircraft_diffusion_cfd.py train --num-epochs 3 --batch-size 8 --dataset-manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --grid-size 16 --latent-dim 16 --save-dir build\airshow_training_20260620\checkpoints --disable-pipeline --disable-checkpointing --solver D3Q27
python CLI\run_airshow_flight_path_tests.py --checkpoint build\airshow_training_20260620\checkpoints\final_optimized_model.pt --manifest build\airshow_grounded_corpus_20260620\manifest.jsonl --output-dir build\airshow_training_20260620\flight_path_tests --grid-size 16 --num-steps 4 --cfd-steps 100
```

## Corpus Summary

| Field | Value |
| --- | ---: |
| Public Airshow model documents observed | 381 |
| Eligible after license and geometry filtering | 357 |
| Converted voxel records | 355 |
| True conversion failures | 2 stale storage 404s |
| Grid | `16^3` |
| Manifest hash | `7bb59bab9cc8ed3a836377a35d3c38d5c0086a56b617b2695e131486451885a6` |
| Split counts | train 250, val 31, test 38, holdout 36 |
| License counts | CC0 208, CC BY 18, CC BY-SA 129 |

The builder excludes Airshow records marked CC BY-NC or CC BY-NC-ND from this
training manifest. It also records source URLs, Airshow document ids, license
ids/names, geometry hashes, voxel hashes, preprocessing provenance, and a
per-record claim boundary.

## Training Summary

| Field | Value |
| --- | --- |
| Checkpoint | `build\airshow_training_20260620\checkpoints\final_optimized_model.pt` |
| Checkpoint SHA-256 | `71e808aa3c35142f145da267bb4eb7050300adc383e3a070d571eec36413d4f6` |
| Epochs | 3 |
| Batch size | 8 |
| Latent dimension | 16 |
| Solver | D3Q27 |
| Grid | `16^3` |
| Global step | 135 |

Terminal-observed epoch metrics:

| Epoch | Loss | MSE | Geometry | Consistency | Connectivity | Aerodynamic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.9388 | 0.9822 | 0.2227 | 0.01025 | 2.5519 | 11.1717 |
| 2 | 20.2200 | 0.8463 | 0.07975 | 0.00219 | 0.04038 | 19.2514 |
| 3 | 21.5905 | 0.7997 | 0.07782 | 0.00109 | 0.00149 | 20.7104 |

These values are useful for reproducing the local smoke run. They are not a
convergence study; the loss increase across epochs is reported as observed.

## Three Generated Flight-Path Checks

Output report:
`build\airshow_training_20260620\flight_path_tests\flight_path_results.json`

| Case | Occupied voxels | Occupancy | D3Q27 Cd | D3Q27 L/D | Validity |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_takeoff_payload` | 58 | 0.014160 | 1.204657 | 0.004383 | fail: `span_sanity` |
| `high_speed_sprint` | 53 | 0.012939 | 0.912229 | 0.030508 | fail: `span_sanity` |
| `endurance_turning` | 51 | 0.012451 | 0.982371 | 0.046629 | fail: `span_sanity` |

All three cases produced nonempty voxel grids, STL files, finite D3Q27
coefficients, and positive reference areas. All three also failed the current
aircraft-specific validity screen on `span_sanity`, and all D3Q27 metrics are
marked `claim_bearing_cfd=false` / `label_tier=lbm_raw`.

## Claim Boundary

This run supports a narrow code-path claim: the current system can build a
public Airshow geometry manifest, train from hundreds of public records, emit a
checkpoint, generate three conditioned artifacts, export STL files, and run the
internal D3Q27 CFD path on those artifacts.

It does not support claims that the generated artifacts are valid aircraft,
that the model has learned manufacturer-certified design rules, that the D3Q27
numbers are validated aerodynamic predictions, or that the method outperforms
mature aircraft-design baselines.
