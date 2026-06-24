# HiLiftAeroML 96 Cubed STL Expansion And Direct Solver Training Report

Generated: `2026-06-24`

## Scope

This report records the 96^3 corpus expansion and direct-solver training pass that pushed the local trainable manifest over 750 records. The added records come from public HiLiftAeroML exact STL surface runs and force/moment CSVs, then merge with the existing Airshow+NASA CRM 96^3 corpus.

This is smoke and overfit evidence, not a final scientific result. The direct solver is in the optimization scalar, but the generated aircraft quality is not established by this run.

## Source And Geometry Counts

The exact CAD/source catalog now contains `2363` records:

| Source collection | Records |
| --- | ---: |
| `hiliftaeroml_crm_hl_surface_runs` | 1800 |
| `hiliftaeroml_crm_hl_variants` | 180 |
| `vsp_airshow_public_models` | 359 |
| `local_nasa_crm_ready_catalog` | 15 |
| `nasa_uam_reference_vehicles` | 9 |

The trainable local 96^3 manifest used for this run contains `752` records:

| Input manifest | Records |
| --- | ---: |
| `build/expanded_aircraft_corpus_20260622/manifest.jsonl` | 370 |
| `build/hiliftaeroml_g96_stream_20260624/manifest.jsonl` | 382 |

The HiLift addition has `382` AoA-labeled records but only `39` unique voxel artifacts because each `geometry_variant_id` is reused across multiple angles of attack. That is intentional: the records preserve different flow labels and source URLs, but they should not be described as 382 independent aircraft geometries.

## Validation Evidence

Claim-bearing manifest validation passed for both the HiLift slice and the combined training manifest:

| Manifest | Records | Hash |
| --- | ---: | --- |
| `build/hiliftaeroml_g96_stream_20260624/manifest.jsonl` | 382 | `140c38ea0c7d9417f588efa1a868456ba566059986136eae1e0091eeb3bbd92e` |
| `build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl` | 752 | `eda12e15898c47b1434114c5f8a7d59c50f66a76f72544dedb1bc0cc1cdb5287` |

The builder streamed STL files one at a time and deleted raw STL payloads after voxelization. The checked-in artifacts remain source catalogs and scripts; the large generated voxel grids and checkpoints stay under `build/`.

## Direct Solver OOM Fix

The first 96^3 three-epoch attempts failed during direct-solver SPSA evaluation with CUDA out-of-memory errors. The failure was traced to retained D3Q27 boundary/link-distance caches:

- `_q_cache`
- `_boundary_cache_key`
- `_boundary_link_cache`
- the same fields on the nested wrapped solver, when present

`DirectSolverSPSAFunction.forward` now clears those geometry-specific caches after each solver sample. This does not remove the solver from the loss; it prevents stale per-geometry tensors from accumulating across SPSA base/plus/minus calls.

Regression coverage:

`python -m pytest tests/test_aerodynamic_loss.py tests/test_exact_cad_catalog.py tests/test_hiliftaeroml_voxel_manifest.py -q`

## 96 Cubed Training Run

Final completed run:

`build/expanded_aircraft_hilift_training_20260624/g96_ep4_direct_nestedcachefix/checkpoints/final_optimized_model.pt`

Key configuration:

- Grid: `96^3`
- Records per epoch: `752`
- Epochs: `4`
- Solver: `D3Q27`
- Direct solver loss: enabled with weight `0.05`
- Direct solver interval: every `64` batches
- SPSA samples per solver call: base/plus/minus finite-difference estimates, `5` solver steps
- Connectivity/aero monitors: disabled as separate legacy monitors, so their logged monitor fields are `0.0`

The zero `connectivity` and `aerodynamic` monitor fields do not mean the solver was absent. The integrated solver contribution is reported as `direct_solver_loss` and is included in `optimization_loss` through the configured nonzero weight.

## Training Metrics

| Epoch | Optimization loss | MSE | Geometry recon | Generation recon | Direct solver loss | Direct solver eval loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6.288045 | 0.976245 | 0.761959 | 0.414643 | 1.214959 | 76.137443 |
| 2 | 1.835346 | 0.925703 | 0.525893 | 0.031614 | 2.852562 | 178.760568 |
| 3 | 2.088015 | 0.891805 | 0.394038 | 0.029906 | 3.298090 | 206.680283 |
| 4 | 1.966317 | 0.851987 | 0.281439 | 0.029927 | 4.429654 | 277.591662 |

The model did train over the expanded corpus, and reconstruction terms improved. The direct solver objective moved the wrong way after epoch 1. That is the useful result: more blind epochs are now evidence of overfit/objective drift, not evidence that the final checkpoint is a better aircraft generator.

Best checkpoint by direct-solver evaluation in this run is epoch 1, but only the final smoke checkpoint was saved by the current trainer. A claim-bearing training protocol should save per-epoch checkpoints and select by held-out solver/validity score.

## Resource Usage

Resource monitor summary for the completed 4-epoch run:

| Metric | Value |
| --- | ---: |
| Elapsed wall time | `922.297 s` |
| Return code | `0` |
| Peak GPU memory used | `7947 MB` |
| Mean GPU memory used | `4926.38 MB` |
| Peak GPU utilization | `100%` |
| Mean GPU utilization | `26.84%` |
| Peak process RSS | `4528.87 MB` |
| Mean process CPU | `548.78%` |

The global GPU memory/utilization fields are usable. Per-process GPU memory remained unavailable from the local monitor on this run and is recorded as zero in the JSON summary.

## Replication Commands

Catalog regeneration:

```bash
python CLI/build_exact_cad_catalog.py --allow-insecure-tls
```

HiLift STL streaming and manifest merge:

```bash
python CLI/build_hiliftaeroml_voxel_manifest.py \
  --target-total-records 752 \
  --output-root build/hiliftaeroml_g96_stream_20260624 \
  --manifest build/hiliftaeroml_g96_stream_20260624/manifest.jsonl \
  --report build/hiliftaeroml_g96_stream_20260624/report.json \
  --combined-manifest build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl \
  --combined-report build/expanded_aircraft_hilift_corpus_20260624/flight_path_manifest_report.json \
  --delete-raw-stl
```

Manifest validation:

```bash
python CLI/validate_manifest.py --manifest build/hiliftaeroml_g96_stream_20260624/manifest.jsonl --level claim-bearing --output build/hiliftaeroml_g96_stream_20260624/manifest_validation.json
python CLI/validate_manifest.py --manifest build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl --level claim-bearing --output build/expanded_aircraft_hilift_corpus_20260624/manifest_validation.json
```

Training:

```bash
python CLI/run_with_resource_monitor.py \
  --output-dir build/expanded_aircraft_hilift_training_20260624/g96_ep4_direct_nestedcachefix/resources \
  --interval 10 \
  --cwd . \
  -- python CLI/aircraft_diffusion_cfd.py train \
  --num-epochs 4 \
  --batch-size 1 \
  --learning-rate 5e-6 \
  --latent-dim 16 \
  --grid-size 96 \
  --dataset-manifest build/expanded_aircraft_hilift_corpus_20260624/manifest.jsonl \
  --save-dir build/expanded_aircraft_hilift_training_20260624/g96_ep4_direct_nestedcachefix/checkpoints \
  --disable-pipeline \
  --enable-checkpointing \
  --solver D3Q27 \
  --coordinate-training-samples 4096 \
  --coordinate-positive-fraction 0.5 \
  --coordinate-decoder-threshold 96 \
  --full-diagnostic-interval 0 \
  --direct-solver-loss-weight 0.05 \
  --direct-solver-interval 64 \
  --direct-solver-steps 5 \
  --direct-solver-perturbation 0.15 \
  --direct-solver-perturbation-grid-size 8 \
  --direct-solver-gradient-clip 1.0 \
  --direct-connectivity-weight 1.0 \
  --direct-solver-target-occupancy 0.004 \
  --connectivity-monitor-interval 0 \
  --aerodynamic-monitor-interval 0
```

## Claim Boundary

This run proves that the code can build a 752-record local 96^3 manifest, train across it with the direct solver loss enabled, and complete without the previous direct-solver OOM. It does not prove that the generated outputs are valid aircraft. The direct solver trend says the opposite: the final epochs improve reconstruction while degrading the measured solver objective.

## Source References

- HiLiftAeroML dataset: https://huggingface.co/datasets/nvidia/HiLiftAeroML
- HiLiftAeroML overview: https://caemldatasets.org/hiliftaeroml/
- VSP Airshow: https://airshow.openvsp.org/
- NASA UAM reference vehicles: https://www.nasa.gov/reference/uam-refs/
- NASA Common Research Model original CAD: https://commonresearchmodel.larc.nasa.gov/geometry/original-cad-files/
