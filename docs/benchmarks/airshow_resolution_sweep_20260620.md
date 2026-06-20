# Airshow Resolution Sweep, 2026-06-20

This report records the requested higher-resolution Airshow reruns. It is a
smoke evidence addendum, not a final validation run. The purpose was to test
whether increasing the voxel lattice from the earlier `16^3` Airshow run to
`32^3` and `64^3` would cause generated samples to pass the current
aircraft-validity gates.

## Source Corpus

Both higher-resolution runs use the same public VSP Airshow source policy as
the earlier Airshow corpus report:

- public Airshow model documents observed: 381
- license-and-geometry-eligible documents: 357
- converted records: 355
- true conversion failures: 2 stale public storage URLs returning 404
- admitted licenses: 208 CC0, 18 CC BY, 129 CC BY-SA
- deterministic split: train 250, val 31, test 38, holdout 36

The public geometry was re-voxelized at each target resolution. The source
metadata remains Airshow/OpenVSP metadata; mission and manufacturing fields in
the manifest remain deterministic repository inferences, not source-author or
manufacturer assertions.

## Commands

```powershell
python CLI\build_airshow_corpus.py --output-dir build\airshow_grounded_corpus_20260620_g32 --grid-size 32 --allowed-licenses 1 2 3
python CLI\validate_manifest.py --manifest build\airshow_grounded_corpus_20260620_g32\manifest.jsonl --level claim-bearing
python CLI\aircraft_diffusion_cfd.py train --num-epochs 1 --batch-size 2 --dataset-manifest build\airshow_grounded_corpus_20260620_g32\manifest.jsonl --grid-size 32 --latent-dim 16 --save-dir build\airshow_training_20260620_g32\checkpoints --disable-pipeline --disable-checkpointing --solver D3Q27
python CLI\run_airshow_flight_path_tests.py --checkpoint build\airshow_training_20260620_g32\checkpoints\final_optimized_model.pt --manifest build\airshow_grounded_corpus_20260620_g32\manifest.jsonl --output-dir build\airshow_training_20260620_g32\flight_path_tests --grid-size 32 --num-steps 4 --cfd-steps 100 --cpu

python CLI\build_airshow_corpus.py --output-dir build\airshow_grounded_corpus_20260620_g64 --grid-size 64 --allowed-licenses 1 2 3
python CLI\validate_manifest.py --manifest build\airshow_grounded_corpus_20260620_g64\manifest.jsonl --level claim-bearing
python CLI\aircraft_diffusion_cfd.py train --num-epochs 1 --batch-size 1 --dataset-manifest build\airshow_grounded_corpus_20260620_g64\manifest.jsonl --grid-size 64 --latent-dim 16 --save-dir build\airshow_training_20260620_g64\checkpoints --disable-pipeline --disable-checkpointing --solver D3Q27
```

## Manifest Results

| Grid | Records | Manifest validation | Manifest SHA-256 |
| ---: | ---: | --- | --- |
| `32^3` | 355 | pass | `7684e70bc6e92214382525e2f96de2311d47735ff0f42f2a5ee4c288f98521f1` |
| `64^3` | 355 | pass | `2627227fc337edf79a323ec63b50a79783eedbcaf3333b77aa02cfcd4a1dbd80` |

The higher-resolution corpus construction itself passed the same manifest gate
at both resolutions.

## Training and Gate Results

| Grid | Training outcome | Checkpoint | Generated flight-path gate outcome |
| ---: | --- | --- | --- |
| `32^3` | completed on CUDA, 1 epoch, batch 2 | `build\airshow_training_20260620_g32\checkpoints\final_optimized_model.pt` | all 3 generated cases failed aircraft-validity gates |
| `64^3` | attempted with 1 epoch, batch 1; no checkpoint was produced before the 15-minute run ceiling | none | not run, because no checkpoint existed |

The `32^3` checkpoint hash is
`7234e1b9b3381ce00e13776be05bff614afc008ffa675194c9b1326783b95444`.
The observed epoch metrics were: loss `41.63597`, MSE `0.93468`,
geometry-reconstruction loss `0.09601`, consistency loss `0.00613`,
connectivity loss `1.05512`, and aerodynamic loss `39.54404`.

Three `32^3` generated flight-path checks were then run on CPU for the
generator pass and internal D3Q27 smoke scoring:

| Case | Occupied voxels | Occupancy | Raw D3Q27 Cd | Raw D3Q27 L/D | Validity status |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_takeoff_payload` | 162 | 0.004944 | 0.760058 | -0.100933 | fail: `nonempty_occupancy`, `symmetry`, `span_sanity` |
| `high_speed_sprint` | 156 | 0.004761 | 0.710940 | -0.088196 | fail: `nonempty_occupancy`, `symmetry`, `span_sanity` |
| `endurance_turning` | 167 | 0.005096 | 0.662959 | -0.091659 | fail: `symmetry`, `span_sanity` |

All three `32^3` cases produced STL files, finite internal D3Q27 outputs, and
positive reference areas. They still failed the aircraft-validity screen. The
figures added from this run are:

- `paper/figures/airshow_flight_path_metrics_g32.png`
- `paper/figures/airshow_generated_geometry_g32.png`

## Implementation Scalability Note

The current `LatentTo3DConverter` in `CLI/aircraft_diffusion_cfd.py` maps the
latent vector through a dense final layer from 2048 hidden units to
`grid_resolution ** 3` output voxels. That makes the final decoder layer scale
directly with voxel count:

- `32^3`: 32,768 output voxels, approximately 67.1 million final-layer
  parameters.
- `64^3`: 262,144 output voxels, approximately 537.1 million final-layer
  parameters.

This explains why `64^3` is a much harder training target on the current 8 GB
GPU. The corpus is valid at `64^3`, but the current dense decoder did not
produce a trainable checkpoint under the local time budget used here.

## Claim Boundary

Increasing voxel count did not make the generated Airshow checkpoint samples
pass aircraft-validity gates. The `32^3` training path is executable, but the
generated samples remain non-claim-bearing. The `64^3` corpus path is
executable and manifest-valid, but the current model architecture did not
produce a checkpoint in the attempted local run. Aircraft-level generation,
condition-response, aerodynamic superiority, structural viability, and
publication-quality validation claims remain blocked.
