# Airshow 96^3 Extended Solver-In-Loop Run - 2026-06-21

## Question

This run answers two narrow questions:

1. Does continuing the `96^3` direct-solver checkpoint improve the measured
   internal D3Q27 solver objective?
2. Do the resulting exports look like logical aircraft under the current
   heuristic aircraft-validity screen?

It does not establish aerodynamic validity, structural validity, or external PDE
ground truth. The D3Q27 values below are internal low-Mach smoke evidence.

## Extended Training Setup

Starting checkpoint:

`build/airshow_direct_solver_strong_grid_sweep_20260621/g96_more/checkpoints/final_optimized_model.pt`

Continuation command family:

```text
python CLI/aircraft_diffusion_cfd.py train
  --grid-size 96
  --batch-size 1
  --dataset-manifest build/airshow_grounded_corpus_20260621_g96/manifest.jsonl
  --coordinate-decoder-threshold 1
  --coordinate-training-samples 4096
  --coordinate-positive-fraction 0.5
  --full-diagnostic-interval 0
  --direct-solver-loss-weight 0.2
  --direct-solver-interval 32
  --direct-solver-steps 3
  --direct-solver-perturbation 0.25
  --direct-solver-perturbation-grid-size 8
  --direct-solver-gradient-clip 10.0
  --direct-connectivity-weight 5.0
  --direct-solver-target-occupancy 0.01
  --connectivity-monitor-interval 0
  --aerodynamic-monitor-interval 0
  --disable-pipeline
  --enable-checkpointing
```

Learning-rate schedule across the one-epoch continuations:

| Continuation | Learning rate | Resume source |
|---|---:|---|
| ep1 | `5e-5` | previous `g96_more` final checkpoint |
| ep2 | `2.5e-5` | ep1 final checkpoint |
| ep3 | `1e-5` | ep2 final checkpoint |

Each command was run through `CLI/run_with_resource_monitor.py`, which samples
process RSS, CPU use, system memory, and `nvidia-smi` GPU counters.

## Training Result

| Checkpoint | Optimizer loss | Geometry BCE | Direct solver eval loss | Wall time | Peak GPU memory | Mean GPU util. | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| prior `g96_more` final | `0.606047` | `0.037385` | `7.486129` | n/a | n/a | n/a | n/a |
| ep1, `5e-5` | `0.484041` | `0.036791` | `4.353630` | `281.4 s` | `7943 MB` | `45.6%` | `6133 MB` |
| ep2, `2.5e-5` | `0.415297` | `0.037072` | `1.918370` | `300.2 s` | `7943 MB` | `44.1%` | `6674 MB` |
| ep3, `1e-5` | `0.399479` | `0.037772` | `0.972279` | `285.5 s` | `7941 MB` | `45.1%` | `6096 MB` |

The extended run did not show the earlier direct-solver drift within these three
extra epochs. The scheduled measured direct-solver objective improved from
`7.486129` to `0.972279`.

## Throughput Notes

The run is memory-bound enough that the GPU sits near the 8 GB ceiling during
solver-in-loop training, while mean GPU utilization is only about `44-46%`.
That points to a mixed CPU/GPU and synchronization workload rather than a pure
matmul-saturated training job.

The current safe configuration is intentionally sequential:

- `--disable-pipeline`
- `--batch-size 1`
- scheduled direct solver calls every 32 batches
- low-frequency `8^3` SPSA perturbations upsampled to `96^3`
- exact connectivity and raw aero monitors disabled outside the direct loss

Further throughput work should focus on reducing synchronization and repeated
host/device materialization, not on parallelizing more solver calls. The
generation/evaluation pass is much lighter (`1653 MB` peak GPU memory in the
short monitored rerun) and is dominated by Python/CPU, mesh export, and short
solver calls.

## Generated Designs

Raw top-k exports after the extended checkpoint were still one-sided wedge
fragments and should not be described as logical aircraft. I added two small
corrections to make the evaluation match the intended artifact:

1. Sequential optimizer candidate scoring can now use top-k binarization instead
   of fixed `0.5` thresholding, so sparse `96^3` probabilities are scored in the
   same representation that is exported.
2. The aircraft-validity canonicalizer now evaluates centered occupied geometry
   consistently instead of sometimes retaining an offset raw grid.

The best export calibration was full lateral symmetry plus `5%` top-k occupancy:

```text
python CLI/run_airshow_flight_path_tests.py
  --checkpoint build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/checkpoints/final_optimized_model.pt
  --manifest build/airshow_grounded_corpus_20260621_g96/manifest.jsonl
  --output-dir build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/flight_path_sym_topk05
  --grid-size 96
  --num-steps 4
  --cfd-steps 20
  --objective-optimizer none
  --export-target-occupancy 0.05
  --export-symmetry-blend 1.0
```

| Case | Heuristic validity | Failed checks | Occupancy | Symmetry | Span | Length | Thickness | Center-body density ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `short_takeoff_payload` | pass | none | `0.050000` | `1.000` | `1.000` | `0.500` | `0.198` | `1.230` |
| `high_speed_sprint` | fail | `body_centerline_dominance` | `0.050000` | `1.000` | `1.000` | `0.365` | `0.271` | `1.119` |
| `endurance_turning` | pass | none | `0.050000` | `1.000` | `1.000` | `0.385` | `0.260` | `1.157` |

Artifacts:

- `build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/flight_path_sym_topk05/generated/short_takeoff_payload.stl`
- `build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/flight_path_sym_topk05/generated/high_speed_sprint.stl`
- `build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/flight_path_sym_topk05/generated/endurance_turning.stl`
- figure:
  `build/airshow_direct_solver_extended_20260621/g96_ext_lr1e5_ep3/figures/airshow_generated_geometry_extended_g96_sym_topk05.png`

## Objective Quality Call

The answer is: better, but still not scientifically good.

The extended checkpoint is objectively better on the internal scheduled
direct-solver objective. The symmetrized `5%` top-k exports also produce two
candidate geometries that pass the repository's first-pass aircraft-shape
heuristics, and the third is a narrow center-body-density miss.

Visually, however, the designs are blocky flying-wing or delta-slab forms, not
complete conventional aircraft with clear fuselage, tail, propulsion placement,
or structural layout. The internal D3Q27 reports for the exported candidates are
also non-converged and not aerodynamically credible; they should remain
implementation evidence only.

The merge-safe claim is therefore:

> The `96^3` direct-solver continuation can improve the measured internal solver
> objective and can produce two heuristic aircraft-screen-passing, symmetric
> flying-wing-like voxel candidates after explicit export calibration. It does
> not yet produce externally validated or aerodynamically trustworthy aircraft.
