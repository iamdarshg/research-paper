# Airshow Corpus Addition Report, 2026-06-20

This report documents the public VSP Airshow corpus addition used by the
current paper revision. It is a corpus and smoke-training report, not a claim
that the repository now produces valid aircraft.

## Summary

The corpus builder converts public VSP Airshow preview geometries into
manifest-backed voxel records for bounded training and generation tests. The
reported run observed 381 public Airshow model documents, admitted 357 records
after license and preview-geometry filtering, and converted 355 records into
centered `16^3` voxel grids. Two eligible records failed because their public
storage URLs returned 404.

| Field | Value |
| --- | ---: |
| Public Airshow documents observed | 381 |
| License-and-geometry eligible | 357 |
| Converted records | 355 |
| Conversion failures | 2 stale public storage URLs |
| Grid | `16^3` |
| Manifest SHA-256 | `7bb59bab9cc8ed3a836377a35d3c38d5c0086a56b617b2695e131486451885a6` |
| Train / val / test / holdout | 250 / 31 / 38 / 36 |
| License mix | 208 CC0, 18 CC BY, 129 CC BY-SA |

## Public Sources

- VSP Airshow: <https://airshow.openvsp.org/>
- OpenVSP project site: <https://openvsp.org/>
- OpenVSP source repository: <https://github.com/OpenVSP/OpenVSP>
- OpenVSP license page: <https://openvsp.org/license.shtml>

OpenVSP is treated as the source ecosystem for the public model exchange. The
Airshow records supply source metadata such as model name, author/manufacturer
fields, license labels, document ids, and public geometry URLs. Those metadata
fields are not treated as manufacturer certification or as proof that a model
represents an official NASA, Lockheed, or agency design.

## Admission Policy

The training manifest admits only Airshow records whose license id maps to one
of the following:

| Airshow license id | Manifest label | Training status |
| ---: | --- | --- |
| 1 | No Rights Reserved (CC0) | admitted |
| 2 | Attribution (CC BY) | admitted |
| 3 | Attribution Share Alike (CC BY-SA) | admitted |

Records marked with noncommercial or no-derivatives licenses are excluded from
the training manifest. A record also must expose a public preview-geometry URL.
This policy keeps the smoke-training corpus traceable and avoids treating
restricted public browsing records as training data.

## Conversion Pipeline

For each admitted Airshow document, `CLI/build_airshow_corpus.py` downloads the
public preview geometry, parses X3D indexed-face-set meshes when present,
normalizes geometry into a centered unit cube, voxelizes the result into a
`16^3` occupancy grid, and writes a manifest row. Each manifest row records:

- source URL and Airshow document id,
- license id, license label, and eligibility status,
- geometry hash and voxel hash,
- voxel dimensions and occupancy summary,
- deterministic split assignment,
- preprocessing parameters,
- generated condition fields used by the repository,
- and an explicit claim boundary for the row.

The generated condition fields are deterministic repository inferences from
geometry and defaults. They are useful for plumbing structured conditions
through the model, but they are not facts asserted by the original Airshow
authors, manufacturers, NASA, Lockheed, or OpenVSP.

## Training Smoke Run

The reported checkpoint was trained for three epochs with batch size 8, latent
dimension 16, grid size `16^3`, and the internal D3Q27 evaluator. The final
checkpoint is:

`build\airshow_training_20260620\checkpoints\final_optimized_model.pt`

Checkpoint SHA-256:

`71e808aa3c35142f145da267bb4eb7050300adc383e3a070d571eec36413d4f6`

Terminal-observed epoch metrics:

| Epoch | Loss | MSE | Geometry | Consistency | Connectivity | Aerodynamic |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.9388 | 0.9822 | 0.2227 | 0.01025 | 2.5519 | 11.1717 |
| 2 | 20.2200 | 0.8463 | 0.07975 | 0.00219 | 0.04038 | 19.2514 |
| 3 | 21.5905 | 0.7997 | 0.07782 | 0.00109 | 0.00149 | 20.7104 |

The increasing diagnostic total and aerodynamic diagnostic are reported as
observed. A later loss-semantics audit showed that the connectivity and
aerodynamic values from this trainer revision are detached diagnostics, not
differentiable solver training signals. They are not convergence evidence.

## Generated Flight-Path Checks

The Airshow checkpoint was used to generate three conditioned smoke cases:

| Case | Occupied voxels | Occupancy | D3Q27 Cd | D3Q27 L/D | Validity |
| --- | ---: | ---: | ---: | ---: | --- |
| `short_takeoff_payload` | 58 | 0.014160 | 1.204657 | 0.004383 | fail: `span_sanity` |
| `high_speed_sprint` | 53 | 0.012939 | 0.912229 | 0.030508 | fail: `span_sanity` |
| `endurance_turning` | 51 | 0.012451 | 0.982371 | 0.046629 | fail: `span_sanity` |

All three cases produced nonempty voxel grids, STL exports, finite internal
D3Q27 values, and positive reference areas. All three also failed the current
aircraft-specific validity screen on `span_sanity`. The solver outputs are
labeled `claim_bearing_cfd=false`, `label_tier=lbm_raw`, and
`lbm_converged=false`.

## Manuscript Figures

The following figures are derived from the corpus report, terminal-observed
training metrics, and generated flight-path report:

- `paper/figures/airshow_corpus_summary.png`
- `paper/figures/airshow_training_losses.png`
- `paper/figures/airshow_flight_path_metrics.png`
- `paper/figures/airshow_generated_geometry.png`

The figures are intended to make the data and failures visible inside the
paper. They do not upgrade the scientific claim beyond code-path and smoke-run
evidence.

## Claim Boundary

Supported by this addition:

- the repository can build a traceable public Airshow geometry manifest,
- train a checkpoint on hundreds of public aircraft-like geometry records,
- generate conditioned voxel artifacts from that checkpoint,
- export geometry artifacts,
- run the internal D3Q27 scoring path on those artifacts,
- and document the resulting failure modes.

Not supported by this addition:

- generated samples are valid aircraft,
- the checkpoint learned manufacturer-certified design rules,
- Airshow metadata are official manufacturer or agency certification,
- raw D3Q27 values are validated aerodynamic predictions,
- or the method outperforms established aircraft-design baselines.
