# Cheapest Viable Plan For A Conditioned Airplane Generator

This README exists because the current repository does **not** yet implement a fully AI-driven airplane generator conditioned on flight profile and manufacturing method.

## What The Repo Can Do Today

- Train a proof-of-concept latent generator on a synthetic voxel dataset.
- Carry a structured condition vector through the current dataset, model, and generator code paths.
- Export generated voxel grids to STL.
- Score shapes with the internal CFD path and benchmark selected meshes with OpenFOAM.
- Reproduce a small sanity run on limited hardware.

## What The Repo Cannot Honestly Claim Today

- Scientifically validated conditioned generation from a full flight profile.
- Scientifically validated conditioning on manufacturing method, material, or production constraints.
- Real aircraft design synthesis from a curated aircraft dataset.
- Structural viability beyond connectivity heuristics.
- Publication-grade aerodynamic optimization on an 8 GB laptop GPU.

## Cheapest Viable Next Step

### 1. Narrow the target problem

Do **not** start with "any airplane for any mission."

Start with one constrained family:
- small fixed-wing aircraft,
- one Reynolds/Mach regime,
- one manufacturing family such as foam-core, FDM 3D print, or simple sheet construction.

### 2. Build a minimal condition schema

Train on a structured condition vector with fields such as:
- cruise speed,
- takeoff distance min/max bounds,
- payload mass min/max bounds,
- wingspan limit,
- minimum thrust-to-weight ratio,
- minimum turn-rate target,
- required static thrust,
- engine diameter, length, and count bounds,
- manufacturing method,
- wall-thickness min/max bounds,
- part-count min/max bounds.

#### Documented seam contract

The repo now documents a minimal condition-vector seam in `CLI/conditioning_schema.yaml`, with `CLI/config.yaml` pointing to that schema file. Treat this as a schema and checkpoint contract first, not as proof that the full generation pipeline is already conditioned.

- Tensor shape: `[batch, 22]`
- Tensor dtype: `float32`
- Scalar slots:
  - `target_speed_mps`
  - `wingspan_limit_m`
  - `thrust_to_weight_min`
  - `turn_rate_min_deg_s`
  - `required_static_thrust_n`
  - `engine_diameter_mm`
  - `engine_length_mm`
  - `engine_count_min`
  - `engine_count_max`
  - `payload_mass_min_g`
  - `payload_mass_max_g`
  - `takeoff_distance_min_m`
  - `takeoff_distance_max_m`
  - `wall_thickness_min_mm`
  - `wall_thickness_max_mm`
  - `part_count_min`
  - `part_count_max`
- One-hot categorical group:
  - `manufacturing_method`: `foam_core_hotwire`, `fdm_pla_0p4mm`, `fdm_pla_0p6mm`, `sheet_balsa_tabbed`, `composite_wet_layup`

That ordering matters. Dataset preprocessing, any future condition encoder, and checkpoint metadata should all emit the same flat vector layout.

The repository now has a partial structured conditioning path: the dataset, latent/model path, and generator all consume the documented condition vector. The public CLI currently exposes only a subset of that schema. The direct `generate` and `batch-generate` paths now expose speed, thrust-to-weight, turn-rate, static thrust, engine geometry, engine count, wingspan, payload bounds, takeoff bounds, wall-thickness bounds, part-count bounds, and manufacturing method. Remaining work is still real: payload, takeoff, wingspan, wall-thickness, part-count, and manufacturing controls need condition-response benchmarks against grounded aircraft-like data before they can support scientific claims, and config-driven evaluation workflows still need to be hardened around the full schema.

### 3. Replace the synthetic dataset

The current synthetic voxel generator is not enough. The minimum viable dataset is:
- 200-1000 aircraft-like examples,
- consistent voxelization or mesh-to-voxel preprocessing,
- metadata for the condition schema above,
- one train/validation split and one holdout split.

Even a noisy, partly procedural dataset is better than the current hand-drawn synthetic fuselage-and-wing voxel prior.

### 4. Upgrade partial plumbing into validated conditioning

The current code already carries a structured condition vector into the dataset, latent construction, denoiser, and generation path. The missing work is not "add conditioning from scratch"; it is to make that path scientifically meaningful:
- expose the remaining condition fields cleanly in public CLI/config surfaces,
- train and evaluate against a grounded aircraft-like corpus rather than only procedural data,
- measure whether changing each condition changes outputs in the intended direction,
- run CFD as reranking or periodic auxiliary supervision rather than full tight-loop optimization.

### 5. Respect the 8 GB GPU boundary

For the RTX 4060 laptop GPU in this session, the realistic budget is:
- voxel grids at `16^3` or `24^3`,
- batch size `1`,
- mixed precision only after correctness is verified,
- short smoke runs locally,
- full sweeps only on rented cloud hardware.

### 6. Cheapest compute plan

Local:
- use the laptop GPU for preprocessing, unit tests, and smoke training.

Cloud:
- rent short bursts on a single 24 GB class GPU only for conditioned training checkpoints and evaluation sweeps.

The cost-minimizing path is:
1. debug locally,
2. train a small conditional model on a reduced dataset,
3. run CFD reranking on a small candidate pool,
4. expand only if conditional consistency is measurable.

## Minimum Acceptance Criteria

Do not call the project a conditioned airplane generator until all of the following are true:
- the model consumes the structured condition vector,
- generated samples change in the expected direction when conditions change,
- aircraft-specific validity checks pass on a non-trivial fraction of samples,
- at least one manufacturing constraint affects the decoded geometry,
- aerodynamic scoring is reported against a named baseline.

## Recommended New Issues

- Add structured conditioning for mission and manufacturing constraints.
- Replace the synthetic voxel dataset with an aircraft-like corpus and metadata pipeline.
- Define an 8 GB smoke-run protocol and a cloud-only final-evaluation protocol.
