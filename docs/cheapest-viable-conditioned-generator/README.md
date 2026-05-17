# Cheapest Viable Plan For A Conditioned Airplane Generator

This README exists because the current repository does **not** yet implement a fully AI-driven airplane generator conditioned on flight profile and manufacturing method.

## What The Repo Can Do Today

- Train a proof-of-concept latent generator on a synthetic voxel dataset.
- Export generated voxel grids to STL.
- Score shapes with the internal CFD path and benchmark selected meshes with OpenFOAM.
- Reproduce a small sanity run on limited hardware.

## What The Repo Cannot Honestly Claim Today

- Conditioned generation from a full flight profile.
- Conditioning on manufacturing method, material, or production constraints.
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
- takeoff distance bucket,
- payload bucket,
- wingspan limit,
- manufacturing method,
- minimum wall-thickness bucket,
- maximum part count bucket.

### 3. Replace the synthetic dataset

The current synthetic voxel generator is not enough. The minimum viable dataset is:
- 200-1000 aircraft-like examples,
- consistent voxelization or mesh-to-voxel preprocessing,
- metadata for the condition schema above,
- one train/validation split and one holdout split.

Even a noisy, partly procedural dataset is better than the current hand-drawn synthetic fuselage-and-wing voxel prior.

### 4. Make the model explicitly conditional

The current code only uses `target_speed` plus simple loss weights. The cheapest workable upgrade is:
- encode the structured condition vector with an MLP,
- inject it into the latent denoiser and decoder,
- train a conditional model first without expensive CFD in every step,
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
