# Handoff: AircraftVerse Corpus Regeneration and Compressed B Run

Updated: 2026-10-30

## Completed before this handoff

- Main branch is pushed and current.
- Paper/repo cleanup and architecture-alignment work is in commit `73bb5c9`.
- Python 3.10 compatibility fix for `run_protocol.py` is in `0268749`.
- GCP startup scripts were saved in `17d712e`.
- Protocol wiring for `reference_evidence`, post-training `run_metadata`, `baseline_statistics`, and manifest target passthrough is in `c852bdf`.
- Full prior verification: 642 tests passed, 4 skipped; focused protocol tests passed.
- The 20-record `first_training` reference protocol was found to be correctly fail-closed: its `final` training class requires 600+ unique canonical geometries, so it cannot pretend that a 20-record fixture is a final claim-bearing training corpus.
- The missing 128^3 `build/final_training_corpus_20260830_v2` artifact was confirmed absent from the repo and local disk.

## Current GCP job

Instance: `research-paper-corpus-builder`

Project: `project-ba289c9c-3c25-4e15-9dc`

Zone: `asia-south1-c`

Machine: `e2-standard-8` (8 vCPU, 32 GB RAM, on-demand because the project has zero preemptible CPU quota)

Disk: 150 GB `pd-balanced`

Expected cost: approximately a few tenths of a dollar per hour; stop/delete it when the corpus job finishes. The overall budget target is under $10.

The startup script is:

```text
scripts/gcp/corpus-builder-startup.sh
```

It runs:

1. Clone `main` from GitHub.
2. Install Python dependencies.
3. Download AircraftVerse ZIP shards from Zenodo automatically (no local 11.7 GB upload required).
4. Build 2,000 canonical AircraftVerse records at 96^3 with provenance and validity metadata.
5. Rebuild a compressed 32^3 corpus from those 96^3 sources using:
   - 2,000 real originals
   - zero perturbations for the first bounded run
   - zero procedural records for the first bounded run
6. Validate the compressed manifest at the 600-unique-geometry claim-bearing target.

The rebuild helper is:

```text
scripts/gcp/rebuild_with_retry.py
```

It exists because `rebuild_final_training_corpus.py` enforces exact expected counts. The helper retries with observed counts if the source admission count differs from the expected 2,000.

## How to inspect the corpus VM

The startup script logs to:

```text
/var/log/corpus-build-setup.log
/home/iamdarshg/corpus_build/aircraftverse_build.log
/home/iamdarshg/corpus_build/rebuild.log
/home/iamdarshg/corpus_build/claim_validation.out
```

The VM has a startup-script metadata workflow. For an easy read-back, use a temporary checkback startup script or SSH/SCP after the host key is accepted. Avoid leaving a checkback script as the permanent metadata script while the corpus build is still running.

Useful commands through gcloud:

```powershell
gcloud compute instances describe research-paper-corpus-builder `
  --project=project-ba289c9c-3c25-4e15-9dc `
  --zone=asia-south1-c

gcloud compute instances get-serial-port-output research-paper-corpus-builder `
  --project=project-ba289c9c-3c25-4e15-9dc `
  --zone=asia-south1-c --port=1
```

When complete, the important files are:

```text
/home/iamdarshg/corpus_build/aircraftverse_g96/aircraftverse_manifest.jsonl
/home/iamdarshg/corpus_build/final_corpus_g32/combined_training_manifest.jsonl
/home/iamdarshg/corpus_build/final_corpus_g32/build_spec.json
/home/iamdarshg/corpus_build/claim_validation.json
```

## Next step: B training

Do **not** run the old 128^3 protocol directly against this compressed manifest. Create a dedicated compressed-corpus protocol with at least:

- `grid_size: 32`
- `latent_dim: 16` or another explicitly budgeted width
- `dataset_manifest: /home/iamdarshg/corpus_build/final_corpus_g32/combined_training_manifest.jsonl`
- `run_class: final` only after verifying the manifest has canonicalization metadata and at least 600 unique geometries
- checkpointing enabled
- a bounded epoch/update budget first

Before starting B, run:

```bash
python CLI/validate_manifest.py \
  --manifest /home/iamdarshg/corpus_build/final_corpus_g32/combined_training_manifest.jsonl \
  --level claim-bearing \
  --unique-geometry-target 600
```

## CFD / surrogate decision

The current paper's `CFD-guided training` claim refers to measured D3Q27 solver calls plus SPSA. Replacing that with a surrogate changes the claim and must be labeled as `surrogate-guided`, not silently called CFD-guided.

Recommended budget-safe sequence:

1. Run B with the direct solver disabled or at a very sparse interval to establish that real-corpus training and checkpointing work.
2. If a surrogate is desired, train/fit it against a declared sample of real D3Q27 evaluations, validate its error and rank correlation on held-out geometries, and record the surrogate provenance.
3. Keep a small periodic real-D3Q27 audit if the paper wants to retain any solver-guided wording.
4. Do not claim validated aerodynamics from a surrogate-only run.

## Critical cleanup rule

The corpus VM is on-demand. After collecting the manifest, build spec, validation report, and any checkpoint needed for B, stop it immediately:

```powershell
gcloud compute instances stop research-paper-corpus-builder `
  --project=project-ba289c9c-3c25-4e15-9dc `
  --zone=asia-south1-c
```

Delete it when the disk artifacts are safely copied or persisted:

```powershell
gcloud compute instances delete research-paper-corpus-builder `
  --project=project-ba289c9c-3c25-4e15-9dc `
  --zone=asia-south1-c --quiet
```

Do not leave either this builder VM or a later training VM running unattended.

## Why the 32^3 corpus is the current compromise

- It is real AircraftVerse source geometry, not the 20-record deterministic reference fixture.
- It retains 2,000 distinct source designs and canonical provenance.
- 32^3 reduces voxel storage and solver/training cost by roughly 27x relative to 96^3.
- It can fit on the local RTX 4060 or a small GCP GPU, unlike the original 128^3 plan.
- It is not equivalent to the original 128^3 publication-scale experiment; the paper must report the compressed protocol and its limitations honestly.
