# Handoff: AircraftVerse Corpus Regeneration and Compressed B Run

Updated: 2026-10-30

## What the next agent needs to know (short version)

- All repo work is pushed to `main`.
- A **self-terminating** GCP VM regenerates the real AircraftVerse corpus,
  compresses it to 32^3, validates it, **uploads artifacts to GCS**, then
  **deletes itself**. You do NOT need to manage or shut down any VM.
- The corpus pipeline is fully reproducible from the repo on ANY machine
  (local, another cloud, anywhere with internet + Python 3.10+).
- Nothing after this document depends on this machine or on your GCP access.

## Completed before this handoff

- Main branch is pushed and current (HEAD is the commit containing this doc).
- Cleanup + paper work: `73bb5c9`.
- Python 3.10 fix: `0268749`.
- GCP startup scripts: `17d712e`.
- Protocol wiring (`reference_evidence`, post-train `run_metadata`,
  `baseline_statistics`, manifest target passthrough): `c852bdf`.
- Full verification: 642 tests passed, 4 skipped; protocol tests green.
- The 20-record `first_training` protocol is correctly fail-closed: its `final`
  train class requires 600+ unique canonical geometries. A 20-record fixture
  cannot pretend to be a final claim-bearing training corpus.
- The original 128^3 `build/final_training_corpus_20260830_v2` artifact was
  confirmed missing from the repo and local disk; it is not recoverable.

## The GCP corpus job (self-contained, no agent action needed)

Instance: created fresh by whoever runs the command below (never left running).

Project: `project-ba289c9c-3c25-4e15-9dc`
Zone: `asia-south1-c`
Machine: `e2-standard-8`, on-demand (project has zero spot quota), 150 GB disk.

How to launch it (or have any agent/CI/other machine launch it):

```bash
gcloud compute instances create research-paper-corpus-builder \
  --project=project-ba289c9c-3c25-4e15-9dc \
  --zone=asia-south1-c \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script=scripts/gcp/corpus-builder-startup.sh
```

The startup script:

1. Clones `main`.
2. Installs Python deps + `gsutil`.
3. Downloads AircraftVerse ZIP shards from Zenodo (no local upload needed).
4. Builds 2,000 canonical 96^3 records.
5. Rebuilds a compressed 32^3 corpus from those (2,000 originals, no
   perturbations/procedural for the first bounded run).
6. Validates claim-bearing at target 600.
7. Uploads everything to GCS, writes `UPLOAD_STATUS.json`, then **deletes the VM**.

## Artifacts (in GCS, any machine can fetch)

Bucket: `gs://research-paper-corpus-20261030`
Prefix: `runs/corpus-builder/`

Key files after a successful run:

```
gs://research-paper-corpus-20261030/runs/corpus-builder/corpus_build/aircraftverse_g96/aircraftverse_manifest.jsonl
gs://research-paper-corpus-20261030/runs/corpus-builder/corpus_build/final_corpus_g32/combined_training_manifest.jsonl
gs://research-paper-corpus-20261030/runs/corpus-builder/corpus_build/final_corpus_g32/build_spec.json
gs://research-paper-corpus-20261030/runs/corpus-builder/corpus_build/claim_validation.json
gs://research-paper-corpus-20261030/runs/corpus-builder/UPLOAD_STATUS.json
```

If `UPLOAD_STATUS.json` says `upload_failed`, the VM was stopped (not deleted)
automatically; check `corpus-builder-setup.log` in the bucket. But the default
path is success → upload → delete.

Fetch from any machine with gsutil, or the GCS web console:

```bash
gsutil -m cp -r gs://research-paper-corpus-20261030/runs/corpus-builder/ .
```

## Entire pipeline is portable (no dependence on this machine)

All logic lives in the repo:

- `scripts/gcp/corpus-builder-startup.sh` — full corpus regeneration.
- `scripts/gcp/rebuild_with_retry.py` — handles strict expected-count gates.
- `scripts/gcp/finish_upload_and_self_terminate.py` — upload + self-delete.

You can run the same steps on any Linux box, or in a container, without GCP.
The only bonus GCP gives us is a fast Zenodo download and cheap on-demand
compute.

## Next step: the compressed B training run

Do **not** run the old 128^3 protocol against the 32^3 manifest. Create a
dedicated compressed protocol with at least:

- `grid_size: 32`
- `latent_dim: 16`
- `dataset_manifest: <combined_training_manifest.jsonl from GCS>`
- `run_class: final` only after the manifest validates at target 600
- checkpointing enabled; a bounded epoch/update budget first

Preflight:

```bash
python CLI/validate_manifest.py \
  --manifest combined_training_manifest.jsonl \
  --level claim-bearing \
  --unique-geometry-target 600
```

B fits on the local RTX 4060 (8 GB) or a small GCP GPU. Anything heavy should
again run on a **self-terminating** spot/on-demand VM that uploads its own
artifacts, following the same pattern as the corpus builder.

## CFD / surrogate honesty rule

The paper's `CFD-guided training` claim refers to measured D3Q27 solver calls
plus SPSA. Swapping in a surrogate changes the claim; it must be labeled
`surrogate-guided`. Recommended sequence:

1. Run B with the direct solver disabled or very sparse first.
2. If a surrogate is used, fit it on a declared set of real D3Q27 evaluations,
   validate error + rank correlation on held-out geometries, record provenance.
3. Keep a small periodic real-D3Q27 audit if any solver-guided wording remains.
4. Never claim validated aerodynamics from a surrogate-only run.

## Cost posture

- Corpus builder: roughly $0.29/hr, runs a few hours max, self-deletes.
- Budget target remains under $10 total.
- If you see a leftover VM (e.g., upload failed before stop), it self-stopped;
  you may delete or restart it — but the happy path requires no action.

## Why 32^3 is the right compromise

- Real AircraftVerse source geometry, not the 20-record reference fixture.
- 2,000 distinct designs with canonical provenance.
- ~27x less voxel storage/compute than 96^3.
- Runs on the local 4060 or a small GPU.
- It is NOT the original 128^3 publication-scale experiment; the paper must
  report the compressed protocol and its limits honestly.