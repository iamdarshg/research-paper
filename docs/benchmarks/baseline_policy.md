# Baseline And Statistics Policy

Claim-bearing comparisons require named baselines and sufficient repeated runs.
Smoke reports must not be described as baseline evidence.

## Required Baseline Set

The minimum claim-bearing baseline set is:

- `retrieval`
- `unconditional_checkpoint`
- `bundled_grounded_stl`

Additional procedural or external baselines may be added, but they do not remove
the minimum set above.

## Statistical Report Requirements

Multi-seed reports must include:

- seed list
- seed count
- metric mean
- metric sample standard deviation
- explicit `blocked` status when fewer than the minimum seed count is present

The helper functions in `CLI/multi_seed_eval.py` implement this report contract.

## Current Boundary

The checked-in `CLI/baseline_config.yaml` is a guardrail configuration, not a
publication-grade baseline package. It must be extended with claim-bearing
baselines and sufficient seeds before any superiority or optimization claim is
allowed.
