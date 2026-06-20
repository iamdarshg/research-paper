# Local Paraphrase and Style Pass

Date: 2026-06-20

## Privacy Boundary

No unpublished manuscript text was pasted into online paraphrasers, browser
forms, or remote AI-writing detectors in this pass. That would transmit the
paper text to a third party and create an avoidable authorship/privacy trail.
The wording pass was performed locally against the repository files.

## Style Goal

The goal was not to make the paper sound less like an AI system. The goal was
to make it sound like an honest engineering paper:

- concrete where the evidence is concrete;
- cautious where the run is only smoke evidence;
- explicit when a generated sample failed a gate;
- plain about what the method is not yet able to claim.

## Edits Made

The paper was updated to include the `32^3` and `64^3` Airshow resolution
sweep. The wording deliberately uses terms such as "attempt", "smoke run",
"manifest-valid", "no checkpoint was produced", and "failed aircraft-validity
gates" because those are the ground-truth outcomes of the commands that were
run.

The highest-risk phrases remain blocked:

- "valid aircraft generation"
- "aerodynamic optimization"
- "publication-grade CFD"
- "outperforms prior approaches"
- "conditioned airplane generator"

The safe replacement wording is:

- "public-corpus code-path evidence"
- "generated aircraft-like voxel artifacts"
- "internal D3Q27 smoke scoring"
- "claim-boundary evidence"
- "structured-conditioning plumbing"

## Result

The revised wording is intentionally less flashy and more defensible. It keeps
the failed high-resolution results in the manuscript rather than smoothing them
away. That makes the paper more original as an evidence-hygiene document and
less vulnerable to the "AI-sounding overclaim" failure mode.
