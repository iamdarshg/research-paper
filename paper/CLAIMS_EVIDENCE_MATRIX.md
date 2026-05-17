# Claims Evidence Matrix

This matrix records the strongest claims that appeared in the paper and the wording currently allowed by the available evidence.

| claim_id | file | claim_type | earlier wording risk | allowed wording now | required citation or evidence | status |
| --- | --- | --- | --- | --- | --- | --- |
| C01 | `paper/main.tex` | abstract scope | implied validated aircraft design | proof-of-concept pipeline for aircraft-like voxel geometries | current repo architecture + sanity run | revised |
| C02 | `paper/main.tex` | abstract performance | implied structural viability and aerodynamic efficiency | CFD-informed scoring path with limited sanity evidence | stronger validation study for upgrade | revised |
| C03 | `paper/sections/introduction.tex` | contribution | implied 3D aircraft generation | latent generative pipeline for aircraft-like voxel artifacts | aircraft-specific validity tests for upgrade | revised |
| C04 | `paper/sections/introduction.tex` | contribution | implied demonstrated CFD-guided optimization | implementation path for CFD-informed evaluation | ablation with and without CFD term | revised |
| C05 | `paper/sections/introduction.tex` | contribution | implied structural viability | connectivity heuristics and validation framework | structural or load-path analysis | revised |
| C06 | `paper/sections/related-work.tex` | novelty | unclear novelty relative to 3D diffusion | novelty is the assembled proof-of-concept workflow and validation discipline | related-work citations already added | revised |
| C07 | `paper/sections/related-work.tex` | comparison | unclear distinction from topology optimization and CFD shape optimization | learned generative prior plus post hoc scoring rather than direct parameter optimization | topology/CFD optimization citations | revised |
| C08 | `paper/sections/methodology.tex` | methods | implied differentiating through CFD solver | current code routes geometries through an aerodynamic score path | differentiable-CFD experiment if upgraded | revised |
| C09 | `paper/sections/results-and-discussion.tex` | empirical | risk of aircraft-level interpretation | sanity-run evidence on freeform objects only | larger controlled benchmark | revised |
| C10 | `paper/sections/conclusion.tex` | project scope | implied fully AI-driven conditioned airplane generator | mission and manufacturing conditioning remain future work | real conditioned dataset + model support | revised |

## Upgrade Rule

Do not strengthen any `allowed wording now` entry until the required evidence exists and the corresponding gate in `paper/FINAL_RUN_GATES.md` is marked as passed.
