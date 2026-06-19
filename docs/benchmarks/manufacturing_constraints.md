# Manufacturing And Structural Condition Gates

This gate rejects contradictory condition payloads before generation, benchmark,
or final-run interpretation.

## Executable Checks

`CLI/condition_feasibility.py` checks:

- engine count bounds are positive and ordered
- nonzero thrust requests require at least one engine
- payload mass bounds are non-negative and ordered
- wall-thickness bounds are positive and ordered
- manufacturing method minimum wall thickness is respected
- part-count bounds are positive and ordered
- speed and maneuverability fields are non-negative or positive as applicable

These checks are also called from `DesignSpec` validation in
`CLI/aircraft_diffusion_cfd.py`, so impossible payloads fail before generation.

## Heuristic Boundary

These checks are stronger than raw connectivity, but they are still feasibility
guards. They do not replace structural analysis, load-path validation, material
simulation, or manufacturing process qualification.
