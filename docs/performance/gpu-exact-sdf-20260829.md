# Exact GPU SDF/EDT backend probe — 2026-08-29

## Scope

The direct-objective flame graph identified the SciPy Euclidean distance
transform in `compute_all_link_distances` as the largest remaining CPU-bound
stage. This change adds an exact CuPy EDT backend without changing the signed
distance or BFL q semantics used by the SciPy reference path.

`training.sdf_backend: auto` is the production default. On CUDA, `auto` uses
CuPy only after a deterministic startup parity probe succeeds; otherwise it
uses the existing SciPy implementation. `gpu_exact` is explicit and
fail-closed: an unavailable dependency or failed parity probe raises instead of
silently selecting the old progressive-dilation approximation.

## Local parity evidence

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU, 8,188 MiB; PyTorch 2.9.1+cu130;
CuPy 14.2.0 (`cupy-cuda12x`); CUDA driver 591.59.

The focused SDF suite passed `5 passed`, including:

- structured 17×19×23 geometry;
- sparse/thin and random 19×17×21 geometry;
- explicit unavailable-backend failure behavior;
- reusable SciPy EDT workspace behavior.

The complete direct-objective parity gate passed `1 passed` on CUDA. It
compared the SciPy and exact-GPU paths for loss, SPSA gradient, and aerodynamic,
connectivity, and aircraft-validity telemetry sinks within `rtol=atol=1e-5`.

## Timing evidence

The runs used the repository’s fresh-init 128³ direct-only profiler with the
configured eight SPSA directions and five solver steps. Each result is one
complete direct objective evaluation.

| Backend | Total objective | `_get_q` phase | `simulate_aerodynamics` |
| --- | ---: | ---: | ---: |
| SciPy reference | 30.043 s | 0.819 s | 1.471 s |
| Exact GPU EDT, first run | 32.637 s | 0.014 s | 0.923 s |
| Exact GPU EDT, warmed repeat | 30.941 s | 0.013 s | 0.430 s |

The monitored warmed CUDA repeat completed with return code 0 in 52.735 s
including process startup, model construction, CuPy initialization, and the
single measured objective. Its peak sampled GPU allocation was 7,933 MiB and
peak process RSS was 4,473 MiB on the 8,188 MiB RTX 4060. The sampler saw 44
resource samples. This is a startup-inclusive envelope, not a claim that the
295M/128³ final model fits in 8 GiB.

The exact GPU path removes almost all measured q-preparation time and reduces
the solver phase, but this RTX 4060 run does not yet show an end-to-end wall
speedup because aircraft-validity and CPU-side objective work dominate the
remaining time. The result is therefore a measured SDF/q improvement, not an
unsupported claim that every GPU will be faster overall.

## Runtime/memory boundary

The warm path now transfers a compact `[D,H,W]` SDF directly to the solver
device and continues to build q on the GPU. It does not retain the larger
`[27,D,H,W]` q tensor in the prewarm pool. CUDA work is synchronized before a
worker future is consumed so the solver cannot observe an incomplete
cross-thread result.

CuPy installation for CUDA hosts is listed in
`CLI/requirements-gpu.txt`. CPU environments remain installable with the main
requirements and automatically retain the exact SciPy reference backend.
