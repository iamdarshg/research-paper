# Training CPU/GPU Residency Audit

Generated: `2026-07-16`

## Scope

This audit describes the current `96^3`, 28.1M-parameter training path after
the fused D3Q27 stream/BFL implementation. It separates work that is actually
CPU-resident from GPU work controlled by Python. Moving a function to CUDA is
recommended only when it preserves every solver evaluation and loss component.

The percentages below come from the pre-fusion 500-geometry profile. They are
useful for ordering work, but must be remeasured during the resumed 1k-corpus
epoch because fused BFL changes the denominator.

## Current Residency

| Work | Current location | Prior measured share | Would GPU help? | Recommendation |
| --- | --- | ---: | --- | --- |
| SciPy signed-distance EDT | CPU, two `distance_transform_edt` calls per new mask | 10.76% | High | Highest-value CPU migration. Implement an exact or parity-bounded CUDA EDT/q path and gate SDF, crossing-link, q, force, coefficient, direct-loss, and SPSA-gradient parity. |
| Probability top-k and binary-mask construction | CPU by explicit design, with GPU-to-CPU logits and CPU-to-GPU mask copies | Included in direct-objective time | High after EDT moves | Keep it on GPU with the SDF/q path. Moving top-k alone gives limited benefit because validity and SciPy EDT still consume the CPU mask. |
| Aircraft connectivity and validity | CPU, NumPy/SciPy connected components plus scalar shape checks | 1.24% | Low to medium | Preserve as a real loss. A CUDA connected-components implementation may help after EDT and solver batching, but it is not a first-order bottleneck. |
| SPSA plus/minus orchestration | Python CPU control around 33 sequential measured objectives | 51.29% direct-objective envelope | High | Add a leading solver batch and execute each antithetic pair together. This retains all 33 values while reducing launch and synchronization gaps. Start at batch two because measured VRAM headroom is about 4.7 GB. |
| Loss-component consolidation | Python/NumPy scalars after each solver return | Not isolated | Medium with batching | Return per-geometry device tensors, join aero/connectivity/validity by index, and copy only the final component record to the host. Do not remove any component. |
| Optimizer state | CPU between updates; copied to GPU for AdamW step and back | 1.16%, about 429 MiB transfer/update | Medium | Test persistent GPU-resident optimizer state. The prior 3.42/8.19 GB peak leaves capacity, but record peak memory during the step before enabling it. |
| Dataset JSONL/NumPy loading and collation | CPU, `DataLoader(num_workers=0)` | No material hotspot | Low | Keep on CPU for now. More workers can increase RAM pressure without improving a batch-size-one loop dominated by CFD. |
| Checkpoint/history serialization | CPU and SSD, epoch boundaries | Not a training-loop hotspot | None | Keep on CPU. It is provenance work and does not affect per-update GPU duty cycle. |
| Promotion-gate reporting | CPU scalar aggregation and validity checks | Outside the dominant update path | Low | Keep until the training critical path is fixed; promotion must remain deterministic and inspectable. |

## GPU Work That Is Still Underutilized

These blocks are already on the GPU and should not be described as CPU work:

- the 28.1M-parameter denoiser, decoder, attention, and autograd;
- the full-grid grounded reconstruction objective;
- SPSA perturbation generation and interpolation;
- MRT collision and macroscopic LBM fields;
- the fused pull-stream/BFL kernel;
- momentum-exchange arithmetic, before scalar results are read by the host.

The dense MRT collision remains a major target, but the needed change is GPU
kernel fusion, not CPU-to-GPU migration. It still materializes intermediate
moment grids and launches PyTorch operations around two `27 x 27` transforms.
A Triton/CUDA collision kernel should be evaluated for register spills and must
pass conservation, field, force, direct-loss, and gradient parity.

The q/link construction after EDT is also GPU-resident but Python-looped. Its
26 directions read velocity components with scalar `.item()` calls and launch
separate masked operations. A fused q/link kernel can remove those host-visible
gaps after the EDT output reaches the GPU.

## Recommended Order

1. Profile the resumed fused-backend epoch and establish a new phase baseline.
2. Implement batch-two antithetic SPSA execution without reducing the 33 direct
   solver values or changing loss consolidation.
3. Replace CPU EDT plus q preparation with a parity-gated CUDA path, keeping the
   SciPy implementation as the reference backend.
4. Keep top-k masks and solver inputs on GPU once no CPU consumer requires the
   mask before the solve.
5. Fuse MRT collision on GPU and re-run conservation/OpenFOAM regression gates.
6. Test persistent GPU optimizer state under a measured VRAM ceiling.
7. Consider GPU connectivity/validity only if the new profile makes it material.

## Non-Negotiable Gates

No migration may substitute a surrogate, constant q, sampled geometry loss,
reduced grid, fewer SPSA directions, or fewer CFD steps. Each candidate must
match the corrected D3Q27 opposite-direction table and preserve the actual
aerodynamic, occupancy, connectivity, and aircraft-validity terms in the final
loss. Unsupported GPU environments must fall back explicitly to the reference
implementation and record that backend in run metadata.
