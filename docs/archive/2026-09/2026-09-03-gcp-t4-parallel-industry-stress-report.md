# GCP T4 16^3 parallel industry-variation stress smoke — 2026-09-03

## Scope and safety boundary

This was a bounded diagnostic smoke, not a production training run. It used
one Spot NVIDIA Tesla T4, an eight-worker CUDA-stream executor, a 16^3 toy
model with 275,136 parameters, and no checkpoint or artifact output on the
VM. The hard maximum lifetime was 480 seconds. The user-authorized test
budget was $0.30; the current official T4 Spot GPU rate used for the estimate
was $0.20/hour, so the GPU portion of an eight-minute maximum run is about
$0.027, before the small VM and boot-disk charges. Actual billing was not
queried, so this is an upper-bound estimate rather than a billing receipt.

The first zone request (`us-central1-a`) was rejected by the provider's
temporary resource-pool capacity error before a VM was created. The same
request was then placed in `us-central1-b`.

## Experiment

The screen phase ran 30 cases in parallel: seven model parameterizations over
four seeds, plus baseline+heat and mHC-like+heat probes. Each screen case ran
256 updates with a 0.002 learning rate, a 0.5 standard-deviation weight, and
eight concurrent CUDA streams.

The stress phase scheduled 20 cases: baseline, LayerNorm, RMSNorm, the
mHC-like residual/RMS-normalization variant, and a gated residual variant over
four seeds. Each case was allowed up to 2,048 updates at learning rate 0.05,
with collapse polling every 64 updates. A case would stop at the first joint
collapse indication (`probability_std < 1e-4` and probability span `< 1e-3`).

The tested variations were:

- baseline residual MLP;
- post-block LayerNorm;
- RMSNorm;
- pre-normalized residual update;
- an explicitly small mHC-like residual/RMS-normalization probe;
- sigmoid-gated MLP; and
- sigmoid-gated residual updates.

These are industry-inspired stabilization patterns, not a claim that the toy
variants implement a production mHC or the complete 128^3 model.

## Observed results

The 30-case screen completed in 55.66 seconds. The observed cases were finite
and non-collapsed. Representative screen results showed probability standard
deviation around 0.37–0.383 and probability spans near 0.99–1.00. On this
particular toy architecture, baseline/LayerNorm/pre-norm/gated variants had
the lowest BCE-derived PPL proxy (roughly 1.000–1.004), while the RMSNorm and
mHC-like variants were around 1.010. This differs from the earlier, different
32-update smoke where the mHC-like probe was better; the disagreement is a
useful warning that these tiny proxy rankings are architecture- and schedule-
dependent.

The stress phase produced completed 2,048-update cases before the VM stopped.
The directly observed completed cases included three baseline seeds, all four
RMSNorm seeds, at least one LayerNorm seed, two mHC-like seeds, and one gated
residual seed. Those observed baseline and comparator cases remained finite,
with final probability standard deviation approximately 0.3834 and no
collapse flag. The VM's auto-deletion removed the serial-console retention
before the final aggregate JSON summary could be independently recovered, so
this report intentionally does not invent a complete 20-case aggregate or
claim that every stress seed finished.

## Interpretation

The requested high-parallel smoke successfully exercised the T4 and quickly
screened multiple stabilization families, but it did **not** reproduce
baseline collapse within the observed 2,048-update cases. Therefore it cannot
identify a winner for the production collapse issue, and no production
configuration was changed or restarted. The next scientifically useful step
would be a deterministic 16^3 multi-seed study with a retained result artifact
and the production loss/optimizer path, followed only by a guarded 128^3
validation if the independent collapse and performance gates pass.

## Cleanup and provenance

The temporary instance
`research-t4-16cubed-parallel-20260903` is absent, and its temporary boot disk
is absent. The pre-existing persistent disk
`research-l4-full-20260829` in `asia-south1-b` remains READY and attached; it
was not touched. The exact startup experiment is retained at
`build/gcp_t4_parallel_stress_startup.sh` and contains no checkpoint-writing
path, so this smoke did not consume local SSD space with checkpoints.

