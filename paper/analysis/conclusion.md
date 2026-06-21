# Conclusion Sentence Audit

Source: `paper/sections/conclusion.tex`

Detector results:
- lmscan: `0.2420`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.186498`, fake max `0.456608`

Overall function: the conclusion restates the working implementation, reduced evidence package, and exact future work required before stronger conditioned-aircraft claims.

## Sentence Units

1. `The present implementation demonstrates a working proof-of-concept generative-design pipeline, a reduced freeform-object experiment, and a public VSP Airshow smoke-training run that can be executed in the installed environment.`
   - Function: restates the achieved result after the Airshow addition.
   - Claim type: implementation and execution claim.
   - Word choices: `present implementation` limits scope; `working proof-of-concept` is positive but bounded; `reduced` and `smoke-training` keep evidence strength modest; `installed environment` ties the claim to what was actually run.
   - Risk status: grounded.

2. `The Airshow addition moves the paper beyond synthetic-only training evidence by converting 355 traceable public geometry records into manifest-backed voxel corpora at \(16^3\), \(32^3\), \(64^3\), and \(96^3\).`
   - Function: states exactly what the new corpus contributes.
   - Claim type: corpus-addition claim.
   - Word choices: `beyond synthetic-only` explains why the addition matters; `355` is concrete; `traceable public geometry records` captures provenance; `manifest-backed voxel corpora` defines reproducibility across the lattice sizes without claiming higher resolution solved validity.
   - Risk status: grounded by the corpus report and manifest validation.

3. `It also exposes the current generator's limit: the generated Airshow-checkpoint flight-path samples are nonempty and solver-runnable at the completed resolutions, but they still fail aircraft-specific validity screens such as \texttt{span\_sanity}.`
   - Function: keeps the new corpus result from becoming an aircraft-validity overclaim.
   - Claim type: limitation and failure disclosure.
   - Word choices: `exposes` frames failure as useful evidence; `nonempty` and `solver-runnable` are the supported positives; `completed resolutions` avoids talking about `64^3` generation as though it ran; `still fail` makes the blocking result unavoidable.
   - Risk status: critical.

4. `Filtering the \(32^3\) corpus to 176 source-valid records and fixing a zero-learning-rate resume issue improved the generated samples to a span-sanity-only failure, but it did not produce a passing aircraft-validity result.`
   - Function: records the best source-valid rerun result without calling it success.
   - Claim type: debugging and generated-validity result.
   - Word choices: `176 source-valid records` anchors the filter size; `zero-learning-rate resume issue` names the concrete bug; `improved` credits the result; `but it did not produce` keeps the gate status honest.
   - Risk status: critical.

5. `The \(64^3\) corpus validates, but the dense decoder did not produce a checkpoint within the local run ceiling.`
   - Function: states the `64^3` outcome precisely.
   - Claim type: manifest and implementation-limit result.
   - Word choices: `corpus validates` is narrower than model succeeds; `dense decoder` names the likely architecture bottleneck; `local run ceiling` bounds the negative result to this environment.
   - Risk status: grounded.

5a. `A \(96^3\) coordinate-decoder run trained for three epochs over the full public corpus and produced one heuristic-screen-passing calibrated top-k export out of three flight-path cases; this is a useful implementation result, not a reliable aircraft-generation or aerodynamic-validation result.`
   - Function: records the strongest new positive result and its boundary in the same sentence.
   - Claim type: high-resolution implementation and limitation claim.
   - Word choices: `coordinate-decoder` names the architecture that made the run feasible; `one...out of three` prevents cherry-picking; `heuristic-screen-passing` avoids certification language; `not` blocks aircraft-generation and aerodynamic-validation overclaims.
   - Risk status: critical.

6. `The OpenFOAM cross-check completes on the shared centered-cube validation object and gives the repository an external CFD comparison path.`
   - Function: summarizes external solver route.
   - Claim type: code-path/validation-route claim.
   - Word choices: `completes` avoids agreement or accuracy claims; `comparison path` is safer than `reference path`.
   - Risk status: grounded.

7. `The reduced protocol also assembles a passing package-level evidence report with manifest validation, heuristic aircraft-shape checks, grounded condition-response sweeps, manufacturing-bounds checks, and finite baseline statistics.`
   - Function: summarizes the final evidence package.
   - Claim type: final report claim.
   - Word choices: `reduced`, `heuristic`, `bounds`, and `finite` all keep the gate claims modest; `passing` is grounded by the final evidence JSON.
   - Risk status: grounded.

8. `The next revision should therefore prioritize generated-sample validity, solver calibration, and larger ablation studies before presenting stronger aerodynamic conclusions.`
   - Function: defines immediate next work.
   - Claim type: future-work recommendation.
   - Word choices: `generated-sample validity` is now first because Airshow generation failed `span_sanity`; `solver calibration` and `larger ablation studies` name the remaining evidence gaps; `before` makes the gate sequence explicit.
   - Risk status: safe.

9. `Most importantly, the repository should not yet be described as a fully AI-driven airplane generator conditioned on flight profile, manufacturing method, or structural requirements.`
   - Function: blocks the most marketable but unsupported claim.
   - Claim type: negative claim.
   - Word choices: `not yet` leaves future room; `fully AI-driven airplane generator` names the overclaim exactly; the three condition types match user-facing expectations.
   - Risk status: critical.

10. `The direct solver-in-loop follow-up changes what the optimizer actually sees.`
   - Function: announces the key correction in plain language without overstating validation.
   - Claim type: implementation-semantics claim.
   - Word choices: `changes what the optimizer actually sees` explains why this is more than a logging cleanup; `actually` is used once to contrast optimized loss against detached diagnostics.
   - Risk status: grounded by the trainer change and metrics schema.

11. `Earlier runs could print aerodynamic and connectivity numbers without letting those numbers alter the weights.`
   - Function: states the old failure mode plainly.
   - Claim type: implementation-semantics claim.
   - Word choices: `print` and `alter the weights` are deliberately concrete; this makes the detached-monitor issue understandable without extra framework jargon.
   - Risk status: grounded by the gradient probe and old metrics.

12. `In the new path, scheduled batches decode a full grid, call the internal D3Q27 LBM evaluator, add the measured score to \texttt{optimization\_loss}, and use two-sided SPSA perturbations as the backward estimate.`
   - Function: states exactly what now makes the solver term optimizer-facing.
   - Claim type: implementation claim.
   - Word choices: `decode`, `call`, `add`, and `use` are active code-path verbs; `measured score` is precise about what is folded into the loss; `backward estimate` prevents pretending this is analytic autograd.
   - Risk status: grounded by `DirectSolverSPSALoss` and the completed runs.

13. `In the local grid sweep, continuing the \(96^3\) run drove the scheduled direct solver objective below the \(32^3\) and \(64^3\) values, but this remains internal-LBM smoke evidence rather than external PDE validation.`
   - Function: reports the improved high-resolution run and immediately bounds it.
   - Claim type: local run result and limitation.
   - Word choices: `continuing` is important because the plotted `96^3` point is not the matched two-epoch result; `drove` is less formulaic than `reduced`; `internal-LBM smoke evidence` keeps ground truth tiering explicit.
   - Risk status: grounded by `g96_more/checkpoints/training_metrics.json`.

14. `The present system is narrower: structured-conditioning plumbing, a reduced evidence package, a grounded public-corpus smoke run, a mixed higher-resolution result, a direct internal-solver training term with black-box SPSA gradients, and a sequential candidate optimizer that can reuse the same measured scores after generation.`
   - Function: states the actual present status in plain language.
   - Claim type: implementation and evidence summary.
   - Word choices: `narrower` blocks the airplane-generator overclaim; `plumbing` is intentionally humble; `mixed higher-resolution result` captures the one-of-three top-k export and the solver-loss continuation; `black-box SPSA gradients` states the derivative route without claiming analytic CFD.
   - Risk status: grounded.

15. `Stronger claims need stronger evidence: aircraft-like training runs that pass validity gates, conditional ablations, stricter structural and aerodynamic checks, decoder and export-calibration studies, repeated solver-in-loop benchmarks, OpenFOAM or comparable PDE validation, and repeated runs beyond the present reduced local protocol.`
   - Function: names the minimum future requirements.
   - Claim type: future-work gate.
   - Word choices: `Stronger claims need stronger evidence` is plain and compact; `pass validity gates` makes generated-shape success the priority; `repeated solver-in-loop benchmarks` is the new evidence bar after adding SPSA; `OpenFOAM or comparable PDE validation` preserves the ground-truth boundary.
   - Risk status: future requirement, not current evidence.

## Audit Notes

The conclusion is deliberately plain. It does not end with a triumphant claim. It ends with the actual state of the work and the evidence still needed.

## 2026-06-20 Resolution Addendum

The conclusion now names `16^3`, `32^3`, `64^3`, and `96^3` Airshow corpora, the
176-record source-valid filter, the zero-learning-rate resume fix, and the
loss-semantics correction. This keeps the ending honest: completed generated
samples remain solver-runnable but fail aircraft-specific validity checks, the
`64^3` corpus validated without a checkpoint being produced in the local run
ceiling, the `96^3` coordinate-decoder run produced only one heuristic-screen
pass out of three calibrated top-k exports, and the CFD-oriented diagnostics
are not yet differentiable training signals.

## 2026-06-21 `96^3` Addendum

The conclusion now includes the `96^3` coordinate-decoder result. The added
sentence says the model trained for three epochs over the full public corpus
and produced one heuristic-screen-passing calibrated top-k export out of three
flight-path cases. Its function is to acknowledge the strongest new positive
result while immediately blocking the overclaim that this is reliable aircraft
generation or aerodynamic validation.

The phrase `mixed higher-resolution result` replaces `higher-resolution
negative result` because the evidence is no longer purely negative. The
following future-work phrase, `decoder and export-calibration studies that make
higher-resolution training practical and reliable`, is intentionally broader
than `decoder changes`: the new run shows that the decoder can train at
`96^3`, but binary extraction and repeated validity remain unsolved.

## 2026-06-21 Direct Solver-In-Loop Addendum

The conclusion now separates three ideas that were previously too easy to blur:
the internal D3Q27 solver can be scheduled as a measured training objective, its
backward signal is an SPSA finite-difference estimate rather than analytic
solver autograd, and OpenFOAM remains the external PDE validation route. The
wording deliberately says `internal-LBM-guided smoke result` because the
improved `96^3` direct solver objective is real training evidence but still not
external ground truth.
