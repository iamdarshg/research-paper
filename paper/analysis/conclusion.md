# Conclusion Sentence Audit

Source: `paper/sections/conclusion.tex`

Detector results:
- lmscan: `0.2412`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.022462`, fake max `0.044746`

Overall function: the conclusion restates the working implementation, reduced evidence package, and exact future work required before stronger conditioned-aircraft claims.

## Sentence Units

1. `The present implementation demonstrates a working proof-of-concept generative-design pipeline, a reduced freeform-object experiment, and a public VSP Airshow smoke-training run that can be executed in the installed environment.`
   - Function: restates the achieved result after the Airshow addition.
   - Claim type: implementation and execution claim.
   - Word choices: `present implementation` limits scope; `working proof-of-concept` is positive but bounded; `reduced` and `smoke-training` keep evidence strength modest; `installed environment` ties the claim to what was actually run.
   - Risk status: grounded.

2. `The Airshow addition moves the paper beyond synthetic-only training evidence by converting 355 traceable public geometry records into manifest-backed voxel corpora at \(16^3\), \(32^3\), and \(64^3\).`
   - Function: states exactly what the new corpus contributes.
   - Claim type: corpus-addition claim.
   - Word choices: `beyond synthetic-only` explains why the addition matters; `355` is concrete; `traceable public geometry records` captures provenance; `manifest-backed voxel corpora` defines reproducibility across the three lattice sizes without claiming higher resolution solved validity.
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

5. `The \(64^3\) corpus validates, but the current dense decoder did not produce a checkpoint within the local run ceiling.`
   - Function: states the `64^3` outcome precisely.
   - Claim type: manifest and implementation-limit result.
   - Word choices: `corpus validates` is narrower than model succeeds; `dense decoder` names the likely architecture bottleneck; `local run ceiling` bounds the negative result to this environment.
   - Risk status: grounded.

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

10. `What exists today is structured-conditioning plumbing plus a reduced evidence package, a grounded public-corpus smoke run, a higher-resolution negative result, and a clarified CFD-oriented scoring path whose aerodynamic and connectivity diagnostics are not yet differentiable training signals.`
   - Function: states the actual present status in plain language.
   - Claim type: implementation and evidence summary.
   - Word choices: `plumbing` is intentionally humble; `reduced` keeps scope honest; `grounded public-corpus smoke run` adds the Airshow achievement without upgrading it to validation; `clarified CFD-oriented scoring path` describes the debug result without claiming solver-guided learning.
   - Risk status: grounded.

11. `That stronger version needs more evidence first: validity-bearing aircraft-like training runs, conditional ablations, stricter structural and aerodynamic checks, decoder changes that make higher-resolution training practical, a sequential candidate-ranking or surrogate-training loop for solver feedback, and repeated runs beyond the present reduced local protocol.`
   - Function: names the minimum future requirements.
   - Claim type: future-work gate.
   - Word choices: `validity-bearing` makes passing generated gates the priority; `decoder changes` reflects the `64^3` bottleneck; `sequential candidate-ranking or surrogate-training loop` is included because raw solver diagnostics are detached from gradients; `first` prevents premature claims.
   - Risk status: future requirement, not current evidence.

## Audit Notes

The conclusion is deliberately plain. It does not end with a triumphant claim. It ends with the actual state of the work and the evidence still needed.

## 2026-06-20 Resolution Addendum

The conclusion now names `16^3`, `32^3`, and `64^3` Airshow corpora, the
176-record source-valid filter, the zero-learning-rate resume fix, and the
loss-semantics correction. This keeps the ending honest: completed generated
samples remain solver-runnable but fail aircraft-specific validity checks, the
`64^3` corpus validated without a checkpoint being produced in the local run
ceiling, and the CFD-oriented diagnostics are not yet differentiable training
signals.
