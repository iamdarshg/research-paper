# Conclusion Sentence Audit

Source: `paper/sections/conclusion.tex`

Detector results:
- lmscan: `0.1767`, verdict `Human-written`, confidence `high`
- RoBERTa detector: fake mean `0.000174`, fake max `0.000174`

Overall function: the conclusion restates the working implementation, reduced evidence package, and exact future work required before stronger conditioned-aircraft claims.

## Sentence Units

1. `The present implementation demonstrates a working proof-of-concept generative-design pipeline, a reduced freeform-object experiment, and a public VSP Airshow smoke-training run that can be executed in the installed environment.`
   - Function: restates the achieved result after the Airshow addition.
   - Claim type: implementation and execution claim.
   - Word choices: `present implementation` limits scope; `working proof-of-concept` is positive but bounded; `reduced` and `smoke-training` keep evidence strength modest; `installed environment` ties the claim to what was actually run.
   - Risk status: grounded.

2. `The Airshow addition moves the paper beyond synthetic-only training evidence by converting 355 traceable public geometry records into a manifest-backed \(16^3\) voxel corpus.`
   - Function: states exactly what the new corpus contributes.
   - Claim type: corpus-addition claim.
   - Word choices: `beyond synthetic-only` explains why the addition matters; `355` is concrete; `traceable public geometry records` captures provenance; `manifest-backed` and `16^3` define reproducibility and resolution.
   - Risk status: grounded by the corpus report and manifest validation.

3. `It also exposes the current generator's limit: the three Airshow-checkpoint flight-path samples are nonempty and solver-runnable, but all fail the aircraft-specific \texttt{span\_sanity} screen.`
   - Function: keeps the new corpus result from becoming an aircraft-validity overclaim.
   - Claim type: limitation and failure disclosure.
   - Word choices: `exposes` frames failure as useful evidence; `nonempty` and `solver-runnable` are the supported positives; `but all fail` makes the blocking result unavoidable.
   - Risk status: critical.

4. `The OpenFOAM cross-check completes on the shared centered-cube validation object and gives the repository an external CFD comparison path.`
   - Function: summarizes external solver route.
   - Claim type: code-path/validation-route claim.
   - Word choices: `completes` avoids agreement or accuracy claims; `comparison path` is safer than `reference path`.
   - Risk status: grounded.

5. `The reduced protocol also assembles a passing package-level evidence report with manifest validation, heuristic aircraft-shape checks, grounded condition-response sweeps, manufacturing-bounds checks, and finite baseline statistics.`
   - Function: summarizes the final evidence package.
   - Claim type: final report claim.
   - Word choices: `reduced`, `heuristic`, `bounds`, and `finite` all keep the gate claims modest; `passing` is grounded by the final evidence JSON.
   - Risk status: grounded.

6. `The next revision should therefore prioritize generated-sample validity, solver calibration, and larger ablation studies before presenting stronger aerodynamic conclusions.`
   - Function: defines immediate next work.
   - Claim type: future-work recommendation.
   - Word choices: `generated-sample validity` is now first because Airshow generation failed `span_sanity`; `solver calibration` and `larger ablation studies` name the remaining evidence gaps; `before` makes the gate sequence explicit.
   - Risk status: safe.

7. `Most importantly, the repository should not yet be described as a fully AI-driven airplane generator conditioned on flight profile, manufacturing method, or structural requirements.`
   - Function: blocks the most marketable but unsupported claim.
   - Claim type: negative claim.
   - Word choices: `not yet` leaves future room; `fully AI-driven airplane generator` names the overclaim exactly; the three condition types match user-facing expectations.
   - Risk status: critical.

8. `What exists today is structured-conditioning plumbing plus a reduced evidence package and a grounded public-corpus smoke run.`
   - Function: states the actual present status in plain language.
   - Claim type: implementation and evidence summary.
   - Word choices: `plumbing` is intentionally humble; `reduced` keeps scope honest; `grounded public-corpus smoke run` adds the Airshow achievement without upgrading it to validation.
   - Risk status: grounded.

9. `That stronger version needs more evidence first: higher-resolution or validity-bearing aircraft-like training runs, conditional ablations, stricter structural and aerodynamic checks, and repeated runs beyond the present reduced local protocol.`
   - Function: names the minimum future requirements.
   - Claim type: future-work gate.
   - Word choices: `higher-resolution or validity-bearing` replaces the old generic dataset request now that a public corpus exists; `needs` is firm; `first` prevents premature claims; the list is concrete and auditable.
   - Risk status: future requirement, not current evidence.

## Audit Notes

The conclusion is deliberately plain. It does not end with a triumphant claim. It ends with the actual state of the work and the evidence still needed.

## 2026-06-20 Resolution Addendum

The conclusion now names `16^3`, `32^3`, and `64^3` Airshow corpora and states
the key limitation: completed generated samples remain solver-runnable but fail
aircraft-specific validity checks, while the `64^3` corpus validated without a
checkpoint being produced in the local run ceiling. This keeps the ending
honest and prevents "higher resolution" from becoming an unsupported substitute
for passing validity gates.
