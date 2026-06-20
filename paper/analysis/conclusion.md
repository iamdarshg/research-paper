# Conclusion Sentence Audit

Source: `paper/sections/conclusion.tex`

Detector results:
- lmscan: `0.2158`, verdict `Likely human`, confidence `medium`
- RoBERTa detector: fake mean `0.000224`, fake max `0.000224`

Overall function: the conclusion restates the working implementation, reduced evidence package, and exact future work required before stronger conditioned-aircraft claims.

## Sentence Units

1. `The present implementation demonstrates a working proof-of-concept generative-design pipeline and a reduced freeform-object experiment that can be executed in the installed environment.`
   - Function: restates the achieved result.
   - Claim type: implementation and execution claim.
   - Word choices: `present implementation` limits scope; `working proof-of-concept` is positive but bounded; `reduced` and `installed environment` limit evidence strength.
   - Risk status: grounded.

2. `The observed step sweep suggests that longer consistency sampling can change the reported $L/D$ in this specific sanity setting, but those numbers remain tied to downsampled freeform objects and CPU CFD evaluation.`
   - Function: summarizes the sweep without claiming improvement.
   - Claim type: limited empirical observation.
   - Word choices: `suggests` and `can change` replace stronger causal language; `specific sanity setting` and `CPU CFD` prevent generalization.
   - Risk status: conservative.

3. `The OpenFOAM cross-check completes on the shared centered-cube validation object and gives the repository an external CFD comparison path.`
   - Function: summarizes external solver route.
   - Claim type: code-path/validation-route claim.
   - Word choices: `completes` avoids agreement or accuracy claims; `comparison path` is safer than `reference path`.
   - Risk status: grounded.

4. `The reduced protocol also assembles a passing package-level evidence report with manifest validation, heuristic aircraft-shape checks, grounded condition-response sweeps, manufacturing-bounds checks, and finite baseline statistics.`
   - Function: summarizes the final evidence package.
   - Claim type: final report claim.
   - Word choices: `reduced`, `heuristic`, `bounds`, and `finite` all keep the gate claims modest; `passing` is grounded by the final evidence JSON.
   - Risk status: grounded.

5. `The next revision should therefore prioritize solver calibration and a larger ablation study before presenting stronger aerodynamic conclusions.`
   - Function: defines immediate next work.
   - Claim type: future-work recommendation.
   - Word choices: `therefore` ties future work to current limits; `before` makes the gate sequence explicit.
   - Risk status: safe.

6. `Most importantly, the repository should not yet be described as a fully AI-driven airplane generator conditioned on flight profile, manufacturing method, or structural requirements.`
   - Function: blocks the most marketable but unsupported claim.
   - Claim type: negative claim.
   - Word choices: `not yet` leaves future room; `fully AI-driven airplane generator` names the overclaim exactly; the three condition types match user-facing expectations.
   - Risk status: critical.

7. `What exists today is structured-conditioning plumbing plus a reduced evidence package.`
   - Function: states the actual present status in plain language.
   - Claim type: implementation and evidence summary.
   - Word choices: `plumbing` is intentionally humble; `plus` is direct; `reduced` keeps scope honest.
   - Risk status: grounded.

8. `That stronger version needs more evidence first: a larger aircraft-like dataset, conditional ablations, stricter structural and aerodynamic checks, and repeated runs beyond the present reduced local protocol.`
   - Function: names the minimum future requirements.
   - Claim type: future-work gate.
   - Word choices: `stronger version` refers to the fully conditioned generator; `needs` is firm; `first` prevents premature claims; the list is concrete and auditable.
   - Risk status: future requirement, not current evidence.

## Audit Notes

The conclusion is deliberately plain. It does not end with a triumphant claim. It ends with the actual state of the work and the evidence still needed.
