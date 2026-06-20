# Validation and Testing Standards Sentence Audit

Source: `paper/sections/validation-and-testing-standards.tex`

Detector results:
- lmscan: `0.2434`, verdict `Likely human`, confidence `medium`
- RoBERTa detector: fake mean `0.000181`, fake max `0.000181`

Overall function: this section separates execution checks, solver comparison, and physical validation so the paper does not imply that code-path success equals aircraft validation.

## Sentence Units

1. `To keep validation claims bounded, we separate three levels of evidence:`
   - Function: introduces the evidence hierarchy.
   - Claim type: validation framing.
   - Word choices: `bounded` is the key limiter; `separate` signals that evidence types must not be merged.
   - Risk status: central guardrail.

2. `Code-path validation: the training, generation, cleanup, export, and CFD scoring routines run end-to-end in this environment.`
   - Function: defines the weakest evidence tier.
   - Claim type: execution claim.
   - Word choices: `run end-to-end` is concrete; `in this environment` limits portability.
   - Risk status: grounded by tests/final report.

3. `Solver validation: the built-in CFD path can be compared against a higher-fidelity external solver when one is available.`
   - Function: defines the middle evidence tier.
   - Claim type: validation-route claim.
   - Word choices: `can be compared` avoids claiming agreement; `when one is available` acknowledges dependency.
   - Risk status: safe.

4. `Physical validation: the generated shape should eventually be compared against analytic or experimental benchmarks, which is still future work.`
   - Function: defines the strongest evidence tier and marks it incomplete.
   - Claim type: future-work claim.
   - Word choices: `eventually` and `still future work` prevent present-tense overclaim.
   - Risk status: essential.

5. `Verification checks that the implemented steps are executed as intended.`
   - Function: distinguishes verification from validation.
   - Claim type: definition.
   - Word choices: `as intended` is safer than `correctly` because correctness requires deeper validation.
   - Risk status: corrected wording.

6. `In this revision we verified that:`
   - Function: introduces the verified items.
   - Claim type: structure sentence.
   - Word choices: `this revision` limits scope to the current branch.
   - Risk status: safe.

7. `the shape-plausibility loss is included in the training objective`
   - Function: lists a verified training-objective inclusion.
   - Claim type: code-path verification.
   - Word choices: `included` is more accurate than `contributes`; it avoids claiming measured effect.
   - Risk status: grounded.

8. `the connected-component cleanup hook removes smaller disconnected voxel components`
   - Function: lists a verified cleanup behavior.
   - Claim type: code-path verification.
   - Word choices: `removes smaller disconnected voxel components` is concrete; it avoids broad claims like `improves geometry`.
   - Risk status: grounded.

9. `STL export succeeds from the cleaned voxel grid`
   - Function: lists mesh export verification.
   - Claim type: code-path verification.
   - Word choices: `succeeds` is an execution claim, not a quality claim.
   - Risk status: grounded.

10. `the OpenFOAM export hook writes a runnable case directory skeleton, including \texttt{blockMeshDict}, \texttt{snappyHexMeshDict}, and surface STL output.`
   - Function: lists OpenFOAM export artifacts.
   - Claim type: code-path verification.
   - Word choices: `runnable case directory skeleton` says the export is concrete but not necessarily a validated simulation campaign; specific filenames make it auditable.
   - Risk status: grounded.

11. `Validation is now runnable on a single local machine with OpenFOAM installed.`
   - Function: states the validation-route execution environment.
   - Claim type: reproducibility/execution claim.
   - Word choices: `runnable` is safer than `reproducible`; `single local machine` and `with OpenFOAM installed` state prerequisites.
   - Risk status: grounded.

12. `The built-in CFD path is exercised on the reduced CPU pipeline, and the OpenFOAM sonicFoam benchmark is run on the shared centered-cube validation object.`
   - Function: states the two paths used for comparison.
   - Claim type: validation procedure.
   - Word choices: `exercised` and `run` are execution verbs; `reduced CPU pipeline` limits scope.
   - Risk status: grounded.

13. `The comparison uses the lower-level \texttt{forces} output when available, with a manual pressure integration fallback only if the function-object output is missing.`
   - Function: explains force extraction path.
   - Claim type: implementation detail.
   - Word choices: `when available` and `only if` clarify fallback logic; `manual` keeps implementation transparent.
   - Risk status: grounded.

14. `The paper therefore claims the following and nothing more: the export path is concrete, the benchmark comparison is measurable, and the shared validation geometry is documented explicitly so the solver-to-solver check can be repeated without ambiguity.`
   - Function: final claim boundary for validation.
   - Claim type: limitation and evidence interpretation.
   - Word choices: `nothing more` is intentionally strict; `concrete`, `measurable`, and `documented` are defensible words; `without ambiguity` refers to geometry specification, not scientific certainty.
   - Risk status: essential close.

## Audit Notes

This is one of the strongest anti-overclaim sections. It does not try to make validation sound bigger than it is. It defines a ladder: code execution now, solver comparison route now, physical validation later.
