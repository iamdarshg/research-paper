# Methodology Sentence Audit

Source: `paper/sections/methodology.tex`

Detector results:
- lmscan: `0.2240`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.001119`, fake max `0.002962`

Overall function: this section describes the implemented architecture while avoiding claims that the solver is fully validated, differentiable through training, or proven to improve aircraft performance.

## Sentence Units

1. `The current repository implements a proof-of-concept generative workflow built from four components: a noise scheduler, a latent diffusion-style denoiser, a latent-to-3D converter, and an internal CFD-oriented evaluator.`
   - Function: defines the architecture at a high level.
   - Claim type: implementation claim.
   - Word choices: `current repository` anchors the claim to code; `proof-of-concept` limits maturity; `diffusion-style` avoids objective novelty; `CFD-oriented` avoids validated CFD overclaim.
   - Risk status: grounded.

2. `The codebase operates on synthetic voxel training data and uses bounded smoke runs together with connectivity heuristics and aerodynamic score terms.`
   - Function: identifies training-data scope and evidence level.
   - Claim type: implementation and limitation claim.
   - Word choices: `synthetic`, `bounded`, `smoke runs`, and `heuristics` all reduce strength to match actual evidence.
   - Risk status: grounded.

3. `In this revision, the same path also carries a documented structured condition vector through dataset generation, latent construction, and checkpointed inference.`
   - Function: records the structured-conditioning plumbing.
   - Claim type: code-path claim.
   - Word choices: `carries` means the vector is routed, not proven effective; `documented` and `checkpointed` point to artifacts.
   - Risk status: grounded if condition-vector code and docs remain.

4. `Figure \ref{fig:architecture} should therefore be read as an implementation diagram for the present repository rather than as evidence of a fully validated aircraft-design system.`
   - Function: limits what the architecture figure proves.
   - Claim type: figure interpretation.
   - Word choices: `should therefore be read` directs interpretation; `rather than` blocks visual overclaim.
   - Risk status: important guardrail.

5. `Implementation-level view of the current repository.`
   - Function: figure caption title.
   - Claim type: figure framing.
   - Word choices: `Implementation-level` states the diagram is about code structure, not scientific proof.
   - Risk status: safe.

6. `A latent code and optional structured condition vector are routed through denoising, voxel decoding, cleanup, internal scoring, and export paths.`
   - Function: explains the diagram flow.
   - Claim type: pipeline claim.
   - Word choices: `optional` matches conditional support; `routed through` avoids claiming the condition has validated causal effect.
   - Risk status: grounded.

7. `The diagram describes the code path under test; it does not by itself establish differentiable training, structural viability, or publication-grade CFD validation.`
   - Function: explicit figure limitation.
   - Claim type: anti-overclaim sentence.
   - Word choices: `under test` and `does not by itself` keep evidence narrow; the three exclusions name likely overreads.
   - Risk status: essential.

8. `The forward diffusion process gradually adds Gaussian noise to the input data over a series of T timesteps.`
   - Function: introduces diffusion mechanics.
   - Claim type: method description.
   - Word choices: standard diffusion terminology; no novelty claim.
   - Risk status: grounded by common diffusion formulation.

9. `The noise schedule is defined by a variance schedule \beta_t, which is typically a linear or cosine schedule.`
   - Function: defines the schedule.
   - Claim type: method description.
   - Word choices: `typically` acknowledges alternatives; notation is standard.
   - Risk status: safe.

10. `The implementation uses a linear schedule, where \beta_t increases linearly from \beta_{start} to \beta_{end}.`
   - Function: states the repository's schedule choice.
   - Claim type: implementation claim.
   - Word choices: `implementation uses` ties to code; no performance claim attached.
   - Risk status: grounded if code matches.

11. `The forward process is defined as:`
   - Function: introduces the equation.
   - Claim type: mathematical setup.
   - Word choices: conventional.
   - Risk status: equation should remain standard and compile.

12. `During reverse sampling, the denoiser estimates the noise component of a noisy latent.`
   - Function: corrects earlier generic wording into implementation-level denoising.
   - Claim type: method description.
   - Word choices: `estimates` avoids certainty; `noise component` matches the loss target.
   - Risk status: grounded.

13. `The training loss compares predicted noise with sampled noise; sampling then uses that estimate to update the latent.`
   - Function: links training target and sampling use.
   - Claim type: method description.
   - Word choices: `compares` is precise for MSE; semicolon keeps cause/effect compact.
   - Risk status: grounded.

14. `The denoising model is a UNet-based architecture that operates in the latent space.`
   - Function: introduces the denoiser.
   - Claim type: architecture claim.
   - Word choices: `UNet-based` allows implementation-specific variation; `latent space` distinguishes from voxel-space denoising.
   - Risk status: grounded.

15. `The UNet takes as input a noisy latent vector, a timestep embedding, and an optional condition embedding, and outputs the predicted noise.`
   - Function: states model inputs and output.
   - Claim type: implementation claim.
   - Word choices: `optional condition embedding` matters because conditioning exists but is not always active; `predicted noise` matches diffusion training.
   - Risk status: grounded.

16. `The implementation projects the latent to a small spatial tensor and applies residual block stacks with additive skip connections.`
   - Function: describes implementation shape without generic UNet exaggeration.
   - Claim type: architecture claim.
   - Word choices: `small spatial tensor` and `additive skip connections` are more concrete than generic down/up-sampling claims.
   - Risk status: grounded if model code matches.

17. `The skip path is intended to preserve intermediate spatial features during denoising.`
   - Function: explains why skip connections exist.
   - Claim type: design rationale.
   - Word choices: `intended` avoids claiming measured effect; `intermediate spatial features` stays architecture-level.
   - Risk status: safe.

18. `The latent-to-3D converter is a multi-layer perceptron (MLP) that maps the denoised latent vector to a 3D voxel grid.`
   - Function: introduces the decoder.
   - Claim type: implementation claim.
   - Word choices: `maps` is neutral; `3D voxel grid` states representation.
   - Risk status: grounded.

19. `The MLP uses fully connected layers with ReLU activations; downstream training and generation code applies a sigmoid to convert logits into voxel probabilities.`
   - Function: corrects the earlier inaccurate final-sigmoid wording.
   - Claim type: implementation claim.
   - Word choices: `downstream` is important because sigmoid is not necessarily in the MLP layer itself; `logits` and `probabilities` are precise.
   - Risk status: grounded.

20. `The repository uses an internal lattice-Boltzmann-style CFD evaluator to score generated voxel grids.`
   - Function: introduces the internal evaluator.
   - Claim type: code-path claim.
   - Word choices: `style` and `score` are deliberately weaker than validated CFD simulation.
   - Risk status: grounded.

21. `The evaluator takes a voxel geometry and flow settings such as Mach number or Reynolds number and returns drag- and lift-related quantities.`
   - Function: describes solver inputs and outputs.
   - Claim type: implementation claim.
   - Word choices: `such as` avoids exhaustive interface claims; `related quantities` is safer than validated coefficients.
   - Risk status: grounded.

22. `In addition, the repository can export benchmark cases for comparison against an external OpenFOAM run.`
   - Function: states external-solver export support.
   - Claim type: implementation claim.
   - Word choices: `can export` means capability; `comparison` avoids claiming agreement.
   - Risk status: grounded.

23. `This combination is useful for code-path validation, but the present paper does not treat it as a substitute for a full aerodynamic validation campaign.`
   - Function: limits solver evidence.
   - Claim type: limitation.
   - Word choices: `useful` gives value; `not a substitute` blocks overclaim.
   - Risk status: essential.

24. `A simplified loss used to describe the main claim-bearing terms combines diffusion, connectivity, and aerodynamic terms.`
   - Function: introduces the loss without claiming full trainer completeness.
   - Claim type: simplified-method description.
   - Word choices: `simplified` is crucial because training code may include additional paths; `claim-bearing` tells the reader why these terms are discussed.
   - Risk status: grounded.

25. `The diffusion loss is the mean squared error between the predicted noise and the actual noise.`
   - Function: defines diffusion loss.
   - Claim type: method claim.
   - Word choices: standard and precise.
   - Risk status: grounded.

26. `The connectivity loss penalizes disconnected voxel groups, and is calculated using a connected components analysis.`
   - Function: defines connectivity term.
   - Claim type: implementation/method claim.
   - Word choices: `penalizes` describes training pressure; `connected components analysis` names mechanism.
   - Risk status: grounded.

27. `The aerodynamic term combines internal drag and lift-related scores as a heuristic pressure toward lower-drag and higher-lift proxy scores.`
   - Function: describes aero term without claiming validated optimization.
   - Claim type: heuristic scoring claim.
   - Word choices: `heuristic pressure` and `proxy scores` are deliberately cautious; `internal` distinguishes from validated external CFD.
   - Risk status: grounded and conservative.

28. `The codebase is designed to couple geometry generation and CFD-informed scoring during training.`
   - Function: opens training-loop coupling section.
   - Claim type: design intent and code-path claim.
   - Word choices: `designed to` allows implementation without claiming effect; `CFD-informed` stays cautious.
   - Risk status: acceptable.

29. `In the present implementation, that coupling should be understood as a practical score path rather than as a demonstrated differentiable CFD-training result.`
   - Function: prevents differentiability overclaim.
   - Claim type: limitation claim.
   - Word choices: `present implementation` and `practical score path` name the true status; `rather than` blocks a strong claim.
   - Risk status: critical.

30. `The current repository evidence is limited to a reduced final protocol, so we only claim that the training code can invoke aerodynamic scoring terms and route generated geometries through the evaluator.`
   - Function: states exactly what the evidence supports.
   - Claim type: evidence boundary.
   - Word choices: `only claim` is intentionally strict; `invoke` and `route` are code-path verbs, not performance verbs.
   - Risk status: central guardrail.

31. `Step 1: Sample noisy latent z_t.`
   - Function: first algorithm step.
   - Claim type: procedural description.
   - Word choices: terse step language supports reproducibility.
   - Risk status: grounded.

32. `Step 2: Predict noise or a denoised latent estimate using the UNet.`
   - Function: second algorithm step.
   - Claim type: procedural description.
   - Word choices: `or` is important because code paths can differ; avoids forcing a single theoretical interpretation.
   - Risk status: grounded.

33. `Step 3: Decode z_0 to voxel grid V.`
   - Function: third algorithm step.
   - Claim type: procedural description.
   - Word choices: notation is compact and standard.
   - Risk status: grounded.

34. `Step 4: Run the internal D3Q27 LBM evaluator on V.`
   - Function: fourth algorithm step.
   - Claim type: procedural description.
   - Word choices: `internal` and `evaluator` avoid pretending this is the final external solver.
   - Risk status: grounded.

35. `Step 5: Compute aerodynamic coefficients C_L, C_D.`
   - Function: fifth algorithm step.
   - Claim type: procedural description.
   - Word choices: coefficient notation is standard; this remains tied to the internal evaluator context.
   - Risk status: acceptable with surrounding caveats.

36. `Step 6: Accumulate scalar loss terms for the optimizer.`
   - Function: sixth algorithm step.
   - Claim type: procedural description.
   - Word choices: `accumulate scalar loss terms` is more accurate than claiming each term directly updates weights.
   - Risk status: grounded.

37. `In a stronger future version of this pipeline, the repository would need ablations showing that the aerodynamic term measurably changes either the training dynamics or the ranking of generated candidates.`
   - Function: defines the future evidence needed for mechanism claims.
   - Claim type: future validation requirement.
   - Word choices: `stronger future version` keeps the current claim limited; `measurably changes` defines what proof would look like.
   - Risk status: good guardrail.

38. `Until such evidence exists, the paper treats the CFD term as an implemented scoring mechanism whose training effect is not yet fully validated.`
   - Function: closes with the correct present claim.
   - Claim type: limitation and grounding claim.
   - Word choices: `implemented scoring mechanism` is the defensible claim; `not yet fully validated` reserves stronger conclusions.
   - Risk status: essential.

## Audit Notes

The methodology was deliberately rewritten away from phrases like "GPU-accelerated solver directly in the training loop" and "total loss" because those read stronger than the verified implementation path. The current version is technical, but its verbs are mostly code verbs: `implements`, `routes`, `exports`, `scores`, `invokes`, and `accumulates`.
