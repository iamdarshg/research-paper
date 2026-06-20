# Methodology Sentence Audit

Source: `paper/sections/methodology.tex`

Detector results:
- lmscan: `0.2701`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.008173`, fake max `0.043167`

Overall function: this section describes the implemented architecture and the public Airshow corpus-construction path while avoiding claims that the solver is fully validated, differentiable through training, or proven to improve aircraft performance.

## Sentence Units

1. `The current repository implements a proof-of-concept generative workflow built from four components: a noise scheduler, a latent diffusion-style denoiser, a latent-to-3D converter, and an internal CFD-oriented evaluator.`
   - Function: defines the architecture at a high level.
   - Claim type: implementation claim.
   - Word choices: `current repository` anchors the claim to code; `proof-of-concept` limits maturity; `diffusion-style` avoids objective novelty; `CFD-oriented` avoids validated CFD overclaim.
   - Risk status: grounded.

2. `The codebase now has two evidence tracks: deterministic synthetic/procedural smoke data retained for continuity, and a public VSP Airshow corpus used for the grounded training smoke run reported in Section \ref{sec:results}.`
   - Function: updates the data-source scope after the Airshow addition.
   - Claim type: evidence inventory.
   - Word choices: `two evidence tracks` makes the split explicit; `retained for continuity` explains why synthetic/procedural results remain; `public VSP Airshow corpus` names the grounded source; `smoke run` prevents treating the new corpus as final validation.
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

## Airshow Corpus Construction Sentence Audit

A1. `The grounded corpus path collects public model documents from the VSP Airshow web app, keeps only records whose Airshow license identifiers map to CC0, CC BY, or CC BY-SA, and requires a public preview-geometry URL before a record can enter the training manifest \cite{openvspGithub,openvspLicense,vspAirshow}.`
   - Function: states the inclusion rule for the new corpus.
   - Claim type: data-provenance and license-filtering claim.
   - Word choices: `public model documents` avoids private or scraped-data implications; `keeps only` makes the license gate strict; `requires` states the geometry availability rule; `training manifest` names the artifact being controlled.
   - Risk status: grounded by `CLI/build_airshow_corpus.py` and the corpus report.

A2. `The builder parses available X3D indexed-face-set geometry, normalizes each mesh into a centered unit cube, voxelizes the occupied geometry into a fixed grid, and records source URLs, license metadata, geometry hashes, voxel hashes, split assignment, and preprocessing provenance.`
   - Function: describes the geometry conversion method.
   - Claim type: preprocessing method claim.
   - Word choices: `available` avoids saying every Airshow record has usable geometry; `centered unit cube` states the normalization; `fixed grid` covers the `16^3`, `32^3`, and `64^3` reruns without pretending resolution alone improves validity; the hash/provenance list makes the corpus auditable.
   - Risk status: grounded by the builder and manifest.

A3. `The deterministic split assignment is stored in the manifest so that later validation, training, and generation commands can be tied back to the same corpus lineage.`
   - Function: explains why split metadata matters.
   - Claim type: reproducibility rationale.
   - Word choices: `deterministic` and `stored` make the split repeatable; `lineage` connects later claims to the same manifest rather than loose files.
   - Risk status: safe.

A4. `Corpus construction summary for the public VSP Airshow smoke run.`
   - Function: figure-caption title.
   - Claim type: figure framing.
   - Word choices: `summary` and `smoke run` keep the visual from sounding like a final benchmark.
   - Risk status: safe.

A5. `The plot shows the observed Airshow model-document count, license-and-geometry filtering, converted voxel records, admitted license mix, and deterministic train/validation/test/holdout split.`
   - Function: tells the reader exactly what Figure \ref{fig:airshow_corpus_summary} encodes.
   - Claim type: figure-content claim.
   - Word choices: `observed` ties counts to one run; `admitted` reminds the reader that licenses are filtered; `deterministic` matches the split policy.
   - Risk status: grounded by `corpus_report.json`.

A6. `These counts document data provenance and coverage; they do not certify manufacturer approval or aircraft performance.`
   - Function: blocks a visual overread of the corpus plot.
   - Claim type: limitation.
   - Word choices: `document` is allowed; `do not certify` directly rejects the strongest unsupported interpretation.
   - Risk status: critical.

A7. `The run used in this paper observed 381 public Airshow model documents, admitted 357 license-and-geometry-eligible documents, and converted 355 records after two public storage URLs returned 404.`
   - Function: gives the exact corpus arithmetic.
   - Claim type: empirical run result.
   - Word choices: `used in this paper` anchors the count; the three numbers make the funnel auditable; `returned 404` explains the two-record gap without inventing a qualitative reason.
   - Risk status: grounded by `build/airshow_grounded_corpus_20260620/corpus_report.json`.

A8. `Figure \ref{fig:airshow_corpus_summary} visualizes that corpus funnel together with the license and split distributions.`
   - Function: links the prose counts to the plot.
   - Claim type: figure-navigation sentence.
   - Word choices: `visualizes` is neutral; `funnel`, `license`, and `split` match the three panels.
   - Risk status: safe.

A9. `Airshow model names, manufacturer fields, URLs, and license labels are treated as source metadata only.`
   - Function: separates source metadata from certification.
   - Claim type: provenance boundary.
   - Word choices: `treated as` describes the paper's evidentiary use; `only` prevents implied endorsement.
   - Risk status: essential.

A10. `The additional mission and manufacturing fields used by the condition vector are deterministic repository inferences from geometry and defaults, not factual claims made by NASA, Lockheed, OpenVSP, or any other named source.`
   - Function: answers the ground-truth concern directly.
   - Claim type: inferred-field limitation.
   - Word choices: `additional` distinguishes these fields from source metadata; `deterministic repository inferences` says exactly where they come from; naming NASA, Lockheed, and OpenVSP blocks accidental attribution.
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

24. `A simplified view of the current training objective must distinguish the differentiable optimizer loss from detached scoring diagnostics.`
   - Function: introduces the loss section by correcting the key implementation distinction.
   - Claim type: method and limitation claim.
   - Word choices: `must distinguish` is deliberately firm because the debug pass showed the older combined-loss wording was misleading; `differentiable optimizer loss` names what actually drives gradients; `detached scoring diagnostics` names what is logged but not backpropagated.
   - Risk status: critical correction.

25. `The optimizer step uses diffusion, decoded-geometry reconstruction, direct generation reconstruction, and consistency terms:`
   - Function: lists the terms that actually enter the backpropagated objective.
   - Claim type: implementation claim.
   - Word choices: `optimizer step uses` ties the sentence to gradient flow; `direct generation reconstruction` identifies the added fast-sampler geometry term; the colon points to the equation rather than overexplaining in prose.
   - Risk status: grounded by `combine_training_loss_terms`.

26. `The diffusion loss is the mean squared error between the predicted noise and the actual noise.`
   - Function: defines diffusion loss.
   - Claim type: method claim.
   - Word choices: standard and precise.
   - Risk status: grounded.

27. `The decoded-geometry and generation-reconstruction terms compare voxel logits against the target voxel grid, while the consistency term supports the reduced-step sampler.`
   - Function: explains the non-diffusion optimizer terms.
   - Claim type: implementation claim.
   - Word choices: `compare` is accurate for the BCE-style loss; `voxel logits` avoids saying probabilities are compared at that point; `supports` avoids claiming validated sample quality improvement.
   - Risk status: grounded.

28. `The repository also logs a detached diagnostic total,`
   - Function: introduces the diagnostic-total equation.
   - Claim type: logging-contract claim.
   - Word choices: `also logs` separates monitoring from optimization; `detached` encodes the gradient boundary.
   - Risk status: grounded.

29. `where \(\mathcal{D}_{conn}\) is a connected-component diagnostic on thresholded voxels and \(\mathcal{D}_{aero}\) is an internal solver score on thresholded generated geometry.`
   - Function: defines the two diagnostic components.
   - Claim type: implementation claim.
   - Word choices: `diagnostic` replaces `loss` to avoid training-pressure overclaim; `thresholded` makes the nondifferentiable step visible; `internal solver score` avoids validated CFD wording.
   - Risk status: critical.

30. `These diagnostics are useful for monitoring and candidate ranking, but the present implementation does not backpropagate through connected-component labeling or through the CFD solver.`
   - Function: states what the diagnostics can and cannot do.
   - Claim type: limitation and future-route claim.
   - Word choices: `useful` preserves their value; `monitoring and candidate ranking` names legitimate uses; `does not backpropagate` directly answers the aero/connectivity-loss concern.
   - Risk status: essential.

31. `The codebase is designed to couple geometry generation and CFD-informed scoring during training.`
   - Function: opens training-loop coupling section.
   - Claim type: design intent and code-path claim.
   - Word choices: `designed to` allows implementation without claiming effect; `CFD-informed` stays cautious.
   - Risk status: acceptable.

32. `In the present implementation, that coupling should be understood as a practical score path rather than as a demonstrated differentiable CFD-training result.`
   - Function: prevents differentiability overclaim.
   - Claim type: limitation claim.
   - Word choices: `present implementation` and `practical score path` name the true status; `rather than` blocks a strong claim.
   - Risk status: critical.

33. `The current repository evidence is limited to a reduced final protocol, so we only claim that the training code can invoke aerodynamic scoring terms and route generated geometries through the evaluator.`
   - Function: states exactly what the evidence supports.
   - Claim type: evidence boundary.
   - Word choices: `only claim` is intentionally strict; `invoke` and `route` are code-path verbs, not performance verbs.
   - Risk status: central guardrail.

34. `In the current loop, training starts by sampling a noisy latent \(z_t\), asking the UNet for a noise estimate or denoised latent estimate, and decoding the resulting \(z_0\) candidate into a voxel grid \(V\).`
   - Function: describes the differentiable model-side training path in prose.
   - Claim type: procedural description.
   - Word choices: `current loop` anchors the description to implementation; `asking the UNet` is plainer than a step list; `candidate` keeps the decoded grid provisional.
   - Risk status: grounded.

35. `That decoded grid can then be routed through the internal D3Q27 LBM evaluator to compute \(C_L\) and \(C_D\) diagnostics.`
   - Function: states where solver diagnostics enter the loop.
   - Claim type: procedural and limitation claim.
   - Word choices: `can then be routed` avoids saying the route always drives gradients; `internal` and `diagnostics` keep solver evidence bounded.
   - Risk status: grounded.

36. `The optimizer update itself still comes from \(\mathcal{L}_{opt}\), while connectivity and aerodynamic scores are logged as detached diagnostics.`
   - Function: closes the loop description with the optimizer/diagnostic split.
   - Claim type: implementation contract.
   - Word choices: `itself still comes from` is intentionally explicit; `logged as detached diagnostics` is the core correction from the debugging pass.
   - Risk status: critical.

37. `In a stronger future version of this pipeline, the repository would need either a validated differentiable surrogate or a sequential candidate-scoring loop showing that solver feedback measurably changes generated candidates.`
   - Function: defines the future evidence needed for solver-guided generation claims.
   - Claim type: future validation requirement.
   - Word choices: `either` gives the two credible paths; `validated differentiable surrogate` and `sequential candidate-scoring loop` avoid pretending the current solver already supplies gradients; `measurably changes` defines the evidence threshold.
   - Risk status: good guardrail.

38. `For now, the CFD path is an implemented scoring mechanism rather than evidence of CFD-guided gradient training.`
   - Function: closes with the correct present claim.
   - Claim type: limitation and grounding claim.
   - Word choices: `For now` keeps future room; `implemented scoring mechanism` is the defensible claim; `rather than evidence` blocks the exact overclaim found during debugging.
   - Risk status: essential.

## Audit Notes

The methodology was deliberately rewritten away from phrases like "GPU-accelerated solver directly in the training loop" and "combined aerodynamic loss" because those read stronger than the verified implementation path. The current version is technical, but its verbs are mostly code verbs: `implements`, `routes`, `exports`, `scores`, `invokes`, `logs`, and `updates`.

## 2026-06-20 Resolution Addendum

The methodology now says the Airshow builder voxelizes into a fixed grid rather
than only a `16^3` grid. That wording is necessary because the same public
corpus path was rerun at `32^3` and `64^3`. The sentence still avoids implying
that higher resolution validated aircraft generation; it only states that the
corpus construction path supports multiple lattice sizes.
