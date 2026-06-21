# Methodology Sentence Audit

Source: `paper/sections/methodology.tex`

Detector results:
- lmscan: `0.2647`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.051535`, fake max `0.412132`

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
   - Word choices: `available` avoids saying every Airshow record has usable geometry; `centered unit cube` states the normalization; `fixed grid` covers the `16^3`, `32^3`, `64^3`, and `96^3` reruns without pretending resolution alone improves validity; the hash/provenance list makes the corpus auditable.
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

18. `The latent-to-3D converter maps the denoised latent vector to voxel logits.`
   - Function: introduces the decoder without locking the paper to one decoder architecture.
   - Claim type: implementation claim.
   - Word choices: `maps` is neutral; `voxel logits` is precise and covers both dense and coordinate decoder paths.
   - Risk status: grounded.

19. `For lower-resolution smoke runs, this is a dense multi-layer perceptron (MLP) whose final layer emits the whole voxel grid.`
   - Function: preserves the earlier dense-decoder description where it still applies.
   - Claim type: implementation claim.
   - Word choices: `lower-resolution smoke runs` prevents applying this description to the `96^3` run; `whole voxel grid` explains the memory scaling problem.
   - Risk status: grounded.

20. `For the \(96^3\) follow-up, the implementation switches to a coordinate decoder that evaluates an MLP over the latent vector concatenated with normalized \((z,y,x)\) coordinates; training samples voxel coordinates with importance-weighted BCE so occupied coordinates can be oversampled without changing the intended sparse reconstruction objective.`
   - Function: records the high-resolution architecture and loss correction.
   - Claim type: implementation and training-objective claim.
   - Word choices: `follow-up` keeps this tied to the new run; `coordinate decoder` names the architecture; `importance-weighted` explains why positive oversampling is not treated as a changed target distribution; `intended sparse reconstruction objective` matches the Airshow occupancy data.
   - Risk status: grounded by `CLI/aircraft_diffusion_cfd.py` and the `96^3` report.

21. `Downstream training and generation code applies a sigmoid to convert logits into voxel probabilities.`
   - Function: keeps the logit/probability boundary clear.
   - Claim type: implementation claim.
   - Word choices: `downstream` is important because sigmoid is not necessarily in the decoder layer itself; `logits` and `probabilities` are precise.
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

24. `A simplified view of the current training objective distinguishes the base differentiable model losses from the measured solver-in-loop term.`
   - Function: opens the loss section with the corrected split.
   - Claim type: method and implementation-semantics claim.
   - Word choices: `base differentiable model losses` names the normal autograd path; `measured solver-in-loop term` says the CFD term is evaluated by the solver rather than invented by the paper.
   - Risk status: critical correction.

25. `The optimizer step uses diffusion, decoded-geometry reconstruction, direct generation reconstruction, consistency, and, when scheduled, a direct internal-solver objective:`
   - Function: lists what can enter the backpropagated optimizer scalar.
   - Claim type: implementation claim.
   - Word choices: `when scheduled` explains why the term is not present on every batch; `direct internal-solver objective` rejects the old detached-monitor-only interpretation.
   - Risk status: grounded by `combine_training_loss_terms` and `DirectSolverSPSALoss`.

26. `The diffusion loss is the mean squared error between the predicted noise and the actual noise.`
   - Function: defines diffusion loss.
   - Claim type: method claim.
   - Word choices: standard and precise.
   - Risk status: grounded.

27. `The decoded-geometry and generation-reconstruction terms compare voxel logits against the target voxel grid, while the consistency term supports the reduced-step sampler.`
   - Function: explains the non-diffusion model losses.
   - Claim type: implementation claim.
   - Word choices: `compare` is accurate for BCE-style reconstruction; `voxel logits` keeps the logit/probability boundary clear; `supports` avoids overclaiming sampler quality.
   - Risk status: grounded.

28. `The solver term materializes voxel probabilities into a full grid, converts them to a binary geometry by thresholding or top-k occupancy selection, calls the internal D3Q27 lattice-Boltzmann evaluator on that geometry, and combines the measured aerodynamic score with an optional exact connected-component penalty.`
   - Function: explains what the direct solver loss actually does.
   - Claim type: implementation claim.
   - Word choices: `materializes`, `converts`, `calls`, and `combines` are code-path verbs; `measured` is important because the scalar is produced by solver evaluation; `optional exact connected-component penalty` accounts for the connectivity contribution without implying analytic differentiability.
   - Risk status: grounded.

29. `This path is not a learned stand-in: the scalar used in \(\mathcal{L}_{solver\mbox{-}SPSA}\) comes from solver calls made during training.`
   - Function: directly addresses the rejected learned-substitute path.
   - Claim type: implementation boundary.
   - Word choices: `not a learned stand-in` is plain enough to prevent misreading; `solver calls made during training` gives the evidence mechanism.
   - Risk status: essential.

30. `The connected-component operation, binary materialization step, and internal CFD evaluator are not ordinary PyTorch operations with analytic gradients.`
   - Function: explains why a derivative estimator is needed.
   - Claim type: limitation and method rationale.
   - Word choices: `ordinary PyTorch operations` and `analytic gradients` are precise about the autograd boundary without saying gradients are impossible in every sense.
   - Risk status: grounded.

31. `The training loop therefore uses a two-sided simultaneous perturbation estimate when this term is active.`
   - Function: names the derivative-estimation method.
   - Claim type: method claim.
   - Word choices: `therefore` ties the estimator to the nondifferentiable solver boundary; `when this term is active` matches scheduled training.
   - Risk status: grounded.

32. `For a probability grid \(V\), a perturbation field \(\Delta\), and perturbation scale \(\epsilon\), it evaluates the measured objective on \(V+\epsilon\Delta\) and \(V-\epsilon\Delta\) and uses ... as the backward signal.`
   - Function: explains the SPSA equation in prose.
   - Claim type: mathematical method claim.
   - Word choices: `measured objective` reinforces that both sides call the real objective; `backward signal` avoids saying exact gradient.
   - Risk status: grounded.

33. `The implementation can draw \(\Delta\) directly at voxel resolution or on a lower-frequency grid that is upsampled to the solver lattice before evaluation.`
   - Function: records the variance-control option used in the final sweep.
   - Claim type: implementation option.
   - Word choices: `can draw` allows both modes; `lower-frequency grid` explains the `8^3` perturbation field without making it a new physics claim.
   - Risk status: grounded.

34. `The latter reduces high-frequency finite-difference noise and keeps the expensive solver work sequential and scheduled rather than evaluating many independent CFD calls in parallel.`
   - Function: gives the engineering reason for low-frequency SPSA.
   - Claim type: design rationale.
   - Word choices: `reduces` refers to estimator noise structure; `sequential and scheduled` matches the user requirement and the implementation; `expensive` justifies the scheduling.
   - Risk status: reasonable, but should remain tied to local smoke evidence.

35. `The exact connected-component score and raw internal solver score can still be run as measured monitors,`
   - Function: introduces the diagnostic-total equation.
   - Claim type: logging-contract claim.
   - Word choices: `still` says monitors were not removed; `measured monitors` separates them from the optimizer-facing scheduled term.
   - Risk status: grounded.

36. `where \(\mathcal{D}_{conn}\) is a connected-component score on thresholded voxels and \(\mathcal{D}_{aero}\) is an internal solver score on thresholded generated geometry.`
   - Function: defines the two detached monitor components.
   - Claim type: implementation claim.
   - Word choices: `score` rather than `loss` prevents accidental optimizer overclaim; `thresholded` exposes the nondifferentiable conversion.
   - Risk status: grounded.

37. `In the direct-solver runs, those detached monitor intervals are set to zero so the limited full-grid compute is spent on the solver term that contributes to \(\mathcal{L}_{opt}\).`
   - Function: explains why logged aero/connectivity monitor values remain zero in the direct sweep.
   - Claim type: run-configuration explanation.
   - Word choices: `detached monitor intervals` names the zero fields correctly; `contributes to L_opt` makes clear where the solver compute goes.
   - Risk status: critical.

38. `The reported \texttt{direct\_solver\_loss} is the optimizer-facing term averaged over all training batches, and \texttt{direct\_solver\_eval\_loss} is the mean over the scheduled solver evaluations.`
   - Function: defines the two direct-solver metrics.
   - Claim type: metric semantics.
   - Word choices: `optimizer-facing` and `scheduled` prevent readers from comparing the two averages incorrectly.
   - Risk status: grounded.

39. `The codebase is designed to couple geometry generation and CFD-informed scoring during training.`
   - Function: opens training-loop coupling section.
   - Claim type: design intent and code-path claim.
   - Word choices: `designed to` allows implementation without claiming final design validity; `CFD-informed` stays cautious.
   - Risk status: acceptable.

40. `In the present implementation, that coupling has two honest forms: direct black-box solver optimization inside the training loop and measured black-box candidate optimization after generation.`
   - Function: states the two solver-feedback routes now implemented.
   - Claim type: implementation claim.
   - Word choices: `honest forms` is blunt but useful; `direct black-box` distinguishes SPSA from analytic differentiability; `after generation` separates sequential candidate optimization from model-weight training.
   - Risk status: grounded.

41. `The raw internal LBM evaluator remains sequential and nondifferentiable in the analytic-autograd sense, but its scalar outputs can still influence model weights through the SPSA estimator described above.`
   - Function: reconciles nondifferentiable solver mechanics with optimizer-facing training.
   - Claim type: method boundary.
   - Word choices: `analytic-autograd sense` keeps the statement precise; `can still influence model weights` is the key new result; `through the SPSA estimator` names the mechanism.
   - Risk status: critical.

42. `In the current loop, training starts by sampling a noisy latent \(z_t\), asking the UNet for a noise estimate or denoised latent estimate, and decoding the resulting \(z_0\) candidate into voxel probabilities \(V\).`
   - Function: describes the model-side loop.
   - Claim type: procedural description.
   - Word choices: `current loop` anchors the description to code; `voxel probabilities` matches the sigmoid materialization boundary.
   - Risk status: grounded.

43. `For high-resolution coordinate decoders, the loop normally trains on sampled coordinates and only materializes the full grid when a full-grid objective is scheduled.`
   - Function: explains why full-grid solver work is scheduled, not constant.
   - Claim type: implementation detail.
   - Word choices: `normally` and `only` describe the compute-saving control flow; `full-grid objective` covers direct solver and other full-grid losses.
   - Risk status: grounded.

44. `When \(\mathcal{L}_{solver\mbox{-}SPSA}\) is active, the loop materializes that grid, evaluates the internal D3Q27 LBM objective on the base geometry and on two perturbed geometries, and folds the resulting measured scalar into the same \texttt{optimization\_loss} that is backpropagated through the decoder and denoiser weights.`
   - Function: states the exact direct-solver training path.
   - Claim type: implementation claim.
   - Word choices: `base geometry and two perturbed geometries` gives the solver-call pattern; `folds` makes consolidation into the loss explicit; `same optimization_loss` addresses the earlier diagnostic-loss confusion.
   - Risk status: grounded.

45. `The OpenFOAM setup is the external PDE-validation foundation.`
   - Function: starts the ground-truth tier boundary.
   - Claim type: validation-route claim.
   - Word choices: `foundation` means route and apparatus, not completed validation.
   - Risk status: grounded.

46. `Its role is to generate independent reference cases and, when residual and field-export gates pass, promote labels to the external-PDE tier defined in the ground-truth specification.`
   - Function: defines how labels would become external ground truth.
   - Claim type: future/conditional validation claim.
   - Word choices: `when` and `gates pass` prevent automatic promotion; `external-PDE tier` matches the repo's claim taxonomy.
   - Risk status: essential.

47. `The present Airshow grid-loss sweep uses the internal D3Q27 solver as a measured training objective and does not use OpenFOAM labels, so it should be read as internal-LBM-guided optimization plus a validation route, not as externally validated aerodynamic learning.`
   - Function: closes the section with the precise claim boundary.
   - Claim type: evidence boundary.
   - Word choices: `measured training objective` gives credit for the new direct loss; `does not use OpenFOAM labels` blocks ground-truth overclaim; `not externally validated` is the necessary caveat.
   - Risk status: critical.

## Audit Notes

The methodology now deliberately distinguishes analytic autograd from black-box
gradient estimation. It no longer says the solver is merely detached from
training, because the direct SPSA path does fold measured solver values into
the optimizer loss. It also refuses to upgrade those values to external PDE
ground truth.

## 2026-06-20 Resolution Addendum

The methodology now says the Airshow builder voxelizes into a fixed grid rather
than only a `16^3` grid. That wording is necessary because the same public
corpus path was rerun at `32^3`, `64^3`, and `96^3`. The sentence still avoids implying
that higher resolution validated aircraft generation; it only states that the
corpus construction path supports multiple lattice sizes.

The decoder subsection now distinguishes the dense lower-resolution decoder
from the `96^3` coordinate decoder. That avoids the inaccurate implication that
the successful `96^3` run used the dense final layer that blocked the earlier
`64^3` attempt.

## 2026-06-21 Sequential Objective Addendum

The methodology now adds a `Sequential Measured-Objective Candidate
Optimization` subsection. Its function is to separate three ideas that are easy
to conflate:

- model-weight training still backpropagates through `L_opt`;
- connectivity, validity, and aerodynamic scores can now drive a sequential
  black-box candidate optimizer through `L_seq`;
- that candidate optimizer is not differentiable CFD and is not yet benchmark
  evidence that generated aircraft validity gates pass.

The wording uses `measured black-box objective terms`, `sequential genetic
search`, and `two-point SPSA-style estimator` because those phrases describe
what the new code actually does. It avoids saying the solver is differentiable
or that the diffusion model learns directly from the solver score.
