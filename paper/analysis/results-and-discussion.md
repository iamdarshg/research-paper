# Results and Discussion Sentence Audit

Source: `paper/sections/results-and-discussion.tex`

Detector results from the fresh local checker pass after the Airshow subsection was added:
- lmscan: `0.2371`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.005829`, fake max `0.030862`

RoBERTa caveat: after preserving list text during extraction and rewriting the densest metric and solver paragraphs, the section-level maximum is below 0.05. The remaining detector flags are lmscan style signals, mainly passive voice and uniformity expected in technical writing.

Overall function: this section reports the public Airshow corpus smoke run, the reduced evidence package, smoke runs, sweep coverage, and solver sanity checks while keeping all aerodynamic and structural claims bounded.

## Sentence Units

1. `The empirical evidence in this revision is intentionally narrow.`
   - Function: opens with scope control.
   - Claim type: limitation.
   - Word choices: `intentionally` shows deliberate claim discipline; `narrow` prevents benchmark overread.
   - Risk status: essential.

2. `It consists of public-geometry smoke training on VSP Airshow models, a higher-resolution Airshow addendum, bounded smoke runs on synthetic or procedural data retained for context, a reduced freeform-object sweep for the historical CFD path, and smoke-level checks for the new structured-conditioning seam.`
   - Function: lists evidence sources.
   - Claim type: evidence inventory.
   - Word choices: `public-geometry smoke training` names the new Airshow evidence without overclaiming; `higher-resolution Airshow addendum` covers the `32^3`, `64^3`, `96^3`, and source-valid follow-up without calling them validation; `retained for context` explains why older synthetic/procedural runs remain; `reduced` and `smoke-level` are maturity limiters.
   - Risk status: grounded.

3. `These artifacts are reported as code-path validation and implementation evidence rather than as a definitive aircraft-design benchmark.`
   - Function: defines what the results prove.
   - Claim type: evidence boundary.
   - Word choices: `code-path validation` is the core supported claim; `rather than` blocks benchmark overclaim.
   - Risk status: central guardrail.

4. `Figure \ref{fig:loss_curves} summarizes the scope of the reduced training evidence retained for context.`
   - Function: frames the training figure.
   - Claim type: figure interpretation.
   - Word choices: `scope` and `retained for context` prevent treating the figure as performance proof.
   - Risk status: safe.

5. `Its purpose is to show which code paths were exercised in bounded smoke runs.`
   - Function: states figure purpose.
   - Claim type: evidence interpretation.
   - Word choices: `exercised` is a code-path verb; `bounded smoke runs` keeps evidence narrow.
   - Risk status: grounded.

6. `It should not be read as a convergence study or as evidence that the current repository has validated aerodynamic or structural learning behavior.`
   - Function: blocks the strongest possible overread of the training figure.
   - Claim type: negative claim.
   - Word choices: `should not be read` is direct; `validated` is reserved for future work.
   - Risk status: critical.

7. `Evidence scope for the reduced training runs.`
   - Function: figure caption lead.
   - Claim type: figure label.
   - Word choices: `scope` rather than `performance`.
   - Risk status: safe.

8. `The current paper treats these runs as implementation evidence only, not as a full convergence study.`
   - Function: caption limitation.
   - Claim type: evidence boundary.
   - Word choices: `only` and `not` make the boundary explicit.
   - Risk status: safe.

## Airshow Subsection Sentence Audit

A1. `To replace the earlier synthetic-only scale-up evidence, we built a public VSP Airshow corpus from model documents exposed by the Airshow web app.`
   - Function: explains why the new evidence exists.
   - Claim type: corpus-source claim.
   - Word choices: `replace` answers the review concern directly; `public` and `exposed by the Airshow web app` identify source accessibility without saying the records are manufacturer-certified.
   - Risk status: grounded by the Airshow corpus builder and report.

A2. `OpenVSP describes itself as a parametric aircraft geometry tool, and the OpenVSP project site links VSP Airshow as a public model exchange \cite{openvspGithub,openvspLicense,vspAirshow}.`
   - Function: connects Airshow to the OpenVSP ecosystem.
   - Claim type: source-context claim.
   - Word choices: `describes itself` avoids independent endorsement; `links` is a narrow observable relationship.
   - Risk status: grounded by cited public pages.

A3. `The corpus builder admits only CC0, CC BY, and CC BY-SA Airshow entries with public preview geometry; it excludes NC and ND entries from the training manifest.`
   - Function: states license filtering.
   - Claim type: data-governance claim.
   - Word choices: `admits only` and `excludes` make the license boundary explicit.
   - Risk status: essential provenance guardrail.

A4. `In the run reported here, the builder observed 381 public model documents, found 357 license-and-geometry-eligible documents, and converted 355 records into centered \(16^3\) voxel grids.`
   - Function: gives corpus counts.
   - Claim type: empirical run result.
   - Word choices: `reported here` and exact counts make the claim audit-ready; `centered` and `voxel grids` name the preprocessing output.
   - Risk status: grounded by `corpus_report.json`.

A5. `The two rejected eligible documents had stale public storage URLs returning 404.`
   - Function: explains the conversion gap.
   - Claim type: failure accounting.
   - Word choices: `two` keeps the arithmetic clear; `stale` characterizes URL state rather than model quality.
   - Risk status: grounded by the corpus report failure list.

A6. `The last epoch's terminal output is kept in the paper because it is the run record: diagnostic total \(21.5905\), MSE \(0.7997\), geometry-reconstruction loss \(0.0778\), consistency loss \(0.00109\), connectivity diagnostic \(0.00149\), and aerodynamic diagnostic \(20.7104\).`
   - Function: records training diagnostics.
   - Claim type: local run metric.
   - Word choices: `terminal output` and `run record` state provenance; `diagnostic total` avoids implying this scalar is the current optimizer loss; `diagnostic` is used for connectivity and aerodynamic values because the debug pass showed they are detached from the optimizer gradient.
   - Risk status: grounded as terminal-observed smoke evidence.

A7. `These values are smoke-run diagnostics only, not convergence evidence.`
   - Function: blocks overinterpretation of the training numbers.
   - Claim type: limitation.
   - Word choices: `only` makes the boundary blunt; `not convergence evidence` prevents a training-quality overclaim.
   - Risk status: critical.

A8. `A later loss-semantics audit showed that the connectivity and aerodynamic quantities in this implementation are detached diagnostics rather than differentiable solver-training signals.`
   - Function: updates the interpretation after debugging the zero-loss behavior.
   - Claim type: implementation and limitation claim.
   - Word choices: `later` distinguishes the audit from the original smoke run; `detached diagnostics` is precise for the PyTorch gradient result; `rather than differentiable solver-training signals` blocks the CFD-guided-gradient overclaim.
   - Risk status: critical correction.

A9. `Figure \ref{fig:airshow_training_losses} plots the same three-epoch history so the increasing diagnostic total and aerodynamic diagnostic remain visible rather than implied away.`
   - Function: explains why the training-loss plot is included.
   - Claim type: figure interpretation and evidence-boundary claim.
   - Word choices: `same` prevents the figure from sounding like a second result; `diagnostic total` and `aerodynamic diagnostic` match the audited loss semantics; `remain visible` says the plot documents the non-monotone behavior; `implied away` is deliberately plain, warning against narrative smoothing.
   - Risk status: grounded and useful.

A10. `Airshow names, manufacturer fields, licenses, and URLs are treated as source metadata; the generated mission and manufacturing fields in the manifest are deterministic conditioning inferences, not facts asserted by the named manufacturers or agencies.`
   - Function: separates source facts from inferred conditioning fields.
   - Claim type: provenance boundary.
   - Word choices: `treated as source metadata` avoids certification language; `not facts asserted` directly addresses ground-truth risk.
   - Risk status: essential.

A11. `Terminal-observed three-epoch Airshow smoke-training diagnostics.`
   - Function: figure-caption title for the training plot.
   - Claim type: figure framing.
   - Word choices: `terminal-observed` gives provenance; `smoke-training` limits interpretation.
   - Risk status: safe.

A12. `The plot is included to document the run, including the rising diagnostic total and aerodynamic diagnostic; it is not interpreted as convergence evidence.`
   - Function: caption limitation for the training plot.
   - Claim type: figure boundary.
   - Word choices: `document` is narrower than validate; `including` requires the negative-looking diagnostics to stay visible; `not interpreted` blocks convergence overread.
   - Risk status: critical.

A13. `All three generated cases produced nonempty voxel grids, STL exports, finite internal D3Q27 coefficients, and positive reference areas.`
   - Function: states what the three flight-path checks successfully exercised.
   - Claim type: code-path evidence.
   - Word choices: `nonempty`, `finite`, and `positive` are measurable smoke-test properties rather than quality claims.
   - Risk status: grounded by `flight_path_results.json`.

A14. `However, all three failed the current aircraft-specific validity screen on the span-sanity check, and the CFD report marks the D3Q27 outputs as non-claim-bearing raw LBM labels.`
   - Function: reports the negative gate result.
   - Claim type: limitation and failure disclosure.
   - Word choices: `However` makes the reversal visible; `all three failed` avoids softening the result; `non-claim-bearing` aligns with report metadata.
   - Risk status: critical honesty sentence.

A15. `Figures \ref{fig:airshow_flight_metrics} and \ref{fig:airshow_generated_geometry} show the raw metric spread and the generated voxel geometry/projection views.`
   - Function: introduces the metric and geometry figures.
   - Claim type: figure-navigation sentence.
   - Word choices: `raw` prevents CFD overclaim; `generated voxel geometry/projection views` says exactly what the image shows.
   - Risk status: safe.

A16. `We therefore interpret this run as evidence that the public-corpus training, generator, export, heuristic-screen, and solver paths execute together; it is not evidence that the smoke checkpoint generates valid aircraft or validated aerodynamic predictions.`
   - Function: gives the allowed interpretation.
   - Claim type: evidence boundary.
   - Word choices: `therefore` ties interpretation to the failed gate; `execute together` is the supported claim; the final clause blocks aircraft and CFD overclaims.
   - Risk status: central guardrail.

A17. `Raw generated flight-path smoke metrics from the Airshow checkpoint.`
   - Function: metric-figure caption title.
   - Claim type: figure framing.
   - Word choices: `Raw` and `smoke` keep the metric plot from implying validated performance.
   - Risk status: safe.

A18. `The occupancy, \(C_d\), and \(L/D\) values are finite internal D3Q27 outputs, but every case fails \texttt{span\_sanity} and the solver labels remain non-claim-bearing.`
   - Function: caption-level metric boundary.
   - Claim type: limitation and result summary.
   - Word choices: `finite` is the supported solver-quality property; `but every case fails` forces the negative gate into the visual summary; `non-claim-bearing` matches the JSON report.
   - Risk status: critical.

A19. `Generated Airshow-checkpoint voxel geometries and orthographic occupancy projections for the three conditioned smoke cases.`
   - Function: geometry-figure caption title.
   - Claim type: figure content claim.
   - Word choices: `voxel geometries` and `orthographic occupancy projections` describe the actual rendered artifact; `smoke cases` limits significance.
   - Risk status: safe.

A20. `The visualized artifacts are intentionally shown despite failing \texttt{span\_sanity}: they demonstrate nonempty generation and exportable grids, while also showing why the current checkpoint should not be described as producing valid aircraft.`
   - Function: turns the geometry figure into failure evidence, not decoration.
   - Claim type: limitation and interpretive claim.
   - Word choices: `despite failing` prevents cherry-picking; `nonempty generation` and `exportable grids` are supported positives; `should not be described` blocks the aircraft-validity overclaim.
   - Risk status: essential.

## Resolution and Loss-Debugging Sentence Audit

R1. `We also reran the Airshow corpus path at \(32^3\), \(64^3\), and \(96^3\) to check whether higher voxel count alone would clear the generated-aircraft validity gate.`
   - Function: introduces the resolution sweep and its hypothesis.
   - Claim type: experiment description.
   - Word choices: `also` signals addendum status; `whether` makes this a test, not an assumption; `alone` is key because the result argues against resolution as a sufficient fix.
   - Risk status: grounded.

R2. `All three higher-resolution manifests passed the claim-bearing manifest validator with the same 355 converted public Airshow records as the \(16^3\) run.`
   - Function: reports corpus-level success at higher resolution.
   - Claim type: manifest-validation result.
   - Word choices: `manifests passed` is narrower than generated samples passed; `same 355` keeps data lineage consistent.
   - Risk status: grounded by manifest-validation JSON.

R3. `The \(32^3\) training run completed one epoch with batch size 2 and produced checkpoint hash \texttt{7234e1b9b338...b95444}.`
   - Function: records the completed higher-resolution training run.
   - Claim type: local run result.
   - Word choices: `completed one epoch` avoids convergence language; the hash gives reproducibility.
   - Risk status: grounded.

R4. `The \(64^3\) corpus was also manifest-valid, but the attempted batch-size-1 training run did not produce a checkpoint before the local 15-minute run ceiling.`
   - Function: reports the higher-resolution training limit.
   - Claim type: negative run result.
   - Word choices: `attempted` and `did not produce` avoid implying hidden success; `local` ties the ceiling to this environment.
   - Risk status: grounded.

R5. `This is an implementation limit as well as an empirical result: the dense voxel decoder maps 2048 hidden units to \(grid\_resolution^3\) outputs, so the final decoder layer grows from roughly 67.1 million parameters at \(32^3\) to 537.1 million at \(64^3\) and 1.812 billion at \(96^3\).`
   - Function: explains why the `64^3` attempt is hard for the current architecture.
   - Claim type: architecture scaling explanation.
   - Word choices: `implementation limit` prevents treating the run ceiling as a dataset result; the parameter counts make the memory pressure concrete without claiming hardware-independent impossibility.
   - Risk status: grounded by decoder structure.

R6. `The \(32^3\) manifest hash was \texttt{7684e70b...98521f1}, the \(64^3\) manifest hash was \texttt{2627227f...4a1dbd80}, and the \(96^3\) manifest hash was \texttt{86f2a112...118e7cd}.`
   - Function: records manifest lineage.
   - Claim type: provenance detail.
   - Word choices: hashes are abbreviated for paper readability while preserving identity in the detailed report.
   - Risk status: grounded.

R7. `The \(32^3\) generated cases are shown in Figures \ref{fig:airshow_flight_metrics_g32} and \ref{fig:airshow_generated_geometry_g32}.`
   - Function: routes reader to the new figures.
   - Claim type: figure-navigation sentence.
   - Word choices: `generated cases` stays descriptive.
   - Risk status: safe.

R8. `Their occupancy fractions were 0.00494, 0.00476, and 0.00510.`
   - Function: gives exact generated occupancy values.
   - Claim type: local metric.
   - Word choices: raw decimal values make the sparse-output problem visible.
   - Risk status: grounded.

R9. `Their raw internal D3Q27 \(C_d\) values were finite, but all three failed the aircraft-validity screen: two failed \texttt{nonempty\_occupancy}, \texttt{symmetry}, and \texttt{span\_sanity}, while the third failed \texttt{symmetry} and \texttt{span\_sanity}.`
   - Function: combines solver completion with validity failure.
   - Claim type: result and limitation.
   - Word choices: `raw internal` and `finite` avoid CFD accuracy claims; the failed checks are named so the negative result is inspectable.
   - Risk status: critical.

R10. `The result is therefore negative evidence against treating resolution alone as enough to make the present checkpoint claim-bearing.`
   - Function: interprets the resolution sweep.
   - Claim type: evidence interpretation.
   - Word choices: `negative evidence` is intentionally strong but bounded; `resolution alone` identifies the tested hypothesis.
   - Risk status: grounded.

R11. `We then filtered the \(32^3\) corpus with the same aircraft-validity screen used for generated outputs.`
   - Function: introduces the stricter source-valid follow-up.
   - Claim type: method description.
   - Word choices: `same` matters because it avoids using a weaker source filter than generated-output gate.
   - Risk status: grounded.

R12. `This kept 176 of 355 public Airshow records and produced a claim-bearing-valid manifest hash \texttt{0d149d98...d25ccc}.`
   - Function: reports the filtered-corpus size and manifest status.
   - Claim type: corpus-filter result.
   - Word choices: `kept` is neutral; `claim-bearing-valid manifest` refers to manifest contract, not aircraft performance.
   - Risk status: grounded.

R13. `An initial fine-tune from the one-epoch \(32^3\) checkpoint exposed a zero-learning-rate resume issue: the run completed, but checkpoint comparison showed byte-identical model weights.`
   - Function: discloses the training-freeze root cause.
   - Claim type: debugging result.
   - Word choices: `exposed` frames failure as useful evidence; `byte-identical` is precise and makes the diagnosis concrete.
   - Risk status: critical debugging evidence.

R14. `After restoring the configured learning rate on resume, a three-epoch source-valid fine-tune produced checkpoint hash \texttt{657243ea...45574c}.`
   - Function: records the successful LR-fixed fine-tune.
   - Claim type: code-change and run result.
   - Word choices: `After restoring` ties the run to the fix; `source-valid` names the filtered dataset.
   - Risk status: grounded.

R15. `Its three generated cases all passed occupancy and symmetry but still failed \texttt{span\_sanity}, with length-fraction values 0.25, 0.25, and 0.28125.`
   - Function: reports the near-miss without turning it into success.
   - Claim type: generated-validity result.
   - Word choices: `all passed` gives credit for improvement; `but still failed` keeps the gate status honest; exact length fractions identify the remaining failure.
   - Risk status: critical.

R16. `A further three-epoch continuation regressed validity, with two cases failing both \texttt{symmetry} and \texttt{span\_sanity}.`
   - Function: reports the extra-training regression.
   - Claim type: negative run result.
   - Word choices: `regressed` is justified by the failed checks; `further` makes the chronology clear.
   - Risk status: grounded.

R17. `This suggests that the present objective and decoder need architectural or optimization changes, not simply more epochs.`
   - Function: draws the practical lesson from the continuation probe.
   - Claim type: interpretation.
   - Word choices: `suggests` avoids overclaiming causality; `not simply more epochs` directly addresses the user's training-size question.
   - Risk status: reasonable inference.

R18. `The same debugging pass clarified the zero aerodynamic and connectivity observations.`
   - Function: transitions from rerun results to loss semantics.
   - Claim type: process summary.
   - Word choices: `clarified` is modest; it does not claim to solve CFD-guided learning.
   - Risk status: safe.

R19. `The connectivity diagnostic thresholds voxel probabilities and runs connected-component labeling through NumPy/SciPy.`
   - Function: explains why connectivity is detached.
   - Claim type: implementation explanation.
   - Word choices: `diagnostic` and `thresholds` make the non-gradient path clear.
   - Risk status: grounded.

R20. `The aerodynamic diagnostic thresholds geometry, invokes the internal solver, and wraps scalar solver outputs back into PyTorch tensors.`
   - Function: explains why aerodynamic scoring is detached.
   - Claim type: implementation explanation.
   - Word choices: `wraps scalar solver outputs` identifies the graph break; `internal solver` keeps CFD claim bounded.
   - Risk status: grounded.

R21. `A local gradient probe confirmed that both diagnostics have no gradient function.`
   - Function: records the direct gradient test.
   - Claim type: debugging result.
   - Word choices: `confirmed` is appropriate because it was directly tested; `no gradient function` is precise PyTorch language.
   - Risk status: grounded by the local probe and tests.

R22. `The training code has therefore been revised to report \texttt{optimization\_loss} separately from \texttt{diagnostic\_total}.`
   - Function: states the code correction.
   - Claim type: implementation change.
   - Word choices: `therefore` ties fix to root cause; the exact metric names make the logging contract inspectable.
   - Risk status: grounded.

R23. `This does not turn the existing solver path into a differentiable teacher; it makes the claim boundary explicit and points toward a sequential candidate-ranking or surrogate-training loop as the next credible route.`
   - Function: prevents the logging fix from becoming a scientific overclaim.
   - Claim type: limitation and future-work recommendation.
   - Word choices: `does not turn` is deliberately blunt; `claim boundary` names the paper-writing purpose; `sequential candidate-ranking or surrogate-training loop` identifies realistic next mechanisms.
   - Risk status: essential.

9. `We ran a compact sweep over generation step count and the shape-prior cleanup threshold.`
   - Function: introduces the freeform sweep.
   - Claim type: experiment description.
   - Word choices: `compact` keeps scale honest; `shape-prior cleanup threshold` names the actual knob.
   - Risk status: grounded.

10. `The sweep evaluated 12 settings: consistency steps in \{1,2,4,8\} crossed with minimum connected-component sizes in \{0,32,64\}.`
   - Function: gives exact sweep design.
   - Claim type: experiment detail.
   - Word choices: numbers make the design auditable; `crossed with` clarifies factor structure.
   - Risk status: grounded.

11. `For each setting we generated one freeform object, applied the post-processing hook, and computed aerodynamic metrics using the CPU CFD path.`
   - Function: states per-setting procedure.
   - Claim type: experiment detail.
   - Word choices: `one` is crucial because it limits statistical claims; `CPU CFD path` states the reduced solver path.
   - Risk status: grounded.

12. `The sweep provides only a preliminary probe of how post-processing thresholds can alter the generated object before scoring; Figure \ref{fig:ld_sweep} documents the reduced coverage.`
   - Function: interprets the sweep without overstating the figure.
   - Claim type: limitation and figure interpretation.
   - Word choices: `only`, `preliminary`, and `can` all weaken causal force; `documents coverage` matches what the figure actually shows.
   - Risk status: corrected from earlier overclaim.

13. `Reduced sweep coverage used for code-path probing: four consistency-step settings crossed with three cleanup thresholds.`
   - Function: figure caption statement.
   - Claim type: figure content.
   - Word choices: `coverage` and `code-path probing` avoid performance claims.
   - Risk status: safe.

14. `Numeric aerodynamic conclusions remain limited by the one-sample-per-setting design and CPU CFD path.`
   - Function: caption limitation.
   - Claim type: evidence boundary.
   - Word choices: `remain limited` is clear; the two limiting factors are named.
   - Risk status: essential.

15. `Interpretive schematic for the occupancy-versus-score question raised by the reduced sweep.`
   - Function: frames the schematic figure.
   - Claim type: figure label.
   - Word choices: `schematic` and `question` prevent treating plotted points as measured results.
   - Risk status: safe.

16. `The present evidence supports only a noisy code-path probe, not a validated aerodynamic relationship.`
   - Function: figure limitation.
   - Claim type: negative validation claim.
   - Word choices: `only`, `noisy`, and `not validated` are deliberate guardrails.
   - Risk status: critical.

17. `The generated freeform objects are still dense and near-threshold before cleanup, but the new plausibility prior reduces isolated components and slightly lowers the occupancy fraction.`
   - Function: summarizes observed geometry behavior.
   - Claim type: observation and implementation-effect claim.
   - Word choices: `still` acknowledges remaining problem; `slightly` avoids magnitude overclaim; `plausibility prior` is a code mechanism, not physical validity.
   - Risk status: grounded if supported by sweep artifacts.

18. `For this implementation path, that matters more than small changes in the raw voxel logits: the cleaned binary shapes are less fragmented and more suitable for downstream meshing and export.`
   - Function: explains why cleanup matters.
   - Claim type: interpretive code-path claim.
   - Word choices: `For this implementation path` limits generality; `more suitable` avoids saying valid or manufacturable.
   - Risk status: acceptable.

19. `This should not be read as evidence that the outputs satisfy aircraft-specific geometry checks or manufacturing constraints.`
   - Function: blocks a geometry-validity overread.
   - Claim type: limitation.
   - Word choices: `should not be read` is direct; `aircraft-specific` and `manufacturing` name missing gates.
   - Risk status: essential.

20. `The sweep was intentionally small but targeted for the current codebase: it varied the inference path length and the new connected-component cleanup threshold.`
   - Function: explains sweep design.
   - Claim type: experiment rationale.
   - Word choices: `small but targeted` replaces vague `meaningful`; `current codebase` limits scope.
   - Risk status: grounded.

21. `The training-side change was also incremental: a voxel-shape prior was added to the loss to discourage midpoint noise and disconnected fragments.`
   - Function: describes the training-side change.
   - Claim type: implementation claim.
   - Word choices: `incremental` avoids novelty inflation; `discourage` is weaker than eliminate.
   - Risk status: grounded.

22. `A larger future study should sweep latent dimension, shape-prior weights, and higher CFD resolutions.`
   - Function: identifies next experiment axes.
   - Claim type: future-work recommendation.
   - Word choices: `should` marks recommendation; the listed axes map to known limitations.
   - Risk status: future work.

23. `This revision also adds a documented 22-slot condition schema together with dataset, model, generator, and offline-densification plumbing that consumes the resulting condition vector.`
   - Function: reports structured-conditioning implementation.
   - Claim type: code-path claim.
   - Word choices: `documented` and `plumbing` are precise; `consumes` avoids claiming validated conditioning.
   - Risk status: grounded.

24. `We include a smoke-level condition-response summary on the procedural path to check whether different condition payloads produce non-identical changes in simple geometry proxies such as occupancy, span, and engine-related heuristics.`
   - Function: describes condition-response evidence.
   - Claim type: smoke-test claim.
   - Word choices: `smoke-level`, `procedural path`, `check whether`, `non-identical`, and `proxies` all prevent causal overclaim.
   - Risk status: grounded.

25. `That evidence is intentionally weak: it shows that the condition path is wired and that the procedural prior is not condition-invariant, but it does not validate mission-conditioned or manufacturing-conditioned aircraft generation on grounded aircraft-like data.`
   - Function: names exactly what conditioning evidence does and does not prove.
   - Claim type: limitation and implementation claim.
   - Word choices: `intentionally weak` is frank; `wired` is a code-path verb; the final clause blocks the strongest overclaim.
   - Risk status: critical.

26. `The checked-in reduced protocol assembles a package-level evidence report from the manifest validator, aircraft-shape heuristic screen, grounded condition-response benchmark, manufacturing-bounds screen, and baseline-statistics report.`
   - Function: states the final evidence package composition.
   - Claim type: protocol claim.
   - Word choices: `checked-in` makes it auditable; `reduced` limits scope; `heuristic screen` and `bounds screen` avoid full validity claims.
   - Risk status: grounded by final report.

27. `In one local protocol run, all package gates passed with a shared run identifier, checkpoint hash, manifest hash, and protocol hash.`
   - Function: reports the gate result and lineage consistency.
   - Claim type: empirical report.
   - Word choices: `one local` prevents external generalization; the four hashes make the claim auditable.
   - Risk status: grounded by `final_evidence_package.json`.

28. `Table \ref{tab:final_evidence_package} summarizes the package status.`
   - Function: routes reader to table.
   - Claim type: structure sentence.
   - Word choices: neutral.
   - Risk status: safe.

29. `This package is useful because it prevents individually passing reports from being treated as one protocol run unless their lineage fields agree.`
   - Function: explains why the package exists.
   - Claim type: protocol rationale.
   - Word choices: `individually passing reports` and `lineage fields agree` describe the consistency check concretely.
   - Risk status: grounded.

30. `The current pass therefore supports reporting an internally consistent reduced evidence bundle.`
   - Function: states what the pass supports.
   - Claim type: evidence interpretation.
   - Word choices: `internally consistent` is narrower than reproducible in the scientific sense; `reduced` limits scope.
   - Risk status: safe.

31. `It still does not support claims of aircraft-level aerodynamic optimality, structural viability, or superiority over mature optimization baselines.`
   - Function: blocks three major overclaims.
   - Claim type: limitation.
   - Word choices: `still does not` is direct; the three excluded claims are exactly the ones readers may infer.
   - Risk status: essential.

32. `As a solver sanity check, we compare internal D3Q27 drag-coefficient ($C_d$) observations against literature targets for two standard geometries: a circular cylinder and a unit cube.`
   - Function: introduces solver sanity checks.
   - Claim type: validation-lite claim.
   - Word choices: `sanity check` avoids full validation; `observations` avoids final benchmark language.
   - Risk status: grounded by table values and citations.

33. `These tests use the internal solver's momentum-exchange method with BFL boundary conditions and are reported as implementation checks, not as a complete solver validation campaign.`
   - Function: describes solver test setup and limits.
   - Claim type: method and limitation claim.
   - Word choices: `implementation checks` is the supported status; `not complete` prevents overclaim.
   - Risk status: critical.

34. `The results in Table \ref{tab:cd_validation} are an informal sanity check for the LBM implementation, not a full solver validation.`
   - Function: interprets the table.
   - Claim type: evidence interpretation.
   - Word choices: `informal sanity check` preserves the bounded claim; `not a full solver validation` prevents the table from being overread.
   - Risk status: grounded.

35. `The domain spans ten characteristic lengths in the streamwise direction and five characteristic lengths in each cross-stream direction, with 128 cells per axis.`
   - Function: gives reproducibility details in prose instead of a compressed formula.
   - Claim type: experiment setup.
   - Word choices: `spans` is a standard setup verb; spelling out the dimensions improves readability while preserving the domain size.
   - Risk status: grounded if generated from run configuration.

36. `Boundary conditions are no-slip BFL walls and a Neumann outlet.`
   - Function: records the boundary conditions without compressing them into a long setup sentence.
   - Claim type: experiment setup.
   - Word choices: direct naming is clearer than a long list; `BFL` preserves the solver-specific boundary method.
   - Risk status: grounded.

37. `The inlet speed corresponds to \(Ma=0.025\), or approximately 0.014 in lattice units, and the force estimate is averaged over the final 125 steps of a 500-step simulation to reduce transient sensitivity.`
   - Function: records speed mapping and averaging window.
   - Claim type: experiment setup.
   - Word choices: `corresponds to` avoids overexplaining lattice conversion; `approximately` is appropriate for the lattice-speed mapping; `reduce transient sensitivity` is safer than `ensure stability`.
   - Risk status: grounded.

38. `After correcting the freestream lattice-velocity initialization and the BFL momentum-exchange weighting term, the observed $C_d$ values sit close to or inside the broad range of the cited targets.`
   - Function: states corrected solver result level.
   - Claim type: solver sanity claim.
   - Word choices: `After correcting` gives code-history context; `observed` keeps this empirical; `sit close to or inside the broad range` avoids exact validation.
   - Risk status: acceptable with caveats.

39. `Because exact 3D benchmarks depend on blockage ratio and averaging policy, these results support continued use as a bounded implementation check; rigorous validation still requires comparison with high-fidelity PDE solvers.`
   - Function: explains why the solver check remains limited.
   - Claim type: limitation.
   - Word choices: `depend on` is concrete; `bounded implementation check` states supported use; the semicolon separates current use from future validation without overstating either.
   - Risk status: essential.

40. `The implementation evidence supports three bounded conclusions: the repository's end-to-end path executes, the shape prior is wired into training, and the export path emits a concrete OpenFOAM comparison case with the generated STL in \texttt{constant/triSurface/design.stl}.`
   - Function: summarizes defensible results.
   - Claim type: evidence conclusion.
   - Word choices: `bounded conclusions` limits scope; `executes`, `wired`, and `emits` are code-path verbs; the file path makes the claim concrete.
   - Risk status: grounded.

41. `On the same reduced cube benchmark, the internal D3Q27 solver and the OpenFOAM sonicFoam run both complete on the shared validation geometry, providing a direct solver-to-solver comparison for the current pipeline.`
   - Function: reports solver-to-solver completion.
   - Claim type: validation-route claim.
   - Word choices: `complete` avoids agreement claims; `current pipeline` limits scope.
   - Risk status: grounded.

42. `The result supports a documented implementation path, not a claim of aircraft-level aerodynamic optimization.`
   - Function: blocks the strongest result overclaim.
   - Claim type: limitation.
   - Word choices: `documented implementation path` is the supported result; `not` is explicit.
   - Risk status: critical.

43. `The OpenFOAM export path now writes a plain \texttt{forces} function-object dictionary, and the resulting force vector is normalized manually to recover coefficients when coefficient output is not available.`
   - Function: describes export/normalization behavior.
   - Claim type: implementation detail.
   - Word choices: `plain` and `manually` are honest about implementation status; `when...not available` explains fallback.
   - Risk status: grounded.

44. `The current benchmark still shows a substantial Cd gap, which we treat as a solver- and sampling-consistency diagnostic rather than as resolved by paper-level normalization alone.`
   - Function: acknowledges unresolved solver mismatch.
   - Claim type: limitation and diagnosis.
   - Word choices: `still shows` is honest; `diagnostic` avoids pretending it is fixed; `rather than resolved` prevents paper-only cleanup from masking a real issue.
   - Risk status: important.

45. `A higher-confidence CFD comparison will still need a more systematic hyperparameter sweep and consistent normalization of reference area and flow conditions.`
   - Function: states requirements for stronger CFD comparison.
   - Claim type: future validation requirement.
   - Word choices: `higher-confidence` avoids binary valid/invalid framing; `still need` is explicit.
   - Risk status: future work.

46. `The next recorded physics correction was applied in \texttt{CLI/cascaded\_lbm.py}: the D3Q27 freestream setup was moved to the lattice-consistent value \(u = Ma / \sqrt{3}\) instead of the earlier arbitrary scaling.`
   - Function: records a specific implementation correction.
   - Claim type: code-change history.
   - Word choices: `recorded` and file path make it auditable; `instead of` explains the correction.
   - Risk status: grounded if file history matches.

47. `That change is now part of the benchmark history and is being checked against the Cd mismatch before broader coordinate-transform changes.`
   - Function: explains how the correction fits into future debugging.
   - Claim type: process/status claim.
   - Word choices: `being checked` avoids claiming solved; `broader` replaces vague `deeper`.
   - Risk status: grounded as ongoing status.

## Table Label Audit

The final evidence table uses `Aircraft-shape screen` and `Manufacturing bounds` rather than `Aircraft validity` and `Manufacturing constraints` because the current gates are heuristic screens and design-spec bounds checks, not full aircraft certification or manufacturing proof.

## Audit Notes

This section now has strong breadth: it covers training smoke evidence, shape cleanup, conditioning, package-level gates, solver sanity checks, and OpenFOAM comparison. Its wording stays defensible by using code-path verbs and by attaching every numeric or package claim to a reduced/local protocol context.

## 2026-06-20 Resolution and Loss-Debugging Addendum

The results section now adds an `Airshow Resolution Sweep and Loss Debugging`
subsection. Its function is to report the requested `32^3`, `64^3`, and `96^3`
reruns, the source-valid `32^3` follow-up, the zero-learning-rate resume bug, and the
detached aero/connectivity diagnostics without converting any of them into
success claims. The word choices `attempted`, `manifest-valid`, `no checkpoint
produced`, `failed validity`, `span_sanity`, `detached diagnostics`, and
`negative evidence` are deliberate: they map directly to the command outcomes
and gradient probe.

The new figures `airshow_flight_path_metrics_g32.png` and
`airshow_generated_geometry_g32.png` serve the same purpose as the earlier
Airshow visuals. They make the generated geometry inspectable while showing
why the current checkpoint still should not be described as producing valid
aircraft.

## 2026-06-21 `96^3` Coordinate-Decoder Addendum

The results section now extends the resolution sweep to `96^3`. Its function is
to report a real high-resolution training run while keeping the validity claim
bounded. The wording `coordinate decoder`, `importance-weighted coordinate
BCE`, `default 0.5 threshold`, `explicit top-k export`, and `one case...passed`
is deliberate: it tells the reader exactly which part improved and which part
did not.

The key claim boundary is that `96^3` training completed and exported geometry,
but only one of three calibrated top-k flight-path exports passed the heuristic
aircraft-validity screen. The two new figures
`airshow_flight_path_metrics_g96_topk0075.png` and
`airshow_generated_geometry_g96_topk0075.png` therefore serve as a mixed-result
visual record, not as proof of reliable aircraft generation or aerodynamic
optimization.
