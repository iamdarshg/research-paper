# Results and Discussion Sentence Audit

Source: `paper/sections/results-and-discussion.tex`

Detector results:
- lmscan: `0.2173`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.013522`, fake max `0.052247`

Overall function: this section reports the reduced evidence package, smoke runs, sweep coverage, and solver sanity checks while keeping all aerodynamic and structural claims bounded.

## Sentence Units

1. `The empirical evidence in this revision is intentionally narrow.`
   - Function: opens with scope control.
   - Claim type: limitation.
   - Word choices: `intentionally` shows deliberate claim discipline; `narrow` prevents benchmark overread.
   - Risk status: essential.

2. `It consists of bounded smoke runs on synthetic or procedural data, a reduced freeform-object sweep for the historical CFD path, and smoke-level checks for the new structured-conditioning seam.`
   - Function: lists evidence sources.
   - Claim type: evidence inventory.
   - Word choices: `bounded`, `synthetic`, `procedural`, `reduced`, and `smoke-level` are all maturity limiters.
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

34. `The results in Table \ref{tab:cd_validation} provide an informal sanity check for our LBM implementation, framing the observations against typical literature targets for 3D flow.`
   - Function: interprets the table.
   - Claim type: evidence interpretation.
   - Word choices: `informal` and `typical` soften exactness; `framing` avoids saying validated.
   - Risk status: grounded.

35. `The tests were conducted in a $10L \times 5L \times 5L$ domain ($L$: characteristic length) using a $128^3$ grid, no-slip BFL wall conditions, and a Neumann outlet.`
   - Function: gives reproducibility details.
   - Claim type: experiment setup.
   - Word choices: exact geometry/grid/boundary terms make the claim checkable.
   - Risk status: grounded if generated from run configuration.

36. `We mapped physical velocity to $Ma=0.025$ ($u_{lattice} \approx 0.014$) and averaged forces over the final 125 steps of a 500-step simulation to reduce transient sensitivity.`
   - Function: records normalization and averaging details.
   - Claim type: experiment setup.
   - Word choices: `reduce transient sensitivity` is safer than `ensure stability`; exact values make it auditable.
   - Risk status: grounded.

37. `Following corrections to the freestream lattice velocity initialization and BFL momentum exchange weighting ($4.0/(1+2q)$), the solver yields $C_d$ values close to or in the broad range of the cited targets.`
   - Function: states corrected solver result level.
   - Claim type: solver sanity claim.
   - Word choices: `close to or in the broad range` avoids exact validation; `cited targets` grounds comparison.
   - Risk status: acceptable with caveats.

38. `Exact 3D benchmarks are sensitive to blockage ratios and averaging criteria, so these results support continued use as a bounded implementation check while rigorous validation still requires comparison with high-fidelity PDE solvers.`
   - Function: explains why the solver check remains limited.
   - Claim type: limitation.
   - Word choices: `sensitive` names sources of variation; `bounded implementation check` states supported use; `still requires` marks future work.
   - Risk status: essential.

39. `The implementation evidence supports three bounded conclusions: the repository's end-to-end path executes, the shape prior is wired into training, and the export path emits a concrete OpenFOAM comparison case with the generated STL in \texttt{constant/triSurface/design.stl}.`
   - Function: summarizes defensible results.
   - Claim type: evidence conclusion.
   - Word choices: `bounded conclusions` limits scope; `executes`, `wired`, and `emits` are code-path verbs; the file path makes the claim concrete.
   - Risk status: grounded.

40. `On the same reduced cube benchmark, the internal D3Q27 solver and the OpenFOAM sonicFoam run both complete on the shared validation geometry, providing a direct solver-to-solver comparison for the current pipeline.`
   - Function: reports solver-to-solver completion.
   - Claim type: validation-route claim.
   - Word choices: `complete` avoids agreement claims; `current pipeline` limits scope.
   - Risk status: grounded.

41. `The result supports a documented implementation path, not a claim of aircraft-level aerodynamic optimization.`
   - Function: blocks the strongest result overclaim.
   - Claim type: limitation.
   - Word choices: `documented implementation path` is the supported result; `not` is explicit.
   - Risk status: critical.

42. `The OpenFOAM export path now writes a plain \texttt{forces} function-object dictionary, and the resulting force vector is normalized manually to recover coefficients when coefficient output is not available.`
   - Function: describes export/normalization behavior.
   - Claim type: implementation detail.
   - Word choices: `plain` and `manually` are honest about implementation status; `when...not available` explains fallback.
   - Risk status: grounded.

43. `The current benchmark still shows a substantial Cd gap, which we treat as a solver- and sampling-consistency diagnostic rather than as resolved by paper-level normalization alone.`
   - Function: acknowledges unresolved solver mismatch.
   - Claim type: limitation and diagnosis.
   - Word choices: `still shows` is honest; `diagnostic` avoids pretending it is fixed; `rather than resolved` prevents paper-only cleanup from masking a real issue.
   - Risk status: important.

44. `A higher-confidence CFD comparison will still need a more systematic hyperparameter sweep and consistent normalization of reference area and flow conditions.`
   - Function: states requirements for stronger CFD comparison.
   - Claim type: future validation requirement.
   - Word choices: `higher-confidence` avoids binary valid/invalid framing; `still need` is explicit.
   - Risk status: future work.

45. `The next recorded physics correction was applied in \texttt{CLI/cascaded\_lbm.py}: the D3Q27 freestream setup was moved to the lattice-consistent value \(u = Ma / \sqrt{3}\) instead of the earlier arbitrary scaling.`
   - Function: records a specific implementation correction.
   - Claim type: code-change history.
   - Word choices: `recorded` and file path make it auditable; `instead of` explains the correction.
   - Risk status: grounded if file history matches.

46. `That change is now part of the benchmark history and is being checked against the Cd mismatch before broader coordinate-transform changes.`
   - Function: explains how the correction fits into future debugging.
   - Claim type: process/status claim.
   - Word choices: `being checked` avoids claiming solved; `broader` replaces vague `deeper`.
   - Risk status: grounded as ongoing status.

## Table Label Audit

The final evidence table uses `Aircraft-shape screen` and `Manufacturing bounds` rather than `Aircraft validity` and `Manufacturing constraints` because the current gates are heuristic screens and design-spec bounds checks, not full aircraft certification or manufacturing proof.

## Audit Notes

This section now has strong breadth: it covers training smoke evidence, shape cleanup, conditioning, package-level gates, solver sanity checks, and OpenFOAM comparison. Its wording stays defensible by using code-path verbs and by attaching every numeric or package claim to a reduced/local protocol context.
