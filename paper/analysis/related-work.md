# Related Work Sentence Audit

Source: `paper/sections/related-work.tex`

Detector results:
- lmscan: `0.2402`, verdict `Likely human`, confidence `high`
- RoBERTa detector: fake mean `0.001538`, fake max `0.004273`

Overall function: this section places the project among 3D diffusion, topology optimization, CFD shape optimization, physics-informed ML, and constraint-aware generation while repeatedly stating what this repository does not yet prove.

## Sentence Units

1. `Diffusion models are now a standard point of reference for high-quality generative modeling.`
   - Function: opens with accepted ML context.
   - Claim type: literature framing.
   - Word choices: `standard point of reference` avoids saying diffusion is always best; `high-quality` is supported by cited diffusion literature.
   - Risk status: grounded by cited diffusion papers.

2. `In 3D generation, recent systems such as Point-E and Shap-E show that diffusion-style methods can generate point-cloud or implicit 3D assets at practical sampling speeds.`
   - Function: narrows diffusion context to 3D generation.
   - Claim type: literature comparison.
   - Word choices: `such as` avoids exhaustive coverage; `diffusion-style` matches the paper's own cautious terminology; `assets` avoids equating generic 3D output with aircraft designs.
   - Risk status: grounded by cited Point-E and Shap-E work.

3. `Those works are important context because they establish that diffusion models can represent 3D geometry, but they do not by themselves solve aircraft-specific design, manufacturability, or validation concerns.`
   - Function: separates relevant prior art from the current engineering problem.
   - Claim type: boundary claim.
   - Word choices: `context` avoids overstating direct baseline status; `by themselves` prevents dismissing the work; the three constraint nouns name what remains unsolved.
   - Risk status: conservative and necessary.

4. `An adjacent engineering-design tradition comes from topology optimization, where the goal is to optimize material layout or geometry under physics-based objectives and constraints.`
   - Function: introduces a mature engineering baseline family.
   - Claim type: field framing.
   - Word choices: `adjacent` makes clear this is not the same task; `physics-based objectives and constraints` names why TO matters.
   - Risk status: grounded by TO references.

5. `That literature is mature and has a well-developed vocabulary for separating optimization variables, constraints, and validation requirements.`
   - Function: explains why TO is included even though the repository is generative.
   - Claim type: interpretive literature claim.
   - Word choices: `mature` is a broad but defensible description; `separating` supports the paper's claim-boundary theme.
   - Risk status: acceptable.

6. `Topology Optimization (TO), including the SIMP method, remains a central baseline for load-path optimization.`
   - Function: identifies a specific baseline family.
   - Claim type: baseline claim.
   - Word choices: `remains` and `central` justify future comparison; `load-path` keeps the claim structural rather than aerodynamic.
   - Risk status: grounded by cited SIMP/TO sources.

7. `Recent neural TO approaches accelerate parts of this process.`
   - Function: acknowledges modern ML in TO.
   - Claim type: literature context.
   - Word choices: `parts of this process` avoids claiming end-to-end replacement.
   - Risk status: grounded by citation.

8. `The present repository is closer to a generative prior over voxelized shapes than to classical topology optimization, but TO practice remains a useful reference for how strongly structural or performance claims should be validated.`
   - Function: positions the project relative to TO without mislabeling it.
   - Claim type: novelty and limitation claim.
   - Word choices: `closer to` avoids categorical overprecision; `generative prior` describes the actual role; `how strongly` sets evidence expectations.
   - Risk status: well bounded.

9. `Conventional aerodynamic shape optimization couples CFD solvers with numerical optimization over parameterized geometries, often using gradient or adjoint information.`
   - Function: introduces the classical CFD optimization comparison.
   - Claim type: field framing.
   - Word choices: `conventional` marks a mature reference path; `often` avoids universalizing; `parameterized geometries` distinguishes it from freeform latent samples.
   - Risk status: grounded by cited ASO literature.

10. `Surrogate modeling is also widely used for efficient design-space exploration.`
   - Function: adds another mature baseline family.
   - Claim type: literature claim.
   - Word choices: `also` links to ASO; `widely used` is broad but citation-supported; `efficient` explains why surrogates matter.
   - Risk status: grounded.

11. `In those settings, the geometry representation, optimizer, and solver are explicit components of a controlled loop.`
   - Function: explains the difference between mature optimization workflows and the repository.
   - Claim type: comparative framing.
   - Word choices: `explicit` and `controlled loop` set the standard the repository has not yet reached.
   - Risk status: conservative.

12. `By contrast, the present repository samples from a learned latent model and then applies CFD-style evaluation.`
   - Function: states the repository's actual workflow in contrast.
   - Claim type: code-path claim.
   - Word choices: `By contrast` makes the comparison clear; `CFD-style` avoids claiming equivalent rigor to mature CFD ASO.
   - Risk status: grounded.

13. `This makes the workflow exploratory, and success in code-path execution should not be confused with the stronger evidence expected in mature CFD-based shape optimization.`
   - Function: prevents the comparison from becoming an overclaim.
   - Claim type: limitation claim.
   - Word choices: `exploratory` is the key maturity limiter; `should not be confused` is intentionally direct.
   - Risk status: important guardrail.

14. `Physics-informed machine learning incorporates PDEs through PINNs, Neural Operators such as FNO, and operator-learning models such as DeepONet.`
   - Function: introduces physics-informed ML categories.
   - Claim type: literature context.
   - Word choices: `incorporates` is broad enough for varied methods; the named methods are recognizable anchors.
   - Risk status: grounded by citations.

15. `Differentiable-simulation work goes further by embedding differentiable fluid solvers into learning and design frameworks.`
   - Function: identifies a stronger simulation-in-the-loop path.
   - Claim type: literature comparison.
   - Word choices: `goes further` is relative to PINN/operator context; `differentiable fluid solvers` marks the capability absent here.
   - Risk status: grounded by citation.

16. `These directions represent stronger forms of "simulation in the loop" than the reduced sanity evidence reported here.`
   - Function: makes the evidence hierarchy explicit.
   - Claim type: limitation and comparison.
   - Word choices: `stronger forms` avoids dismissing the current work; `reduced sanity evidence` is deliberately modest.
   - Risk status: conservative.

17. `The current repository uses an internal LBM-style scoring path as an implementation mechanism; using that path to promote labels across fidelity levels after validation remains future work.`
   - Function: clarifies the current role of the solver path.
   - Claim type: code-path and future-work claim.
   - Word choices: `implementation mechanism` avoids presenting it as validated physics; `after validation` is a crucial condition.
   - Risk status: grounded and cautious.

18. `Constraint-aware generation is another relevant line of work.`
   - Function: transitions to manufacturability and validity constraints.
   - Claim type: structure.
   - Word choices: short sentence reduces list fatigue and keeps prose human.
   - Risk status: no issue.

19. `Enforcing physical validity, including manufacturability and connectivity, remains a core challenge.`
   - Function: states why constraints matter for generated geometry.
   - Claim type: field framing.
   - Word choices: `remains` implies unresolved challenge; `including` signals examples rather than exhaustive list.
   - Risk status: citation-supported.

20. `Oh et al. demonstrated generative design for structures, while Steinmetz et al. used differentiable projection for manufacturable designs.`
   - Function: names representative constraint/generative design work.
   - Claim type: literature comparison.
   - Word choices: `demonstrated` is assigned to cited work, not this repository; `manufacturable` is tied to Steinmetz et al., not our outputs.
   - Risk status: grounded by citations.

21. `The current repository evidence is more limited: connectivity penalties, post-processing cleanup, and manufacturing-aware scoring proxies are useful for smoke-testing the code path, but they are not aircraft-specific validity guarantees.`
   - Function: explicitly limits this repository against stronger constraint-aware design work.
   - Claim type: limitation claim.
   - Word choices: `more limited`, `proxies`, and `smoke-testing` all reduce evidentiary strength; `not aircraft-specific validity guarantees` blocks a common overread.
   - Risk status: central guardrail.

22. `The implementation also borrows standard architectural ingredients from modern deep learning, including attention-style modules inspired by the Transformer literature.`
   - Function: acknowledges reused architecture components.
   - Claim type: implementation/literature claim.
   - Word choices: `borrows` and `standard` make clear this is reuse; `attention-style` avoids claiming a novel Transformer.
   - Risk status: grounded.

23. `In this paper, those ingredients are treated as reused components rather than as a novelty claim.`
   - Function: blocks novelty overclaim around standard components.
   - Claim type: claim-boundary sentence.
   - Word choices: `reused components` is the important phrase; `rather than` contrasts against novelty.
   - Risk status: conservative.

24. `The present repository does not claim a new diffusion objective, a new adjoint CFD method, or a validated aircraft-design benchmark.`
   - Function: opens the novelty subsection with explicit exclusions.
   - Claim type: negative novelty claim.
   - Word choices: repeated `new` is intentional because it rules out three possible overclaims; `validated` is reserved for a future benchmark.
   - Risk status: very important.

25. `Its defensible contribution is narrower: a proof-of-concept assembly of a latent generative model, voxel decoding path, CFD-informed evaluation hook, STL export, and baseline and claim-gate checks in a single reproducible codebase.`
   - Function: states the actual novelty.
   - Claim type: contribution claim.
   - Word choices: `defensible` and `narrower` frame the claim as evidence-aware; `assembly` emphasizes integration; `hook` weakens CFD overclaim; `reproducible codebase` refers to runnable artifacts.
   - Risk status: grounded if repo artifacts remain.

26. `Relative to general 3D diffusion work, the emphasis here is on engineering-shaped voxel outputs rather than text-to-3D asset generation.`
   - Function: compares to 3D diffusion.
   - Claim type: positioning.
   - Word choices: `emphasis` avoids superiority; `engineering-shaped` is broader than aircraft-valid; `rather than` clarifies distinction.
   - Risk status: acceptable.

27. `Relative to topology optimization and classical CFD shape optimization, the repository explores a learned generative prior rather than directly optimizing a single parameterized shape.`
   - Function: compares to optimization workflows.
   - Claim type: positioning.
   - Word choices: `explores` is modest; `learned generative prior` describes the role; `rather than directly optimizing` avoids claiming TO/ASO equivalence.
   - Risk status: grounded.

28. `Relative to surrogate and differentiable-simulation papers, the current implementation is lighter-weight and presently supported only by sanity-run evidence.`
   - Function: compares to stronger simulation-aware ML work.
   - Claim type: limitation comparison.
   - Word choices: `lighter-weight` describes engineering scope; `presently` and `only` restrict claims to current evidence.
   - Risk status: conservative.

29. `This positioning is intentionally conservative.`
   - Function: summarizes the section's stance.
   - Claim type: rhetorical framing.
   - Word choices: `intentionally` signals discipline, not weakness.
   - Risk status: no issue.

30. `The repository should not yet be read as demonstrating superior exploration of an engineering design space, validated surrogate replacement, mission-conditioned aircraft generation, or manufacturing-conditioned aircraft generation.`
   - Function: names the strongest claims that are not supported.
   - Claim type: exclusion sentence.
   - Word choices: `should not yet` leaves room for future work; the list is explicit to prevent accidental implication.
   - Risk status: critical anti-overclaim sentence.

31. `Those claims require the stronger evidence gates summarized below and in Section \ref{sec:validation}.`
   - Function: routes the reader to evidence requirements.
   - Claim type: roadmap and gate claim.
   - Word choices: `require` is intentionally strict; `stronger evidence gates` matches the paper's validation architecture.
   - Risk status: grounded by validation section.

32. `The reduced final evidence package requires three executable baseline families: retrieval records from the grounded manifest, unconditional samples from the checkpoint, and bundled grounded STL examples.`
   - Function: states the baseline families required in the current reduced package.
   - Claim type: protocol claim.
   - Word choices: `reduced final evidence package` distinguishes current gates from publication baselines; `executable` means artifacts must exist, not just be proposed.
   - Risk status: grounded by final evidence package.

33. `These are guardrail baselines: they verify that the report contains concrete comparison artifacts and repeated-run statistics, but they do not establish superiority over classical optimization or mature generative-model baselines.`
   - Function: limits what the current baselines prove.
   - Claim type: limitation claim.
   - Word choices: `guardrail` means minimum check; `concrete` emphasizes files/results; `do not establish superiority` blocks benchmark overclaim.
   - Risk status: essential.

34. `Publication-grade baseline comparisons should still include four stronger families:`
   - Function: introduces future baseline requirements.
   - Claim type: protocol recommendation.
   - Word choices: `Publication-grade` separates future standard from current reduced package; `should still` keeps the requirement open.
   - Risk status: future-work framing.

35. `Classical primitives: hand-designed parametric aircraft primitives, such as swept or tapered wings extruded from NACA airfoil profiles, with controlled aspect-ratio, sweep, and taper ranges.`
   - Function: names a transparent hand-designed baseline.
   - Claim type: future baseline requirement.
   - Word choices: `hand-designed` and `controlled` emphasize comparability; NACA profiles make the primitive baseline concrete.
   - Risk status: proposed, not claimed executed.

36. `Unconstrained generative 3D models: representative voxel, point-cloud, and implicit-shape baselines such as voxel-GAN, point-cloud diffusion, and occupancy networks.`
   - Function: names model-family baselines outside aircraft constraints.
   - Claim type: future baseline requirement.
   - Word choices: `representative` avoids needing every model; the three shape representations make coverage broad.
   - Risk status: proposed, not claimed executed.

37. `Independent optimization search: a SIMP-style topology-optimization comparison and a random-search baseline in which sampled candidates are ranked by the same CFD evaluation protocol.`
   - Function: adds non-learned search comparisons.
   - Claim type: future baseline requirement.
   - Word choices: `independent` means outside the generator; `same CFD evaluation protocol` is essential for fair ranking.
   - Risk status: proposed.

38. `Differentiable aerodynamic shape optimization: an adjoint-based refinement baseline for a standard fuselage-wing primitive.`
   - Function: names a mature aerodynamic optimization baseline.
   - Claim type: future baseline requirement.
   - Word choices: `standard fuselage-wing primitive` keeps the target controlled; `adjoint-based` ties to ASO literature.
   - Risk status: proposed.

39. `Ablations should isolate the impact of the aerodynamic loss, connectivity loss, connected-component cleanup hook, semantic constraint projector, and consistency-step count.`
   - Function: states mechanism-level tests needed later.
   - Claim type: ablation protocol.
   - Word choices: `isolate` is the key scientific requirement; each listed component maps to a claim-bearing mechanism.
   - Risk status: future requirement, not current evidence.

40. `Until those ablations are executed on larger grounded aircraft-like data with shared solver settings, they remain a protocol rather than evidence for mechanism-level causality.`
   - Function: prevents the protocol from being mistaken for results.
   - Claim type: limitation.
   - Word choices: `Until` creates a clear gate; `larger grounded aircraft-like data` names the missing dataset standard; `mechanism-level causality` is deliberately reserved.
   - Risk status: critical guardrail.

## Table Claim Labels

The final table is a plan, not evidence. Its labels were chosen to name gated claims (`Aerodynamic efficiency`, `Manifold integrity`, `Computational efficiency`, `Generative diversity`) and pair them with the baseline, metric, and execution requirements that would be needed before stronger paper claims are justified.

## Audit Notes

The section is relatively expansive: it covers five adjacent literatures and proposes future baselines. It is not hype-heavy because every comparison is paired with an evidence boundary or a future-work gate.
