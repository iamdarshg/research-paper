# Abstract Sentence Audit

Source: `paper/main.tex`

Detector results:
- lmscan: `0.2578`, verdict `Likely human`, confidence `medium`
- RoBERTa detector: fake mean `0.035332`, fake max `0.035332`

Overall function: the abstract deliberately frames the work as a proof-of-concept repository paper, not as a completed aircraft-design breakthrough.

## Sentence Units

1. `This paper documents a proof-of-concept generative-design pipeline for aircraft-like voxel geometries.`
   - Function: opens the paper by defining the artifact and scope.
   - Claim type: implementation claim.
   - Word choices: `documents` avoids implying a new theory; `proof-of-concept` limits the maturity claim; `aircraft-like` avoids claiming flight-ready aircraft; `voxel geometries` states the actual representation.
   - Risk status: grounded and conservative.

2. `The repository combines a latent diffusion-style model, a voxel decoder, connectivity heuristics, and a CFD-informed scoring path built around an internal lattice-Boltzmann solver plus an external OpenFOAM export route.`
   - Function: lists the main technical components.
   - Claim type: architecture and code-path claim.
   - Word choices: `diffusion-style` is safer than claiming a novel diffusion formulation; `heuristics` signals non-final validity checks; `CFD-informed` is weaker than CFD-optimized; `internal` and `external` distinguish the local solver from OpenFOAM export.
   - Risk status: grounded if the named components remain present in the codebase and final evidence package.

3. `The present implementation is intentionally scoped: the public Airshow run uses a 355-record low-resolution voxel corpus, the protocol is a reduced final-evidence run, and the evidence should be read as code-path validation rather than publication-grade aerodynamic or structural validation.`
   - Function: puts claim boundaries in the abstract before the reader sees results.
   - Claim type: limitation and evidence-quality claim.
   - Word choices: `intentionally scoped` explains the narrowness without apologizing; `355-record` gives the real corpus scale; `low-resolution` prevents the Airshow count from implying high-fidelity geometry; `reduced` prevents overclaiming; `code-path validation` names the actual evidence class; `publication-grade` marks what is not being claimed.
   - Risk status: important guardrail sentence.

4. `In addition to describing the architecture and benchmark workflow, we use the current experiments to clarify which claims the codebase supports today and which stronger claims, including validated mission-conditioned or manufacturing-conditioned aircraft generation, remain future work.`
   - Function: tells the reader the paper is partly a claim-boundary document.
   - Claim type: roadmap and limitation claim.
   - Word choices: `current experiments` ties the claim to existing runs; `supports today` prevents future-looking claims from being treated as achieved; `validated mission-conditioned` and `manufacturing-conditioned` name the stronger claims explicitly so they are not implied by the condition-vector plumbing.
   - Risk status: grounded as long as the body keeps those future-work boundaries.

## Audit Notes

The abstract is deliberately expansive in component coverage but conservative in evidentiary force. The wording was chosen to be readable, not flashy: most nouns name real code paths, and most adjectives reduce rather than inflate claims.
