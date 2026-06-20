# Local AI Checker and Evidence Summary

Source paper: `paper/main.tex` and `paper/sections/*.tex`

Generated artifacts:
- Plain text extraction: `build/ai_check_current/*.txt`
- Local lmscan JSON: `build/ai_check_current/*.lmscan.json`
- lmscan summary: `build/ai_check_current/lmscan_summary.json`
- Local RoBERTa detector summary: `build/ai_check_current/roberta_openai_detector_summary.json`
- Final evidence report: `build/protocol_final/final_evidence_package.json`

Detector sources:
- `lmscan` package page: https://pypi.org/project/lmscan/
- RoBERTa detector model card: https://huggingface.co/openai-community/roberta-base-openai-detector

## Grounding Status

The final evidence package was rerun after the paper edits and returned `status: pass`.

Shared lineage fields:
- `run_id`: `protocol-43207a893be9`
- `checkpoint_hash`: `3736dbd5c8521d14bf6aa6f42a5cf94ece3b4bf1f38df033e66097a0405dd0f3`
- `manifest_hash`: `b1d32c523de68c7095ae8803b10b9d9584af41ac342cc58736e3684da6a16c98`
- `protocol_hash`: `43207a893be9965d5c44cc0322821f64dd23bc831e91c541c37cabd2342ef075`

All final evidence gates passed:
- Manifest validation
- Aircraft-shape heuristic screen
- Condition-response benchmark
- Manufacturing-bounds screen
- Baseline statistics

Interpretation: the paper can describe a reduced, internally consistent evidence package. It still should not claim publication-grade aircraft optimization, full physical validation, structural viability, or superiority over mature baselines.

## Local AI Checker Results

These detectors are screening tools, not ground truth. A low score does not prove human authorship, and a high score does not prove AI authorship. They are useful here as style diagnostics.

| Section | lmscan AI probability | lmscan verdict | RoBERTa fake mean | RoBERTa fake max |
| --- | ---: | --- | ---: | ---: |
| Abstract | 0.2571 | Likely human | 0.039344 | 0.039344 |
| Introduction | 0.2217 | Likely human | 0.000229 | 0.000229 |
| Related Work | 0.2248 | Likely human | 0.003821 | 0.011105 |
| Methodology | 0.2193 | Likely human | 0.000178 | 0.000186 |
| Results and Discussion | 0.1880 | Human-written | 0.155046 | 0.907542 |
| Validation and Testing Standards | 0.2434 | Likely human | 0.000181 | 0.000181 |
| Conclusion | 0.1767 | Human-written | 0.000174 | 0.000174 |
| Full Paper | 0.2256 | Likely human | 0.086482 | 0.902928 |

## Style Signals

The final full-paper lmscan pass reports:
- `burstiness`: 0.220973
- `bigram_repetition`: 0.125275
- `passive_voice_ratio`: 0.260664
- `sentence_opening_diversity`: 0.829384
- `chatbot_marker_score`: 0.000510
- `word_count`: 3922 after LaTeX stripping
- `sentence_count`: 211 after LaTeX stripping

The most important style cleanup was not cosmetic. The paper now repeatedly separates:
- what the code path actually demonstrates,
- what the final evidence package proves,
- what remains future work,
- and what stronger claims would require.

## Detector Caveats

`lmscan` is useful for local style screening but is not a formal authorship oracle. Its passive-voice and repetition signals are expected to be higher in academic writing.

The RoBERTa detector is an older OpenAI detector model trained around GPT-2-era generated text. It is useful as a local second opinion, but modern LLM authorship cannot be proved or disproved from this score.

The high RoBERTa maximum on `Results and Discussion` still comes from technical chunks after LaTeX stripping converts mathematical notation, tables, benchmark dimensions, lattice speed, and \(C_d\) notation into plain text. The lmscan result for the same section is now `Human-written`; the RoBERTa maximum is treated as a style-screening flag, not authorship evidence.

## 2026-06-20 Resolution and Originality Addendum

New supporting reports:
- `docs/benchmarks/airshow_resolution_sweep_20260620.md`
- `paper/analysis/originality-and-relevance-comparison.md`
- `paper/analysis/local-paraphrase-and-style-pass.md`

The paper now includes the `32^3` and `64^3` Airshow addendum. The update is
negative evidence, not stronger claim evidence: the `32^3` generated samples
failed aircraft-validity gates, and the `64^3` corpus validated but did not
produce a checkpoint within the local run ceiling. No online paraphraser upload
was used; the wording pass stayed local to avoid transmitting unpublished paper
text. The refreshed local detector artifacts were written under
`build/ai_check_current/`.
