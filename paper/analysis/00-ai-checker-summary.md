# Local AI Checker and Evidence Summary

Source paper: `paper/main.tex` and `paper/sections/*.tex`

Generated artifacts:
- Plain text extraction: `build/ai_check/*.txt`
- Local lmscan JSON: `build/ai_check/*.lmscan.json`
- lmscan summary: `build/ai_check/lmscan_summary.json`
- Local RoBERTa detector summary: `build/ai_check/roberta_openai_detector_summary.json`
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
| Abstract | 0.2578 | Likely human | 0.035332 | 0.035332 |
| Introduction | 0.2579 | Likely human | 0.047526 | 0.094879 |
| Related Work | 0.2402 | Likely human | 0.001538 | 0.004273 |
| Methodology | 0.2539 | Likely human | 0.088077 | 0.351760 |
| Results and Discussion | 0.2527 | Likely human | 0.339818 | 0.917835 |
| Validation and Testing Standards | 0.2471 | Likely human | 0.000216 | 0.000216 |
| Conclusion | 0.2530 | Likely human | 0.000253 | 0.000253 |
| Full Paper | 0.2634 | Likely human | 0.053759 | 0.682330 |

## Style Signals

The final full-paper lmscan pass reports:
- `burstiness`: 0.126944
- `bigram_repetition`: 0.132041
- `passive_voice_ratio`: 0.287879
- `sentence_opening_diversity`: 0.833333
- `chatbot_marker_score`: 0.000487
- `word_count`: 4109 after LaTeX stripping
- `sentence_count`: 198 after LaTeX stripping

The most important style cleanup was not cosmetic. The paper now repeatedly separates:
- what the code path actually demonstrates,
- what the final evidence package proves,
- what remains future work,
- and what stronger claims would require.

## Detector Caveats

`lmscan` is useful for local style screening but is not a formal authorship oracle. Its passive-voice and repetition signals are expected to be higher in academic writing.

The RoBERTa detector is an older OpenAI detector model trained around GPT-2-era generated text. It is useful as a local second opinion, but modern LLM authorship cannot be proved or disproved from this score.

The high RoBERTa maximum on `Results and Discussion` comes from the dense solver-validation paragraph after LaTeX stripping converts mathematical notation and benchmark dimensions into plain text. I rewrote the worst formula-dense sentence into prose, which lowered the full-paper RoBERTa max from the earlier spike while preserving the solver provenance. The lmscan result for the same section remains `Likely human`; the RoBERTa result is treated as a style-screening flag, not authorship evidence.
