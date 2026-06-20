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
| Abstract | 0.2671 | Likely human | 0.105105 | 0.105105 |
| Introduction | 0.2220 | Likely human | 0.000229 | 0.000229 |
| Related Work | 0.2151 | Likely human | 0.012215 | 0.036288 |
| Methodology | 0.2235 | Likely human | 0.002193 | 0.006217 |
| Results and Discussion | 0.2149 | Likely human | 0.164410 | 0.618411 |
| Validation and Testing Standards | 0.2086 | Likely human | 0.000202 | 0.000202 |
| Conclusion | 0.2158 | Likely human | 0.000224 | 0.000224 |
| Full Paper | 0.2314 | Likely human | 0.001994 | 0.016500 |

## Style Signals

The final full-paper lmscan pass reports:
- `burstiness`: 0.231946
- `bigram_repetition`: 0.118151
- `passive_voice_ratio`: 0.272727
- `sentence_opening_diversity`: 0.843434
- `chatbot_marker_score`: 0.000561
- `word_count`: 3562 after LaTeX stripping
- `sentence_count`: 198 after LaTeX stripping

The most important style cleanup was not cosmetic. The paper now repeatedly separates:
- what the code path actually demonstrates,
- what the final evidence package proves,
- what remains future work,
- and what stronger claims would require.

## Detector Caveats

`lmscan` is useful for local style screening but is not a formal authorship oracle. Its passive-voice and repetition signals are expected to be higher in academic writing.

The RoBERTa detector is an older OpenAI detector model trained around GPT-2-era generated text. It is useful as a local second opinion, but modern LLM authorship cannot be proved or disproved from this score.

The high RoBERTa maximum on `Results and Discussion` comes from one short formula-heavy tail chunk after LaTeX stripping mangled \(u = Ma / \sqrt{3}\) into incomplete plain text. The full-paper RoBERTa mean and max remain low, and the lmscan result for the same section remains `Likely human`.
