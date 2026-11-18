# RP-G6 — held-out reconstructions: literal PASS, but a constant shape also passes

Gate (handoff): ≥90 % of held-out reconstructions pass the screen, and mean IoU vs GT ≥ 0.40.
Screen: the frozen old 12-check screen (`diversity_novelty_gate.sample_report(...)["old_gate_pass"]`, gate spec hash `4eafa948…`).
Reconstruction: the arm's converter applied to each holdout record's **true** latent (the 16-dim structured descriptor), passed through a sigmoid and the solver binarizer at 0.5.
Holdout: all 96 records of `div_corpus_660_raw/holdout.jsonl`; train has 660, and the split guard is in place.
Scripts: `rp_g6_recon_screen.py`, `rp_g6_baselines.py`. Raw: `recon_screen.json`, `baselines.json`.

| decoder | screen pass | Wilson 95 % | mean IoU vs GT | min IoU | RP-G6 literal |
|---|---:|---|---:|---:|---|
| C4 | 87/96 = 90.6 % | [83.1, 95.0] | **0.767** | 0.568 | PASS (by 1 record) |
| C3 | 95/96 = 99.0 % | [94.3, 99.8] | 0.605 | 0.304 | PASS |
| D3 | 96/96 = 100 % | [96.2, 100] | 0.607 | 0.307 | PASS |
| *GT voxels (ceiling)* | 96/96 | | 1.000 | | |
| *constant: majority-vote training shape, the same array for all 96* | **96/96** | | **0.661** | | **PASS** |
| *random training record (exploratory, RP-COND)* | | | 0.563 | | |
| *nearest training record by 16-dim latent (exploratory, RP-COND)* | | | 0.916 | | |

All 9 C4 screen failures are `grounded_occupancy_envelope`.

## Reading
- **Literal: PASS for all three arms.** C4 clears the 90 % line by exactly one record, and its CI spans 83–95 %, so a re-draw of the holdout could fail it.
- **The gate does not discriminate.** A model-free constant passes the screen 96/96 and scores IoU 0.661, well above the 0.40 line.
  - C3 and D3 score **below** that constant on IoU.
  - Only C4 beats it, by +0.106.
  - A nearest-neighbour lookup on the same latent reaches 0.916, beating every decoder.

  So "≥90 % pass, IoU ≥ 0.40" is met without learning anything about the specific shape. That is the same failure pattern as RP-G4 (X-13) and as the paper's own "a model-free constant passes the same screen".
- **What is real:** C4's decoder does carry per-record information beyond the corpus average (+0.106 IoU over the constant). It is still well short of what the latent itself allows (0.916).

Gate unchanged. Recorded as owner note X-15: an informative RP-G6 would score against the constant and nearest-neighbour baselines, not against a fixed 0.40.
