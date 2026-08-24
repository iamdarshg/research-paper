# Cheap Training Paths: Full GPU Trade-off Analysis

Date: 2026-08-24. Branch: main.

## Key Discovery: GCP L4 Spot at $0.70/hr

The L4 is the cheapest viable GCP GPU for this workload. At 121 TFLOPS bf16,
it handles our compute-bound decoder MLP efficiently and costs 3x less than A100 spot.

## Answer 1: Largest model at 128^3 within $270

| GPU | Config | Params | Capacity | s/u | Cost/ep | Epochs | Total |
|---|---|---|---|---|---|---|---|
| L4 ($0.70) | w=3072 d=22 | 426M | 17x MEMORIZE | 63s | $99 | 2 | $198 |
| **L4 ($0.70)** | **w=1792 d=12** | **100M** | **2.9x GOOD** | **18s** | **$28** | **9** | **$253** |
| L40S ($1.80) | w=3072 d=16 | 313M | 12x MEMORIZE | 21s | $85 | 3 | $255 |
| PRO6000 ($2.25) | w=3072 d=16 | 313M | 12x MEMORIZE | 24s | $121 | 2 | $242 |

**Largest USEFUL model (capacity < 5x): w=1792 d=12 = 100M on L4 spot.**
Exactly the requested param count at the grokking sweet spot.

## Answer 2: Largest voxel size at ~100M params within $270

| Grid | L4 cost/ep | L4 epochs | L40S cost/ep | L40S epochs | Verdict |
|---|---|---|---|---|---|
| 96^3 | $11 | 23 | $20 | 13 | Comfortable |
| **128^3** | **$28** | **9** | **$45** | **5** | **Sweet spot** |
| 160^3 | $58 | 4 | $91 | 2 | Possible but thin |
| 192^3 | $100 | 1 | $157 | 1 | Single pass only |
| 224^3 | $158 | 0 | $249 | 1 | Marginal on L40S only |

128^3 remains the sweet spot - enough resolution for meaningful geometry quality
while leaving room for multiple training epochs.

## Recommended Training Plan: $253 total

- GPU: L4 spot on GCP ($0.70/hr)
- Model: w=1792 d=7 lat=512 = 100M trainable params
- Resolution: 128^3
- Corpus: 8,027 shapes
- Epochs: 9 (72,243 optimizer updates)
- Per-update: ~18 seconds
- Total time: ~144 hours (6 days continuous)
- Capacity ratio: 2.9x (good balance)

## All GPU options compared at 128^3 / 100M params

| GPU | Rate/hr | TFLOPS | s/u | hrs/ep | cost/ep | epochs in $270 |
|---|---|---|---|---|---|---|
| T4 spot | $0.35 | 65 | 33 | 74h | $26 | 10 |
| **L4 spot** | **$0.70** | **121** | **18** | **28h** | **$20-28** | **9-13** |
| A100 spot | $2.50 | 228 | 14 | 31h | $78 | 3 |
| L40S spot | $1.80 | 362 | 11 | 26h | $45 | 5 |
| PRO6000 spot | $2.25 | 325 | 12 | 27h | $60 | 4 |