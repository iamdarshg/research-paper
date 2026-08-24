# 128^3 Scaling Study with Expanded Corpus

Date: 2026-08-24. Branch: main at 979ea3c.

## Corpus expansion results

| Source | Count | Description |
|---|---|---|
| Original CAD | 1,069 | AircraftVerse + fixedwing + airshow, canonical-validated |
| Perturbed | 4,958 | Six aerodynamic shape modifications with validity gating |
| Procedural | 2,000 | Seven parametric aircraft types from scratch |
| **Total** | **8,027** | All unique by SHA-256 canonical voxel hash |

Estimated information content: ~4.07 Mbit (vs 1.2 Mbit for the original corpus).

### Perturbation transforms and acceptance rates

| Transform | What it does | Accepted | Rate |
|---|---|---|---|
| tail_widen_30 | Rear fuselage z-thickness +30% | 796/1069 | 74.5% |
| tail_widen_50 | Rear fuselage z-thickness +50% | 796/1069 | 74.5% |
| wing_dihedral_up | Outer wing shifted upward | 801/1069 | 74.9% |
| wing_dihedral_down | Outer wing shifted downward | 801/1069 | 74.9% |
| nose_thin | Forward fuselage thickness -30% | 733/1069 | 68.6% |
| airfoil_thicken | 1-voxel z-axis dilation | 1,031/1069 | 96.4% |

### Procedural types generated

| Type | Count | Key features |
|---|---|---|
| flying_wing | 551 | No fuselage, blended lifting surface |
| delta_wing | 463 | Triangular planform, vertical fins only |
| glider | 431 | High aspect ratio, dihedral wings, thin fuselage |
| anhedral | 276 | Negative dihedral, wings angle downward |
| canard | 156 | Forward stabilizer + main aft wing |
| biplane | 92 | Two stacked wing planes with struts |
| swept_wing | 31 | Aft-swept main and tail surfaces |

Lifting bodies were attempted but rejected by symmetry gates.

## Grokking-regime analysis at 8k corpus / 128^3

Corpus information: ~4,167 kbit (4.07 Mbit), up from ~1,203 kbit for 1,069 shapes.

| Config | Params | Effective cap | Ratio | Regime | bf16 peak | Verdict |
|---|---|---|---|---|---|---|
| ref_96 w=928 d=5 | 35M | 1.3 Mbit | 0.3x | UNDERFITTING | 7.7 GiB | FITS |
| **128 w=2048 d=8** | **88M** | **11.1 Mbit** | **2.6x** | **GOOD BALANCE** | **20.7 GiB** | **FITS** |
| 128 w=3072 d=12 | 237M | 39.4 Mbit | 9.2x | memorization risk | 23.6 GiB | FITS |
| 128 w=4096 d=16 | 534M | 96.6 Mbit | 22.6x | DEEP MEMORIZATION | 28.1 GiB | FITS |
| 128 w=6144 d=12 | 862M | 171 Mbit | 40.1x | DEEP MEMORIZATION | 34.3 GiB | FITS |
| 128 w=8192 d=12 | 1509M | 314 Mbit | 73.6x | DEEP MEMORIZATION | 44.0 GiB | FITS |
| 128 w=16384 d=5 | 2185M | 564 Mbit | 132x | DEEP MEMORIZATION | 61.6 GiB | FITS |

Grokking boundary: w=2048-3072 d=8-12 (~88-237M params).

## A100 cost estimate at 128^3

First-principles FLOP calculation:

| Config | s/update | hrs/epoch | 2ep on-demand | 2ep spot | Budget fit |
|---|---|---|---|---|---|
| w=2048 d=8 (88M) | 13.2s | 29.3h | $187 | $70 | YES |
| w=3072 d=12 (237M) | 31.0s | 69.1h | $442 | $166 | Spot only |
| w=4096 d=16 (534M) | 66.4s | 148.1h | $948 | $356 | NO |

## Recommendation

Train w=2048 d=8 (~88M total params) at 128^3 on the 8027-shape corpus.

Why this config:
1. Right at the grokking balance point - enough capacity without trivial memorization
2. Fits the $270 budget ($187 on-demand or $70 spot for 2 epochs)
3. All shapes fit A100-80GB in bf16 with 57 GiB headroom
4. At 13 s/u, one epoch takes ~29 hours; 2 epochs gives meaningful convergence

Why NOT larger:
- 500M+ configs will memorize within 1-2 epochs
- They exceed the budget for even a single epoch on-demand
- Marginal quality gain does not justify cost when already past generalization threshold