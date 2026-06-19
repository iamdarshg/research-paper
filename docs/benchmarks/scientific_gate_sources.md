# Scientific Gate Source Map

This file records the sources used to keep the repository's claim gates
scientifically conservative. Code comments cite these sources where the gate is
implemented; this document explains the mapping.

## Gate List

| Gate | Current executable artifact | Evidence needed to pass scientifically |
| --- | --- | --- |
| Grounded corpus contract | `CLI/validate_manifest.py`, `docs/dataset/manifest_schema.example.json` | Provenance-complete corpus manifest with real aircraft-like examples, units, preprocessing version, splits, and complete `design_spec` records |
| Aircraft-specific validity | `CLI/aircraft_validity.py` | Multiple generated artifacts passing shape checks plus corpus-backed evidence that those proxies map to aircraft-like geometry |
| Grounded condition response | `CLI/run_condition_benchmark.py` | Fixed A/B sweeps over grounded samples with response metrics traceable to checkpoint outputs |
| Structural/manufacturing feasibility | `CLI/condition_feasibility.py`, `DesignSpec` validation | Executed feasibility reports plus stronger structural/load-path analysis for any structural viability claim |
| CFD/aerodynamic credibility | `evaluate-baselines`, future V&V reports | Grid/time-step sensitivity, solver agreement, uncertainty quantification, and baseline comparison under fixed settings |
| CFD-guided training ablation | planned ablation report | Matched-seed runs with and without the CFD/scoring term, candidate-ranking comparison, and uncertainty intervals |
| Prior-method comparison | planned method-comparison report | Named prior/internal baselines under the same metrics, seeds, and evaluation protocol |
| Publication-quality validation | planned solver validation study | Grid convergence, timestep sensitivity, external/reference solver comparison, and uncertainty qualification |
| Baseline/statistical reporting | `CLI/multi_seed_eval.py` helpers | Named baselines, sufficient seeds, mean/std or stronger uncertainty intervals, and explicit insufficient-evidence states |
| Final evidence package | `CLI/final_evidence.py`, `paper/FINAL_EVIDENCE_PACKAGE.md` | All required reports present and `status: pass` |

## Authoritative Sources

### NASA-STD-7009B: Standard For Models And Simulations

Use for the repo-level credibility posture: final claims require an assembled
evidence package across model lifecycle, verification, validation, uncertainty,
sensitivity, and reporting.

Source: [NASA-STD-7009B](https://standards.nasa.gov/standard/nasa/nasa-std-7009)

Used in:
- `CLI/final_evidence.py`
- `CLI/condition_feasibility.py`
- `CLI/aircraft_validity.py`
- `paper/FINAL_EVIDENCE_PACKAGE.md`

### NASA CFD Verification And Validation Overview

Use for the distinction between verification, validation, uncertainty/error,
and physical-reality comparison. This supports the repo's refusal to treat
connectivity, directional response, or smoke CFD as validated aircraft evidence.

Source: [NASA Glenn CFD V&V overview](https://www.grc.nasa.gov/WWW/wind/valid/tutorial/overview.html)

Used in:
- `CLI/aircraft_validity.py`
- `CLI/run_condition_benchmark.py`
- `paper/FINAL_RUN_GATES.md`

### ASME V&V 20

Use for CFD validation requirements around quantifying accuracy at validation
points and considering errors/uncertainties in both solution and data.

Source: [ASME V&V 20](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer/2009/print-book/)

Used in:
- `CLI/run_condition_benchmark.py`
- future CFD report schema

### AIAA Guide For Verification And Validation Of CFD Simulations

Use for CFD V&V terminology and the separation of verification, validation,
and prediction. This is relevant before any aerodynamic-optimization wording is
unlocked.

Source: [AIAA G-077-1998](https://doi.org/10.2514/4.472855.001)

Recommended future use:
- solver validation study
- aerodynamic baseline comparison report

### Oberkampf And Trucano Survey Of V&V In Computational Simulation

Use as background for uncertainty-aware computational simulation claims and
for avoiding overclaiming from unvalidated numerical outputs.

Source: [Oberkampf and Trucano, 2002](https://doi.org/10.1016/S0376-0421(02)00005-2)

Recommended future use:
- validation study rationale
- final evidence package interpretation

### Grid Convergence Index

Use for future grid-convergence studies rather than accepting one CFD grid as a
claim-bearing result.

Source: [Celik et al., 2008](https://doi.org/10.1115/1.2960953)

Recommended future use:
- `solver_validation_study.json`
- CFD baseline comparison methods

### NASA Turbulence Modeling Resource

Use for future solver-verification fixtures and grid-convergence reference
cases. It provides verification/validation cases and reference data for
turbulence modeling work.

Source: [NASA Turbulence Modeling Resource](https://www.nasa.gov/nasa-turbulence-modeling-resource/)

Recommended future use:
- CFD validation report docs
- solver agreement tests
- grid-convergence protocol docs

### NIST CFD VVUQ Summary

Use for credibility factors in CFD: physical modeling quality, analyst quality,
V&V activities, and uncertainty quantification.

Source: [NISTIR 8298](https://www.nist.gov/publications/summary-industrial-verification-validation-and-uncertainty-quantification-procedures)

Recommended future use:
- final CFD credibility report
- baseline policy docs

### NIST Measurement Uncertainty

Use for reporting uncertainty as standard deviations or intervals rather than
single-run values.

Source: [NIST Measurement Uncertainty](https://www.nist.gov/itl/sed/topic-areas/measurement-uncertainty)

Used in:
- `CLI/multi_seed_eval.py`
- `docs/benchmarks/baseline_policy.md`

### NIST/SEMATECH Confidence Limits

Use for future confidence-interval reporting when gate outputs compare generated
samples against baselines.

Source: [NIST/SEMATECH e-Handbook, confidence limits](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

Recommended future use:
- baseline statistics report
- condition-response effect-size intervals

### ASA Statement On P-Values

Use to keep p-values from being treated as standalone proof of superiority.

Source: [Wasserstein and Lazar, 2016](https://doi.org/10.1080/00031305.2016.1154108)

Recommended future use:
- prior-method comparison report
- baseline statistics interpretation

### Datasheets For Datasets

Use for manifest/corpus provenance requirements: dataset motivation,
composition, collection process, preprocessing, uses, and limitations.

Source: [Gebru et al., Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

Used in:
- `CLI/validate_manifest.py`
- `docs/dataset/GROUNDED_CORPUS_SPEC.md`
- `docs/dataset/manifest_schema.example.json`

### Model Cards For Model Reporting

Use for checkpoint/report documentation once a claim-bearing model exists:
model details, intended use, factors, metrics, evaluation data, and limitations.

Source: [Mitchell et al., Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

Recommended future use:
- claim-bearing checkpoint card
- final evidence package report

### Reproducibility In Deep Reinforcement Learning And NLP

Use for multi-seed reporting, sensitivity to implementation details, and
avoiding single-run conclusions in learned generative workflows.

Sources:
- [Henderson et al., 2018](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
- [Dodge et al., 2019](https://aclanthology.org/D19-1224/)

Recommended future use:
- multi-seed evaluation policy
- CFD-guided training ablation

### FAIR Data Principles

Use for future corpus packaging so claim-bearing data can be found, accessed,
interoperated with, and reused.

Source: [Wilkinson et al., 2016](https://doi.org/10.1038/sdata.2016.18)

Recommended future use:
- grounded corpus release checklist
- manifest provenance fields
