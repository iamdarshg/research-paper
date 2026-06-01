#!/usr/bin/env python3
"""Fail-closed machine-readable scaffolds for claim-bearing scientific gates.

These helpers check whether required evidence metadata exists and is internally
consistent. They do not decide that a scientific claim is true.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


CLAIM_BOUNDARY = (
    "This report checks evidence presence and schema consistency only; it does "
    "not establish superiority, CFD-guided benefit, or publication-quality validity."
)


def _is_present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _get_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _missing_paths(payload: Mapping[str, Any], paths: Iterable[str]) -> List[str]:
    return [path for path in paths if not _is_present(_get_path(payload, path))]


def _base_report(gate_id: str, required_evidence: Sequence[str], blockers: Sequence[str]) -> Dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": "blocked" if blockers else "pass",
        "required_evidence": list(required_evidence),
        "blockers": list(blockers),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_cfd_guided_training_ablation_report(metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Build a fail-closed scaffold for matched CFD-guided training ablations."""

    metadata = metadata or {}
    required = [
        "arms.cfd_guided.seeds",
        "arms.cfd_guided.checkpoints",
        "arms.cfd_guided.training_curves",
        "arms.cfd_guided.candidate_rankings",
        "arms.cfd_guided.cfd_metrics",
        "arms.control.seeds",
        "arms.control.checkpoints",
        "arms.control.training_curves",
        "arms.control.candidate_rankings",
        "arms.control.cfd_metrics",
        "statistical_comparison",
        "matched_config_fields",
        "changed_config_fields",
    ]
    blockers = [f"missing required evidence: {path}" for path in _missing_paths(metadata, required)]

    arms = metadata.get("arms") if isinstance(metadata.get("arms"), Mapping) else {}
    cfd_arm = arms.get("cfd_guided", {}) if isinstance(arms, Mapping) else {}
    control_arm = arms.get("control", {}) if isinstance(arms, Mapping) else {}
    cfd_seeds = sorted(cfd_arm.get("seeds", [])) if isinstance(cfd_arm, Mapping) else []
    control_seeds = sorted(control_arm.get("seeds", [])) if isinstance(control_arm, Mapping) else []
    if cfd_seeds and control_seeds and cfd_seeds != control_seeds:
        blockers.append("ablation arms must use identical matched seeds")
    if cfd_seeds and len(cfd_seeds) < 3:
        blockers.append("ablation arms require at least three matched seeds")

    changed_fields = metadata.get("changed_config_fields", [])
    if _is_present(changed_fields) and changed_fields != ["aero_loss_weight"]:
        blockers.append("changed_config_fields must be exactly ['aero_loss_weight']")

    report = _base_report("cfd_guided_training_ablation", required, blockers)
    report.update(
        {
            "matched_seeds": cfd_seeds if cfd_seeds == control_seeds else [],
            "checked_contract": "matched-ablation metadata",
        }
    )
    return report


def build_prior_method_comparison_report(metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Build a fail-closed scaffold for comparable prior-method evidence."""

    metadata = metadata or {}
    required = [
        "methods",
        "comparison.sample_set_id",
        "comparison.seeds",
        "comparison.metrics",
        "comparison.statistical_tests",
        "comparison.result_interpretation",
    ]
    blockers = [f"missing required evidence: {path}" for path in _missing_paths(metadata, required)]

    methods = metadata.get("methods", [])
    if not isinstance(methods, list) or len(methods) < 2:
        blockers.append("at least two method definitions are required")
        methods = []

    method_required = [
        "method_id",
        "citation",
        "implementation_source",
        "version_or_commit",
        "evaluation_protocol",
        "metric_mapping",
        "reproduction_status",
        "sample_set_id",
    ]
    method_sample_sets = set()
    for index, method in enumerate(methods):
        if not isinstance(method, Mapping):
            blockers.append(f"methods[{index}] must be an object")
            continue
        for key in method_required:
            if not _is_present(method.get(key)):
                blockers.append(f"methods[{index}] missing {key}")
        if _is_present(method.get("sample_set_id")):
            method_sample_sets.add(str(method["sample_set_id"]))

    comparison = metadata.get("comparison", {})
    comparison_sample_set = comparison.get("sample_set_id") if isinstance(comparison, Mapping) else None
    comparison_seeds = comparison.get("seeds", []) if isinstance(comparison, Mapping) else []
    if _is_present(comparison_seeds) and (not isinstance(comparison_seeds, list) or len(comparison_seeds) < 3):
        blockers.append("comparison.seeds must contain at least three seeds")
    if comparison_sample_set and method_sample_sets and method_sample_sets != {str(comparison_sample_set)}:
        blockers.append("all methods must use the comparison sample_set_id")

    report = _base_report("prior_method_comparison", required, blockers)
    report.update(
        {
            "method_count": len(methods),
            "checked_contract": "comparable prior-method metadata",
        }
    )
    return report


def build_publication_quality_validation_report(metadata: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Build a fail-closed scaffold for publication-quality validation metadata."""

    metadata = metadata or {}
    required = [
        "solver_settings",
        "convergence_study.resolution_ladder",
        "convergence_study.metrics",
        "sensitivity_study.parameters",
        "sensitivity_study.metrics",
        "external_validation.reference_cases",
        "external_validation.agreement_metrics",
        "residuals_or_forces",
    ]
    blockers = [f"missing required evidence: {path}" for path in _missing_paths(metadata, required)]

    ladder = _get_path(metadata, "convergence_study.resolution_ladder")
    if _is_present(ladder) and (not isinstance(ladder, list) or len(ladder) < 3):
        blockers.append("convergence_study.resolution_ladder must contain at least three resolutions")

    reference_cases = _get_path(metadata, "external_validation.reference_cases")
    if _is_present(reference_cases) and (not isinstance(reference_cases, list) or len(reference_cases) < 1):
        blockers.append("external_validation.reference_cases must contain at least one case")

    report = _base_report("publication_quality_validation", required, blockers)
    report.update(
        {
            "checked_contract": "validation-study metadata",
            "resolution_count": len(ladder) if isinstance(ladder, list) else 0,
        }
    )
    return report


REPORT_BUILDERS: Dict[str, Callable[[Mapping[str, Any] | None], Dict[str, Any]]] = {
    "cfd-guided-training-ablation": build_cfd_guided_training_ablation_report,
    "prior-method-comparison": build_prior_method_comparison_report,
    "publication-quality-validation": build_publication_quality_validation_report,
}


def _read_metadata(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("metadata JSON must contain an object")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write fail-closed scientific claim scaffold reports.")
    subparsers = parser.add_subparsers(dest="report_type", required=True)
    for report_type in REPORT_BUILDERS:
        subparser = subparsers.add_parser(report_type)
        subparser.add_argument("--metadata", default=None, help="Optional input metadata JSON.")
        subparser.add_argument("--output", required=True, help="Output report JSON path.")

    args = parser.parse_args(argv)
    metadata = _read_metadata(args.metadata)
    report = REPORT_BUILDERS[args.report_type](metadata)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
