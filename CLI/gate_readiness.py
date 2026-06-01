#!/usr/bin/env python3
"""Report scientific gate implementation readiness separately from evidence.

This module is intentionally not a scientific validation shortcut. It answers:
"does the repository have documented, testable, machine-readable gate
infrastructure?" Claim-bearing evidence remains blocked until real grounded
reports exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


GATES: List[Dict[str, Any]] = [
    {
        "id": "manifest_validation",
        "name": "Manifest validation",
        "documentation_artifacts": ["docs/dataset/README.md", "docs/dataset/manifest_schema.example.json"],
        "machine_readable_artifacts": ["CLI/validate_manifest.py"],
        "tests_or_verification": ["tests/test_manifest_contract.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "minimal manifest intentionally lacks claim-bearing provenance",
    },
    {
        "id": "aircraft_validity",
        "name": "Aircraft validity",
        "documentation_artifacts": ["docs/benchmarks/aircraft_validity_suite.md"],
        "machine_readable_artifacts": ["CLI/aircraft_validity.py"],
        "tests_or_verification": ["tests/test_aircraft_validity.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no generated claim-eval voxel set",
    },
    {
        "id": "grounded_condition_response",
        "name": "Grounded condition response",
        "documentation_artifacts": ["docs/benchmarks/condition_response_benchmark.md"],
        "machine_readable_artifacts": ["CLI/run_condition_benchmark.py"],
        "tests_or_verification": ["tests/test_condition_benchmark.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no grounded response metrics/corpus",
    },
    {
        "id": "manufacturing_structural_feasibility",
        "name": "Manufacturing and structural condition feasibility",
        "documentation_artifacts": ["docs/benchmarks/manufacturing_constraints.md"],
        "machine_readable_artifacts": ["CLI/condition_feasibility.py"],
        "tests_or_verification": ["tests/test_manufacturing_constraints.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no geometry-aware structural/load-path report",
    },
    {
        "id": "baseline_statistics",
        "name": "Baseline statistics",
        "documentation_artifacts": ["docs/benchmarks/baseline_policy.md"],
        "machine_readable_artifacts": ["CLI/multi_seed_eval.py"],
        "tests_or_verification": ["tests/test_baseline_policy.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "required baselines and final metric tables missing",
    },
    {
        "id": "final_evidence_package",
        "name": "Final evidence package",
        "documentation_artifacts": ["paper/FINAL_EVIDENCE_PACKAGE.md"],
        "machine_readable_artifacts": ["CLI/final_evidence.py"],
        "tests_or_verification": ["tests/test_final_evidence_package.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "required report bundle missing",
    },
    {
        "id": "generates_aircraft_structures",
        "name": "Generates aircraft structures",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/aircraft_validity_suite.md"],
        "machine_readable_artifacts": ["CLI/aircraft_validity.py", "CLI/validate_manifest.py"],
        "tests_or_verification": ["tests/test_aircraft_validity.py", "tests/test_manifest_contract.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no claim-bearing generated sample package",
    },
    {
        "id": "aerodynamically_optimized",
        "name": "Aerodynamically optimized",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/scientific_gate_sources.md"],
        "machine_readable_artifacts": ["CLI/aircraft_diffusion_cfd.py", "CLI/advanced_lbm_solver.py", "CLI/run_protocol.py"],
        "tests_or_verification": ["tests/test_cfd_solver_contract.py", "tests/test_solver.py", "tests/test_protocol_runner.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no converged CFD comparison against baselines",
    },
    {
        "id": "structurally_viable",
        "name": "Structurally viable",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/manufacturing_constraints.md"],
        "machine_readable_artifacts": ["CLI/condition_feasibility.py"],
        "tests_or_verification": ["tests/test_manufacturing_constraints.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no structural analysis or load-case evidence",
    },
    {
        "id": "cfd_guided_training",
        "name": "CFD-guided training",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/scientific_gate_sources.md"],
        "machine_readable_artifacts": ["CLI/aircraft_diffusion_cfd.py", "CLI/advanced_lbm_solver.py", "CLI/scientific_claim_scaffolds.py"],
        "tests_or_verification": ["tests/test_cfd_solver_contract.py", "tests/test_solver.py", "tests/test_scientific_claim_scaffolds.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no matched ablation with and without CFD term",
    },
    {
        "id": "outperforms_prior_approaches",
        "name": "Outperforms prior approaches",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/baseline_policy.md"],
        "machine_readable_artifacts": ["CLI/multi_seed_eval.py", "CLI/scientific_claim_scaffolds.py", "CLI/baseline_config.yaml"],
        "tests_or_verification": ["tests/test_baseline_policy.py", "tests/test_scientific_claim_scaffolds.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no prior-method or superiority comparison package",
    },
    {
        "id": "publication_quality_validation",
        "name": "Publication-quality validation",
        "documentation_artifacts": ["paper/FINAL_RUN_GATES.md", "docs/benchmarks/scientific_gate_sources.md"],
        "machine_readable_artifacts": ["CLI/final_evidence.py", "CLI/run_protocol.py", "CLI/scientific_claim_scaffolds.py", "CLI/advanced_lbm_solver.py"],
        "tests_or_verification": ["tests/test_final_evidence_package.py", "tests/test_protocol_runner.py", "tests/test_scientific_claim_scaffolds.py", "tests/test_solver.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no convergence/sensitivity/external-validation study",
    },
    {
        "id": "conditioned_flight_profile_manufacturing",
        "name": "Conditioned on flight profile and manufacturing method",
        "documentation_artifacts": ["CLI/conditioning_schema.yaml", "docs/benchmarks/condition_response_benchmark.md"],
        "machine_readable_artifacts": ["CLI/run_condition_benchmark.py", "CLI/aircraft_diffusion_cfd.py"],
        "tests_or_verification": ["tests/test_conditioning.py", "tests/test_condition_benchmark.py"],
        "claim_bearing_evidence_status": "blocked",
        "claim_blocker": "no grounded generated-output response evidence",
    },
]


def _with_implementation_status(gate: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(gate)
    implementation_complete = all(
        enriched.get(key)
        for key in ("documentation_artifacts", "machine_readable_artifacts", "tests_or_verification")
    )
    enriched["implementation_status"] = "complete" if implementation_complete else "incomplete"
    return enriched


def build_gate_readiness_report() -> Dict[str, Any]:
    gates = [_with_implementation_status(gate) for gate in GATES]
    completed = [gate for gate in gates if gate["implementation_status"] == "complete"]
    evidence_passed = [
        gate
        for gate in gates
        if gate["claim_bearing_evidence_status"] == "pass"
    ]
    gate_count = len(gates)
    completed_ratio = len(completed) / max(gate_count, 1)
    return {
        "status": "pass" if completed_ratio >= 0.90 else "blocked",
        "gate_count": gate_count,
        "implementation_readiness": {
            "status": "pass" if completed_ratio >= 0.90 else "blocked",
            "completed_count": len(completed),
            "completed_ratio": completed_ratio,
            "threshold": 0.90,
            "meaning": (
                "Documentation mapping, machine-readable scaffolding, and tests "
                "exist for the gate. This is not claim-bearing science evidence."
            ),
        },
        "claim_bearing_evidence": {
            "status": "pass" if len(evidence_passed) == gate_count else "blocked",
            "passed_count": len(evidence_passed),
            "passed_ratio": len(evidence_passed) / max(gate_count, 1),
            "meaning": (
                "A gate only passes here when publication-grade evidence reports "
                "exist. The current repo intentionally keeps these blocked."
            ),
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report scientific gate implementation readiness.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = build_gate_readiness_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
