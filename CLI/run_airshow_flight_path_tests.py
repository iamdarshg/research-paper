#!/usr/bin/env python3
"""Run three generated flight-path smoke checks from an Airshow-trained checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from aircraft_diffusion_cfd import (
    AdvancedCFDSimulator,
    CFDConfig,
    DesignSpec,
    OptimizedAircraftGenerator,
)
from aircraft_validity import evaluate_aircraft_validity
from sequential_diagnostic_optimizer import (
    SequentialDiagnosticOptimizationConfig,
    SequentialDiagnosticOptimizer,
)


CASES = [
    {
        "case_id": "short_takeoff_payload",
        "seed": 31415,
        "spec": {
            "target_speed": 9.0,
            "thrust_to_weight_min": 0.65,
            "turn_rate_min_deg_s": 16.0,
            "required_static_thrust_n": 260.0,
            "payload_mass_min_g": 900,
            "payload_mass_max_g": 2200,
            "takeoff_distance_min_m": 45,
            "takeoff_distance_max_m": 120,
            "wingspan_limit_m": 1.7,
            "manufacturing_method": "fdm_pla_0p6mm",
            "engine_diameter_mm": 160,
            "engine_length_mm": 320,
            "engine_count_min": 1,
            "engine_count_max": 2,
            "wall_thickness_min_mm": 1,
            "wall_thickness_max_mm": 3,
            "part_count_min": 1,
            "part_count_max": 8,
        },
    },
    {
        "case_id": "high_speed_sprint",
        "seed": 27182,
        "spec": {
            "target_speed": 26.0,
            "thrust_to_weight_min": 0.85,
            "turn_rate_min_deg_s": 14.0,
            "required_static_thrust_n": 320.0,
            "payload_mass_min_g": 250,
            "payload_mass_max_g": 900,
            "takeoff_distance_min_m": 80,
            "takeoff_distance_max_m": 180,
            "wingspan_limit_m": 1.35,
            "manufacturing_method": "composite_wet_layup",
            "engine_diameter_mm": 180,
            "engine_length_mm": 360,
            "engine_count_min": 1,
            "engine_count_max": 2,
            "wall_thickness_min_mm": 1,
            "wall_thickness_max_mm": 2,
            "part_count_min": 1,
            "part_count_max": 6,
        },
    },
    {
        "case_id": "endurance_turning",
        "seed": 16180,
        "spec": {
            "target_speed": 12.0,
            "thrust_to_weight_min": 0.5,
            "turn_rate_min_deg_s": 30.0,
            "required_static_thrust_n": 170.0,
            "payload_mass_min_g": 350,
            "payload_mass_max_g": 1200,
            "takeoff_distance_min_m": 55,
            "takeoff_distance_max_m": 160,
            "wingspan_limit_m": 2.2,
            "manufacturing_method": "sheet_balsa_tabbed",
            "engine_diameter_mm": 120,
            "engine_length_mm": 240,
            "engine_count_min": 1,
            "engine_count_max": 2,
            "wall_thickness_min_mm": 1,
            "wall_thickness_max_mm": 2,
            "part_count_min": 2,
            "part_count_max": 10,
        },
    },
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_artifacts_dir(output_dir: Path) -> Path:
    path = output_dir / "generated"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().numpy())
    return value


def _prepare_voxel_for_solver(voxel_grid: torch.Tensor, device: torch.device) -> torch.Tensor:
    if voxel_grid.ndim == 4:
        voxel_grid = voxel_grid.max(dim=0).values
    if voxel_grid.ndim != 3:
        raise ValueError(f"Expected a 3D voxel grid after channel reduction, got {tuple(voxel_grid.shape)}")
    return voxel_grid.to(device=device, dtype=torch.float32)


def _binarize_voxel(
    voxel_grid: torch.Tensor,
    threshold: float,
    target_occupancy: float | None = None,
) -> np.ndarray:
    grid = voxel_grid.detach().cpu().float()
    if target_occupancy is None:
        return (grid.numpy() > threshold).astype(np.float32)

    target = float(np.clip(target_occupancy, 0.0, 1.0))
    flat = grid.reshape(-1)
    if flat.numel() == 0 or target <= 0.0:
        return np.zeros(tuple(grid.shape), dtype=np.float32)
    keep = min(flat.numel(), max(1, int(round(flat.numel() * target))))
    selected = torch.topk(flat, keep, largest=True).indices
    binary = torch.zeros_like(flat, dtype=torch.float32)
    binary[selected] = 1.0
    return binary.reshape_as(grid).numpy().astype(np.float32)


def _blend_lateral_symmetry(voxel_grid: torch.Tensor, blend: float) -> torch.Tensor:
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0:
        return voxel_grid
    mirrored = torch.flip(voxel_grid, dims=[1])
    symmetric = 0.5 * (voxel_grid + mirrored)
    return ((1.0 - blend) * voxel_grid + blend * symmetric).clamp(0.0, 1.0)


def _build_objective_optimizer(
    args: argparse.Namespace,
    cfd: AdvancedCFDSimulator,
    device: torch.device,
    seed: int,
) -> SequentialDiagnosticOptimizer | None:
    if args.objective_optimizer == "none":
        return None
    config = SequentialDiagnosticOptimizationConfig(
        method=args.objective_optimizer,
        population_size=args.objective_population_size,
        generations=args.objective_generations,
        elite_count=args.objective_elite_count,
        mutation_rate=args.objective_mutation_rate,
        mutation_sigma=args.objective_mutation_sigma,
        symmetry_blend=args.objective_symmetry_blend,
        spsa_steps=args.objective_spsa_steps,
        spsa_perturbation=args.objective_spsa_perturbation,
        spsa_learning_rate=args.objective_spsa_learning_rate,
        connectivity_weight=args.objective_connectivity_weight,
        aerodynamic_weight=args.objective_aerodynamic_weight,
        validity_weight=args.objective_validity_weight,
        occupancy_weight=args.objective_occupancy_weight,
        target_occupancy=args.objective_target_occupancy,
        binarization_target_occupancy=(
            args.objective_binarization_target_occupancy
            if args.objective_binarization_target_occupancy is not None
            else args.export_target_occupancy
        ),
        enable_aerodynamic=not args.no_objective_cfd,
        cfd_steps=args.cfd_steps,
        seed=seed + args.objective_seed_offset,
    )
    return SequentialDiagnosticOptimizer(cfd, config=config, device=device)


def run_cases(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir = _case_artifacts_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    generator = OptimizedAircraftGenerator(str(checkpoint_path), device=device)
    cfd = AdvancedCFDSimulator(
        CFDConfig(solver_type="D3Q27", base_grid_resolution=args.grid_size),
        device,
    )

    results: List[Dict[str, Any]] = []
    for case in CASES:
        seed = int(case["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        spec = DesignSpec(**case["spec"])
        with torch.no_grad():
            generated = generator.generate(spec, num_steps=args.num_steps)
        voxel = _prepare_voxel_for_solver(generated, device)
        voxel = _blend_lateral_symmetry(voxel, args.export_symmetry_blend)
        objective_optimizer = _build_objective_optimizer(args, cfd, device, seed)
        optimization_report = None
        pre_optimization_binary = _binarize_voxel(
            voxel,
            threshold=args.export_threshold,
            target_occupancy=args.export_target_occupancy,
        )

        if objective_optimizer is not None:
            optimized = objective_optimizer.optimize(voxel, spec)
            voxel = optimized["voxel_grid"].to(device=device, dtype=torch.float32)
            optimization_report = {
                key: value
                for key, value in optimized.items()
                if key not in {"voxel_grid", "binary_grid"}
            }

        binary = _binarize_voxel(
            voxel,
            threshold=args.export_threshold,
            target_occupancy=args.export_target_occupancy,
        )
        solver_voxel = torch.as_tensor(binary, device=device, dtype=torch.float32)

        npy_path = generated_dir / f"{case['case_id']}.npy"
        stl_path = generated_dir / f"{case['case_id']}.stl"
        pre_npy_path = generated_dir / f"{case['case_id']}_pre_objective_optimization.npy"
        if optimization_report is not None:
            np.save(pre_npy_path, pre_optimization_binary)
        np.save(npy_path, binary)
        generator.voxels_to_stl(solver_voxel.detach().cpu(), str(stl_path), use_marching_cubes=True)

        cfd_metrics = cfd.simulate_aerodynamics(solver_voxel, steps=args.cfd_steps)
        validity = evaluate_aircraft_validity(binary)
        occupied = int(binary.sum())
        results.append(
            {
                "case_id": case["case_id"],
                "seed": seed,
                "design_spec": asdict(spec),
                "artifact_paths": {
                    "voxels_npy": str(npy_path),
                    "stl": str(stl_path),
                    "pre_objective_optimization_voxels_npy": (
                        str(pre_npy_path) if optimization_report is not None else None
                    ),
                },
                "artifact_hashes": {
                    "voxels_npy_sha256": _sha256_file(npy_path),
                    "stl_sha256": _sha256_file(stl_path),
                    "pre_objective_optimization_voxels_npy_sha256": (
                        _sha256_file(pre_npy_path) if optimization_report is not None else None
                    ),
                },
                "geometry_summary": {
                    "shape": list(binary.shape),
                    "occupied_voxels": occupied,
                    "occupancy_ratio": float(occupied / max(1, binary.size)),
                },
                "sequential_objective_optimization": _to_jsonable(optimization_report),
                "cfd_metrics": _to_jsonable(cfd_metrics),
                "validity": _to_jsonable(validity),
            }
        )

    report = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "manifest_path": str(manifest_path),
        "manifest_hash": _sha256_file(manifest_path),
        "grid_size": args.grid_size,
        "num_steps": args.num_steps,
        "cfd_steps": args.cfd_steps,
        "objective_optimizer": args.objective_optimizer,
        "objective_binarization_target_occupancy": args.objective_binarization_target_occupancy,
        "binarization": {
            "export_threshold": args.export_threshold,
            "export_target_occupancy": args.export_target_occupancy,
            "method": "top_k_target_occupancy" if args.export_target_occupancy is not None else "fixed_threshold",
            "export_symmetry_blend": args.export_symmetry_blend,
        },
        "case_count": len(results),
        "cases": results,
        "claim_boundary": (
            "Generated flight-path smoke checks from a public Airshow-corpus checkpoint. "
            "CFD metrics are internal D3Q27 implementation outputs and are not validated "
            "aircraft-performance claims. When enabled, sequential objective optimization "
            "uses measured connectivity, validity, and CFD scores as black-box candidate "
            "selection losses; it is not gradient backpropagation through the solver."
        ),
    }
    output_path = output_dir / "flight_path_results.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="build/airshow_training_20260620/flight_path_tests")
    parser.add_argument("--run-id", default="airshow-grounded-flight-path-20260620")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--cfd-steps", type=int, default=100)
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument(
        "--objective-optimizer",
        choices=["none", "genetic", "spsa"],
        default="genetic",
        help="Sequential black-box optimizer for measured connectivity/aero/validity losses.",
    )
    parser.add_argument("--objective-population-size", type=int, default=4)
    parser.add_argument("--objective-generations", type=int, default=2)
    parser.add_argument("--objective-elite-count", type=int, default=1)
    parser.add_argument("--objective-mutation-rate", type=float, default=0.08)
    parser.add_argument("--objective-mutation-sigma", type=float, default=0.20)
    parser.add_argument("--objective-symmetry-blend", type=float, default=0.25)
    parser.add_argument("--objective-spsa-steps", type=int, default=4)
    parser.add_argument("--objective-spsa-perturbation", type=float, default=0.18)
    parser.add_argument("--objective-spsa-learning-rate", type=float, default=0.04)
    parser.add_argument("--objective-connectivity-weight", type=float, default=50.0)
    parser.add_argument("--objective-aerodynamic-weight", type=float, default=1.0)
    parser.add_argument("--objective-validity-weight", type=float, default=10.0)
    parser.add_argument("--objective-occupancy-weight", type=float, default=0.0)
    parser.add_argument("--objective-target-occupancy", type=float, default=0.03)
    parser.add_argument(
        "--objective-binarization-target-occupancy",
        type=float,
        default=None,
        help=(
            "If set, score sequential optimizer candidates with top-k binarization "
            "at this occupancy. Defaults to --export-target-occupancy when provided."
        ),
    )
    parser.add_argument("--objective-seed-offset", type=int, default=1000)
    parser.add_argument("--no-objective-cfd", action="store_true", help="Disable CFD calls inside the sequential objective optimizer.")
    parser.add_argument("--export-threshold", type=float, default=0.5, help="Fixed probability threshold for binary export when no target occupancy is set.")
    parser.add_argument("--export-target-occupancy", type=float, default=None, help="If set, export the top-k probability voxels at this target occupancy instead of a fixed threshold.")
    parser.add_argument(
        "--export-symmetry-blend",
        type=float,
        default=0.0,
        help="Blend generated probabilities with their lateral mirror before objective optimization and export.",
    )
    args = parser.parse_args()
    report = run_cases(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
