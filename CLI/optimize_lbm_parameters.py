import os
import sys
import json
import argparse
import itertools
import numpy as np
import torch
from pathlib import Path

# Add CLI to path to import solver and benchmark utilities
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'CLI'))
sys.path.insert(0, str(REPO_ROOT))

from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig
from advanced_lbm_solver import D3Q27CascadedSolver
import run_internal_benchmark as benchmark

def calculate_error(internal_results, openfoam_results):
    """Calculate error between internal and OpenFOAM drag coefficients."""
    cd_internal = internal_results['drag_coefficient']
    cd_of = openfoam_results['cd_total']
    if cd_of == 0: return float('inf')
    return abs(cd_internal - cd_of) / abs(cd_of) * 100.0

def run_optimization_sweep(args):
    """Run a parameter sweep to find optimal relaxation rates."""
    # Discover STLs to evaluate
    stl_dir = Path(args.stl_dir).resolve()
    stls = benchmark.discover_root_stls(stl_dir)
    if not stls:
        print("No STL files found for evaluation.")
        return

    # Evaluation cases
    grid_sizes = [16, 32]
    speeds = [60.0, 80.0]

    # Parameters to optimize (D3Q27CascadedSolver uses s_e and s_h, and s_ghost/s_e in D3Q27Solver)
    # Note: D3Q27CascadedSolver uses self._solver (D3Q27Solver)
    s_e_values = [1.0, 1.1, 1.2, 1.3]
    s_h_values = [1.2, 1.4, 1.6, 1.8]

    best_params = None
    min_avg_error = float('inf')
    results_log = []

    for se, sh in itertools.product(s_e_values, s_h_values):
        print(f"Testing parameters: s_e={se}, s_h={sh}")
        total_error = 0.0
        case_count = 0

        for stl_path in stls:
            mesh = benchmark._load_trimesh(stl_path)
            for res, speed in itertools.product(grid_sizes, speeds):
                print(f"  Evaluating {stl_path.name} at {res}^3, speed={speed}...")

                sweep_case = {
                    'grid_resolution': res,
                    'domain_scale': 2.0,
                    'freestream_speed': speed,
                    'reynolds_number': 1e5,
                    'steps': 200
                }

                # Setup internal solver with candidate parameters
                domain_min, domain_max, domain_size, max_extent = benchmark.compute_geometry_frame(mesh, sweep_case['domain_scale'])
                geometry_mask = benchmark.mesh_to_geometry_mask(mesh, res, domain_min, domain_size)

                cfg = CFDConfig(
                    base_grid_resolution=res,
                    mach_number=speed / 343.0,
                    reynolds_number=1e5,
                    simulation_steps=200,
                )

                ref_length = max_extent
                cfg.lbm_config.physical_length_scale = ref_length
                cfg.lbm_config.grid_spacing = domain_size / res

                solver = D3Q27CascadedSolver(cfg, torch.device('cpu'), LBMPhysicsConfig)
                # Apply parameters to the underlying D3Q27Solver
                solver._solver.s_e = se
                solver._solver.s_ghost = sh

                solver.collide_stream(geometry_mask, steps=200, use_inlet_outlet=True)
                internal_res = solver.compute_aerodynamic_coefficients(geometry_mask)

                # Run or get OpenFOAM results (using benchmark utility)
                # In a real run, this would invoke OpenFOAM
                # For this script, we'll try to run it once and cache if possible, or just run it.
                of_case_res = benchmark.run_benchmark_case(stl_path, mesh, sweep_case, args)

                if of_case_res.get('openfoam', {}).get('status') == 'completed':
                    of_res = of_case_res['openfoam']['force']
                    error = calculate_error(internal_res, of_res)
                    total_error += error
                    case_count += 1
                else:
                    print(f"    OpenFOAM failed for {stl_path.name}")

        if case_count > 0:
            avg_error = total_error / case_count
            print(f"Average Error for s_e={se}, s_h={sh}: {avg_error:.4f}%")
            results_log.append({'s_e': se, 's_h': sh, 'avg_error': avg_error})

            if avg_error < min_avg_error:
                min_avg_error = avg_error
                best_params = (se, sh)
        else:
            print("No valid cases completed.")

    print("\nOptimization Results:")
    print(f"Best parameters: s_e={best_params[0]}, s_h={best_params[1]}")
    print(f"Minimum average error: {min_avg_error:.4f}%")

    with open('lbm_optimization_results.json', 'w') as f:
        json.dump(results_log, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize LBM parameters to match OpenFOAM.')
    parser.add_argument('--stl-dir', default='.', help='Directory with STL files')
    parser.add_argument('--install-openfoam', action='store_true', help='Install OpenFOAM if missing')
    parser.add_argument('--openfoam-package', default='openfoam', help='OpenFOAM package name')
    parser.add_argument('--openfoam-timeout', type=int, default=1200, help='Timeout for OpenFOAM')

    args = parser.parse_args()
    run_optimization_sweep(args)
