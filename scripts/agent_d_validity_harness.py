"""Geometry validity pass/fail harness (Agent D, Workstream C item 2).

Runs CLI/aircraft_validity.py evaluate_aircraft_validity on:
  a) the frozen NASA CRM whole-aircraft voxel artifacts (validation clone), and
  b) the bundled root STLs voxelized into a 96^3 lattice.
Reports pass/fail counts per the documented criteria set.
"""
import glob
import os
import sys

import numpy as np
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_validity import evaluate_aircraft_validity  # noqa: E402

VALIDATION_CLONE_VOXELS = (
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\repos\research-paper-validation"
    r"\docs\dataset\nasa_crm_whole_aircraft\voxels"
)
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def check(name, array, canonicalize=True):
    r = evaluate_aircraft_validity(
        torch.as_tensor(array, dtype=torch.float32), canonicalize=canonicalize
    )
    return r


def main():
    print("=" * 72)
    print("A. Frozen NASA CRM voxel artifacts (grounded corpus, 32^3)")
    print("=" * 72)
    files = sorted(glob.glob(os.path.join(VALIDATION_CLONE_VOXELS, "*.npy")))
    counts = {"pass": 0, "fail": 0}
    failed = []
    for f in files:
        arr = np.load(f)
        r = check(os.path.basename(f), arr)
        status = r["status"]
        counts[status] += 1
        if status != "pass":
            failed.append((os.path.basename(f), r["failed_checks"]))
        else:
            print(f"  PASS {os.path.basename(f)}  occ={r['metrics']['occupancy_ratio']:.5f}")
    print(f"  -> {counts['pass']} pass / {counts['fail']} fail / {len(files)} total")
    for name, checks in failed:
        print(f"  FAIL {name}: {checks}")

    print()
    print("=" * 72)
    print("B. Bundled root STLs voxelized into 96^3 lattices")
    print("=" * 72)
    import trimesh

    stls = ["biplane.stl", "F-18_Hornet.stl", "test_cylinder.stl", "20mm_cube.stl"]
    stl_counts = {"pass": 0, "fail": 0}
    for name in stls:
        path = os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            print(f"  SKIP {name} (missing)")
            continue
        mesh = trimesh.load(path)
        pitch = max(mesh.extents) / 95.0
        matrix = mesh.voxelized(pitch).matrix.astype(np.float32)
        padded = np.zeros((96, 96, 96), dtype=np.float32)
        z0 = (96 - matrix.shape[0]) // 2
        y0 = (96 - matrix.shape[1]) // 2
        x0 = (96 - matrix.shape[2]) // 2
        padded[z0:z0 + matrix.shape[0],
               y0:y0 + matrix.shape[1],
               x0:x0 + matrix.shape[2]] = matrix
        r = check(name, padded)
        stl_counts[r["status"]] += 1
        print(f"  {r['status'].upper()} {name}: occ={r['metrics']['occupancy_ratio']:.5f} "
              f"failed={r['failed_checks']}")
    print(f"  -> {stl_counts['pass']} pass / {stl_counts['fail']} fail")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    total_pass = counts["pass"] + stl_counts["pass"]
    total = len(files) + sum(stl_counts.values())
    print(f"  total {total_pass} pass / {total - total_pass} fail / {total} artifacts")
    print("  NOTE: passing the heuristic validity screen is NOT structural,")
    print("        aerodynamic, or manufacturing validation (see FINAL_RUN_GATES).")


if __name__ == "__main__":
    main()