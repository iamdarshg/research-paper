"""Recompute the frozen low-Mach corpus gate tables on a CONSISTENT reference area.

Publication blocker fixed 2026-11-01: the frozen comparison corpus normalized
OpenFOAM C_d/C_l on a fixed sweep reference area (0.27734375) while the internal
LBM solver normalized on its own voxel-projected frontal area, which is
resolution-dependent (0.2850196 @ g32, 0.2385490 @ g64, 0.3965490 @ g16,
0.3084270 @ g24).  Dividing the raw coefficients directly contaminated
``cd_error_percent`` by up to +/-16% and produced one fake "0.25%" passing row.

This script re-derives every frozen row's error on a single convention:
C_corrected_LBM = C_raw_LBM * Aref_LBM / Aref_sweep  (a force comparison).
OpenFOAM is not installed on this host, so the frozen OpenFOAM coefficients are
kept as-is; only the comparison convention is corrected.  Outputs the corrected
CSV tables and the corrected gate summary (rows under the repo's 5% gate).
"""
import csv
import json
import math
import os
from pathlib import Path

VALIDATION_DOCS = Path(
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\repos\research-paper-validation\docs\benchmarks"
)
OUTPUT_DIR = Path(
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\evidence\research-paper-publish"
).parent / "research-paper-publish" / "corrected_corpus"
SWEEP_REFERENCE_AREA = 0.27734375

# Per-grid LBM voxel-projected frontal areas recorded in the frozen artifacts:
LBM_REF_AREA = {
    32: 0.2850196089378276,
    64: 0.23854902052405133,
    16: 0.3965490211308905,
    24: 0.3084270164351371,
}

GATE = 5.0


def corrected_cd(lbm_cd, lbm_grid):
    factor = LBM_REF_AREA[int(lbm_grid)] / SWEEP_REFERENCE_AREA
    if lbm_cd is None or not math.isfinite(float(lbm_cd)):
        return None
    return float(lbm_cd) * factor


def pct_error(corrected, reference):
    if corrected is None or reference is None:
        return None
    if not math.isfinite(float(reference)) or abs(float(reference)) < 1e-12:
        return None
    return abs(float(corrected) - float(reference)) / abs(float(reference)) * 100.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("A. sonicFoam sweep (low_mach_openfoam_lbm_sweep_20260613, 16 rows)")
    print("=" * 78)
    src = VALIDATION_DOCS / "low_mach_openfoam_lbm_sweep_20260613" / "comparison_rows.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    corrected_rows = []
    old_under = 0
    new_under = 0
    print(f"{'mach':>5} {'OFg':>3} {'LBMg':>4} {'OF Cd':>9} {'LBM Cd raw':>11} "
          f"{'LBM Cd corr':>11} {'old err%':>9} {'new err%':>9}")
    for r in rows:
        lbm_grid = int(r["lbm_grid"])
        of_cd = float(r["openfoam_cd"])
        of_cl = float(r["openfoam_cl"])
        lbm_cd_raw = float(r["lbm_cd"])
        lbm_cl_raw = float(r["lbm_cl"])
        lbm_cd_new = corrected_cd(lbm_cd_raw, lbm_grid)
        lbm_cl_new = corrected_cd(lbm_cl_raw, lbm_grid)
        old_cd_err = pct_error(lbm_cd_raw, of_cd)
        new_cd_err = pct_error(lbm_cd_new, of_cd)
        new_cl_err = pct_error(lbm_cl_new, of_cl)
        if old_cd_err is not None and old_cd_err < GATE:
            old_under += 1
        if new_cd_err is not None and new_cd_err < GATE:
            new_under += 1
        row = dict(r)
        row["lbm_reference_area"] = LBM_REF_AREA[lbm_grid]
        row["openfoam_reference_area"] = SWEEP_REFERENCE_AREA
        row["reference_area_convention"] = "lbm_renormalized_to_openfoam_reference_area"
        row["lbm_cd_corrected"] = f"{lbm_cd_new:.12g}" if lbm_cd_new is not None else ""
        row["lbm_cl_corrected"] = f"{lbm_cl_new:.12g}" if lbm_cl_new is not None else ""
        row["cd_error_percent_corrected"] = f"{new_cd_err:.12g}" if new_cd_err is not None else ""
        row["cl_error_percent_corrected"] = f"{new_cl_err:.12g}" if new_cl_err is not None else ""
        corrected_rows.append(row)
        print(f"{r['mach']:>5} {r['openfoam_grid']:>3} {lbm_grid:>4} {of_cd:9.4f} "
              f"{lbm_cd_raw:11.5f} {lbm_cd_new:11.5f} "
              f"{old_cd_err:9.3f} {new_cd_err:9.3f}")
    print(f"\nsonicFoam rows < 5% gate: OLD {old_under}/16  ->  CORRECTED {new_under}/16")

    out_csv = OUTPUT_DIR / "sonicfoam_comparison_rows_corrected.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(corrected_rows[0].keys()))
        writer.writeheader()
        writer.writerows(corrected_rows)
    print(f"wrote {out_csv}")

    print()
    print("=" * 78)
    print("B. simpleFoam suite (simplefoam_low_mach_suite_20260613, 14 rows)")
    print("=" * 78)
    src2 = VALIDATION_DOCS / "simplefoam_low_mach_suite_20260613" / "suite_rows.json"
    srows = json.loads(src2.read_text(encoding="utf-8"))
    old_under = 0
    new_under = 0
    print(f"{'mach':>5} {'g':>3} {'OF Cd':>9} {'LBM Cd raw':>11} {'LBM Cd corr':>11} "
          f"{'old err%':>9} {'new err%':>9}")
    corrected_srows = []
    for r in srows:
        grid = int(r["grid"])
        of_cd = r.get("openfoam_cd")
        of_cl = r.get("openfoam_cl")
        lbm_cd_raw = r.get("lbm_cd")
        lbm_cl_raw = r.get("lbm_cl")
        lbm_cd_new = corrected_cd(lbm_cd_raw, grid) if lbm_cd_raw is not None else None
        lbm_cl_new = corrected_cd(lbm_cl_raw, grid) if lbm_cl_raw is not None else None
        old_cd_err = pct_error(lbm_cd_raw, of_cd)
        new_cd_err = pct_error(lbm_cd_new, of_cd)
        new_cl_err = pct_error(lbm_cl_new, of_cl)
        if old_cd_err is not None and old_cd_err < GATE:
            old_under += 1
        if new_cd_err is not None and new_cd_err < GATE:
            new_under += 1
        r2 = dict(r)
        r2["lbm_reference_area"] = LBM_REF_AREA[grid]
        r2["openfoam_reference_area"] = SWEEP_REFERENCE_AREA
        r2["reference_area_convention"] = "lbm_renormalized_to_openfoam_reference_area"
        r2["lbm_cd_corrected"] = f"{lbm_cd_new:.12g}" if lbm_cd_new is not None else ""
        r2["lbm_cl_corrected"] = f"{lbm_cl_new:.12g}" if lbm_cl_new is not None else ""
        r2["cd_error_percent_corrected"] = f"{new_cd_err:.12g}" if new_cd_err is not None else ""
        r2["cl_error_percent_corrected"] = f"{new_cl_err:.12g}" if new_cl_err is not None else ""
        corrected_srows.append(r2)
        print(f"{r['mach']:>5} {grid:>3} {of_cd if of_cd is None else round(of_cd,4)!s:>9} "
              f"{lbm_cd_raw if lbm_cd_raw is None else round(lbm_cd_raw,5)!s:>11} "
              f"{lbm_cd_new if lbm_cd_new is None else round(lbm_cd_new,5)!s:>11} "
              f"{old_cd_err if old_cd_err is None else round(old_cd_err,3)!s:>9} "
              f"{new_cd_err if new_cd_err is None else round(new_cd_err,3)!s:>9}")
    print(f"\nsimpleFoam rows < 5% gate: OLD {old_under}/14  ->  CORRECTED {new_under}/14")

    out_json = OUTPUT_DIR / "simplefoam_suite_rows_corrected.json"
    out_json.write_text(json.dumps(corrected_srows, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()