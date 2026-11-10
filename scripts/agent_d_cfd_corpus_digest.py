"""Quantitative digest of the frozen low-Mach OpenFOAM-vs-LBM corpus
(Agent D, Workstream C item 4, CFD verification)."""
import csv
import json
import os

VALIDATION_DOCS = (
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\repos\research-paper-validation"
    r"\docs\benchmarks"
)


def main():
    print("=" * 78)
    print("A. sonicFoam vs internal LBM sweep (low_mach_openfoam_lbm_sweep_20260613)")
    print("=" * 78)
    rows = list(csv.DictReader(open(os.path.join(
        VALIDATION_DOCS,
        "low_mach_openfoam_lbm_sweep_20260613", "comparison_rows.csv"))))
    print(f"rows: {len(rows)}")
    for of_grid in ("32", "64"):
        subs = [r for r in rows if r["openfoam_grid"] == of_grid]
        print(f"--- OpenFOAM grid {of_grid} ---")
        for r in sorted(subs, key=lambda x: (int(x["lbm_grid"]), float(x["mach"]))):
            print(
                f"  OFg{r['openfoam_grid']} LBMg{r['lbm_grid']} M{r['mach']}: "
                f"OF_Cd={float(r['openfoam_cd']):.4f} LBM_Cd={float(r['lbm_cd']):.4f} "
                f"CdErr={float(r['cd_error_percent']):8.2f}% "
                f"ClErr={float(r['cl_error_percent']):9.2f}%"
            )
    cd_errors = [float(r["cd_error_percent"]) for r in rows]
    cl_errors = [float(r["cl_error_percent"]) for r in rows]
    print(f"\nCd error: min {min(cd_errors):.2f}%  max {max(cd_errors):.2f}%  "
          f"mean {sum(cd_errors)/len(cd_errors):.2f}%  "
          f"rows <5%: {sum(1 for e in cd_errors if e < 5)}/{len(cd_errors)}")
    print(f"Cl error: min {min(cl_errors):.2f}%  max {max(cl_errors):.2f}%  "
          f"rows <5%: {sum(1 for e in cl_errors if e < 5)}/{len(cl_errors)}")

    print()
    print("=" * 78)
    print("B. simpleFoam suite (simplefoam_low_mach_suite_20260613)")
    print("=" * 78)
    with open(os.path.join(VALIDATION_DOCS,
                           "simplefoam_low_mach_suite_20260613",
                           "suite_rows.json")) as f:
        srows = json.load(f)
    if isinstance(srows, list):
        errs = [float(s["cd_error_percent"]) for s in srows
                if s.get("cd_error_percent") is not None]
        cl_errs = [float(s.get("cl_error_percent"))
                   for s in srows if s.get("cl_error_percent") is not None]
        print(f"rows: {len(srows)}; Cd errors: "
              f"{[round(e, 2) for e in errs]}")
        print(f"Cd <5%: {sum(1 for e in errs if e < 5)}/{len(errs)}; "
              f"Cl <5%: {sum(1 for e in cl_errs if e < 5)}/{len(cl_errs)}")
    else:
        print("suite_rows.json is not a list:", list(srows.keys())[:10])

    print()
    print("=" * 78)
    print("C. Reference-area normalization audit")
    print("=" * 78)
    summary = json.load(open(os.path.join(
        VALIDATION_DOCS, "low_mach_openfoam_lbm_sweep_20260613",
        "grid_speed_study_summary.json")))
    sweep_ref_area = summary["reference_area"]
    print(f"sweep-level reference area (OpenFOAM normalization): "
          f"{sweep_ref_area}")
    for case in summary["lbm_cases"]:
        coeff = case.get("coefficients", {})
        print(f"  LBM grid {case['grid']:>2} M{case['mach']:<5}: "
              f"internal reference_area = {coeff.get('reference_area'):.9f} "
              f"(source {coeff.get('reference_area_source')})")
    print("\nNOTE: the LBM Cd is normalized by the internal voxel-projected "
          "area (resolution-dependent), the OpenFOAM Cd by the fixed sweep "
          "reference area; 'cd_error_percent' therefore includes a "
          "reference-area mismatch that varies with grid resolution.")


if __name__ == "__main__":
    main()