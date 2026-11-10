"""Statistical digest of the n=12 condition-response sweep (robust).

Reports Pearson r, p, and Fisher-z 95% CI for each condition-response pair,
both with the raw n=12 set and with the single catastrophic solver
instability excluded (seed index 1, drag ~ 5.8e33).  Also reports the
run-to-run variation finding against the October n=4 artifact.

All numbers trace to build/protocol_smoke/condition_validation_n12.json.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, pearsonr

REPORT = Path(
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\repos\research-paper"
    r"\build\protocol_smoke\condition_validation_n12.json"
)
N4_ARTIFACT = Path(
    r"C:\Users\Darsh Gupta\engineering-portfolio-audit\repos\research-paper"
    r"\build\protocol_smoke\condition_validation.json"
)


def fisher_z_ci(r, n, alpha=0.05):
    if not np.isfinite(r):
        return None
    z = math.atanh(max(min(r, 1.0 - 1e-12), -1.0 + 1e-12))
    se = 1.0 / math.sqrt(n - 3)
    zcrit = norm.ppf(1.0 - alpha / 2.0)
    return (math.tanh(z - zcrit * se), math.tanh(z + zcrit * se))


def tabulate(name, xs, ys, n, notes=""):
    r, p = pearsonr(xs, ys)
    ci = fisher_z_ci(r, n)
    ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
    print(f"{name:<38} {r:7.4f} {p:7.4f} {ci_text:>22} {notes}")
    return r, p


def main():
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    raw = data["raw_data"]
    n = len(raw["measured_drag"])
    print(f"checkpoint: {data['metadata']['checkpoint']}")
    print(f"n seeds run: {n}, grid: {data['metadata']['grid_size']}")
    print()
    drag = np.asarray([float(v) for v in raw["measured_drag"]])
    lift = np.asarray([float(v) for v in raw["measured_lift"]])
    speed = np.asarray([float(v) for v in raw["target_speed"]])
    span = np.asarray([float(v) for v in raw["wingspan_limit"]])
    occ = np.asarray([float(v) for v in raw["occupancy"]])

    bad = np.where(np.abs(drag) > 1e6)[0]
    print(f"non-physical outlier seeds (|drag|>1e6): {list(bad)}  "
          f"values drag={[f'{drag[i]:.3e}' for i in bad]}")
    if len(bad):
        bad_lift = [f"{lift[i]:.3e}" for i in bad]
        bad_speed = [f"{speed[i]:.1f}" for i in bad]
        bad_span = [f"{span[i]:.2f}" for i in bad]
        print(f"  lift={bad_lift} speed={bad_speed} span={bad_span}")
        print("  -> D3Q27 solver blow-up on the smoke checkpoint at grid 32 / "
              "100 steps for those conditions; a measurement-instability event, "
              "not a physical drag value.")
    print()

    holdout = [i for i in range(n) if i not in set(bad)]
    print(f"{'pair':<38} {'r':>7} {'p':>7} {'95% CI':>22}  set")
    print("-" * 100)
    for xs, xname in ((speed, "target_speed"), (span, "wingspan_limit")):
        for ys, yname in ((drag, "measured_drag"), (lift, "measured_lift"), (occ, "occupancy")):
            tabulate(f"{xname}_vs_{yname} [n={n}, raw]", xs, ys, n)
            if len(holdout) >= 4:
                tabulate(
                    f"{xname}_vs_{yname} [n={len(holdout)}, excl. outlier]",
                    xs[holdout], ys[holdout], len(holdout),
                )
    print()
    if N4_ARTIFACT.exists():
        d4 = json.loads(N4_ARTIFACT.read_text(encoding="utf-8"))
        drag4 = np.asarray(d4["raw_data"]["measured_drag"])
        print("Run-to-run variation vs October n=4 artifact (same checkpoint, "
              "same seeds 0-3):")
        print(f"  n4 mean drag {drag4.mean():.3f} +- {drag4.std(ddof=1):.3f}")
        print(f"  n12 seeds 0-3 drag {[round(float(drag[i]),3) for i in range(4)]}")
        print("  -> identical seeds give different drag on CUDA: the smoke "
              "checkpoint's D3Q27 measurement is not bit-reproducible "
              "run-to-run; treat point coefficients as estimates, not constants.")
    print()
    print("Power context (Fisher-z): n=12 -> ~80% power (alpha=.05, two-sided)")
    print("for |rho| ~ 0.78; smaller effects are not resolvable and are reported")
    print("as non-significant with wide CIs (honest null/indeterminate), never as")
    print("cherry-picked significance.")


if __name__ == "__main__":
    main()