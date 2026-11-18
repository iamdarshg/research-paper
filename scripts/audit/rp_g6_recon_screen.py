#!/usr/bin/env python3
"""RP-G6: held-out reconstructions. For every holdout record, decode the TRUE latent with
the arm's converter (sigmoid, binarize 0.5 with the solver binarizer), run the repo's frozen
old 12-check screen (diversity_novelty_gate.sample_report -> old_gate_pass), and IoU vs GT.
Control: the same screen on the GT voxels (ceiling). Gate: >=90% pass and mean IoU >=0.40.
Run from the research-paper code dir. usage: --build <repo>/build --arms C4 ... --output out.json"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "CLI"); sys.path.insert(0, "scripts")
from aircraft_diffusion_cfd import AircraftDesignDataset, OptimizedAircraftGenerator, _binarize_probability_grid_for_solver  # noqa
from diversity_novelty_gate import sample_report, iou, GATE_SPEC_HASH  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True, type=Path)
ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--output", required=True, type=Path)
a = ap.parse_args(); R = a.build
ds = AircraftDesignDataset(num_samples=0, grid_size=32, latent_dim=16, manifest_path=str(R / "div_corpus_660_raw/holdout.jsonl"))
N = len(ds); gt = [ds[i]["geometry"].numpy() > 0.5 for i in range(N)]
def summary(rep_list, ious=None):
    from collections import Counter
    passed = [r["old_gate_pass"] for r in rep_list]
    fails = Counter(c for r in rep_list for c in r["old_gate_failed_checks"])
    o = {"n": len(passed), "screen_pass": int(sum(passed)), "screen_pass_rate": round(float(np.mean(passed)), 4),
         "failed_check_counts": dict(fails.most_common())}
    if ious is not None:
        o.update(mean_iou_vs_gt=round(float(np.mean(ious)), 4), median_iou=round(float(np.median(ious)), 4),
                 min_iou=round(float(np.min(ious)), 4), frac_iou_ge_0p40=round(float(np.mean(np.array(ious) >= 0.40)), 4))
    return o
out = {"gate_spec_hash": GATE_SPEC_HASH, "holdout_n": N, "gate": ">=0.90 screen pass and mean IoU >= 0.40",
       "gt_control": summary([sample_report(g) for g in gt]), "arms": {}}
print("GT control", json.dumps(out["gt_control"]))
for arm in a.arms:
    g = OptimizedAircraftGenerator(str(R / f"rp_g1_{arm}/ckpt/final_optimized_model.pt"), device=torch.device("cpu"))
    reps, ious = [], []
    with torch.no_grad():
        for i in range(N):
            p = torch.sigmoid(g.converter(ds[i]["latent"].reshape(1, -1).float()))[0]
            v = _binarize_probability_grid_for_solver(p, threshold=0.5).numpy() > 0.5
            reps.append(sample_report(v)); ious.append(iou(v, gt[i]))
    s = summary(reps, ious)
    s["rp_g6_pass"] = bool(s["screen_pass_rate"] >= 0.90 and s["mean_iou_vs_gt"] >= 0.40)
    out["arms"][arm] = s; print(arm, json.dumps(s), flush=True)
a.output.write_text(json.dumps(out, indent=2) + "\n")
