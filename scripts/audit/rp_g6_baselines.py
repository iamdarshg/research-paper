"""RP-G6 baselines: constant majority-vote training shape (one array for every record), and
Wilson 95% CIs on the arms' screen pass rates. Run from research-paper code dir."""
import json, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0, "CLI"); sys.path.insert(0, "scripts")
from aircraft_diffusion_cfd import AircraftDesignDataset  # noqa
from diversity_novelty_gate import sample_report, iou  # noqa
R = Path(sys.argv[1]); res = json.load(open(sys.argv[2]))
ho = AircraftDesignDataset(num_samples=0, grid_size=32, latent_dim=16, manifest_path=str(R / "div_corpus_660_raw/holdout.jsonl"))
tr = AircraftDesignDataset(num_samples=0, grid_size=32, latent_dim=16, manifest_path=str(R / "div_corpus_660_raw/train.jsonl"))
gt = [ho[i]["geometry"].numpy() > 0.5 for i in range(len(ho))]
freq = np.mean([tr[i]["geometry"].numpy() > 0.5 for i in range(len(tr))], axis=0)
out = {"train_n": len(tr)}
for thr in (0.5, 0.3, 0.2):
    c = freq >= thr; rep = sample_report(c)
    out[f"constant_majority_thr{thr}"] = {"occupied_voxels": int(c.sum()), "screen_pass": rep["old_gate_pass"],
        "failed_checks": rep["old_gate_failed_checks"], "mean_iou_vs_gt": round(float(np.mean([iou(c, g) for g in gt])), 4)}
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]
out["screen_pass_wilson95"] = {k: {"k": v["screen_pass"], "n": v["n"], "ci": wilson(v["screen_pass"], v["n"])} for k, v in res["arms"].items()}
print(json.dumps(out, indent=1)); Path(sys.argv[3]).write_text(json.dumps(out, indent=2) + "\n")
