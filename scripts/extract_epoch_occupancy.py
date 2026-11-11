import ast
import re
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", errors="replace")
for m in re.finditer(r"Epoch (\d+) Metrics: (\{.*?\})\r?\n", text, re.S):
    try:
        d = ast.literal_eval(m.group(2))
    except Exception as exc:
        print(f"epoch {m.group(1)}: parse error {exc}")
        continue
    print(
        f"epoch {m.group(1):>2} occ_loss={d['occupancy_mean_loss']:.4f} "
        f"pred_clean={d['predicted_clean_occupancy']:.5f} "
        f"pred_gen={d['predicted_generation_occupancy']:.5f} "
        f"target={d['target_occupancy']:.5f} "
        f"clean_geom={d['clean_geometry_reconstruction']:.4f} "
        f"gen_geom={d['generation_reconstruction']:.4f}"
    )