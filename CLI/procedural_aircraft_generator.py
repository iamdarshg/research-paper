#!/usr/bin/env python3
# Procedurally generate diverse aircraft voxel geometries.
# Axis convention: g[z_vertical, y_spanwise, x_lengthwise]
from __future__ import annotations
import argparse, hashlib, json, os, sys, random, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.ndimage import label as scipy_label

GRID = 96
MIN_OCC = 0.0005
MIN_SYMMETRY = 0.10
MIN_SPAN = 0.03
MIN_LENGTH = 0.08
MIN_COMPONENT_FRAC = 0.5

AIRCRAFT_TYPES = [
    "glider", "delta_wing", "flying_wing", "biplane",
    "canard", "anhedral", "swept_wing", "lifting_body",
]


def _ellipsoid(g, cz, cy_center, x_center, rz, ry, rx):
    for zi in range(max(0,cz-rz),min(GRID,cz+rz+1)):
        for yi in range(max(0,cy_center-ry),min(GRID,cy_center+ry+1)):
            for xi in range(max(0,x_center-rx),min(GRID,x_center+rx+1)):
                nz=(zi-cz)/max(rz,1); ny=(yi-cy_center)/max(ry,1); nx=(xi-x_center)/max(rx,1)
                if nz*nz+ny*ny+nx*nx<=1.0: g[zi,yi,xi]=1.0


def _box(g, z0,z1, y0,y1, x0,x1):
    g[max(0,int(z0)):min(GRID,int(z1)+1), max(0,int(y0)):min(GRID,int(y1)+1), max(0,int(x0)):min(GRID,int(x1)+1)]=1.0


def _wing_symmetric(g, y_center, x_le_root, root_chord, half_span, thickness,
                    sweep_frac=0.0, dihedral_deg=0.0, taper=0.5):
    """Draw a symmetric tapered wing pair centered at y_center."""
    dih = math.radians(dihedral_deg)
    for side in [-1, 1]:
        for s in range(int(half_span)):
            t = s / max(half_span, 1)
            ch = max(2, int(round(root_chord * (1.0 - t * (1.0 - taper)))))
            sw = int(round(sweep_frac * s))
            zo = int(round(math.tan(dih) * s))
            yi = y_center + side * s
            if yi < 0 or yi >= GRID: continue
            xi_end = x_le_root - sw  # leading edge sweeps back
            xi_start = xi_end + ch   # trailing edge
            zc = GRID // 2 + zo
            th = max(1, int(round(thickness * (1.0 - 0.3*t))))
            for xi in range(max(0,xi_end), min(GRID,xi_start+1)):
                for dz in range(-th, th+1):
                    zi = zc + dz
                    if zi < 0 or zi >= GRID: continue
                    cf = abs(xi - (xi_end + xi_start)/2.0) / max(ch/2.0, 1)
                    if cf > 0.8 and dz != 0: continue
                    g[zi, yi, xi] = 1.0


def generate_aircraft(rng, aircraft_type):
    g = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    cz = GRID // 2
    margin = 4

    if aircraft_type == "glider":
        fl = rng.randint(35, 55)
        frz = rng.randint(2, 3); fry = rng.randint(2, 3)
        whs = rng.randint(28, 40); wch = rng.randint(5, 9)
        wt = rng.randint(1, 2); dih = rng.randint(2, 6)
        # Fuselage along x-axis (length)
        _ellipsoid(g, cz, cz, GRID//2, frz, fry, fl//2)
        # Main wing at mid-fuselage
        wy = cz
        wx = GRID//2 + rng.randint(-5, 5)
        _wing_symmetric(g, wy, wx+wch, wch, whs, wt,
                       sweep_frac=rng.uniform(0, 0.05), dihedral_deg=dih,
                       taper=rng.uniform(0.3, 0.5))
        # Horizontal tail
        ty = GRID//2 + fl//2 - 6
        _wing_symmetric(g, wy, ty, 2, whs//3, 1, sweep_frac=0.05, taper=0.6)
        # Vertical tail
        _box(g, cz, cz+rng.randint(4,7), ty-1, ty+1, GRID//2-2, GRID//2+2)

    elif aircraft_type == "delta_wing":
        span = rng.randint(32, 44); rc = rng.randint(22, 36)
        th = rng.randint(1, 2)
        nose_x = margin + rc
        for s in range(span):
            t = s / max(span, 1)
            ch = max(2, int(rc * (1.0-t) * 0.85))
            xs = max(margin, nose_x - ch)
            for side in [-1, 1]:
                yi = cz + side * s
                if not (0 <= yi < GRID): continue
                for xi in range(xs, min(nose_x, GRID-margin)):
                    g[cz, yi, xi] = 1.0
                    if th > 1 and cz+1 < GRID:
                        g[cz+1, yi, xi] = 1.0
        # Small vertical fins near trailing edge
        fin_y = cz - rc // 4
        _box(g, cz, cz+rng.randint(3,5), fin_y-1, fin_y+1, margin+2, margin+6)

    elif aircraft_type == "flying_wing":
        span = rng.randint(35, 45); rc = rng.randint(16, 26)
        th = rng.randint(2, 3); swf = rng.uniform(0.25, 0.45)
        for s in range(span):
            ch = max(3, int(rc * (1.0 - (s/max(span,1)) * 0.85)))
            sw = int(swf * s)
            for side in [-1, 1]:
                yi = cz + side * s
                if not (0 <= yi < GRID): continue
                xe = min(GRID-margin, cz + rc//2 - sw)
                xs = max(margin, xe - ch)
                if xs >= xe: continue
                g[cz, yi, xs:xe] = 1.0
                if th > 1 and cz+1 < GRID:
                    g[cz+1, yi, xs:xe] = 1.0
        _ellipsoid(g, cz, cz, cz, 2, rc//3, 3)

    elif aircraft_type == "biplane":
        fl = rng.randint(32, 48); fr = rng.randint(2, 3)
        whs = rng.randint(20, 30); wch = rng.randint(4, 7)
        gap = rng.randint(5, 9)
        _ellipsoid(g, cz, cz, GRID//2, fr, fr, fl//2)
        wx = GRID//2
        for dz_off in [-(gap//2), gap//2]:
            zw = cz + dz_off
            if 0 <= zw < GRID:
                # Draw both wings at this z-offset
                for side in [-1, 1]:
                    for s in range(whs):
                        yi = cz + side * s
                        if 0 <= yi < GRID:
                            g[zw, yi, wx-wch//2:wx+wch//2+1] = 1.0
        ty = GRID//2 + fl//2 - 5
        _box(g, cz-1, cz+gap//2+1, ty-1, ty+1, GRID//2-3, GRID//2+3)

    elif aircraft_type == "canard":
        fl = rng.randint(38, 52); fr = rng.randint(2, 3)
        mhs = rng.randint(22, 34); mch = rng.randint(5, 8)
        chs = rng.randint(7, 13); chc = rng.randint(2, 4)
        _ellipsoid(g, cz, cz, GRID//2, fr, fr, fl//2)
        # Main wing aft
        mwx = GRID//2 + fl//4
        _wing_symmetric(g, cz, mwx+mch, mch, mhs, 1,
                       sweep_frac=rng.uniform(0.05, 0.15),
                       dihedral_deg=rng.randint(-2, 4), taper=rng.uniform(0.3, 0.6))
        # Canard forward
        cwx = GRID//2 - fl//2 + 6
        _wing_symmetric(g, cz, cwx+chc, chc, chs, 1, sweep_frac=0.0, taper=0.8)
        ty = GRID//2 + fl//2 - 4
        _box(g, cz-1, cz+4, ty-1, ty+1, GRID//2-2, GRID//2+2)

    elif aircraft_type == "anhedral":
        fl = rng.randint(40, 55); fr = rng.randint(2, 4)
        whs = rng.randint(26, 38); wch = rng.randint(5, 8)
        ad = -rng.randint(3, 8)
        _ellipsoid(g, cz, cz, GRID//2, fr, fr, fl//2)
        wx = GRID//2
        _wing_symmetric(g, cz, wx+wch, wch, whs, 1,
                       sweep_frac=rng.uniform(0.0, 0.08), dihedral_deg=ad,
                       taper=rng.uniform(0.4, 0.6))
        twy = GRID//2 + fl//2 - 5
        _wing_symmetric(g, cz, twy, 2, whs//3, 1, sweep_frac=0.12, taper=0.5)
        _box(g, cz, cz+rng.randint(4, 7), twy-1, twy+1, GRID//2-2, GRID//2+2)

    elif aircraft_type == "swept_wing":
        fl = rng.randint(42, 58); fr = rng.randint(2, 4)
        whs = rng.randint(24, 36); wch = rng.randint(5, 9)
        sw = rng.uniform(0.15, 0.35)
        _ellipsoid(g, cz, cz, GRID//2, fr, fr, fl//2)
        wx = GRID//2 + rng.randint(-4, 4)
        _wing_symmetric(g, cz, wx+wch, wch, whs, 1,
                       sweep_frac=sw, dihedral_deg=rng.randint(0, 5),
                       taper=rng.uniform(0.2, 0.4))
        twy = GRID//2 + fl//2 - 5
        _wing_symmetric(g, cz, twy, 2, whs//3, 1, sweep_frac=0.2, taper=0.5)
        _box(g, cz, cz+rng.randint(4, 6), twy-1, twy+1, GRID//2-2, GRID//2+2)

    elif aircraft_type == "lifting_body":
        bl = rng.randint(42, 58); brx = rng.randint(6, 10)
        brz = rng.randint(4, 6)
        _ellipsoid(g, cz, cz, GRID//2, brz, brx, bl//2)
        shs = rng.randint(7, 15)
        swx = GRID//2 + bl//3
        _wing_symmetric(g, cz, swx, 3, shs, 1, sweep_frac=0.15, taper=0.5)
        ty = GRID//2 + bl//2 - 3
        _box(g, cz, cz+brz+2, ty-1, ty+1, GRID//2-2, GRID//2+2)

    return (g > 0.5).astype(np.float32)


def chash(v):
    return hashlib.sha256(((v>0.5).astype(np.uint8)).tobytes()).hexdigest()


def validate(v):
    try:
        occ = v > 0.5
        if float(occ.sum()) / v.size < MIN_OCC: return False
        lab, n = scipy_label(occ)
        if n == 0: return False
        sz = np.bincount(lab.ravel())[1:]
        if sz.max() / max(sz.sum(), 1) < MIN_COMPONENT_FRAC: return False
        coords = np.argwhere(occ)
        dims = (coords.max(axis=0) - coords.min(axis=0) + 1).astype(float)
        if dims[1] / GRID < MIN_SPAN or dims[2] / GRID < MIN_LENGTH: return False
        mid = GRID // 2
        left = occ[:, :mid, :]; right = occ[:, GRID-mid:, :]
        ml = min(left.shape[1], right.shape[1])
        asym = np.abs(left[:,:ml,:].astype(int) - right[:,-ml:,:].astype(int)).sum() / max(occ.sum(), 1)
        if max(0.0, 1.0 - asym) < MIN_SYMMETRY: return False
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Generate procedural aircraft")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    (od / "voxels").mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    seen = set(); out = []
    type_counts = {t: 0 for t in AIRCRAFT_TYPES}
    stats = {"target": args.count, "generated": 0, "accepted": 0,
             "rejected_invalid": 0, "rejected_duplicate": 0, "per_type": type_counts}
    attempts = 0
    while stats["accepted"] < args.count and attempts < args.count * 10:
        attempts += 1
        atype = rng.choice(AIRCRAFT_TYPES)
        vox = generate_aircraft(rng, atype)
        stats["generated"] += 1
        c = chash(vox)
        if c in seen: stats["rejected_duplicate"] += 1; continue
        if not validate(vox): stats["rejected_invalid"] += 1; continue
        vid = f"proc:{atype}:{stats['accepted']}"
        fn = vid.replace(":", "_") + ".npy"
        np.save(str(od / "voxels" / fn), vox.astype(np.uint8))
        out.append({
            "source_id": vid, "source_type": "procedural",
            "aircraft_type": atype, "canonical_content_sha256": c,
            "voxel_sha256": c, "geometry_path": "voxels/" + fn,
            "conditioning_mode": "unconditioned_source_metadata_only",
            "split": "train",
            "provenance": {
                "generator_seed": args.seed, "attempt": attempts,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "description": f"Parametrically generated {atype}",
                "claim_boundary": "Procedurally generated design, NOT real CAD."
            }
        })
        seen.add(c); stats["accepted"] += 1; type_counts[atype] += 1
    stats["created_at"] = datetime.now(timezone.utc).isoformat()
    mp = od / "manifest.jsonl"; tmp = mp.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(tmp, mp)
    rp = od / "report.json"
    rp.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


