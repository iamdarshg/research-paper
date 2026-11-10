#!/usr/bin/env python3
# Procedurally generate diverse aircraft voxel geometries.
# Axis convention: g[z_vertical, y_spanwise, x_lengthwise]
from __future__ import annotations
import argparse, hashlib, json, os, sys, random, math
from pathlib import Path
from typing import Iterator
import numpy as np
from scipy.ndimage import label as scipy_label

GRID = 96
# Occupancy band for a normalized procedural sample, expressed as a fraction of
# the full lattice. The declared aircraft screen in CLI/aircraft_validity.py
# requires 0.002 <= occupancy <= 0.02 at 96^3; this band sits strictly inside it
# with headroom for the perturb/cleanup passes downstream.
TARGET_OCCUPANCY_MIN = 0.0026
TARGET_OCCUPANCY_MAX = 0.0170
# Largest of (span, length) as a fraction of the lattice edge after
# normalization. The screen requires span and length >= 0.35 and <= 0.90.
TARGET_FILL = 0.72
TARGET_LENGTH = 0.78
MIN_OCC = 0.0005
MIN_SYMMETRY = 0.10
MIN_SPAN = 0.03
MIN_LENGTH = 0.08
MIN_COMPONENT_FRAC = 0.5

AIRCRAFT_TYPES = [
    "glider", "delta_wing", "flying_wing", "biplane",
    "canard", "anhedral", "swept_wing", "lifting_body",
]
PROCEDURAL_MIN_ACCEPTED_PER_TYPE = {aircraft_type: 1 for aircraft_type in AIRCRAFT_TYPES}
PROCEDURAL_GENERATOR_VERSION = "procedural-aircraft-v2"


def _ellipsoid(g, cz, cy_center, x_center, rz, ry, rx):
    for zi in range(max(0,cz-rz),min(GRID,cz+rz+1)):
        for yi in range(max(0,cy_center-ry),min(GRID,cy_center+ry+1)):
            for xi in range(max(0,x_center-rx),min(GRID,x_center+rx+1)):
                nz=(zi-cz)/max(rz,1); ny=(yi-cy_center)/max(ry,1); nx=(xi-x_center)/max(rx,1)
                if nz*nz+ny*ny+nx*nx<=1.0: g[zi,yi,xi]=1.0


def _box(g, z0,z1, y0,y1, x0,x1):
    g[max(0,int(z0)):min(GRID,int(z1)+1), max(0,int(y0)):min(GRID,int(y1)+1), max(0,int(x0)):min(GRID,int(x1)+1)]=1.0


def _ensure_center_body(g, radius=2, x_start_frac=0.12, x_end_frac=0.88):
    """Give the aircraft a continuous longitudinal body with blunt ends.

    Every station inside ``[x_start_frac, x_end_frac]`` of the lattice receives a
    body section of at least ``radius`` voxels, centred on that station's own
    occupied centroid (or on the previous station's body when the station is
    empty). Tailless planforms would otherwise expose no centre body and no
    fuselage ends to the declared screen, and thin ellipsoid fuselages taper to
    one voxel at the nose and tail. A real aircraft carries a continuous centre
    body between its nose and its tail, so the body is made explicit instead of
    being inferred from whichever primitive happened to be drawn.
    """
    occupied = g > 0.5
    columns = np.nonzero(occupied.any(axis=(0, 1)))[0]
    if columns.size == 0:
        return g
    minimum_side = 2 * int(radius) + 1
    x_first = int(round(GRID * x_start_frac))
    x_last = int(round(GRID * x_end_frac)) - 1
    body_center: np.ndarray | None = None
    for xi in range(x_first, min(GRID, x_last + 1)):
        section = np.argwhere(occupied[:, :, xi])
        if section.size:
            z_extent = int(section[:, 0].max() - section[:, 0].min() + 1)
            y_extent = int(section[:, 1].max() - section[:, 1].min() + 1)
            body_center = np.array(
                [
                    int(round(float(section[:, 0].mean()))),
                    int(round(float(section[:, 1].mean()))),
                ]
            )
            if min(z_extent, y_extent) >= minimum_side:
                # Station already carries a full-size body section.
                continue
        if body_center is None:
            continue
        zc, yc = int(body_center[0]), int(body_center[1])
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dz * dz + dy * dy > radius * radius:
                    continue
                zi, yi = zc + dz, yc + dy
                if 0 <= zi < GRID and 0 <= yi < GRID:
                    g[zi, yi, xi] = 1.0
    return g


def _resample(binary, factor):
    """Nearest-neighbour resample of a binary lattice by ``factor`` (any > 0)."""
    if abs(factor - 1.0) < 1e-6:
        return binary
    shape = binary.shape
    out_shape = tuple(max(1, int(round(size * factor))) for size in shape)
    indices = [
        np.clip((np.arange(size) / factor).astype(np.int64), 0, old - 1)
        for size, old in zip(out_shape, shape)
    ]
    return binary[np.ix_(indices[0], indices[1], indices[2])]


def _resample_axis(binary, axis, factor):
    """Resample a single axis by ``factor``, leaving the other axes unchanged."""
    if abs(factor - 1.0) < 1e-6:
        return binary
    shape = binary.shape
    old = shape[axis]
    new = max(1, int(round(old * factor)))
    axis_indices = np.clip((np.arange(new) / factor).astype(np.int64), 0, old - 1)
    if axis == 0:
        return binary[axis_indices, :, :]
    if axis == 1:
        return binary[:, axis_indices, :]
    return binary[:, :, axis_indices]


def _canonicalize_and_normalize(g):
    """Bilateral symmetry, then scale into the declared occupancy/fill band.

    The declared screen compares the two spanwise halves about the lattice
    mid-plane, and it bounds span, length, thickness and total occupancy. Real
    aircraft are bilaterally symmetric and occupy a bounded fraction of their
    bounding box, so a procedural aircraft is mirrored about the same mid-plane
    and scaled to a declared fill target rather than being emitted at whatever
    scale the random primitives happened to produce.
    """
    binary = g > 0.5
    if not binary.any():
        return g
    # Mirror about the lattice mid-plane (the plane the screen measures).
    binary = binary | binary[:, ::-1, :]

    coords = np.argwhere(binary)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    crop = binary[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]]

    length = max(int(crop.shape[2]), 1)
    factor = (TARGET_LENGTH * GRID) / float(length)
    chosen = _resample(crop, factor)

    # Per-axis cap so the materialized sample stays inside the screen's
    # normalization margins (span/length <= 0.90, thickness <= 0.40).
    for axis, cap in ((0, 0.38), (1, 0.86), (2, 0.86)):
        extent = max(int(chosen.shape[axis]), 1)
        limit = cap * GRID
        if extent > limit:
            ratio = limit / float(extent)
            if axis == 0:
                chosen = _resample_axis(chosen, 0, ratio)
            elif axis == 1:
                chosen = _resample_axis(chosen, 1, ratio)
            else:
                chosen = _resample_axis(chosen, 2, ratio)
    if max(chosen.shape) > GRID:
        factor *= 0.95
        chosen = _resample(crop, factor)
    if float(chosen.sum()) / float(GRID ** 3) > TARGET_OCCUPANCY_MAX:
        factor *= 0.90
        chosen = _resample(crop, factor)

    # Grow thin shapes with a face-connected dilation until the corpus band is met.
    from scipy.ndimage import binary_dilation
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[0, 1, 1] = structure[2, 1, 1] = True
    structure[1, 0, 1] = structure[1, 2, 1] = True
    structure[1, 1, 0] = structure[1, 1, 2] = True
    for _ in range(4):
        occupancy = float(chosen.sum()) / float(GRID ** 3)
        if occupancy >= TARGET_OCCUPANCY_MIN:
            break
        candidate = binary_dilation(chosen, structure=structure)
        if max(candidate.shape) > GRID:
            break
        if float(candidate.sum()) / float(GRID ** 3) > TARGET_OCCUPANCY_MAX:
            break
        chosen = candidate

    canvas = np.zeros((GRID, GRID, GRID), dtype=np.float32)
    if max(chosen.shape) > GRID:
        chosen = chosen[:GRID, :GRID, :GRID]
    starts = [(GRID - size) // 2 for size in chosen.shape]
    canvas[
        starts[0]:starts[0] + chosen.shape[0],
        starts[1]:starts[1] + chosen.shape[1],
        starts[2]:starts[2] + chosen.shape[2],
    ] = chosen.astype(np.float32)
    # Re-impose exact bilateral symmetry after resampling and centring, which
    # can each shift features by one voxel relative to the mid-plane.
    canvas = np.maximum(canvas, canvas[:, ::-1, :])
    return canvas


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
        bl = rng.randint(42, 58); brx = rng.randint(4, 6)
        brz = rng.randint(4, 6)
        _ellipsoid(g, cz, cz, GRID//2, brz, brx, bl//2)
        shs = rng.randint(24, 34)
        swx = GRID//2 + bl//3
        _wing_symmetric(g, cz, swx, 11, shs, 1, sweep_frac=0.15, taper=0.5)
        tail_x = GRID//2 + bl//2 - 3
        _box(g, cz, cz+brz+2, cz-1, cz+1, tail_x-1, tail_x+1)
        # The legacy corpus validator compares the two spanwise halves about
        # the plane between voxels 47 and 48.  The primitives above are drawn
        # about voxel 48, so mirror their union one cell toward the lower half
        # to preserve the intended bilateral geometry on that established
        # validation plane.  The aircraft is comfortably clear of y=0/95, so
        # this shift cannot clip or wrap occupied voxels.
        shifted = np.zeros_like(g)
        shifted[:, :-1, :] = g[:, 1:, :]
        g = np.maximum(g, shifted)

    g = _ensure_center_body(g)
    g = _canonicalize_and_normalize(g)
    return (g > 0.5).astype(np.float32)


def canonicalize_voxels(v: np.ndarray) -> np.ndarray:
    """Return a detached binary uint8 array for deterministic persistence."""
    return np.ascontiguousarray((np.asarray(v) > 0.5).astype(np.uint8, copy=False))


def canonical_content_hash(v: np.ndarray) -> str:
    return hashlib.sha256(canonicalize_voxels(v).tobytes()).hexdigest()


def chash(v):
    return canonical_content_hash(v)


def validate(v):
    """Screen a procedural sample with the repository's declared aircraft checks.

    This module used to carry its own laxer gate (occupancy >= 0.0005, span >=
    0.03, symmetry >= 0.10). That gate disagreed with the paper's declared
    screen (`CLI/aircraft_validity.py`), so the "accepted" procedural corpus was
    mostly invalid under the very screen the generator is later measured
    against. There is now one definition of aircraft-like validity in the
    repository, and the procedural generator uses it.
    """
    try:
        from aircraft_validity import evaluate_aircraft_validity

        report = evaluate_aircraft_validity(np.asarray(v, dtype=np.float32), canonicalize=False)
        return bool(report.get("status") == "pass")
    except Exception:
        return False


def iter_procedural_samples(
    count: int,
    seed: int,
    *,
    seen_hashes: set[str] | None = None,
    stats: dict | None = None,
) -> Iterator[dict]:
    """Yield accepted procedural samples without writing filesystem artifacts."""
    if count < 0:
        raise ValueError("count must be non-negative")
    known_hashes = seen_hashes if seen_hashes is not None else set()
    state = stats if stats is not None else {}
    state.clear()
    state.update(
        {
            "target": count,
            "attempts": 0,
            "generated": 0,
            "accepted": 0,
            "rejected_invalid": 0,
            "rejected_duplicate": 0,
            "per_type": {aircraft_type: 0 for aircraft_type in AIRCRAFT_TYPES},
        }
    )
    rng = random.Random(seed)
    while state["accepted"] < count and state["attempts"] < count * 10:
        state["attempts"] += 1
        aircraft_type = rng.choice(AIRCRAFT_TYPES)
        voxels = generate_aircraft(rng, aircraft_type)
        state["generated"] += 1
        canonical = canonicalize_voxels(voxels)
        content_hash = canonical_content_hash(canonical)
        if content_hash in known_hashes:
            state["rejected_duplicate"] += 1
            continue
        if not validate(canonical):
            state["rejected_invalid"] += 1
            continue

        accepted_index = state["accepted"]
        known_hashes.add(content_hash)
        state["accepted"] += 1
        state["per_type"][aircraft_type] += 1
        yield {
            "aircraft_type": aircraft_type,
            "accepted_index": accepted_index,
            "attempt": state["attempts"],
            "seed": seed,
            "canonical_content_sha256": content_hash,
            "voxels": canonical,
        }


def generate_procedural_samples(
    count: int,
    seed: int,
    *,
    seen_hashes: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Materialize the deterministic stream for small standalone callers."""
    stats: dict = {}
    samples = list(iter_procedural_samples(count, seed, seen_hashes=seen_hashes, stats=stats))
    return samples, stats


def _task2_builder_module():
    try:
        from . import rebuild_final_training_corpus as builder
    except ImportError:
        import rebuild_final_training_corpus as builder
    return builder


def main():
    ap = argparse.ArgumentParser(description="Generate procedural aircraft")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    builder = _task2_builder_module()
    samples, stats = generate_procedural_samples(args.count, args.seed)
    if stats["accepted"] != args.count:
        raise ValueError(f"procedural generator accepted {stats['accepted']} of requested {args.count}")
    builder._assert_procedural_family_minimums(stats, args.count)
    od = builder._safe_output_target(Path(args.output_dir))
    od.mkdir()
    builder._assert_no_reparse_components(od, role="standalone procedural output")
    (od / "voxels").mkdir(exist_ok=True)
    out = []
    for sample in samples:
        atype = sample["aircraft_type"]
        vox = sample["voxels"]
        c = sample["canonical_content_sha256"]
        vid = f"proc:{atype}:{sample['accepted_index']}"
        fn = c + ".npy"
        temporary = od / "voxels" / ("." + fn + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vox, allow_pickle=False)
        os.replace(temporary, od / "voxels" / fn)
        out.append(builder.build_procedural_record(
            aircraft_type=atype,
            accepted_index=sample["accepted_index"],
            attempt=sample["attempt"],
            seed=sample["seed"],
            child_hash=c,
            voxel_file_hash=builder._file_sha256(od / "voxels" / fn),
            geometry_path="voxels/" + fn,
        ))
    stats["generator_version"] = PROCEDURAL_GENERATOR_VERSION
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


