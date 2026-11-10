#!/usr/bin/env python3
"""Build a diverse mission-conditioned 32^3 aircraft voxel corpus.

Fix A of the diversity/novelty pass (2026-11-06): the first recovery trained on
six near-identical CRM records, whose structured latents were pairwise 0.0027
apart.  This builder uses the repository's own mission-conditioned procedural
geometry (`aircraft_diffusion_cfd._procedural_aircraft_geometry`), in which the
geometry responds to the design spec (span, fuselage length/thickness, engine
pods, vertical tail), so a corpus of screen-valid distinct aircraft with
per-record condition vectors is available on this 8 GiB host.

Every record passes the declared screen (old 12-check heuristic) at 32^3, is
distinct at the voxel level, and carries a NORMALIZED condition vector so the
diffusion model sees all 22 slots on comparable scales (Fix C; previously raw
slots spanned 9 -> 1.6e5 in magnitude).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CLI")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for directory in (CLI_DIR, SCRIPTS_DIR):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from aircraft_diffusion_cfd import (  # noqa: E402
    _normalize_manifest_design_spec,
    _procedural_aircraft_geometry,
    build_condition_vector,
    normalize_condition_vector_tensor,
    sample_design_spec,
)
from aircraft_validity import evaluate_aircraft_validity  # noqa: E402
from diversity_novelty_gate import structural_report  # noqa: E402

GRID = 32
HOLDOUT_SEED_OFFSET = 1_000_000


def _voxel_sha(geometry: torch.Tensor) -> str:
    binary = (np.asarray(geometry.detach().cpu().numpy()) > 0.5).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(binary).tobytes()).hexdigest()


def _candidate(i: int, base_seed: int, torch_rng) -> tuple:
    seed = base_seed + i
    rng = random.Random(seed)
    spec = sample_design_spec(rng)
    torch.manual_seed(seed)
    if torch_rng is not None:
        torch_rng.manual_seed(seed)
    try:
        geometry = _procedural_aircraft_geometry(spec, GRID, generator=torch_rng)
    except Exception as error:  # pragma: no cover - defensive
        print(f"  generator raised for seed {seed}: {error}")
        return None, None, None
    return spec, geometry, seed


def build_corpus(
    train_count: int,
    holdout_count: int,
    seed: int,
    output_dir: Path,
    max_attempts_per_split: int = 6000,
    require_structural: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_dir = output_dir / "voxels"
    voxel_dir.mkdir(exist_ok=True)
    torch_rng = torch.Generator()

    seen = set()
    records = []
    stats = {"train_requested": train_count, "holdout_requested": holdout_count,
             "accepted": 0, "attempts": 0, "rejected_invalid": 0, "rejected_duplicate": 0,
             "require_structural": require_structural}
    for split, count, base_seed in (("train", train_count, seed),
                                    ("holdout", holdout_count, seed + HOLDOUT_SEED_OFFSET)):
        accepted = 0
        attempts = 0
        while accepted < count and attempts < max_attempts_per_split:
            attempts += 1
            stats["attempts"] += 1

            spec, geometry, candidate_seed = _candidate(accepted + attempts * 7, base_seed, torch_rng)
            if geometry is None:
                continue
            content_hash = _voxel_sha(geometry)
            if content_hash in seen:
                stats["rejected_duplicate"] += 1
                continue
            report = evaluate_aircraft_validity(geometry.numpy(), canonicalize=True)
            if report["status"] != "pass":
                stats["rejected_invalid"] += 1
                continue
            if require_structural:
                structural = structural_report(geometry.numpy())
                if not structural["pass"]:
                    stats["rejected_invalid"] += 1
                    continue

            seen.add(content_hash)
            record_id = f"proc_{split}_{accepted:04d}"
            condition_vector = normalize_condition_vector_tensor(
                build_condition_vector(spec)
            ).tolist()
            voxel_path = f"voxels/{content_hash}.npy"
            file_path = voxel_dir / f"{content_hash}.npy"
            file_sha = ""
            if not file_path.exists():
                np.save(file_path, (geometry.numpy() > 0.5).astype(np.float32))
            file_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
            records.append({
                "sample_id": record_id,
                "split": split,
                "design_spec": _normalize_manifest_design_spec(spec.__dict__.copy()),
                "condition_vector": condition_vector,
                "geometry_path": voxel_path,
                "voxel_sha256": file_sha,
                "canonical_voxel_sha256": content_hash,
                "source_seed": candidate_seed,
                "source": "procedural_mission_conditioned_v1",
                "old_gate_status": "pass",
                "structural_gate_status": "pass" if require_structural else "not_applied",
                "grid_size": GRID,
            })
            accepted += 1
            stats["accepted"] += 1
            if accepted % 100 == 0:
                print(f"  {split}: {accepted}/{count} accepted (attempts={attempts})", flush=True)
        if accepted < count:
            raise RuntimeError(f"split {split} only accepted {accepted}/{count} in {attempts} attempts")
        stats[f"{split}_accepted"] = accepted
        stats[f"{split}_attempts"] = attempts

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    stats["records"] = len(records)
    stats["generator"] = "aircraft_diffusion_cfd._procedural_aircraft_geometry"
    stats["grid_size"] = GRID
    stats["seed"] = seed
    (output_dir / "corpus_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=720)
    parser.add_argument("--holdout-count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-structural", default=True, action="store_true",
                        help="Accept only records that also pass the frozen structural articulation gate.")
    parser.add_argument("--no-require-structural", dest="require_structural", action="store_false")
    parser.set_defaults(require_structural=True)
    args = parser.parse_args()
    return 0 if build_corpus(args.train_count, args.holdout_count, args.seed,
                             Path(args.output_dir), require_structural=args.require_structural) else 1


if __name__ == "__main__":
    raise SystemExit(main())