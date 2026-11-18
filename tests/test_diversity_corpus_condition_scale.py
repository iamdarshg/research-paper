"""The diversity corpus must store RAW condition vectors (2026-11-12 audit).

`scripts/build_diversity_procedural_corpus.py` stored
`normalize_condition_vector_tensor(build_condition_vector(spec))`, but the
dataset loader passes a manifest's stored vector through unchanged and both
`LatentDiffusionUNet._forward_impl` and `_project_condition_signature`
normalize again. The model therefore saw every scalar slot divided by its
scale twice: measured on the 660-record corpus, scalar-slot std fell to
4e-6..0.04 (wingspan 0.0033), leaving only the 5 one-hot slots informative,
and the standardized generator emitted the same 11 shapes for any condition.
Other manifests (NASA CRM) carry no vector and are built raw, so a corpus
that stores pre-normalized vectors is also on a different scale from them.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for sub in ("CLI", "scripts"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)

import build_diversity_procedural_corpus as builder  # noqa: E402
from aircraft_diffusion_cfd import build_condition_vector, sample_design_spec  # noqa: E402


def test_corpus_condition_vector_is_raw():
    spec = sample_design_spec(__import__("random").Random(7))
    stored = torch.tensor(builder.corpus_condition_vector(spec))
    assert torch.allclose(stored, build_condition_vector(spec).float(), atol=1e-6)


def test_corpus_condition_vector_is_not_prenormalized():
    spec = sample_design_spec(__import__("random").Random(11))
    raw = build_condition_vector(spec).float()
    stored = torch.tensor(builder.corpus_condition_vector(spec))
    # target speed (tens of m/s) must not have been divided by its scale
    assert stored[0].item() == raw[0].item() and raw[0].item() > 5.0
