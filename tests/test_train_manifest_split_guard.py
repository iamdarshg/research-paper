"""The ``train`` CLI must not silently train on held-out records.

Context (2026-11-12 audit).  ``scripts/build_diversity_procedural_corpus.py``
writes ONE ``manifest.jsonl`` holding both ``split: train`` and
``split: holdout`` records, while ``train`` built its loader from every record
in the manifest and ignored the ``split`` field.  Passing the builder's
manifest straight to ``train`` therefore trained on the 96 records the
diversity/novelty evaluation treats as held out, contaminating every
"held-out" generation, novelty and conditioning number without any warning.

The guard fails closed: a manifest whose records carry a held-out split label
is rejected before any model is built.  Callers pass a train-only manifest.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import (  # noqa: E402
    HELD_OUT_SPLIT_LABELS,
    reject_held_out_training_records,
)


def _records(*splits):
    return [{"sample_id": f"r{i}", "split": s} for i, s in enumerate(splits)]


def test_held_out_labels_cover_holdout_and_test():
    assert {"holdout", "test"} <= set(HELD_OUT_SPLIT_LABELS)


def test_train_only_manifest_is_accepted():
    reject_held_out_training_records(_records("train", "train"), "m.jsonl")


def test_unlabelled_manifest_is_accepted():
    reject_held_out_training_records([{"sample_id": "a"}, {"sample_id": "b"}], "m.jsonl")


@pytest.mark.parametrize("label", ["holdout", "test", "HoldOut"])
def test_manifest_with_held_out_records_is_rejected(label):
    with pytest.raises(ValueError, match="held-out"):
        reject_held_out_training_records(_records("train", label), "m.jsonl")


def test_train_cli_rejects_combined_manifest(tmp_path):
    from click.testing import CliRunner

    from aircraft_diffusion_cfd import cli

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(r) for r in _records("train", "holdout")) + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        cli,
        [
            "train",
            "--num-epochs", "1",
            "--dataset-manifest", str(manifest),
            "--save-dir", str(tmp_path / "ckpt"),
            "--grid-size", "32",
            "--latent-dim", "16",
        ],
    )
    assert result.exit_code != 0
    assert "held-out" in str(result.exception) or "held-out" in result.output
