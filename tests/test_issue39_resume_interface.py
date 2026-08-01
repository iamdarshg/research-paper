"""Issue #39: interruption-safe resume interface tests.

Covers the CLI surface added for issue #39 on top of the existing
``run_monitored_training.py`` trainer:

* ``--max-optimizer-updates`` — cap on ACTUAL optimizer updates per
  invocation (additional on resume); the cumulative count is retained.
* ``--checkpoint-every-updates`` — atomic per-N-update checkpoint cadence.
* ``--fixed-validation-seeds`` — explicit promotion/generation seed list,
  used verbatim and preserved across resume sessions.
* checkpoint lineage (self/source/config/corpus hashes, threshold,
  cumulative updates), ``resume_manifest.json`` bookkeeping, atomic-write
  durability, and legacy-checkpoint backward compatibility.

All training runs are CPU-only with a tiny synthetic manifest (8^3 grid,
1 train + 1 val sample), 1 direct-solver step and 2 SPSA directions so the
suite stays fast. Resuming a 1-sample-per-epoch run lands exactly on an
epoch boundary, which exercises the clean resume path.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CLI_DIR = REPO_ROOT / "CLI"
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))

import run_monitored_training  # noqa: E402  (module under test, CLI on path)


# ---------------------------------------------------------------------------
# Shared tiny synthetic manifest (built once per test session)
# ---------------------------------------------------------------------------
def _write_tiny_manifest(directory: Path) -> Path:
    """Two deterministic 8^3 geometries (1 train, 1 val) + grounded manifest."""
    geometries = [
        np.zeros((8, 8, 8), dtype=np.float32),
        np.zeros((8, 8, 8), dtype=np.float32),
    ]
    geometries[0][2:6, 2:6, 2:6] = 1.0
    geometries[0][4, 2:7, 3] = 1.0  # wing-ish protrusion
    geometries[1][2:6, 2:6, 2:6] = 1.0
    geometries[1][3, 2:7, 4] = 1.0
    records = []
    for index, geometry in enumerate(geometries):
        np.save(directory / f"geo{index}.npy", geometry)
        records.append(
            {
                "geometry_path": f"geo{index}.npy",
                "split": "train" if index == 0 else "val",
                "design_spec": {
                    "target_speed": 42.0,
                    "wingspan_limit_m": 1.8,
                    "thrust_to_weight_min": 0.42,
                    "turn_rate_min_deg_s": 16.0,
                    "required_static_thrust_n": 160.0,
                    "engine_diameter_mm": 120,
                    "engine_length_mm": 240,
                    "engine_count_min": 1,
                    "engine_count_max": 1,
                    "payload_mass_min_g": 400,
                    "payload_mass_max_g": 900,
                    "takeoff_distance_min_m": 90,
                    "takeoff_distance_max_m": 180,
                    "wall_thickness_min_mm": 1,
                    "wall_thickness_max_mm": 2,
                    "part_count_min": 1,
                    "part_count_max": 6,
                    "manufacturing_method": "sheet_balsa_tabbed",
                },
            }
        )
    manifest_path = directory / "tiny_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return manifest_path


_MANIFEST_DIR = Path(tempfile.mkdtemp(prefix="issue39_manifest_"))
TINY_MANIFEST = _write_tiny_manifest(_MANIFEST_DIR)


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _updates_global_steps(save_dir: Path):
    lines = (save_dir / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["global_step"] for line in lines if line.strip()]


def _update_losses(save_dir: Path):
    lines = (save_dir / "updates.jsonl").read_text(encoding="utf-8").splitlines()
    return [
        json.loads(line)["losses"]["optimization"] for line in lines if line.strip()
    ]


def _load_history(save_dir: Path):
    return json.loads((save_dir / "history.json").read_text(encoding="utf-8"))


def _run_training(
    save_dir: Path,
    *,
    max_updates: int,
    checkpoint_every: int = 1,
    resume_from: Path | None = None,
    fixed_seeds: str | None = None,
    extra: list | None = None,
    manifest: Path = TINY_MANIFEST,
    timeout: int = 900,
):
    """Run the monitored training CLI in a subprocess; return CompletedProcess."""
    command = [
        sys.executable,
        str(CLI_DIR / "run_monitored_training.py"),
        "--manifest",
        str(manifest),
        "--num-epochs",
        "20",
        "--batch-size",
        "1",
        "--latent-dim",
        "64",
        "--grid-size",
        "8",
        "--direct-solver-steps",
        "1",
        "--direct-solver-directions",
        "2",
        "--direct-solver-perturbation-grid-size",
        "8",
        "--cpu-threads",
        "2",
        "--promotion-evaluation-samples",
        "2",
        "--promotion-generation-seeds",
        "6",
        "--save-dir",
        str(save_dir),
        "--history-output",
        str(save_dir / "history.json"),
        "--updates-output",
        str(save_dir / "updates.jsonl"),
        "--save-every",
        "0",
        "--no-save-final-checkpoint",
        "--max-optimizer-updates",
        str(max_updates),
        "--checkpoint-every-updates",
        str(checkpoint_every),
    ]
    if resume_from is not None:
        command += ["--resume-from", str(resume_from)]
    if fixed_seeds is not None:
        command += ["--fixed-validation-seeds", fixed_seeds]
    if extra:
        command += list(extra)
    env = dict(os.environ, PYTHONPATH=str(CLI_DIR))
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _assert_ok(testcase: unittest.TestCase, result: subprocess.CompletedProcess):
    tail = "\n".join(
        (result.stdout or "").splitlines()[-20:] + (result.stderr or "").splitlines()[-20:]
    )
    testcase.assertEqual(result.returncode, 0, msg=f"rc={result.returncode}\n{tail}")


class Issue39ResumeInterfaceTests(unittest.TestCase):
    """End-to-end issue #39 resume interface behavior on a tiny CPU manifest."""

    def setUp(self):
        self.run_dir = Path(tempfile.mkdtemp(prefix="issue39_run_"))
        self.addCleanup(lambda: _rmtree(self.run_dir))

    # -- update-cap exactness -------------------------------------------------

    def test_max_optimizer_updates_caps_fresh_and_resume_invocations(self):
        # Fresh session: exactly 2 updates, no more.
        fresh = _run_training(self.run_dir, max_updates=2)
        _assert_ok(self, fresh)
        self.assertEqual(_updates_global_steps(self.run_dir), [1, 2])
        manifest = json.loads(
            (self.run_dir / "resume_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["cumulative_optimizer_updates"], 2)

        # Resume session: exactly 2 ADDITIONAL updates; cumulative count
        # retained and contiguous (no repeat, no skip).
        resumed = _run_training(
            self.run_dir,
            max_updates=2,
            resume_from=self.run_dir / "checkpoint_updates_000002.pt",
        )
        _assert_ok(self, resumed)
        self.assertEqual(_updates_global_steps(self.run_dir), [1, 2, 3, 4])
        manifest = json.loads(
            (self.run_dir / "resume_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["cumulative_optimizer_updates"], 4)
        history = _load_history(self.run_dir)
        self.assertEqual(history["config"]["cumulative_updates_at_start"], 2)

    # -- checkpoint cadence ---------------------------------------------------

    def test_checkpoint_every_updates_cadence_and_lineage(self):
        # One fresh invocation capped at 4 updates with a cadence of 2:
        # checkpoints must exist at exactly steps 2 and 4 and load cleanly.
        result = _run_training(self.run_dir, max_updates=4, checkpoint_every=2)
        _assert_ok(self, result)
        self.assertEqual(_updates_global_steps(self.run_dir), [1, 2, 3, 4])

        second = torch.load(
            self.run_dir / "checkpoint_updates_000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        fourth = torch.load(
            self.run_dir / "checkpoint_updates_000004.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(second["global_step"], 2)
        self.assertEqual(fourth["global_step"], 4)

        # Lineage block: self/config/corpus hashes, threshold, cumulative
        # updates; source hash is null on a fresh run.
        for payload in (second, fourth):
            lineage = payload["lineage"]
            self.assertTrue(lineage["checkpoint_self_sha256"])
            self.assertIsNone(lineage["source_checkpoint_sha256"])
            self.assertTrue(lineage["config_hash"])
            self.assertTrue(lineage["corpus_manifest_hash"])
            self.assertEqual(lineage["cumulative_optimizer_updates"], payload["global_step"])
            self.assertIsInstance(lineage["geometry_probability_threshold"], float)
            self.assertIn("command_line", lineage)
            self.assertIn("data_position", payload)
            self.assertEqual(payload["cumulative_optimizer_updates"], payload["global_step"])

        # resume_manifest.json tracks the latest save and stays self-consistent.
        manifest = json.loads(
            (self.run_dir / "resume_manifest.json").read_text(encoding="utf-8")
        )
        latest = Path(manifest["latest_checkpoint"])
        self.assertTrue(latest.exists(), msg=str(latest))
        self.assertEqual(
            manifest["latest_checkpoint_sha256"], _sha256_file(latest)
        )
        self.assertEqual(manifest["cumulative_optimizer_updates"], 4)
        self.assertIn("command_line", manifest)

    # -- resume equivalence ---------------------------------------------------

    def test_resume_is_bit_identical_to_uninterrupted_run(self):
        # Run A: uninterrupted 2-update session.
        run_a = Path(tempfile.mkdtemp(prefix="issue39_eqA_"))
        self.addCleanup(lambda: _rmtree(run_a))
        _assert_ok(self, _run_training(run_a, max_updates=2))

        # Run B: 1 update, then resume for 1 more from the step-1 checkpoint.
        run_b = Path(tempfile.mkdtemp(prefix="issue39_eqB_"))
        self.addCleanup(lambda: _rmtree(run_b))
        _assert_ok(self, _run_training(run_b, max_updates=1))
        resumed_result = _run_training(
            run_b,
            max_updates=1,
            resume_from=run_b / "checkpoint_updates_000001.pt",
        )
        _assert_ok(self, resumed_result)

        # Identical update stream: same losses for update 1 AND the
        # next-update loss after the interruption point (update 2). Exact
        # float equality proves model weights, optimizer momentum, scheduler,
        # RNG state and data ordering were all restored.
        self.assertEqual(_update_losses(run_a), _update_losses(run_b))
        self.assertEqual(_updates_global_steps(run_b), [1, 2])

        # Threshold after resume equals the checkpoint's threshold, NOT a
        # recalibrated value: the calibration metadata is copied verbatim from
        # the checkpoint and the CLI explicitly skips recalibration.
        history = _load_history(run_b)
        self.assertIn("skipping recalibration", resumed_result.stdout)
        checkpoint_payload = torch.load(
            run_b / "checkpoint_updates_000001.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            history["config"]["geometry_threshold_calibration"],
            checkpoint_payload["geometry_threshold_calibration"],
        )
        self.assertEqual(
            history["config"]["geometry_probability_threshold"],
            checkpoint_payload["geometry_probability_threshold"],
        )
        self.assertEqual(history["config"]["cumulative_updates_at_start"], 1)

        # Lineage chains: the resumed checkpoint records the source file's hash.
        resumed_payload = torch.load(
            run_b / "checkpoint_updates_000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            resumed_payload["lineage"]["source_checkpoint_sha256"],
            _sha256_file(run_b / "checkpoint_updates_000001.pt"),
        )

    # -- atomicity ------------------------------------------------------------

    def test_failed_checkpoint_save_preserves_previous_checkpoint_and_manifest(self):
        # Drive main() in-process so we can inject an os.replace failure on
        # the second cadence save. The first save (step 1) must survive, the
        # resume manifest must not be clobbered, and no .tmp debris may remain.
        argv = [
            "run_monitored_training.py",
            "--manifest",
            str(TINY_MANIFEST),
            "--num-epochs",
            "20",
            "--batch-size",
            "1",
            "--latent-dim",
            "64",
            "--grid-size",
            "8",
            "--direct-solver-steps",
            "1",
            "--direct-solver-directions",
            "2",
            "--direct-solver-perturbation-grid-size",
            "8",
            "--cpu-threads",
            "2",
            "--promotion-evaluation-samples",
            "2",
            "--promotion-generation-seeds",
            "6",
            "--save-dir",
            str(self.run_dir),
            "--history-output",
            str(self.run_dir / "history.json"),
            "--updates-output",
            str(self.run_dir / "updates.jsonl"),
            "--save-every",
            "0",
            "--no-save-final-checkpoint",
            "--max-optimizer-updates",
            "2",
            "--checkpoint-every-updates",
            "1",
        ]
        real_replace = os.replace
        failed = {"count": 0}

        def flaky_replace(src, dst):
            if str(dst).endswith("checkpoint_updates_000002.pt"):
                failed["count"] += 1
                raise OSError("injected save failure")
            return real_replace(src, dst)

        with mock.patch.object(sys, "argv", argv), mock.patch(
            "os.replace", side_effect=flaky_replace
        ):
            with self.assertRaises(OSError):
                run_monitored_training.main()

        self.assertEqual(failed["count"], 1)
        self.assertFalse((self.run_dir / "checkpoint_updates_000002.pt").exists())

        # The previously saved checkpoint is intact and valid.
        previous = torch.load(
            self.run_dir / "checkpoint_updates_000001.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(previous["global_step"], 1)

        # resume_manifest.json was not clobbered by the failed save: it still
        # parses and points at a real file whose hash matches.
        manifest = json.loads(
            (self.run_dir / "resume_manifest.json").read_text(encoding="utf-8")
        )
        latest = Path(manifest["latest_checkpoint"])
        self.assertTrue(latest.exists(), msg=str(latest))
        self.assertEqual(manifest["latest_checkpoint_sha256"], _sha256_file(latest))
        self.assertEqual(manifest["cumulative_optimizer_updates"], 1)

        # No temp-file debris from the interrupted atomic write.
        self.assertEqual(list(self.run_dir.glob("*.tmp")), [])
        self.assertEqual(list(self.run_dir.glob("*.pt.tmp")), [])

    # -- fixed validation seeds ------------------------------------------------

    def test_fixed_validation_seeds_honored_verbatim_across_resume(self):
        seeds = "3,7,11,2,5,9"  # non-sequential: distinguishable from legacy derivation
        expected = [3, 7, 11, 2, 5, 9]
        _assert_ok(
            self,
            _run_training(self.run_dir, max_updates=1, fixed_seeds=seeds),
        )
        _assert_ok(
            self,
            _run_training(
                self.run_dir,
                max_updates=1,
                fixed_seeds=seeds,
                resume_from=self.run_dir / "checkpoint_updates_000001.pt",
            ),
        )

        # Both sessions used the exact seed list verbatim in the gate.
        fresh_history = json.loads(
            (self.run_dir / "history.json").read_text(encoding="utf-8")
        )
        self.assertEqual(fresh_history["config"]["fixed_validation_seeds"], expected)
        self.assertEqual(
            fresh_history["history"][0]["promotion_generation_seeds_used"],
            expected,
        )
        resumed_history = _load_history(self.run_dir)
        self.assertEqual(resumed_history["config"]["fixed_validation_seeds"], expected)
        self.assertEqual(
            resumed_history["history"][0]["promotion_generation_seeds_used"],
            expected,
        )
        self.assertEqual(resumed_history["config"]["cumulative_updates_at_start"], 1)

        # Seeds are preserved in the checkpoint lineage and the resume manifest.
        fresh_payload = torch.load(
            self.run_dir / "checkpoint_updates_000001.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(fresh_payload["fixed_validation_seeds"], expected)
        self.assertEqual(fresh_payload["lineage"]["fixed_validation_seeds"], expected)
        manifest = json.loads(
            (self.run_dir / "resume_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["fixed_validation_seeds"], expected)

    # -- legacy backward compatibility ------------------------------------------

    def test_legacy_checkpoint_without_issue39_keys_loads_and_resumes(self):
        # Build a full checkpoint (2 updates -> global_step 2), then strip every
        # issue-#39 key so it looks like a pre-issue-#39 checkpoint
        # (weights + optimizer + configs only).
        _assert_ok(self, _run_training(self.run_dir, max_updates=2))
        source = torch.load(
            self.run_dir / "checkpoint_updates_000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        legacy = {
            key: value
            for key, value in source.items()
            if key
            not in {
                "lineage",
                "data_position",
                "rng_state",
                "cumulative_optimizer_updates",
                "fixed_validation_seeds",
                "geometry_probability_threshold",
                "geometry_threshold_calibrated",
                "geometry_threshold_calibration",
                "scheduler_step_per_update",
                "scheduler",
                "scaler",
            }
        }
        self.assertIn("global_step", legacy)
        legacy_path = self.run_dir / "legacy_checkpoint.pt"
        torch.save(legacy, legacy_path)

        legacy_dir = Path(tempfile.mkdtemp(prefix="issue39_legacy_"))
        self.addCleanup(lambda: _rmtree(legacy_dir))
        result = _run_training(
            legacy_dir,
            max_updates=1,
            resume_from=legacy_path,
        )
        _assert_ok(self, result)

        # Resumed from global_step 2 without repeats: exactly one new update.
        self.assertEqual(_updates_global_steps(legacy_dir), [3])
        history = _load_history(legacy_dir)
        self.assertEqual(history["config"]["cumulative_updates_at_start"], 2)
        # No checkpoint-derived threshold message: legacy behavior recalibrates.
        self.assertNotIn("from checkpoint", result.stdout)

        # New checkpoint chains back to the legacy source file.
        resumed_payload = torch.load(
            legacy_dir / "checkpoint_updates_000003.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(
            resumed_payload["lineage"]["source_checkpoint_sha256"],
            _sha256_file(legacy_path),
        )

    # -- CLI surface ------------------------------------------------------------

    def test_cli_help_lists_issue39_flags(self):
        result = subprocess.run(
            [sys.executable, str(CLI_DIR / "run_monitored_training.py"), "--help"],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=str(CLI_DIR)),
            cwd=str(REPO_ROOT),
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        for flag in (
            "--max-optimizer-updates",
            "--checkpoint-every-updates",
            "--fixed-validation-seeds",
        ):
            self.assertIn(flag, result.stdout)


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
