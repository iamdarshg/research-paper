import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import run_protocol


class TestProtocolRunner(unittest.TestCase):
    def test_build_protocol_commands_resolves_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            cli_dir = repo_root / "CLI"
            protocol_dir = cli_dir / "run_protocols"
            protocol_dir.mkdir(parents=True)

            config_path = protocol_dir / "smoke.yaml"
            payload = {
                "train": {
                    "enabled": True,
                    "num_epochs": 1,
                    "batch_size": 1,
                    "num_samples": 2,
                    "save_dir": "../../checkpoints_test",
                    "run_class": "final",
                    "dataset_manifest": "../../docs/dataset/minimal_grounded_manifest.jsonl",
                    "baseline_config": "../baseline_config.yaml",
                    "claim_gates": "../../paper/FINAL_RUN_GATES.md",
                },
                "evaluate_baselines": {
                    "enabled": True,
                    "output": "../../build/baseline_report.json",
                },
                "validate_conditions": {
                    "enabled": True,
                    "num_seeds": 3,
                    "output": "../../build/condition_validation.json",
                },
                "multi_seed_eval": {
                    "enabled": True,
                    "num_seeds": 3,
                    "output_dir": "../../build/multi_seed_eval",
                },
            }
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = run_protocol.load_protocol_config(str(config_path))
            commands = run_protocol.build_protocol_commands(config)

            self.assertEqual(commands[0][1], str((cli_dir / "aircraft_diffusion_cfd.py").resolve()))
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[0])
            self.assertIn(str((repo_root / "CLI" / "baseline_config.yaml").resolve()), commands[0])
            self.assertIn(str((repo_root / "paper" / "FINAL_RUN_GATES.md").resolve()), commands[0])
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[2])
            self.assertEqual(commands[1][2], "evaluate-baselines")
            self.assertEqual(commands[2][2], "validate-conditions")
            self.assertEqual(commands[3][1], str((cli_dir / "multi_seed_eval.py").resolve()))
