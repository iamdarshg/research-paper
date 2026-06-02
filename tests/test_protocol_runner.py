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
                "validate_manifest": {
                    "enabled": True,
                    "manifest": "../../docs/dataset/minimal_grounded_manifest.jsonl",
                    "level": "claim-bearing",
                    "output": "../../build/manifest_validation.json",
                },
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
                "condition_benchmark": {
                    "enabled": True,
                    "manifest": "../../docs/dataset/minimal_grounded_manifest.jsonl",
                    "num_seeds": 3,
                    "output": "../../build/condition_benchmark.json",
                    "min_grounded_records": 4,
                },
                "manufacturing_constraints": {
                    "enabled": True,
                    "manifest": "../../docs/dataset/minimal_grounded_manifest.jsonl",
                    "output": "../../build/manufacturing_constraints.json",
                },
                "aircraft_validity": {
                    "enabled": True,
                    "input_dir": "../../build/generated_voxels",
                    "output": "../../build/aircraft_validity.json",
                },
                "multi_seed_eval": {
                    "enabled": True,
                    "num_seeds": 3,
                    "output_dir": "../../build/multi_seed_eval",
                },
                "baseline_statistics": {
                    "enabled": True,
                    "baseline_config": "../baseline_config.yaml",
                    "records_json": "../../build/baseline_records.json",
                    "metric_keys": ["lift_to_drag"],
                    "output": "../../build/baseline_statistics.json",
                    "min_seeds": 3,
                },
                "final_evidence": {
                    "enabled": True,
                    "baseline_statistics": "../../build/baseline_statistics.json",
                    "require_run_consistency": True,
                    "output": "../../build/final_evidence_package.json",
                },
                "gate_readiness": {
                    "enabled": True,
                    "output": "../../build/gate_readiness.json",
                },
            }
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = run_protocol.load_protocol_config(str(config_path))
            commands = run_protocol.build_protocol_commands(config)

            self.assertEqual(commands[0][1], str((cli_dir / "validate_manifest.py").resolve()))
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[0])
            self.assertEqual(commands[0][2], "--manifest")
            self.assertEqual(commands[1][1], str((cli_dir / "aircraft_diffusion_cfd.py").resolve()))
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[1])
            self.assertIn(str((repo_root / "CLI" / "baseline_config.yaml").resolve()), commands[1])
            self.assertIn(str((repo_root / "paper" / "FINAL_RUN_GATES.md").resolve()), commands[1])
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[3])
            self.assertEqual(commands[2][2], "evaluate-baselines")
            self.assertEqual(commands[3][2], "validate-conditions")
            self.assertEqual(commands[4][1], str((cli_dir / "run_condition_benchmark.py").resolve()))
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[4])
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[4])
            self.assertEqual(commands[5][1], str((cli_dir / "condition_feasibility.py").resolve()))
            self.assertEqual(commands[6][1], str((cli_dir / "aircraft_validity.py").resolve()))
            self.assertEqual(commands[7][1], str((cli_dir / "multi_seed_eval.py").resolve()))
            self.assertEqual(commands[8][1], str((cli_dir / "multi_seed_eval.py").resolve()))
            self.assertIn("--baseline-statistics-output", commands[8])
            self.assertEqual(commands[9][1], str((cli_dir / "final_evidence.py").resolve()))
            self.assertIn("--require-run-consistency", commands[9])
            self.assertEqual(commands[10][1], str((cli_dir / "gate_readiness.py").resolve()))

    def test_checked_in_protocols_resolve_repo_assets(self):
        repo_root = Path(__file__).resolve().parents[1]
        protocol_paths = [
            repo_root / "CLI" / "run_protocols" / "smoke_8gb.yaml",
            repo_root / "CLI" / "run_protocols" / "final_cloud.yaml",
        ]

        for path in protocol_paths:
            config = run_protocol.load_protocol_config(str(path))
            commands = run_protocol.build_protocol_commands(config)
            self.assertTrue(commands, f"{path} should produce runnable commands")
            self.assertTrue(Path(commands[0][1]).exists())

        final_config = run_protocol.load_protocol_config(str(protocol_paths[1]))
        final_commands = run_protocol.build_protocol_commands(final_config)
        self.assertEqual(final_commands[0][1], str((repo_root / "CLI" / "build_reference_evidence.py").resolve()))
        self.assertEqual(final_commands[1][1], str((repo_root / "CLI" / "validate_manifest.py").resolve()))
        self.assertIn(
            str((repo_root / "build" / "protocol_final" / "grounded_corpus" / "manifest.jsonl").resolve()),
            final_commands[1],
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "run_condition_benchmark.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "condition_feasibility.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "aircraft_validity.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "final_evidence.py").resolve()) for command in final_commands)
        )
        final_evidence_command = next(
            command for command in final_commands
            if command[1] == str((repo_root / "CLI" / "final_evidence.py").resolve())
        )
        self.assertIn("--run-metadata", final_evidence_command)
        gate_readiness_command = next(
            command for command in final_commands
            if command[1] == str((repo_root / "CLI" / "gate_readiness.py").resolve())
        )
        self.assertIn("--final-evidence", gate_readiness_command)
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "gate_readiness.py").resolve()) for command in final_commands)
        )
        baseline_stats_index = next(
            idx for idx, command in enumerate(final_commands)
            if command[1] == str((repo_root / "CLI" / "multi_seed_eval.py").resolve())
            and "--baseline-statistics-output" in command
        )
        final_evidence_index = next(
            idx for idx, command in enumerate(final_commands)
            if command[1] == str((repo_root / "CLI" / "final_evidence.py").resolve())
        )
        self.assertLess(baseline_stats_index, final_evidence_index)
