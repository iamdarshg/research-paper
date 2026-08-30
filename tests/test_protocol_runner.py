import inspect
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
                    "grid_size": 24,
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
                "prepare_aircraft_validity_inputs": {
                    "enabled": True,
                    "output_dir": "../../build/generated_voxels",
                    "metadata": "../../build/generated_voxels_metadata.json",
                    "num_samples": 6,
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
                "final_evidence": {
                    "enabled": True,
                    "baseline_statistics": "../../build/baseline_statistics.json",
                    "require_run_consistency": True,
                    "output": "../../build/final_evidence_package.json",
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
            self.assertIn("--grid-size", commands[1])
            self.assertIn("24", commands[1])
            self.assertIn(str((repo_root / "CLI" / "baseline_config.yaml").resolve()), commands[1])
            self.assertIn(str((repo_root / "paper" / "FINAL_RUN_GATES.md").resolve()), commands[1])
            self.assertEqual(commands[2][2], "evaluate-baselines")
            self.assertIn("--baseline-config", commands[2])
            self.assertIn(str((repo_root / "CLI" / "baseline_config.yaml").resolve()), commands[2])
            self.assertIn("--manifest", commands[2])
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[2])
            self.assertIn("--checkpoint", commands[2])
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[2])
            self.assertEqual(commands[3][2], "validate-conditions")
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[3])
            self.assertEqual(commands[4][1], str((cli_dir / "run_condition_benchmark.py").resolve()))
            self.assertIn(str((repo_root / "docs" / "dataset" / "minimal_grounded_manifest.jsonl").resolve()), commands[4])
            self.assertIn(str((repo_root / "checkpoints_test" / "final_optimized_model.pt").resolve()), commands[4])
            self.assertEqual(commands[5][1], str((cli_dir / "condition_feasibility.py").resolve()))
            self.assertEqual(commands[6][1], str((cli_dir / "build_aircraft_validity_inputs.py").resolve()))
            self.assertEqual(commands[7][1], str((cli_dir / "aircraft_validity.py").resolve()))
            self.assertEqual(commands[8][1], str((cli_dir / "multi_seed_eval.py").resolve()))
            self.assertEqual(commands[9][1], str((cli_dir / "validate_manifest.py").resolve()))
            self.assertEqual(commands[10][1], str((cli_dir / "final_evidence.py").resolve()))
            self.assertIn("--require-run-consistency", commands[10])

    def test_build_protocol_commands_supports_disable_flags(self):
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
                    "enable_consistency": False,
                    "enable_pipeline": False,
                    "enable_checkpointing": False,
                    "enable_compile": False,
                },
            }
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

            config = run_protocol.load_protocol_config(str(config_path))
            commands = run_protocol.build_protocol_commands(config)

            train_cmd = commands[0]
            self.assertIn("--disable-consistency", train_cmd)
            self.assertIn("--disable-pipeline", train_cmd)
            self.assertIn("--disable-checkpointing", train_cmd)
            self.assertNotIn("--enable-compile", train_cmd)

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
        self.assertEqual(final_commands[0][1], str((repo_root / "CLI" / "validate_manifest.py").resolve()))
        train_command = final_commands[1]

        self.assertIn(
            str((repo_root / "build" / "faa_geometry_case_corpus_20260624" / "geometry_case_manifest_5k.jsonl").resolve()),
            train_command,
        )
        self.assertIn("--grid-size", train_command)
        self.assertIn("32", train_command)
        self.assertIn(str((repo_root / "CLI" / "baseline_config.yaml").resolve()), train_command)
        self.assertIn(str((repo_root / "paper" / "FINAL_RUN_GATES.md").resolve()), train_command)
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "run_condition_benchmark.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "condition_feasibility.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "build_aircraft_validity_inputs.py").resolve()) for command in final_commands)
        )
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "aircraft_validity.py").resolve()) for command in final_commands)
        )
        builder_command = next(
            command
            for command in final_commands
            if command[1] == str((repo_root / "CLI" / "build_aircraft_validity_inputs.py").resolve())
        )
        self.assertIn(str((repo_root / "build" / "protocol_final" / "generated_voxels").resolve()), builder_command)
        self.assertTrue(
            any(command[1] == str((repo_root / "CLI" / "final_evidence.py").resolve()) for command in final_commands)
        )
        self.assertGreaterEqual(
            sum(1 for command in final_commands if command[1] == str((repo_root / "CLI" / "validate_manifest.py").resolve())),
            2,
        )

    def test_gcp_128_production_protocol_runs_exactly_two_epochs_with_production_cadence(self):
        repo_root = Path(__file__).resolve().parents[1]
        protocol_path = repo_root / "CLI" / "run_protocols" / "gcp_128_295m.yaml"
        config = run_protocol.load_protocol_config(str(protocol_path))

        commands = run_protocol.build_protocol_commands(config)

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(
            Path(command[1]).resolve(),
            (repo_root / "CLI" / "run_monitored_training.py").resolve(),
        )

        expected_values = {
            "--manifest": str(
                (repo_root / "build" / "final_training_corpus_20260830_v2" / "combined_training_manifest.jsonl").resolve()
            ),
            "--num-epochs": "2",
            "--batch-size": "1",
            "--learning-rate": "2e-05",
            "--latent-dim": "512",
            "--grid-size": "128",
            "--precision": "bfloat16",
            "--solver": "D3Q27",
            "--lbm-stream-bfl-backend": "fused_stream_bfl",
            "--coordinate-training-samples": "65536",
            "--full-lattice-interval": "64",
            "--sparse-samples-per-full": "262144",
            "--direct-solver-interval": "32",
            "--direct-solver-steps": "5",
            "--direct-solver-directions": "8",
            "--direct-solver-batch-chunk": "4",
            "--checkpoint-every-updates": "25",
        }
        for flag, value in expected_values.items():
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], value)

        self.assertIn("--no-require-direct-solver-every-iteration", command)
        self.assertIn("--no-enable-consistency", command)
        self.assertIn("--enable-compile", command)
        self.assertIn("--enable-gradient-checkpointing", command)
        self.assertIn("--no-stop-on-promotion-pass", command)
        self.assertNotIn("--no-save-final-checkpoint", command)
        self.assertNotIn("--stop-after-updates", command)

    def test_gcp_128_smoke_mode_is_explicitly_bounded(self):
        repo_root = Path(__file__).resolve().parents[1]
        protocol_path = repo_root / "CLI" / "run_protocols" / "gcp_128_295m.yaml"
        config = run_protocol.load_protocol_config(str(protocol_path))
        mode_parameter = inspect.signature(
            run_protocol.build_protocol_commands
        ).parameters.get("mode")

        self.assertIsNotNone(mode_parameter)
        commands = run_protocol.build_protocol_commands(config, mode="smoke")

        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[command.index("--num-epochs") + 1], "2")
        self.assertEqual(command[command.index("--checkpoint-every-updates") + 1], "1")
        self.assertEqual(command[command.index("--stop-after-updates") + 1], "5")
        self.assertNotIn("--no-save-final-checkpoint", command)
        self.assertTrue(command[command.index("--save-dir") + 1].endswith("checkpoints_128_295m_smoke"))

    def test_monitored_smoke_followups_consume_isolated_smoke_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "CLI" / "run_protocols" / "monitored.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "train": {
                            "enabled": True,
                            "runner": "monitored",
                            "save_dir": "../../checkpoints",
                        },
                        "smoke": {"enabled": True, "stop_after_updates": 2},
                        "evaluate_baselines": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )

            train_command, baseline_command = run_protocol.build_protocol_commands(
                run_protocol.load_protocol_config(str(config_path)), mode="smoke"
            )

            smoke_save_dir = str((root / "checkpoints_smoke").resolve())
            smoke_checkpoint = str(
                (root / "checkpoints_smoke" / "final_monitored_model.pt").resolve()
            )
            self.assertEqual(
                train_command[train_command.index("--save-dir") + 1], smoke_save_dir
            )
            self.assertEqual(
                baseline_command[baseline_command.index("--checkpoint") + 1],
                smoke_checkpoint,
            )

    def test_monitored_smoke_rejects_production_resume_without_isolated_state(self):
        config = {
            "_config_path": str(Path(__file__).resolve()),
            "_config_dir": str(Path(__file__).resolve().parent),
            "train": {
                "enabled": True,
                "runner": "monitored",
                "resume_run_state": "production/latest_run_state.pt",
            },
            "smoke": {"enabled": True, "stop_after_updates": 2},
        }

        with self.assertRaisesRegex(
            ValueError, "isolated resume_run_state"
        ):
            run_protocol.build_protocol_commands(config, mode="smoke")

    def test_fresh_monitored_smoke_isolates_configured_production_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "CLI" / "run_protocols" / "monitored.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "train": {
                            "enabled": True,
                            "runner": "monitored",
                            "save_dir": "../../production",
                            "history_output": "../../production/history.json",
                            "updates_output": "../../production/updates.jsonl",
                        },
                        "smoke": {"enabled": True, "stop_after_updates": 2},
                    }
                ),
                encoding="utf-8",
            )

            command = run_protocol.build_protocol_commands(
                run_protocol.load_protocol_config(str(config_path)), mode="smoke"
            )[0]

            smoke_dir = root / "production_smoke"
            self.assertEqual(
                command[command.index("--history-output") + 1],
                str((smoke_dir / "history.json").resolve()),
            )
            self.assertEqual(
                command[command.index("--updates-output") + 1],
                str((smoke_dir / "updates.jsonl").resolve()),
            )

    def test_monitored_smoke_uses_isolated_resume_state_and_updates_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "CLI" / "run_protocols" / "monitored.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "train": {
                            "enabled": True,
                            "runner": "monitored",
                            "resume_run_state": "../../production/latest_run_state.pt",
                            "updates_output": "../../production/updates.jsonl",
                        },
                        "smoke": {
                            "enabled": True,
                            "stop_after_updates": 2,
                            "resume_run_state": "../../smoke/latest_run_state.pt",
                            "updates_output": "../../smoke/updates.jsonl",
                        },
                    }
                ),
                encoding="utf-8",
            )

            command = run_protocol.build_protocol_commands(
                run_protocol.load_protocol_config(str(config_path)), mode="smoke"
            )[0]

            self.assertEqual(
                command[command.index("--resume-run-state") + 1],
                str((root / "smoke" / "latest_run_state.pt").resolve()),
            )
            self.assertEqual(
                command[command.index("--updates-output") + 1],
                str((root / "smoke" / "updates.jsonl").resolve()),
            )
            self.assertEqual(
                command[command.index("--history-output") + 1],
                str(
                    (
                        config_path.parent
                        / "checkpoints_monitored_smoke"
                        / "history.json"
                    ).resolve()
                ),
            )

    def test_monitored_smoke_rejects_paths_aliasing_production_artifacts(self):
        cases = {
            "resume_run_state": {
                "train": "production/latest_run_state.pt",
                "smoke": "production/latest_run_state.pt",
            },
            "updates_output": {
                "train": "production/updates.jsonl",
                "smoke": "production/updates.jsonl",
            },
            "history_output": {
                "train": "production/history.json",
                "smoke": "production/history.json",
            },
        }
        for field, paths in cases.items():
            with self.subTest(field=field):
                train = {
                    "enabled": True,
                    "runner": "monitored",
                    "resume_run_state": "production/latest_run_state.pt",
                    "updates_output": "production/updates.jsonl",
                    "history_output": "production/history.json",
                }
                smoke = {
                    "enabled": True,
                    "stop_after_updates": 2,
                    "resume_run_state": "smoke/latest_run_state.pt",
                    "updates_output": "smoke/updates.jsonl",
                    "history_output": "smoke/history.json",
                }
                train[field] = paths["train"]
                smoke[field] = paths["smoke"]
                config = {
                    "_config_path": str(Path(__file__).resolve()),
                    "_config_dir": str(Path(__file__).resolve().parent),
                    "train": train,
                    "smoke": smoke,
                }

                with self.assertRaisesRegex(ValueError, "aliases production"):
                    run_protocol.build_protocol_commands(config, mode="smoke")

    def test_monitored_smoke_mode_rejects_missing_or_nonpositive_update_bound(self):
        for stop_after_updates in (None, 0, -1):
            with self.subTest(stop_after_updates=stop_after_updates):
                config = {
                    "_config_path": str(Path(__file__).resolve()),
                    "_config_dir": str(Path(__file__).resolve().parent),
                    "train": {"enabled": True, "runner": "monitored"},
                    "smoke": {"enabled": True},
                }
                if stop_after_updates is not None:
                    config["smoke"]["stop_after_updates"] = stop_after_updates

                with self.assertRaisesRegex(
                    ValueError, r"smoke\.stop_after_updates.*positive"
                ):
                    run_protocol.build_protocol_commands(config, mode="smoke")

    def test_monitored_protocol_forwards_exact_resume_and_uses_monitored_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            protocol_dir = repo_root / "CLI" / "run_protocols"
            protocol_dir.mkdir(parents=True)
            config_path = protocol_dir / "monitored.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "train": {
                            "enabled": True,
                            "runner": "monitored",
                            "dataset_manifest": "../../build/corpus/manifest.jsonl",
                            "save_dir": "../../checkpoints",
                            "resume_run_state": "../../checkpoints/latest_run_state.pt",
                        },
                        "evaluate_baselines": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )

            commands = run_protocol.build_protocol_commands(
                run_protocol.load_protocol_config(str(config_path))
            )

            train_command, baseline_command = commands
            expected_state = str(
                (repo_root / "checkpoints" / "latest_run_state.pt").resolve()
            )
            expected_checkpoint = str(
                (repo_root / "checkpoints" / "final_monitored_model.pt").resolve()
            )
            self.assertEqual(
                train_command[train_command.index("--resume-run-state") + 1],
                expected_state,
            )
            self.assertEqual(
                baseline_command[baseline_command.index("--checkpoint") + 1],
                expected_checkpoint,
            )
