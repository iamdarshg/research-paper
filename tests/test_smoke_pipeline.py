import os
import sys
import unittest
import json
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import torch
from click.testing import CliRunner


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import aircraft_diffusion_cfd as cli_module
import offline_densify as densify_module
from aircraft_diffusion_cfd import CFDConfig, DiffusionConfig, LBMPhysicsConfig, ModelConfig, TrainingConfig


class _FakeStateModule:
    def __init__(self, *args, **kwargs):
        self.loaded_state = None

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def load_state_dict(self, state):
        self.loaded_state = state

    def state_dict(self):
        return {"ok": True}


class _FakeConsistencyModel(_FakeStateModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student_model = _FakeStateModule()


class TestCLISmokePipeline(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    def test_cli_help_lists_current_commands(self):
        result = self.runner.invoke(cli_module.cli, ["--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("train", result.output)
        self.assertIn("generate", result.output)
        self.assertIn("batch-generate", result.output)
        self.assertIn("condition-response-smoke", result.output)
        self.assertIn("densify-dataset", result.output)
        self.assertIn("performance-benchmark", result.output)
        self.assertIn("info", result.output)

    def test_train_help_lists_dataset_artifact_option(self):
        result = self.runner.invoke(cli_module.cli, ["train", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("--dataset-artifact", result.output)
        self.assertIn("--dataset-manifest", result.output)
        self.assertIn("--run-class", result.output)
        self.assertIn("--baseline-config", result.output)
        self.assertIn("--claim-gates", result.output)
        self.assertIn("--disable-consistency", result.output)
        self.assertIn("--disable-pipeline", result.output)
        self.assertIn("--disable-checkpointing", result.output)

    def test_manifest_dataset_loads_grounded_samples(self):
        with self.runner.isolated_filesystem():
            geometry = torch.zeros((16, 16, 16), dtype=torch.float32)
            geometry[4:12, 4:12, 4:12] = 1.0
            os.makedirs("dataset", exist_ok=True)
            geometry_path = os.path.join("dataset", "sample.npy")
            manifest_path = os.path.join("dataset", "manifest.jsonl")

            import numpy as np

            np.save(geometry_path, geometry.numpy())
            record = {
                "geometry_path": "sample.npy",
                "design_spec": {
                    "target_speed": 44.0,
                    "wingspan_limit_m": 1.7,
                    "thrust_to_weight_min": 0.4,
                    "turn_rate_min_deg_s": 15.0,
                    "required_static_thrust_n": 150.0,
                    "engine_diameter_mm": 120,
                    "engine_length_mm": 240,
                    "engine_count_min": 1,
                    "engine_count_max": 1,
                    "payload_mass_min_g": 300,
                    "payload_mass_max_g": 900,
                    "takeoff_distance_min_m": 80,
                    "takeoff_distance_max_m": 150,
                    "wall_thickness_min_mm": 1,
                    "wall_thickness_max_mm": 2,
                    "part_count_min": 1,
                    "part_count_max": 5,
                    "manufacturing_method": "fdm_pla_0p4mm",
                },
                "split": "train",
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            dataset = cli_module.AircraftDesignDataset(
                manifest_path=manifest_path,
                grid_size=16,
                latent_dim=16,
                seed=7,
            )

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.metadata["data_source"], "grounded_manifest")
            self.assertEqual(dataset.metadata["split_assignments"], ["train"])
            sample = dataset[0]
            self.assertEqual(sample["geometry"].shape, (16, 16, 16))
            self.assertEqual(sample["design_spec"].target_speed, 44.0)
            self.assertEqual(sample["condition_vector"].numel(), cli_module.infer_conditioning_dim())

    def test_manifest_dataset_accepts_schema_target_speed_mps(self):
        with self.runner.isolated_filesystem():
            geometry = torch.zeros((8, 8, 8), dtype=torch.float32)
            geometry[2:6, 2:6, 2:6] = 1.0
            os.makedirs("dataset", exist_ok=True)
            geometry_path = os.path.join("dataset", "sample.npy")
            manifest_path = os.path.join("dataset", "manifest.jsonl")

            import numpy as np

            np.save(geometry_path, geometry.numpy())
            record = {
                "geometry_path": "sample.npy",
                "design_spec": {
                    "target_speed_mps": 51.0,
                    "wingspan_limit_m": 1.7,
                    "thrust_to_weight_min": 0.4,
                    "turn_rate_min_deg_s": 15.0,
                    "required_static_thrust_n": 150.0,
                    "engine_diameter_mm": 120,
                    "engine_length_mm": 240,
                    "engine_count_min": 1,
                    "engine_count_max": 1,
                    "payload_mass_min_g": 300,
                    "payload_mass_max_g": 900,
                    "takeoff_distance_min_m": 80,
                    "takeoff_distance_max_m": 150,
                    "wall_thickness_min_mm": 1,
                    "wall_thickness_max_mm": 2,
                    "part_count_min": 1,
                    "part_count_max": 5,
                    "manufacturing_method": "fdm_pla_0p4mm",
                },
                "split": "train",
            }
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

            dataset = cli_module.AircraftDesignDataset(
                manifest_path=manifest_path,
                grid_size=8,
                latent_dim=8,
                seed=7,
            )

            self.assertEqual(dataset[0]["design_spec"].target_speed, 51.0)

    def test_checked_in_minimal_manifest_loads_through_repo_dataset_path(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        manifest_path = os.path.join(repo_root, "docs", "dataset", "minimal_grounded_manifest.jsonl")

        dataset = cli_module.AircraftDesignDataset(
            manifest_path=manifest_path,
            grid_size=32,
            latent_dim=16,
            seed=11,
        )

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.metadata["data_source"], "grounded_manifest")
        self.assertEqual(
            dataset.metadata["split_assignments"],
            ["train", "holdout"],
        )
        sample = dataset[0]
        self.assertEqual(sample["geometry"].shape, (32, 32, 32))
        self.assertEqual(sample["condition_vector"].numel(), cli_module.infer_conditioning_dim())

    def test_generate_help_lists_current_options(self):
        result = self.runner.invoke(cli_module.cli, ["generate", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("--checkpoint", result.output)
        self.assertIn("--output", result.output)
        self.assertIn("--target-speed", result.output)
        self.assertIn("--thrust-to-weight-min", result.output)
        self.assertIn("--turn-rate-min-deg-s", result.output)
        self.assertIn("--required-static-thrust-n", result.output)
        self.assertIn("--engine-diameter-mm", result.output)
        self.assertIn("--engine-length-mm", result.output)
        self.assertIn("--engine-count-min", result.output)
        self.assertIn("--engine-count-max", result.output)
        self.assertIn("--wingspan-limit-m", result.output)
        self.assertIn("--payload-mass-min-g", result.output)
        self.assertIn("--payload-mass-max-g", result.output)
        self.assertIn("--takeoff-distance-min-m", result.output)
        self.assertIn("--takeoff-distance-max-m", result.output)
        self.assertIn("--wall-thickness-min-mm", result.output)
        self.assertIn("--wall-thickness-max-mm", result.output)
        self.assertIn("--part-count-min", result.output)
        self.assertIn("--part-count-max", result.output)
        self.assertIn("--manufacturing-method", result.output)
        self.assertIn("--num-steps", result.output)
        self.assertIn("--use-marching-cubes", result.output)
        self.assertIn("--no-marching-cubes", result.output)
        self.assertIn("--solver", result.output)

    def test_densify_help_lists_current_options(self):
        result = self.runner.invoke(cli_module.cli, ["densify-dataset", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("--output-artifact", result.output)
        self.assertIn("--checkpoint", result.output)
        self.assertIn("--report-dir", result.output)
        self.assertIn("--num-samples", result.output)
        self.assertIn("--num-conditions", result.output)
        self.assertIn("--num-candidates-per-condition", result.output)

    def test_validate_conditions_help_avoids_scientific_validation_claim(self):
        result = self.runner.invoke(cli_module.cli, ["validate-conditions", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("condition-response", result.output)
        self.assertNotIn("scientific study", result.output.lower())
        self.assertNotIn("scientific condition-response validation", result.output.lower())

    def test_save_checkpoint_includes_cfd_config_payload(self):
        trainer = object.__new__(cli_module.OptimizedDiffusionTrainer)
        trainer.diffusion_model = _FakeStateModule()
        trainer.consistency_model = _FakeConsistencyModel()
        trainer.converter = _FakeStateModule()
        trainer.ema_model = _FakeStateModule()
        trainer.optimizer = _FakeStateModule()
        trainer.scheduler = _FakeStateModule()
        trainer.scaler = _FakeStateModule()
        trainer.global_step = 11
        trainer.model_config = ModelConfig(latent_dim=8, base_grid_resolution=16, grid_resolution=16)
        trainer.diffusion_config = DiffusionConfig()
        trainer.training_config = TrainingConfig(num_epochs=1, batch_size=1)
        trainer.cfd_config = CFDConfig(base_grid_resolution=24, solver_type="D3Q27")
        trainer.cfd_config.lbm_config = LBMPhysicsConfig(grid_spacing=0.125)

        with mock.patch.object(cli_module.torch, "save") as mock_save, \
             mock.patch.object(cli_module.os, "replace") as mock_replace:
            trainer.save_checkpoint("fake-checkpoint.pt")

        saved_payload, saved_handle = mock_save.call_args.args
        # R10: the checkpoint serializes to a write-handle on the .tmp sibling
        # (which gets fsynced) before the atomic replace onto the final path.
        self.assertEqual(saved_handle.name, "fake-checkpoint.pt.tmp")
        mock_replace.assert_called_once_with(
            Path("fake-checkpoint.pt.tmp"),
            Path("fake-checkpoint.pt"),
        )
        self.assertIn("cfd_config", saved_payload)
        self.assertEqual(saved_payload["cfd_config"]["base_grid_resolution"], 24)
        self.assertEqual(saved_payload["cfd_config"]["solver_type"], "D3Q27")
        self.assertEqual(saved_payload["cfd_config"]["lbm_config"]["grid_spacing"], 0.125)
        # torch.save is mocked (nothing written) and os.replace is mocked (no
        # final move), so the .tmp handle left in cwd is litter — remove it.
        for litter in (Path("fake-checkpoint.pt.tmp"), Path("fake-checkpoint.pt")):
            if litter.exists():
                litter.unlink()

    def test_generator_restores_cfd_config_from_checkpoint(self):
        checkpoint = {
            "model_config": asdict(ModelConfig(latent_dim=8, base_grid_resolution=16, grid_resolution=16)),
            "diffusion_config": asdict(DiffusionConfig()),
            "consistency_model": {"weights": 1},
            "diffusion_model": {"weights": 2},
            "converter": {"weights": 3},
            "cfd_config": asdict(CFDConfig(base_grid_resolution=24, solver_type="D3Q27")),
        }
        checkpoint["cfd_config"]["lbm_config"]["grid_spacing"] = 0.125

        with mock.patch.object(cli_module.torch, "load", return_value=checkpoint), \
             mock.patch.object(cli_module, "LatentDiffusionUNet", _FakeStateModule), \
             mock.patch.object(cli_module, "LatentTo3DConverter", _FakeStateModule), \
             mock.patch.object(cli_module, "ConsistencyModel", _FakeConsistencyModel), \
             mock.patch.object(cli_module, "NoiseSchedule", _FakeStateModule), \
             mock.patch.object(cli_module.torch.cuda, "is_available", return_value=False):
            generator = cli_module.OptimizedAircraftGenerator("fake-checkpoint.pt", device=torch.device("cpu"))

        self.assertEqual(generator.config.base_grid_resolution, 24)
        self.assertEqual(generator.config.solver_type, "D3Q27")
        self.assertIsInstance(generator.config.lbm_config, LBMPhysicsConfig)
        self.assertEqual(generator.config.lbm_config.grid_spacing, 0.125)

    def test_generator_refuses_invalid_voxel_export_before_meshing(self):
        generator = object.__new__(cli_module.OptimizedAircraftGenerator)

        with self.assertRaisesRegex(ValueError, "aircraft-invalid"):
            generator.voxels_to_stl(
                torch.zeros((16, 16, 16)),
                "invalid.stl",
                use_marching_cubes=True,
            )

    def test_generate_uses_generated_voxel_resolution_for_final_cfd(self):
        fake_generator = mock.Mock()
        fake_generator.generate.return_value = torch.ones((12, 12, 12))
        fake_generator.voxels_to_stl.return_value = None

        fake_simulator = mock.Mock()
        fake_simulator.simulate_aerodynamics.return_value = {
            "drag_coefficient": 0.1,
            "lift_coefficient": 0.2,
        }

        with mock.patch.object(cli_module.os.path, "exists", return_value=True), \
             mock.patch.object(cli_module, "OptimizedAircraftGenerator", return_value=fake_generator) as mock_generator_cls, \
             mock.patch.object(cli_module, "AdvancedCFDSimulator", return_value=fake_simulator) as mock_simulator_cls, \
             mock.patch.object(cli_module.torch.cuda, "is_available", return_value=False):
            result = self.runner.invoke(
                cli_module.cli,
                [
                    "generate",
                    "--checkpoint",
                    "fake-checkpoint.pt",
                    "--output",
                    "design.stl",
                    "--target-speed",
                    "42.0",
                    "--required-static-thrust-n",
                    "180.0",
                    "--engine-diameter-mm",
                    "140",
                    "--engine-length-mm",
                    "260",
                    "--engine-count-min",
                    "1",
                    "--engine-count-max",
                    "2",
                    "--num-steps",
                    "6",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_generator_cls.assert_called_once()
        fake_generator.generate.assert_called_once()
        design_spec = fake_generator.generate.call_args.args[0]
        self.assertEqual(design_spec.target_speed, 42.0)
        self.assertEqual(design_spec.required_static_thrust_n, 180.0)
        self.assertEqual(design_spec.engine_diameter_mm, 140)
        self.assertEqual(design_spec.engine_length_mm, 260)
        self.assertEqual(design_spec.engine_count_min, 1)
        self.assertEqual(design_spec.engine_count_max, 2)
        self.assertEqual(fake_generator.generate.call_args.kwargs["num_steps"], 6)
        self.assertEqual(mock_simulator_cls.call_args.args[0].base_grid_resolution, 12)
        fake_generator.voxels_to_stl.assert_called_once_with(
            fake_generator.generate.return_value,
            "design.stl",
            use_marching_cubes=True,
        )

    def test_generate_creates_parent_output_directory(self):
        fake_generator = mock.Mock()
        fake_generator.generate.return_value = torch.ones((12, 12, 12))
        fake_generator.voxels_to_stl.return_value = None

        fake_simulator = mock.Mock()
        fake_simulator.simulate_aerodynamics.return_value = {
            "drag_coefficient": 0.1,
            "lift_coefficient": 0.2,
        }

        nested_dir_created = False
        with self.runner.isolated_filesystem():
            with mock.patch.object(cli_module.os.path, "exists", return_value=True), \
                 mock.patch.object(cli_module, "OptimizedAircraftGenerator", return_value=fake_generator), \
                 mock.patch.object(cli_module, "AdvancedCFDSimulator", return_value=fake_simulator), \
                 mock.patch.object(cli_module.torch.cuda, "is_available", return_value=False):
                result = self.runner.invoke(
                    cli_module.cli,
                    [
                        "generate",
                        "--checkpoint",
                        "fake-checkpoint.pt",
                        "--output",
                        os.path.join("nested", "design.stl"),
                    ],
                )
            nested_dir_created = os.path.isdir("nested")

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(nested_dir_created)

    def test_train_passes_dataset_artifact_and_enables_conditioning_for_fresh_model(self):
        fake_dataset = mock.Mock()
        fake_loader = mock.Mock()
        fake_trainer = mock.Mock()
        fake_trainer.train.return_value = []

        with mock.patch.object(cli_module.torch._logging, "set_logs"), \
             mock.patch.object(cli_module.torch.cuda, "is_available", return_value=False), \
             mock.patch.object(cli_module, "AircraftDesignDataset", return_value=fake_dataset) as mock_dataset_cls, \
             mock.patch.object(cli_module, "DataLoader", return_value=fake_loader) as mock_loader_cls, \
             mock.patch.object(cli_module, "OptimizedDiffusionTrainer", return_value=fake_trainer) as mock_trainer_cls:
            result = self.runner.invoke(
                cli_module.cli,
                [
                    "train",
                    "--num-epochs",
                    "1",
                    "--num-samples",
                    "2",
                    "--dataset-artifact",
                    "artifact.pt",
                    "--save-dir",
                    "tmp-checkpoints",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(mock_dataset_cls.call_args.kwargs["artifact_path"], "artifact.pt")
        self.assertEqual(
            mock_loader_cls.call_args.kwargs["collate_fn"].__name__,
            "aircraft_collate_fn",
        )
        model_config = mock_trainer_cls.call_args.args[0]
        self.assertEqual(model_config.conditioning_dim, cli_module.infer_conditioning_dim())
        fake_trainer.train.assert_called_once_with(fake_loader)

    def test_final_run_class_requires_artifact_baselines_and_claim_gates(self):
        result = self.runner.invoke(
            cli_module.cli,
            [
                "train",
                "--run-class",
                "final",
                "--num-epochs",
                "1",
                "--num-samples",
                "2",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("dataset artifact", result.output.lower())
        self.assertIn("baseline", result.output.lower())
        self.assertIn("claim", result.output.lower())

    def test_final_run_class_rejects_manifest_that_has_not_passed_grounded_claim_gates(self):
        with self.runner.isolated_filesystem():
            manifest_path = "manifest.jsonl"
            baseline_config_path = "baseline_config.yaml"
            claim_gates_path = "FINAL_RUN_GATES.md"
            with open(manifest_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"split": "train"}) + "\n")
            with open(baseline_config_path, "w", encoding="utf-8") as handle:
                handle.write("baseline_set: []\n")
            with open(claim_gates_path, "w", encoding="utf-8") as handle:
                handle.write("# Gates\n")

            with self.assertRaises(cli_module.click.UsageError):
                cli_module._validate_run_class_inputs(
                    cli_module.RUN_CLASS_FINAL,
                    dataset_artifact=None,
                    dataset_manifest=manifest_path,
                    baseline_config=baseline_config_path,
                    claim_gates=claim_gates_path,
                )

    def test_advanced_cfd_simulator_resets_flow_field_per_geometry(self):
        class FakeSolver:
            def collide_stream(self, geometry_mask, steps):
                self.last_shape = tuple(geometry_mask.shape)

            def compute_aerodynamic_coefficients(self, geometry_mask):
                return {"drag_coefficient": 1.0, "lift_coefficient": 0.5}

        simulator = object.__new__(cli_module.AdvancedCFDSimulator)
        simulator.config = mock.Mock(solver_type="D3Q27", use_amr=False)
        simulator.lbm_solver = FakeSolver()
        simulator.amr_solver = None
        simulator.init_flow_field = mock.Mock()
        simulator._run_fluidx3d_validation = mock.Mock(return_value=None)

        geometry = torch.zeros((4, 4, 4), dtype=torch.float32)
        simulator.simulate_aerodynamics(geometry, steps=1)
        simulator.simulate_aerodynamics(geometry, steps=1)

        self.assertEqual(simulator.init_flow_field.call_count, 2)

    def test_densify_dataset_cli_delegates_to_checkpoint_densifier(self):
        with mock.patch.object(densify_module, "densify_from_checkpoint", return_value={"num_candidates": 6, "num_accepted": 2, "output_path": "artifact.pt"}) as mock_densify, \
             mock.patch.object(densify_module, "bootstrap_dataset") as mock_bootstrap:
            result = self.runner.invoke(
                cli_module.cli,
                [
                    "densify-dataset",
                    "--checkpoint",
                    "fake-checkpoint.pt",
                    "--output-artifact",
                    "artifact.pt",
                    "--num-conditions",
                    "2",
                    "--num-candidates-per-condition",
                    "3",
                    "--min-total-reward",
                    "0.2",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_densify.assert_called_once()
        mock_bootstrap.assert_not_called()
        self.assertIn("accepted=2", result.output)

    def test_batch_generate_writes_condition_manifest(self):
        fake_generator = mock.Mock()
        fake_generator.generate.return_value = torch.ones((12, 12, 12))

        def _write_stl(_voxel_grid, output_path, use_marching_cubes=True):
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("solid mock\nendsolid mock\n")

        fake_generator.voxels_to_stl.side_effect = _write_stl

        with self.runner.isolated_filesystem(), \
             mock.patch.object(cli_module, "OptimizedAircraftGenerator", return_value=fake_generator), \
             mock.patch.object(cli_module.os.path, "exists", return_value=True), \
             mock.patch.object(cli_module.torch.cuda, "is_available", return_value=False):
            result = self.runner.invoke(
                cli_module.cli,
                [
                    "batch-generate",
                    "--checkpoint",
                    "fake-checkpoint.pt",
                    "--output-dir",
                    "batch-out",
                    "--num-designs",
                    "2",
                    "--seed",
                    "7",
                ],
            )

            manifest_path = os.path.join("batch-out", "batch_manifest.json")
            self.assertTrue(os.path.exists(manifest_path))
            import json

            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(len(manifest["designs"]), 2)
        self.assertIn("design_spec", manifest["designs"][0])
        self.assertIn("condition_vector", manifest["designs"][0])


if __name__ == "__main__":
    unittest.main()
