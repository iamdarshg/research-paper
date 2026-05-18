import os
import sys
import unittest
from dataclasses import asdict
from unittest import mock

import torch
from click.testing import CliRunner


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import aircraft_diffusion_cfd as cli_module
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
        self.assertIn("performance-benchmark", result.output)
        self.assertIn("info", result.output)

    def test_generate_help_lists_current_options(self):
        result = self.runner.invoke(cli_module.cli, ["generate", "--help"])

        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("--checkpoint", result.output)
        self.assertIn("--output", result.output)
        self.assertIn("--target-speed", result.output)
        self.assertIn("--num-steps", result.output)
        self.assertIn("--use-marching-cubes", result.output)
        self.assertIn("--solver", result.output)

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

        with mock.patch.object(cli_module.torch, "save") as mock_save:
            trainer.save_checkpoint("fake-checkpoint.pt")

        saved_payload, saved_path = mock_save.call_args.args
        self.assertEqual(saved_path, "fake-checkpoint.pt")
        self.assertIn("cfd_config", saved_payload)
        self.assertEqual(saved_payload["cfd_config"]["base_grid_resolution"], 24)
        self.assertEqual(saved_payload["cfd_config"]["solver_type"], "D3Q27")
        self.assertEqual(saved_payload["cfd_config"]["lbm_config"]["grid_spacing"], 0.125)

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
                    "--num-steps",
                    "6",
                ],
            )

        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_generator_cls.assert_called_once()
        fake_generator.generate.assert_called_once()
        design_spec = fake_generator.generate.call_args.args[0]
        self.assertEqual(design_spec.target_speed, 42.0)
        self.assertEqual(fake_generator.generate.call_args.kwargs["num_steps"], 6)
        self.assertEqual(mock_simulator_cls.call_args.args[0].base_grid_resolution, 12)
        fake_generator.voxels_to_stl.assert_called_once_with(
            fake_generator.generate.return_value,
            "design.stl",
            use_marching_cubes=True,
        )


if __name__ == "__main__":
    unittest.main()
