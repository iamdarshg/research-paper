import os
import sys
from unittest import mock

import torch
from torch import nn


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import aircraft_diffusion_cfd as cli_module


class _ConstantConverter(nn.Module):
    def forward(self, latent):
        return torch.full(
            (latent.shape[0], 4, 4, 4),
            1.3862944,
            device=latent.device,
        )


class _SeedRecordingConsistency:
    def __init__(self):
        self.student_model = nn.Identity()
        self.seeds = []

    def fast_inference(self, shape, num_steps, condition=None):
        self.seeds.append(torch.initial_seed())
        return torch.zeros(shape)


def test_promotion_uses_fixed_threshold_unique_seeds_and_training_frame():
    trainer = cli_module.OptimizedDiffusionTrainer.__new__(
        cli_module.OptimizedDiffusionTrainer
    )
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.training_config = cli_module.TrainingConfig(
        overfit_geometry_gate_samples=2,
        promotion_generation_seeds=2,
    )
    trainer.diffusion_config = cli_module.DiffusionConfig(student_steps=4)
    trainer.converter = _ConstantConverter()
    trainer.consistency_model = _SeedRecordingConsistency()
    trainer.geometry_probability_threshold = 0.9
    trainer.geometry_threshold_calibrated = True
    trainer.geometry_threshold_calibration = {
        "source": "test",
        "threshold": 0.9,
    }

    first_target = torch.zeros((4, 4, 4))
    first_target[0, 0, 0] = 1.0
    second_target = torch.zeros((4, 4, 4))
    second_target[:2, :2, :2] = 1.0
    loader = [
        {
            "latent": torch.zeros((2, 3)),
            "geometry": torch.stack((first_target, second_target)),
            "condition_vector": torch.zeros((2, cli_module.infer_conditioning_dim())),
        }
    ]

    with mock.patch.object(
        cli_module,
        "evaluate_aircraft_validity",
        return_value={
            "status": "fail",
            "metrics": {
                "largest_component_fraction": 0.0,
                "normalization_boundary_fraction": 0.0,
            },
            "failed_checks": ["empty_geometry"],
        },
    ) as validity_mock:
        report = trainer.evaluate_geometry_promotion_gate(loader)

    assert trainer.consistency_model.seeds == [0, 1, 2, 3]
    assert report["materialization_mode"] == "fixed_global_threshold"
    assert report["generated_evaluation_count"] == 4
    assert report["generated_unique_count"] == 1
    assert report["generated_mean_occupied_fraction"] == 0.0
    assert report["target_mean_occupied_fraction"] > 0.0
    assert all(
        call.kwargs.get("canonicalize") is False
        for call in validity_mock.call_args_list
    )
