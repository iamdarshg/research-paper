import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import DesignSpec, DirectSolverSPSALoss, TrainingConfig
from diagnose_training_branches import (
    DIRECT_SCALAR_COMPONENT_FIELDS,
    DiagnosticRuntime,
    PreflightConfig,
    _student_gradient_branch,
    build_argument_parser,
    exact_inference_timesteps,
    run_preflight,
    select_split_indices,
    summarize_values,
)


class _ToyNoiseSchedule:
    def q_sample(self, latent, timestep, noise):
        scale = (timestep.float().unsqueeze(1) + 1.0) / 1000.0
        return latent + scale * noise

    def predict_x0(self, noised_latent, timestep, prediction):
        return noised_latent - 0.1 * prediction


class _ToyPredictor(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor(weight, dtype=torch.float32)
        )

    def forward(self, latent, timestep, condition=None):
        result = latent @ self.weight
        if condition is not None:
            result = result + 0.01 * condition[:, : result.shape[1]]
        return result


class _ToyConverter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.linspace(-0.4, 0.4, 16).reshape(8, 2)
            )
            self.linear.bias.copy_(torch.linspace(-0.2, 0.2, 8))

    def forward(self, latent):
        return self.linear(latent).reshape(-1, 2, 2, 2)


class _ToyConsistency(torch.nn.Module):
    def __init__(self, nonfinite=False):
        super().__init__()
        self.teacher_model = _ToyPredictor([[0.8, 0.1], [0.2, 0.7]])
        self.student_model = _ToyPredictor([[0.6, -0.2], [0.3, 0.5]])
        self.nonfinite = nonfinite
        if nonfinite:
            original_forward = self.student_model.forward

            def forward_with_nan(latent, timestep, condition=None):
                result = original_forward(latent, timestep, condition)
                return result * torch.tensor(float("nan"))

            self.student_model.forward = forward_with_nan


class _TinyDataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        geometry = torch.zeros((2, 2, 2), dtype=torch.float32)
        geometry.reshape(-1)[[0, 3, 5, 7]] = 1.0
        return {
            "latent": torch.tensor(
                [0.2 + 0.01 * index, 0.7 - 0.01 * index],
                dtype=torch.float32,
            ),
            "geometry": geometry,
            "condition_vector": torch.tensor([0.3, 0.6]),
            "design_spec": DesignSpec(
                space_weight=0.3,
                drag_weight=0.4,
                lift_weight=0.3,
            ),
        }


class _TinySimulator:
    def __init__(self):
        self.device = torch.device("cpu")

    def simulate_aerodynamics(self, geometry, steps=100):
        weights = torch.linspace(
            0.2,
            1.0,
            geometry.numel(),
            dtype=geometry.dtype,
            device=geometry.device,
        ).reshape_as(geometry)
        weighted_occupancy = float(
            (geometry * weights).sum().item() / weights.sum().item()
        )
        return {
            "training_drag_coefficient": 0.1 + weighted_occupancy,
            "lift_coefficient": 0.25 + 0.1 * weighted_occupancy,
        }


def _runtime(nonfinite=False):
    training = TrainingConfig(
        precision="float32",
        geometry_dice_weight=1.0,
        consistency_huber_delta=0.5,
        direct_solver_loss_weight=1.0,
        direct_solver_perturbation=0.8,
        direct_solver_perturbation_grid_size=0,
        direct_solver_gradient_clip=100.0,
        direct_aero_gradient_max_norm=100.0,
        direct_connectivity_gradient_max_norm=100.0,
        direct_validity_gradient_max_norm=100.0,
        direct_connectivity_weight=1.0,
        direct_aircraft_validity_weight=1.0,
    )
    consistency = _ToyConsistency(nonfinite=nonfinite)
    return DiagnosticRuntime(
        model_config=SimpleNamespace(latent_dim=2),
        diffusion_config=SimpleNamespace(timesteps=1000, student_steps=4),
        training_config=training,
        consistency_model=consistency,
        converter=_ToyConverter(),
        noise_schedule=_ToyNoiseSchedule(),
        dtype=torch.float32,
        latent_value_min=0.0,
        latent_value_max=1.0,
    )


def _write_inputs(root: Path, records):
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    checkpoint = root / "checkpoint.pt"
    torch.save(
        {
            "global_step": 42,
            "probe_tensor": torch.tensor([1.0, 2.0]),
            "optimizer": {
                "param_groups": [
                    {"name": "consistency_student", "lr": 2.0e-4}
                ]
            },
            "training_config": {
                "learning_rate": 2.0e-4,
                "converter_learning_rate": 1.0e-3,
                "consistency_student_learning_rate": 2.0e-4,
            },
        },
        checkpoint,
    )
    return manifest, checkpoint


def _run_tiny_preflight(tmp_path, *, nonfinite=False):
    records = [
        {"sample_id": "train-0", "split": "train"},
        {"sample_id": "val-0", "split": "val"},
        {"sample_id": "val-1", "split": "val"},
    ]
    manifest, checkpoint = _write_inputs(tmp_path, records)
    output = tmp_path / "report.json"
    checkpoint_before = checkpoint.read_bytes()
    report = run_preflight(
        PreflightConfig(
            manifest=manifest,
            checkpoint=checkpoint,
            output_json=output,
            sample_count=1,
            split="val",
            grid_size=2,
            device="cpu",
            solver_backend="pytorch_reference",
            direct_solver_directions=1,
            direct_solver_steps=2,
        ),
        runtime_factory=lambda checkpoint, device, grid: _runtime(
            nonfinite=nonfinite
        ),
        dataset_factory=lambda manifest, grid, latent_dim: _TinyDataset(
            len(records)
        ),
        simulator_factory=lambda checkpoint, device, grid, backend: (
            _TinySimulator()
        ),
        direct_loss_factory=lambda training, directions, steps: (
            DirectSolverSPSALoss(
                cfd_steps=steps,
                perturbation=training.direct_solver_perturbation,
                perturbation_grid_size=0,
                gradient_clip=training.direct_solver_gradient_clip,
                aero_gradient_max_norm=training.direct_aero_gradient_max_norm,
                connectivity_gradient_max_norm=(
                    training.direct_connectivity_gradient_max_norm
                ),
                validity_gradient_max_norm=(
                    training.direct_validity_gradient_max_norm
                ),
                connectivity_weight=training.direct_connectivity_weight,
                aircraft_validity_weight=(
                    training.direct_aircraft_validity_weight
                ),
                directions=directions,
                input_is_logits=True,
            )
        ),
    )
    assert checkpoint.read_bytes() == checkpoint_before
    return report, output, manifest, checkpoint


def test_deterministic_split_selection_uses_manifest_order_and_split():
    records = [
        {"sample_id": "a", "split": "train"},
        {"sample_id": "b", "split": "val"},
        {"sample_id": "c", "split": "val"},
        {"sample_id": "d", "split": "val"},
    ]

    first = select_split_indices(records, split="val", sample_count=2)
    second = select_split_indices(records, split="val", sample_count=2)

    assert first == [1, 2]
    assert second == first


def test_summary_statistics_include_requested_percentiles_and_max():
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    assert summary == {
        "p50": pytest.approx(2.5),
        "p95": pytest.approx(3.85),
        "p99": pytest.approx(3.97),
        "max": pytest.approx(4.0),
    }


def test_exact_inference_schedule_matches_four_training_levels():
    assert exact_inference_timesteps(1000, 4) == [999, 666, 333, 0]


def test_student_gradient_snapshots_are_parked_on_cpu():
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cpu"))
    (parameter.square().sum()).backward()

    gradients, norm = _student_gradient_branch((parameter,), "probe")

    assert gradients[0] is not None
    assert gradients[0].device.type == "cpu"
    assert norm == pytest.approx(2.0)


def test_preflight_hashes_inputs_records_exact_fields_and_solver_calls(tmp_path):
    report, output, manifest, checkpoint = _run_tiny_preflight(tmp_path)

    assert report["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert report["checkpoint_sha256"] == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert report["selection"]["manifest_indices"] == [1]
    assert report["selection"]["sample_ids"] == ["val-0"]
    assert report["inference_timesteps"] == [999, 666, 333, 0]
    assert report["checkpoint"]["global_step"] == 42
    assert report["checkpoint"]["optimizer_groups"] == [
        {"name": "consistency_student", "learning_rate": 2.0e-4}
    ]
    assert report["solver_calls"] == {
        "actual": 12,
        "expected": 12,
        "per_sample_timestep": 3,
    }

    timestep_record = report["samples"][0]["timesteps"][0]
    assert set(timestep_record["direct_solver"]["component_scalars"]) == set(
        DIRECT_SCALAR_COMPONENT_FIELDS
    )
    assert set(timestep_record["direct_solver"]["spsa_gradient_norms"]) == {
        "aero",
        "connectivity",
        "validity",
        "total",
    }
    for component in ("aero", "connectivity", "validity", "total"):
        assert set(
            timestep_record["direct_solver"]["spsa_gradient_norms"][component]
        ) == {"raw", "applied"}
    assert set(timestep_record["student_gradient_branches"]) == {
        "generated_reconstruction",
        "consistency",
        "direct_measured",
    }
    assert set(timestep_record["student_gradient_cosines"]) == {
        "generated_reconstruction__consistency",
        "generated_reconstruction__direct_measured",
        "consistency__direct_measured",
    }
    assert timestep_record["predictions"]["teacher"]["finite"] is True
    assert timestep_record["predictions"]["student"]["finite"] is True
    assert timestep_record["predictions"]["residual"]["finite"] is True
    assert set(report["summaries"]["consistency_raw_mse"]) == {
        "p50",
        "p95",
        "p99",
        "max",
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_preflight_fails_closed_and_writes_no_report_for_nonfinite_model_output(
    tmp_path,
):
    records = [
        {"sample_id": "val-0", "split": "val"},
    ]
    manifest, checkpoint = _write_inputs(tmp_path, records)
    output = tmp_path / "report.json"

    with pytest.raises(FloatingPointError, match="student prediction"):
        run_preflight(
            PreflightConfig(
                manifest=manifest,
                checkpoint=checkpoint,
                output_json=output,
                sample_count=1,
                split="val",
                grid_size=2,
                device="cpu",
                solver_backend="pytorch_reference",
                direct_solver_directions=1,
                direct_solver_steps=1,
            ),
            runtime_factory=lambda checkpoint, device, grid: _runtime(
                nonfinite=True
            ),
            dataset_factory=lambda manifest, grid, latent_dim: _TinyDataset(1),
            simulator_factory=lambda checkpoint, device, grid, backend: (
                _TinySimulator()
            ),
        )

    assert not output.exists()


def test_cli_exposes_required_inputs_and_defaults():
    parser = build_argument_parser()
    args = parser.parse_args(
        [
            "--manifest",
            "manifest.jsonl",
            "--checkpoint",
            "model.pt",
            "--output-json",
            "report.json",
            "--grid-size",
            "96",
            "--device",
            "cpu",
            "--solver-backend",
            "pytorch_reference",
            "--direct-solver-directions",
            "16",
            "--direct-solver-steps",
            "5",
        ]
    )

    assert args.sample_count == 16
    assert args.split == "val"
    assert args.device == "cpu"
    assert args.solver_backend == "pytorch_reference"
    assert args.direct_solver_directions == 16
    assert args.direct_solver_steps == 5
