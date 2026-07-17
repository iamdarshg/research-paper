#!/usr/bin/env python3
"""Read-only stability preflight for consistency and measured solver branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from aircraft_diffusion_cfd import (
    AdvancedCFDSimulator,
    AircraftDesignDataset,
    CFDConfig,
    ConsistencyModel,
    DiffusionConfig,
    DirectSolverSPSALoss,
    LBMPhysicsConfig,
    LatentTo3DConverter,
    ModelConfig,
    NoiseSchedule,
    TrainingConfig,
    bound_latent_to_corpus_support,
    config_value,
    deterministic_split_assignments,
    load_grounded_manifest_records,
    sparse_voxel_reconstruction_loss,
)
from multiobjective_gradients import (
    capture_gradients,
    gradient_cosine_similarity,
    gradient_l2_norm,
)


REPORT_SCHEMA_VERSION = 1
DIRECT_SCALAR_COMPONENT_FIELDS = (
    "total_loss",
    "aero_loss",
    "drag_coefficient",
    "drag_loss",
    "lift_coefficient",
    "lift_loss",
    "occupancy",
    "occupancy_loss",
    "connectivity_loss",
    "aircraft_validity_loss",
)
DIRECT_COMPONENT_PREFIXES = {
    "aero": "aero",
    "connectivity": "connectivity",
    "validity": "aircraft_validity",
}
@dataclass(frozen=True)
class PreflightConfig:
    manifest: Path
    checkpoint: Path
    output_json: Path
    sample_count: int = 16
    split: str = "val"
    grid_size: int = 96
    device: str = "auto"
    solver_backend: str = "fused_stream_bfl"
    direct_solver_directions: int = 16
    direct_solver_steps: int = 5


@dataclass
class DiagnosticRuntime:
    model_config: ModelConfig
    diffusion_config: DiffusionConfig
    training_config: TrainingConfig
    consistency_model: ConsistencyModel
    converter: LatentTo3DConverter
    noise_schedule: NoiseSchedule
    dtype: torch.dtype
    latent_value_min: float
    latent_value_max: float


class CountingSimulator:
    """Count actual solver invocations while preserving the simulator API."""

    def __init__(self, simulator: Any):
        self._simulator = simulator
        self.call_count = 0

    def simulate_aerodynamics(
        self,
        geometry: torch.Tensor,
        steps: int = 100,
    ) -> Dict[str, Any]:
        self.call_count += 1
        return self._simulator.simulate_aerodynamics(geometry, steps=steps)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._simulator, name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value sequence")
    array = np.asarray(values, dtype=np.float64)
    if not bool(np.isfinite(array).all()):
        raise FloatingPointError("Summary input contains nonfinite values")
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def exact_inference_timesteps(
    diffusion_timesteps: int,
    inference_steps: int,
) -> list[int]:
    if int(diffusion_timesteps) <= 0 or int(inference_steps) <= 0:
        raise ValueError("Diffusion and inference step counts must be positive")
    return (
        torch.linspace(
            int(diffusion_timesteps) - 1,
            0,
            steps=int(inference_steps),
        )
        .round()
        .long()
        .tolist()
    )


def select_split_indices(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    sample_count: int,
) -> list[int]:
    """Select the first fixed records in a reproducible split assignment."""

    if int(sample_count) <= 0:
        raise ValueError("sample_count must be greater than zero")
    requested_split = str(split).strip()
    if not requested_split:
        raise ValueError("split must not be empty")

    fallback = deterministic_split_assignments(len(records), seed=0)
    assignments = [
        str(record.get("split", fallback[index]))
        for index, record in enumerate(records)
    ]
    candidates = [
        index
        for index, assignment in enumerate(assignments)
        if assignment == requested_split
    ]
    if len(candidates) < int(sample_count):
        raise ValueError(
            f"Split {requested_split!r} contains {len(candidates)} records, "
            f"but {sample_count} were requested"
        )
    return candidates[: int(sample_count)]


def _dataclass_from_payload(cls: type, payload: Optional[Mapping[str, Any]]) -> Any:
    allowed = {field.name for field in fields(cls)}
    values = {
        key: value
        for key, value in dict(payload or {}).items()
        if key in allowed
    }
    return cls(**values)


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    dtypes = {
        "float64": torch.float64,
        "double": torch.float64,
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtypes.get(str(precision), torch.float32)
    if device.type == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def resolve_device(requested: str) -> torch.device:
    value = str(requested).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but is unavailable")
    return device


def load_checkpoint_read_only(path: Path, device: torch.device) -> Mapping[str, Any]:
    del device
    with path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint root must be a mapping")
    return checkpoint


def _iter_checkpoint_tensors(
    value: Any,
    path: str = "checkpoint",
) -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_checkpoint_tensors(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_checkpoint_tensors(child, f"{path}[{index}]")


def checkpoint_fingerprint(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    tensor_count = 0
    tensor_element_count = 0
    nonfinite_tensors: list[str] = []
    for tensor_path, tensor in _iter_checkpoint_tensors(checkpoint):
        tensor_count += 1
        tensor_element_count += int(tensor.numel())
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all().item()):
                nonfinite_tensors.append(tensor_path)

    state_element_counts: Dict[str, int] = {}
    for state_name in (
        "diffusion_model",
        "consistency_model",
        "converter",
        "ema_model",
    ):
        state = checkpoint.get(state_name)
        if isinstance(state, Mapping):
            state_element_counts[state_name] = sum(
                int(value.numel())
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )

    optimizer_groups = []
    optimizer = checkpoint.get("optimizer")
    if isinstance(optimizer, Mapping):
        for index, group in enumerate(optimizer.get("param_groups", [])):
            if not isinstance(group, Mapping):
                continue
            learning_rate = float(group.get("lr", 0.0))
            _require_finite_scalar(
                f"checkpoint.optimizer.param_groups[{index}].lr",
                learning_rate,
            )
            optimizer_groups.append(
                {
                    "name": str(group.get("name", f"group_{index}")),
                    "learning_rate": learning_rate,
                }
            )

    training_payload = checkpoint.get("training_config")
    configured_learning_rates: Dict[str, float] = {}
    if isinstance(training_payload, Mapping):
        for name in (
            "learning_rate",
            "converter_learning_rate",
            "consistency_student_learning_rate",
        ):
            if name in training_payload:
                value = float(training_payload[name])
                _require_finite_scalar(f"checkpoint.training_config.{name}", value)
                configured_learning_rates[name] = value

    return {
        "tensor_finiteness": {
            "all_finite": not nonfinite_tensors,
            "tensor_count": tensor_count,
            "tensor_element_count": tensor_element_count,
            "nonfinite_tensor_paths": nonfinite_tensors,
        },
        "state_element_counts": state_element_counts,
        "global_step": int(checkpoint.get("global_step", 0)),
        "optimizer_groups": optimizer_groups,
        "configured_learning_rates": configured_learning_rates,
    }


def build_runtime(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    grid_size: int,
) -> DiagnosticRuntime:
    model_config = _dataclass_from_payload(
        ModelConfig,
        checkpoint.get("model_config"),
    )
    if int(model_config.grid_resolution) != int(grid_size):
        raise ValueError(
            "Checkpoint grid resolution does not match --grid-size: "
            f"{model_config.grid_resolution} != {grid_size}"
        )
    diffusion_config = _dataclass_from_payload(
        DiffusionConfig,
        checkpoint.get("diffusion_config"),
    )
    training_config = _dataclass_from_payload(
        TrainingConfig,
        checkpoint.get("training_config"),
    )
    dtype = _precision_dtype(training_config.precision, device)

    consistency_model = ConsistencyModel(
        model_config,
        diffusion_config,
        dtype,
    ).to(device=device, dtype=dtype)
    converter = LatentTo3DConverter(
        model_config.latent_dim,
        model_config.grid_resolution,
        coordinate_decoder_threshold=training_config.coordinate_decoder_threshold,
        coordinate_chunk_size=model_config.coordinate_chunk_size,
        coordinate_decoder_width=model_config.coordinate_decoder_width,
        coordinate_decoder_depth=model_config.coordinate_decoder_depth,
        coordinate_fourier_bands=model_config.coordinate_fourier_bands,
        enable_coordinate_gradient_checkpointing=bool(
            config_value("model", "coordinate_gradient_checkpointing", True)
        ),
    ).to(device=device, dtype=dtype)

    consistency_model.load_state_dict(checkpoint["consistency_model"])
    converter.load_state_dict(checkpoint["converter"])
    ema_state = checkpoint.get("ema_model")
    if isinstance(ema_state, Mapping):
        consistency_model.teacher_model.load_state_dict(ema_state)

    consistency_model.eval()
    consistency_model.student_model.train()
    converter.train()
    for parameter in consistency_model.teacher_model.parameters():
        parameter.requires_grad_(False)

    noise_schedule = consistency_model.noise_schedule.to(device, dtype)
    return DiagnosticRuntime(
        model_config=model_config,
        diffusion_config=diffusion_config,
        training_config=training_config,
        consistency_model=consistency_model,
        converter=converter,
        noise_schedule=noise_schedule,
        dtype=dtype,
        latent_value_min=float(consistency_model.latent_value_min),
        latent_value_max=float(consistency_model.latent_value_max),
    )


def build_dataset(
    manifest_path: Path,
    grid_size: int,
    latent_dim: int,
) -> AircraftDesignDataset:
    dataset = AircraftDesignDataset(
        num_samples=0,
        grid_size=int(grid_size),
        seed=0,
        latent_dim=int(latent_dim),
        manifest_path=str(manifest_path),
    )
    if int(dataset.grid_size) != int(grid_size):
        raise ValueError(
            "Manifest grid resolution does not match --grid-size: "
            f"{dataset.grid_size} != {grid_size}"
        )
    return dataset


def build_simulator(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    grid_size: int,
    solver_backend: str,
) -> AdvancedCFDSimulator:
    cfd_payload = dict(checkpoint.get("cfd_config") or {})
    lbm_payload = cfd_payload.get("lbm_config")
    if isinstance(lbm_payload, Mapping):
        cfd_payload["lbm_config"] = _dataclass_from_payload(
            LBMPhysicsConfig,
            lbm_payload,
        )
    cfd_payload["base_grid_resolution"] = int(grid_size)
    cfd_payload["resolution"] = int(grid_size)
    cfd_payload["solver_type"] = "D3Q27"
    cfd_payload["use_amr"] = False
    cfd_payload["enable_external_validation"] = False
    cfd_payload["use_fused_stream_bfl"] = (
        solver_backend == "fused_stream_bfl"
    )
    cfd_config = _dataclass_from_payload(CFDConfig, cfd_payload)
    return AdvancedCFDSimulator(cfd_config, device)


def build_direct_solver_loss(
    training_config: TrainingConfig,
    directions: int,
    steps: int,
) -> DirectSolverSPSALoss:
    return DirectSolverSPSALoss(
        cfd_steps=int(steps),
        perturbation=float(training_config.direct_solver_perturbation),
        perturbation_grid_size=int(
            training_config.direct_solver_perturbation_grid_size
        ),
        gradient_clip=float(training_config.direct_solver_gradient_clip),
        aero_gradient_max_norm=float(
            training_config.direct_aero_gradient_max_norm
        ),
        connectivity_gradient_max_norm=float(
            training_config.direct_connectivity_gradient_max_norm
        ),
        validity_gradient_max_norm=float(
            training_config.direct_validity_gradient_max_norm
        ),
        connectivity_weight=float(training_config.direct_connectivity_weight),
        aircraft_validity_weight=float(
            training_config.direct_aircraft_validity_weight
        ),
        threshold=0.5,
        target_occupancy=training_config.direct_solver_target_occupancy,
        directions=int(directions),
        seed=0,
        input_is_logits=True,
    )


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains nonfinite values")


def _require_finite_scalar(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise FloatingPointError(f"{name} is nonfinite")
    return numeric


def _tensor_metrics(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise FloatingPointError(f"{name} contains nonfinite values")
    value = tensor.detach().float()
    return {
        "rms": float(value.square().mean().sqrt().item()),
        "abs_max": float(value.abs().max().item()),
        "finite": True,
    }


def _clear_module_gradients(*modules: torch.nn.Module) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


def _stable_seed(
    manifest_sha256: str,
    manifest_index: int,
    timestep: int,
    purpose: str,
) -> int:
    payload = (
        f"{manifest_sha256}:{manifest_index}:{timestep}:{purpose}".encode("ascii")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (
        2**63 - 1
    )


def _deterministic_noise(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(
        tuple(shape),
        device=device,
        dtype=dtype,
        generator=generator,
    )


def _generated_latent(
    runtime: DiagnosticRuntime,
    noised_latent: torch.Tensor,
    timestep: torch.Tensor,
    condition: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    student_prediction = runtime.consistency_model.student_model(
        noised_latent,
        timestep,
        condition=condition,
    )
    _require_finite_tensor("student prediction", student_prediction)
    generated_latent = runtime.noise_schedule.predict_x0(
        noised_latent,
        timestep,
        student_prediction,
    )
    _require_finite_tensor("generated latent before bounding", generated_latent)
    generated_latent = bound_latent_to_corpus_support(
        generated_latent,
        runtime.latent_value_min,
        runtime.latent_value_max,
    )
    _require_finite_tensor("generated latent", generated_latent)
    return student_prediction, generated_latent


def _student_gradient_branch(
    parameters: Sequence[torch.nn.Parameter],
    branch_name: str,
) -> tuple[tuple[Optional[torch.Tensor], ...], float]:
    # The three completed branches are needed together only for scalar cosine
    # measurements. Park each detached snapshot in system memory so the active
    # 96^3 decoder or LBM phase owns the GPU working set.
    gradients = tuple(
        None
        if gradient is None
        else gradient.to(device="cpu", copy=True)
        for gradient in capture_gradients(parameters)
    )
    norm = gradient_l2_norm(gradients, branch_name=branch_name)
    return gradients, _require_finite_scalar(f"{branch_name} gradient norm", norm)


def _extract_direct_diagnostics(
    components: Mapping[str, Any],
) -> tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    scalar_components: Dict[str, float] = {}
    for field_name in DIRECT_SCALAR_COMPONENT_FIELDS:
        if field_name not in components:
            raise KeyError(f"Direct solver diagnostics are missing {field_name!r}")
        scalar_components[field_name] = _require_finite_scalar(
            f"direct solver {field_name}",
            components[field_name],
        )

    spsa_norms: Dict[str, Dict[str, float]] = {}
    for component_name, prefix in DIRECT_COMPONENT_PREFIXES.items():
        raw_field = f"{prefix}_spsa_gradient_norm_unclipped"
        applied_field = f"{prefix}_spsa_gradient_norm"
        if raw_field not in components or applied_field not in components:
            raise KeyError(
                "Direct solver diagnostics are missing component SPSA norms "
                f"for {component_name!r}"
            )
        spsa_norms[component_name] = {
            "raw": _require_finite_scalar(raw_field, components[raw_field]),
            "applied": _require_finite_scalar(
                applied_field,
                components[applied_field],
            ),
        }

    for field_name in (
        "spsa_gradient_norm_unclipped",
        "spsa_gradient_norm",
    ):
        if field_name not in components:
            raise KeyError(f"Direct solver diagnostics are missing {field_name!r}")
    spsa_norms["total"] = {
        "raw": _require_finite_scalar(
            "spsa_gradient_norm_unclipped",
            components["spsa_gradient_norm_unclipped"],
        ),
        "applied": _require_finite_scalar(
            "spsa_gradient_norm",
            components["spsa_gradient_norm"],
        ),
    }
    return scalar_components, spsa_norms


def _sample_identifier(
    record: Mapping[str, Any],
    manifest_index: int,
) -> str:
    for key in ("sample_id", "source_id", "id"):
        if record.get(key) is not None:
            return str(record[key])
    return str(manifest_index)


def _runtime_parameter_counts(runtime: DiagnosticRuntime) -> Dict[str, int]:
    teacher = runtime.consistency_model.teacher_model
    student = runtime.consistency_model.student_model
    return {
        "teacher": sum(int(parameter.numel()) for parameter in teacher.parameters()),
        "student": sum(int(parameter.numel()) for parameter in student.parameters()),
        "converter": sum(
            int(parameter.numel()) for parameter in runtime.converter.parameters()
        ),
    }


def _runtime_objective_configuration(
    runtime: DiagnosticRuntime,
) -> Dict[str, Any]:
    training = runtime.training_config
    positive_fields = (
        "consistency_huber_delta",
        "generation_reconstruction_weight",
        "minimum_denoising_geometry_confidence",
        "direct_solver_loss_weight",
        "direct_solver_perturbation",
        "direct_solver_gradient_clip",
        "direct_aero_gradient_max_norm",
        "direct_connectivity_gradient_max_norm",
        "direct_validity_gradient_max_norm",
        "direct_connectivity_weight",
        "direct_aircraft_validity_weight",
    )
    values: Dict[str, float] = {}
    for field_name in positive_fields:
        value = _require_finite_scalar(
            f"training configuration {field_name}",
            getattr(training, field_name),
        )
        if value <= 0.0:
            raise ValueError(f"{field_name} must be greater than zero")
        values[field_name] = value

    dice_weight = _require_finite_scalar(
        "training configuration geometry_dice_weight",
        training.geometry_dice_weight,
    )
    if dice_weight < 0.0:
        raise ValueError("geometry_dice_weight must be non-negative")
    target_occupancy = training.direct_solver_target_occupancy
    if target_occupancy is not None:
        target_occupancy = _require_finite_scalar(
            "training configuration direct_solver_target_occupancy",
            target_occupancy,
        )

    return {
        "consistency": {
            "robust_loss_type": "smooth_l1",
            "huber_delta": values["consistency_huber_delta"],
        },
        "generated_reconstruction": {
            "dice_weight": dice_weight,
            "loss_weight": values["generation_reconstruction_weight"],
            "minimum_denoising_geometry_confidence": values[
                "minimum_denoising_geometry_confidence"
            ],
        },
        "direct_solver": {
            "loss_weight": values["direct_solver_loss_weight"],
            "perturbation": values["direct_solver_perturbation"],
            "perturbation_grid_size": int(
                training.direct_solver_perturbation_grid_size
            ),
            "total_gradient_max_norm": values[
                "direct_solver_gradient_clip"
            ],
            "component_gradient_max_norms": {
                "aero": values["direct_aero_gradient_max_norm"],
                "connectivity": values[
                    "direct_connectivity_gradient_max_norm"
                ],
                "validity": values["direct_validity_gradient_max_norm"],
            },
            "connectivity_weight": values["direct_connectivity_weight"],
            "aircraft_validity_weight": values[
                "direct_aircraft_validity_weight"
            ],
            "target_occupancy": target_occupancy,
            "input_is_logits": True,
        },
    }


def _evaluate_timestep(
    *,
    runtime: DiagnosticRuntime,
    sample: Mapping[str, Any],
    timestep_value: int,
    manifest_sha256: str,
    manifest_index: int,
    simulator: CountingSimulator,
    direct_solver_loss: DirectSolverSPSALoss,
    direct_solver_directions: int,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    student_model = runtime.consistency_model.student_model
    teacher_model = runtime.consistency_model.teacher_model
    converter = runtime.converter
    student_parameters = tuple(student_model.parameters())

    latent = torch.as_tensor(
        sample["latent"],
        device=next(student_model.parameters()).device,
        dtype=runtime.dtype,
    ).reshape(1, -1)
    geometry = torch.as_tensor(
        sample["geometry"],
        device=latent.device,
        dtype=runtime.dtype,
    )
    if geometry.ndim == 3:
        geometry = geometry.unsqueeze(0)
    condition_value = sample.get("condition_vector")
    condition = (
        None
        if condition_value is None
        else torch.as_tensor(
            condition_value,
            device=latent.device,
            dtype=runtime.dtype,
        ).reshape(1, -1)
    )
    _require_finite_tensor("sample latent", latent)
    _require_finite_tensor("sample geometry", geometry)
    if condition is not None:
        _require_finite_tensor("sample condition", condition)

    timestep = torch.full(
        (1,),
        int(timestep_value),
        device=latent.device,
        dtype=torch.long,
    )
    noise = _deterministic_noise(
        latent.shape,
        device=latent.device,
        dtype=runtime.dtype,
        seed=_stable_seed(
            manifest_sha256,
            manifest_index,
            timestep_value,
            "noise",
        ),
    )
    noised_latent = runtime.noise_schedule.q_sample(latent, timestep, noise)
    _require_finite_tensor("noised latent", noised_latent)

    # Generated reconstruction branch. Its graph is released before consistency.
    _clear_module_gradients(student_model, converter)
    _, generated_latent = _generated_latent(
        runtime,
        noised_latent,
        timestep,
        condition,
    )
    generated_logits = converter(generated_latent)
    _require_finite_tensor("generated reconstruction logits", generated_logits)
    reconstruction_loss = sparse_voxel_reconstruction_loss(
        generated_logits.float(),
        geometry.float(),
        dice_weight=float(runtime.training_config.geometry_dice_weight),
    )
    _require_finite_tensor("generated reconstruction loss", reconstruction_loss)
    reconstruction_branch_weight = float(
        runtime.training_config.generation_reconstruction_weight
    )
    confidence_schedule = getattr(
        runtime.noise_schedule,
        "sqrt_alphas_cumprod",
        None,
    )
    if isinstance(confidence_schedule, torch.Tensor):
        reconstruction_branch_weight *= max(
            float(
                confidence_schedule[int(timestep_value)]
                .detach()
                .float()
                .item()
            ),
            float(
                runtime.training_config.minimum_denoising_geometry_confidence
            ),
        )
    _require_finite_scalar(
        "generated reconstruction branch weight",
        reconstruction_branch_weight,
    )
    weighted_reconstruction_loss = (
        reconstruction_branch_weight * reconstruction_loss
    )
    _require_finite_tensor(
        "weighted generated reconstruction loss",
        weighted_reconstruction_loss,
    )
    weighted_reconstruction_loss.backward()
    reconstruction_gradients, reconstruction_gradient_norm = (
        _student_gradient_branch(
            student_parameters,
            "generated_reconstruction",
        )
    )
    reconstruction_loss_value = float(reconstruction_loss.detach().item())
    del (
        weighted_reconstruction_loss,
        reconstruction_loss,
        generated_logits,
        generated_latent,
    )
    _clear_module_gradients(student_model, converter)

    # Consistency branch uses the same latent, noise realization, and timestep.
    with torch.no_grad():
        teacher_prediction = teacher_model(
            noised_latent,
            timestep,
            condition=condition,
        )
    student_prediction = student_model(
        noised_latent,
        timestep,
        condition=condition,
    )
    teacher_metrics = _tensor_metrics("teacher prediction", teacher_prediction)
    student_metrics = _tensor_metrics("student prediction", student_prediction)
    residual = student_prediction.float() - teacher_prediction.detach().float()
    residual_metrics = _tensor_metrics("consistency residual", residual)
    raw_mse = residual.square().mean()
    robust_loss = F.smooth_l1_loss(
        student_prediction.float(),
        teacher_prediction.detach().float(),
        beta=float(runtime.training_config.consistency_huber_delta),
    )
    _require_finite_tensor("raw consistency MSE", raw_mse)
    _require_finite_tensor("robust consistency loss", robust_loss)
    robust_loss.backward()
    consistency_gradients, consistency_gradient_norm = _student_gradient_branch(
        student_parameters,
        "consistency",
    )
    raw_mse_value = float(raw_mse.detach().item())
    robust_loss_value = float(robust_loss.detach().item())
    del (
        raw_mse,
        robust_loss,
        residual,
        student_prediction,
        teacher_prediction,
    )
    _clear_module_gradients(student_model, converter)

    # Create a detached generated-logit snapshot before the measured objective.
    with torch.no_grad():
        _, generated_latent_snapshot = _generated_latent(
            runtime,
            noised_latent,
            timestep,
            condition,
        )
        generated_logit_snapshot = converter(generated_latent_snapshot).detach()
    _require_finite_tensor("direct generated logits", generated_logit_snapshot)

    direct_logit_leaf = generated_logit_snapshot.float().requires_grad_(True)
    reference_occupancy = geometry.float().mean(
        dim=tuple(range(1, geometry.ndim))
    )
    calls_before = simulator.call_count
    direct_loss = direct_solver_loss(
        direct_logit_leaf,
        sample["design_spec"],
        simulator,
        seed=_stable_seed(
            manifest_sha256,
            manifest_index,
            timestep_value,
            "spsa",
        ),
        reference_occupancy=reference_occupancy,
    )
    _require_finite_tensor("direct measured solver loss", direct_loss)
    direct_loss.backward()
    direct_logit_gradient = direct_logit_leaf.grad
    if direct_logit_gradient is None:
        raise RuntimeError("Direct solver objective did not produce a logit gradient")
    direct_logit_gradient = direct_logit_gradient.detach()
    _require_finite_tensor("direct measured logit gradient", direct_logit_gradient)
    direct_loss_value = float(direct_loss.detach().item())
    scalar_components, spsa_norms = _extract_direct_diagnostics(
        dict(direct_solver_loss.last_components)
    )
    actual_calls = int(simulator.call_count - calls_before)
    expected_calls = 1 + 2 * int(direct_solver_directions)
    if actual_calls != expected_calls:
        raise RuntimeError(
            "Direct measured solver call count mismatch: "
            f"{actual_calls} != {expected_calls}"
        )
    del (
        direct_loss,
        direct_logit_leaf,
        generated_logit_snapshot,
        generated_latent_snapshot,
    )
    _clear_module_gradients(student_model, converter)

    # Recompute the neural path and inject the measured logit-space gradient.
    _, direct_generated_latent = _generated_latent(
        runtime,
        noised_latent,
        timestep,
        condition,
    )
    direct_generated_logits = converter(direct_generated_latent)
    _require_finite_tensor(
        "recomputed direct generated logits",
        direct_generated_logits,
    )
    direct_generated_logits.backward(
        gradient=(
            float(runtime.training_config.direct_solver_loss_weight)
            * direct_logit_gradient.to(
                device=direct_generated_logits.device,
                dtype=direct_generated_logits.dtype,
            )
        )
    )
    direct_gradients, direct_gradient_norm = _student_gradient_branch(
        student_parameters,
        "direct_measured",
    )
    del direct_generated_logits, direct_generated_latent, direct_logit_gradient
    _clear_module_gradients(student_model, converter)

    cosine_similarities = {
        "generated_reconstruction__consistency": gradient_cosine_similarity(
            reconstruction_gradients,
            consistency_gradients,
            first_name="generated_reconstruction",
            second_name="consistency",
        ),
        "generated_reconstruction__direct_measured": gradient_cosine_similarity(
            reconstruction_gradients,
            direct_gradients,
            first_name="generated_reconstruction",
            second_name="direct_measured",
        ),
        "consistency__direct_measured": gradient_cosine_similarity(
            consistency_gradients,
            direct_gradients,
            first_name="consistency",
            second_name="direct_measured",
        ),
    }
    for name, value in cosine_similarities.items():
        _require_finite_scalar(f"student gradient cosine {name}", value)

    record = {
        "timestep": int(timestep_value),
        "predictions": {
            "teacher": teacher_metrics,
            "student": student_metrics,
            "residual": residual_metrics,
        },
        "consistency": {
            "raw_mse": raw_mse_value,
            "robust_loss": robust_loss_value,
            "robust_loss_type": "smooth_l1",
            "huber_delta": float(
                runtime.training_config.consistency_huber_delta
            ),
            "student_gradient_norm": consistency_gradient_norm,
        },
        "generated_reconstruction": {
            "loss": reconstruction_loss_value,
            "optimization_weight": reconstruction_branch_weight,
            "student_gradient_norm": reconstruction_gradient_norm,
        },
        "direct_solver": {
            "loss": direct_loss_value,
            "component_scalars": scalar_components,
            "spsa_gradient_norms": spsa_norms,
            "solver_call_count": actual_calls,
            "expected_solver_call_count": expected_calls,
        },
        "student_gradient_branches": {
            "generated_reconstruction": reconstruction_gradient_norm,
            "consistency": consistency_gradient_norm,
            "direct_measured": direct_gradient_norm,
        },
        "student_gradient_cosines": cosine_similarities,
    }

    summary_values = {
        "teacher_prediction_rms": teacher_metrics["rms"],
        "teacher_prediction_abs_max": teacher_metrics["abs_max"],
        "student_prediction_rms": student_metrics["rms"],
        "student_prediction_abs_max": student_metrics["abs_max"],
        "consistency_residual_rms": residual_metrics["rms"],
        "consistency_residual_abs_max": residual_metrics["abs_max"],
        "consistency_raw_mse": raw_mse_value,
        "consistency_robust_loss": robust_loss_value,
        "generated_reconstruction_loss": reconstruction_loss_value,
        "generated_reconstruction_optimization_weight": (
            reconstruction_branch_weight
        ),
        "student_generated_reconstruction_gradient_norm": (
            reconstruction_gradient_norm
        ),
        "student_consistency_gradient_norm": consistency_gradient_norm,
        "student_direct_measured_gradient_norm": direct_gradient_norm,
    }
    for field_name, value in scalar_components.items():
        summary_values[f"direct_{field_name}"] = value
    for component_name, norms in spsa_norms.items():
        for norm_name, value in norms.items():
            summary_values[
                f"direct_{component_name}_spsa_gradient_norm_{norm_name}"
            ] = value
    for pair_name, value in cosine_similarities.items():
        summary_values[f"student_gradient_cosine_{pair_name}"] = value
    return record, summary_values


def _validate_json_finiteness(value: Any, path: str = "report") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, np.integer)):
        return
    if isinstance(value, (float, np.floating)):
        _require_finite_scalar(path, float(value))
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_json_finiteness(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_finiteness(child, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                report,
                handle,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run_preflight(
    config: PreflightConfig,
    *,
    checkpoint_loader: Optional[
        Callable[[Path, torch.device], Mapping[str, Any]]
    ] = None,
    runtime_factory: Optional[
        Callable[[Mapping[str, Any], torch.device, int], DiagnosticRuntime]
    ] = None,
    dataset_factory: Optional[
        Callable[[Path, int, int], AircraftDesignDataset]
    ] = None,
    simulator_factory: Optional[
        Callable[[Mapping[str, Any], torch.device, int, str], Any]
    ] = None,
    direct_loss_factory: Optional[
        Callable[[TrainingConfig, int, int], DirectSolverSPSALoss]
    ] = None,
) -> Dict[str, Any]:
    if int(config.sample_count) <= 0:
        raise ValueError("sample_count must be greater than zero")
    if int(config.grid_size) <= 0:
        raise ValueError("grid_size must be greater than zero")
    if int(config.direct_solver_directions) <= 0:
        raise ValueError("direct_solver_directions must be greater than zero")
    if int(config.direct_solver_steps) <= 0:
        raise ValueError("direct_solver_steps must be greater than zero")
    if config.solver_backend not in {"pytorch_reference", "fused_stream_bfl"}:
        raise ValueError(f"Unsupported solver backend: {config.solver_backend}")

    manifest_path = Path(config.manifest).resolve()
    checkpoint_path = Path(config.checkpoint).resolve()
    output_path = Path(config.output_json).resolve()
    for input_name, input_path in (
        ("manifest", manifest_path),
        ("checkpoint", checkpoint_path),
    ):
        if not input_path.is_file():
            raise FileNotFoundError(f"{input_name} not found: {input_path}")
    if output_path in {manifest_path, checkpoint_path}:
        raise ValueError("Output JSON must not overwrite the manifest or checkpoint")

    manifest_hash_before = sha256_file(manifest_path)
    checkpoint_hash_before = sha256_file(checkpoint_path)
    device = resolve_device(config.device)
    load_checkpoint = checkpoint_loader or load_checkpoint_read_only
    make_runtime = runtime_factory or build_runtime
    make_dataset = dataset_factory or build_dataset
    make_simulator = simulator_factory or build_simulator
    make_direct_loss = direct_loss_factory or build_direct_solver_loss

    try:
        checkpoint = load_checkpoint(checkpoint_path, device)
        checkpoint_metadata = checkpoint_fingerprint(checkpoint)
        if not checkpoint_metadata["tensor_finiteness"]["all_finite"]:
            paths = checkpoint_metadata["tensor_finiteness"][
                "nonfinite_tensor_paths"
            ]
            raise FloatingPointError(
                "Checkpoint contains nonfinite tensors: " + ", ".join(paths)
            )

        records = load_grounded_manifest_records(str(manifest_path))
        selected_indices = select_split_indices(
            records,
            split=config.split,
            sample_count=config.sample_count,
        )
        runtime = make_runtime(checkpoint, device, int(config.grid_size))
        objective_configuration = _runtime_objective_configuration(runtime)
        dataset = make_dataset(
            manifest_path,
            int(config.grid_size),
            int(runtime.model_config.latent_dim),
        )
        if len(dataset) != len(records):
            raise ValueError(
                "Manifest record count does not match loaded dataset length: "
                f"{len(records)} != {len(dataset)}"
            )

        simulator = CountingSimulator(
            make_simulator(
                checkpoint,
                device,
                int(config.grid_size),
                config.solver_backend,
            )
        )
        direct_solver_loss = make_direct_loss(
            runtime.training_config,
            int(config.direct_solver_directions),
            int(config.direct_solver_steps),
        )
        timesteps = exact_inference_timesteps(
            int(runtime.diffusion_config.timesteps),
            int(runtime.diffusion_config.student_steps),
        )
        if len(timesteps) != 4:
            raise ValueError(
                "The checkpoint does not define the required four-step "
                f"inference path: {timesteps}"
            )

        summary_accumulator: Dict[str, list[float]] = {}
        sample_records = []
        for manifest_index in selected_indices:
            sample = dataset[manifest_index]
            sample_record = {
                "manifest_index": int(manifest_index),
                "sample_id": _sample_identifier(
                    records[manifest_index],
                    manifest_index,
                ),
                "split": str(config.split),
                "timesteps": [],
            }
            for timestep_value in timesteps:
                print(
                    "Preflight "
                    f"sample={len(sample_records) + 1}/{len(selected_indices)} "
                    f"id={sample_record['sample_id']} "
                    f"timestep={int(timestep_value)}",
                    flush=True,
                )
                timestep_record, summary_values = _evaluate_timestep(
                    runtime=runtime,
                    sample=sample,
                    timestep_value=int(timestep_value),
                    manifest_sha256=manifest_hash_before,
                    manifest_index=int(manifest_index),
                    simulator=simulator,
                    direct_solver_loss=direct_solver_loss,
                    direct_solver_directions=int(
                        config.direct_solver_directions
                    ),
                )
                sample_record["timesteps"].append(timestep_record)
                for name, value in summary_values.items():
                    summary_accumulator.setdefault(name, []).append(
                        _require_finite_scalar(name, value)
                    )
            sample_records.append(sample_record)

        summaries = {
            name: summarize_values(values)
            for name, values in sorted(summary_accumulator.items())
        }
        expected_solver_calls = (
            int(config.sample_count)
            * len(timesteps)
            * (1 + 2 * int(config.direct_solver_directions))
        )
        if simulator.call_count != expected_solver_calls:
            raise RuntimeError(
                "Total direct measured solver call count mismatch: "
                f"{simulator.call_count} != {expected_solver_calls}"
            )

        component_nonzero = {
            component_name: any(
                timestep_record["direct_solver"]["spsa_gradient_norms"][
                    component_name
                ]["raw"]
                > 0.0
                for sample_record in sample_records
                for timestep_record in sample_record["timesteps"]
            )
            for component_name in DIRECT_COMPONENT_PREFIXES
        }
        component_retained = {
            component_name: all(
                norms["raw"] == 0.0 or norms["applied"] > 0.0
                for sample_record in sample_records
                for timestep_record in sample_record["timesteps"]
                for norms in (
                    timestep_record["direct_solver"]["spsa_gradient_norms"][
                        component_name
                    ],
                )
            )
            for component_name in DIRECT_COMPONENT_PREFIXES
        }
        failed_checks = [
            f"nonzero_{component_name}_spsa_gradient"
            for component_name, nonzero in component_nonzero.items()
            if not nonzero
        ]
        failed_checks.extend(
            f"retained_{component_name}_spsa_gradient"
            for component_name, retained in component_retained.items()
            if not retained
        )

        report: Dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "pass" if not failed_checks else "fail",
            "failed_checks": failed_checks,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": checkpoint_hash_before,
            "manifest_sha256": manifest_hash_before,
            "inputs": {
                "manifest": str(manifest_path),
                "checkpoint": str(checkpoint_path),
                "output_json": str(output_path),
                "sample_count": int(config.sample_count),
                "split": str(config.split),
                "grid_size": int(config.grid_size),
                "device": str(device),
                "solver_backend": str(config.solver_backend),
                "direct_solver_directions": int(
                    config.direct_solver_directions
                ),
                "direct_solver_steps": int(config.direct_solver_steps),
            },
            "checkpoint": {
                **checkpoint_metadata,
                "parameter_counts": _runtime_parameter_counts(runtime),
                "read_only": True,
                "unchanged_after_preflight": True,
            },
            "objective_configuration": objective_configuration,
            "selection": {
                "strategy": "manifest_order_with_explicit_or_seed0_split",
                "manifest_indices": [int(index) for index in selected_indices],
                "sample_ids": [
                    _sample_identifier(records[index], index)
                    for index in selected_indices
                ],
            },
            "inference_timesteps": [int(value) for value in timesteps],
            "solver_calls": {
                "actual": int(simulator.call_count),
                "expected": int(expected_solver_calls),
                "per_sample_timestep": (
                    1 + 2 * int(config.direct_solver_directions)
                ),
            },
            "component_nonzero": component_nonzero,
            "component_retained": component_retained,
            "samples": sample_records,
            "summaries": summaries,
        }
        _validate_json_finiteness(report)
    finally:
        manifest_hash_after = sha256_file(manifest_path)
        checkpoint_hash_after = sha256_file(checkpoint_path)
        if checkpoint_hash_after != checkpoint_hash_before:
            raise RuntimeError("Checkpoint bytes changed during read-only preflight")
        if manifest_hash_after != manifest_hash_before:
            raise RuntimeError("Manifest bytes changed during read-only preflight")

    _write_json_atomic(output_path, report)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure consistency, reconstruction, and real direct-solver "
            "branch stability without modifying the source checkpoint."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        required=True,
    )
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--split", default="val")
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--solver-backend",
        "--lbm-stream-bfl-backend",
        dest="solver_backend",
        choices=("pytorch_reference", "fused_stream_bfl"),
        required=True,
    )
    parser.add_argument(
        "--direct-solver-directions",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--direct-solver-steps",
        type=int,
        required=True,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = PreflightConfig(
        manifest=Path(args.manifest),
        checkpoint=Path(args.checkpoint),
        output_json=Path(args.output_json),
        sample_count=args.sample_count,
        split=args.split,
        grid_size=args.grid_size,
        device=args.device,
        solver_backend=args.solver_backend,
        direct_solver_directions=args.direct_solver_directions,
        direct_solver_steps=args.direct_solver_steps,
    )
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        report = run_preflight(config)
    except Exception as exc:
        print(f"training branch preflight failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Wrote {report['status']} training branch preflight to "
        f"{Path(args.output_json).resolve()}"
    )
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
