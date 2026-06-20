#!/usr/bin/env python3
"""Sequential black-box optimization for non-differentiable geometry scores.

Connectivity labels, aircraft-validity checks, and the internal CFD scorer cross
thresholding and non-PyTorch code. They cannot provide autograd gradients, but
they can still be used as measured objective terms in a sequential optimizer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.ndimage import label

from aircraft_validity import evaluate_aircraft_validity


@dataclass
class SequentialDiagnosticOptimizationConfig:
    """Configuration for measured-objective candidate optimization."""

    method: str = "genetic"
    threshold: float = 0.5
    population_size: int = 4
    generations: int = 2
    elite_count: int = 1
    mutation_rate: float = 0.08
    mutation_sigma: float = 0.20
    symmetry_blend: float = 0.25
    spsa_steps: int = 4
    spsa_perturbation: float = 0.18
    spsa_learning_rate: float = 0.04
    spsa_max_update: float = 0.20
    connectivity_weight: float = 50.0
    aerodynamic_weight: float = 1.0
    validity_weight: float = 10.0
    occupancy_weight: float = 0.0
    target_occupancy: float = 0.03
    enable_aerodynamic: bool = True
    cfd_steps: int = 100
    seed: Optional[int] = None

    def normalized(self) -> "SequentialDiagnosticOptimizationConfig":
        config = SequentialDiagnosticOptimizationConfig(**asdict(self))
        config.method = str(config.method).lower()
        if config.method not in {"genetic", "spsa"}:
            raise ValueError(f"Unsupported sequential optimizer method: {self.method}")
        config.threshold = float(np.clip(config.threshold, 0.0, 1.0))
        config.population_size = max(1, int(config.population_size))
        config.generations = max(0, int(config.generations))
        config.elite_count = max(1, min(int(config.elite_count), config.population_size))
        config.mutation_rate = float(np.clip(config.mutation_rate, 0.0, 1.0))
        config.mutation_sigma = max(0.0, float(config.mutation_sigma))
        config.symmetry_blend = float(np.clip(config.symmetry_blend, 0.0, 1.0))
        config.spsa_steps = max(0, int(config.spsa_steps))
        config.spsa_perturbation = max(1e-6, float(config.spsa_perturbation))
        config.spsa_learning_rate = max(0.0, float(config.spsa_learning_rate))
        config.spsa_max_update = max(0.0, float(config.spsa_max_update))
        config.connectivity_weight = max(0.0, float(config.connectivity_weight))
        config.aerodynamic_weight = max(0.0, float(config.aerodynamic_weight))
        config.validity_weight = max(0.0, float(config.validity_weight))
        config.occupancy_weight = max(0.0, float(config.occupancy_weight))
        config.target_occupancy = float(np.clip(config.target_occupancy, 0.0, 1.0))
        config.cfd_steps = max(1, int(config.cfd_steps))
        return config


def _to_3d_probability_grid(voxel_grid: torch.Tensor) -> torch.Tensor:
    grid = voxel_grid.detach().float()
    if grid.ndim == 5 and grid.shape[0] == 1:
        grid = grid.squeeze(0)
    if grid.ndim == 4:
        grid = grid.squeeze(0) if grid.shape[0] == 1 else grid.max(dim=0).values
    if grid.ndim != 3:
        raise ValueError(f"Expected a 3D or channel-first voxel grid, got shape {tuple(grid.shape)}")
    return grid.clamp(0.0, 1.0)


def _largest_component_fraction(binary: np.ndarray) -> float:
    labeled, num_components = label(binary)
    occupied = int(binary.sum())
    if occupied <= 0:
        return 0.0
    if num_components <= 1:
        return 1.0
    sizes = np.bincount(labeled.ravel())
    largest = int(sizes[1:].max()) if sizes.size > 1 else 0
    return float(largest) / float(max(occupied, 1))


def _select_drag_coefficient(cfd_results: Dict[str, Any]) -> float:
    for key in ("training_drag_coefficient", "calibrated_drag_coefficient", "drag_coefficient"):
        value = cfd_results.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)) and float(value) > 0.0:
            return float(value)
    return 0.1


def _validity_loss(validity: Dict[str, Any]) -> float:
    metrics = validity.get("metrics", {}) or {}
    checks = validity.get("checks", {}) or {}
    failed = [name for name, passed in checks.items() if not passed]

    def lower(name: str, target: float) -> float:
        return max(0.0, target - float(metrics.get(name, 0.0))) / max(target, 1e-6)

    def upper(name: str, target: float) -> float:
        return max(0.0, float(metrics.get(name, 0.0)) - target) / max(target, 1e-6)

    occupancy = float(metrics.get("occupancy_ratio", 0.0))
    loss = 0.0
    loss += max(0.0, 0.005 - occupancy) / 0.005
    loss += max(0.0, occupancy - 0.50) / 0.50
    loss += lower("symmetry_score", 0.55)
    loss += lower("span_fraction_y", 0.35)
    loss += lower("length_fraction_x", 0.35)
    loss += upper("thickness_fraction_z", 0.35)
    loss += lower("center_body_fraction", 0.10)
    loss += lower("left_wing_fraction", 0.05)
    loss += lower("right_wing_fraction", 0.05)
    loss += lower("center_body_density_ratio", 1.15)
    loss += lower("longitudinal_profile_cv", 0.18)
    loss += upper("tail_fraction", 0.20)
    loss += max(0.0, max(
        float(metrics.get("low_end_fraction", 0.0)),
        float(metrics.get("high_end_fraction", 0.0)),
    ) - 0.50) / 0.50
    loss += 0.25 * len(failed)
    return float(loss)


class SequentialDiagnosticOptimizer:
    """Optimize generated voxel candidates using true sequential score calls."""

    def __init__(
        self,
        cfd_simulator: Optional[Any] = None,
        config: Optional[SequentialDiagnosticOptimizationConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfd_simulator = cfd_simulator
        self.config = (config or SequentialDiagnosticOptimizationConfig()).normalized()
        self.device = device or torch.device("cpu")
        self._rng = torch.Generator()
        if self.config.seed is not None:
            self._rng.manual_seed(int(self.config.seed))

    def evaluate(self, candidate: torch.Tensor, design_spec: Any) -> Dict[str, Any]:
        grid = _to_3d_probability_grid(candidate).to(self.device)
        binary = (grid > self.config.threshold).detach().cpu().numpy().astype(np.uint8)
        occupancy = float(binary.mean()) if binary.size else 0.0
        connected_fraction = _largest_component_fraction(binary)
        connectivity_loss = 1.0 - connected_fraction

        validity = evaluate_aircraft_validity(binary.astype(np.float32))
        validity_loss = _validity_loss(validity)

        cfd_results: Dict[str, Any] = {}
        aerodynamic_loss = 0.0
        if (
            self.config.enable_aerodynamic
            and self.config.aerodynamic_weight > 0.0
            and self.cfd_simulator is not None
        ):
            geometry = torch.as_tensor(binary, device=self.device, dtype=torch.float32)
            cfd_results = self.cfd_simulator.simulate_aerodynamics(geometry, steps=self.config.cfd_steps)
            cd = _select_drag_coefficient(cfd_results)
            cl = abs(float(cfd_results.get("lift_coefficient", 0.0) or 0.0))
            space_weight = float(getattr(design_spec, "space_weight", 1.0))
            drag_weight = float(getattr(design_spec, "drag_weight", 1.0))
            lift_weight = float(getattr(design_spec, "lift_weight", 1.0))
            aerodynamic_loss = (
                space_weight * occupancy
                + drag_weight * cd
                + lift_weight * (1.0 - float(np.clip(cl, 0.0, 1.0)))
            )

        occupancy_loss = abs(occupancy - self.config.target_occupancy)
        total_loss = (
            self.config.connectivity_weight * connectivity_loss
            + self.config.aerodynamic_weight * aerodynamic_loss
            + self.config.validity_weight * validity_loss
            + self.config.occupancy_weight * occupancy_loss
        )

        return {
            "total_loss": float(total_loss),
            "terms": {
                "connectivity_loss": float(connectivity_loss),
                "aerodynamic_loss": float(aerodynamic_loss),
                "validity_loss": float(validity_loss),
                "occupancy_loss": float(occupancy_loss),
            },
            "metrics": {
                "occupancy_ratio": float(occupancy),
                "connected_fraction": float(connected_fraction),
            },
            "validity": validity,
            "cfd_metrics": cfd_results,
        }

    def optimize(self, voxel_grid: torch.Tensor, design_spec: Any) -> Dict[str, Any]:
        initial = _to_3d_probability_grid(voxel_grid).to(self.device)
        if self.config.method == "spsa":
            return self._optimize_spsa(initial, design_spec)
        return self._optimize_genetic(initial, design_spec)

    def _mutate(self, parent: torch.Tensor) -> torch.Tensor:
        rand = torch.rand(tuple(parent.shape), generator=self._rng, dtype=parent.dtype).to(parent.device)
        mask = rand < self.config.mutation_rate
        noise = torch.randn(tuple(parent.shape), generator=self._rng, dtype=parent.dtype).to(parent.device)
        child = parent + mask.float() * noise * self.config.mutation_sigma
        child = child.clamp(0.0, 1.0)
        return self._apply_symmetry_blend(child)

    def _apply_symmetry_blend(self, child: torch.Tensor) -> torch.Tensor:
        if self.config.symmetry_blend > 0.0 and child.ndim == 3:
            mirrored = torch.flip(child, dims=[1])
            symmetric = 0.5 * (child + mirrored)
            child = (
                (1.0 - self.config.symmetry_blend) * child
                + self.config.symmetry_blend * symmetric
            ).clamp(0.0, 1.0)
        return child

    def _record_candidate(
        self,
        generation: int,
        index: int,
        candidate: torch.Tensor,
        design_spec: Any,
    ) -> Dict[str, Any]:
        evaluation = self.evaluate(candidate, design_spec)
        evaluation["generation"] = int(generation)
        evaluation["candidate_index"] = int(index)
        return evaluation

    def _optimize_genetic(self, initial: torch.Tensor, design_spec: Any) -> Dict[str, Any]:
        population = [initial]
        for _ in range(self.config.population_size - 1):
            population.append(self._mutate(initial))

        history: List[Dict[str, Any]] = []
        best_candidate = initial
        best_eval: Optional[Dict[str, Any]] = None
        candidates_evaluated = 0

        for generation in range(self.config.generations + 1):
            evaluated = []
            for index, candidate in enumerate(population):
                record = self._record_candidate(generation, index, candidate, design_spec)
                candidates_evaluated += 1
                evaluated.append((record["total_loss"], candidate, record))

            evaluated.sort(key=lambda item: item[0])
            generation_records = [item[2] for item in evaluated]
            history.append({"generation": generation, "candidates": generation_records})

            if best_eval is None or evaluated[0][0] < float(best_eval["total_loss"]):
                best_candidate = evaluated[0][1].detach().clone()
                best_eval = dict(evaluated[0][2])

            if generation >= self.config.generations:
                break

            elites = [item[1].detach().clone() for item in evaluated[: self.config.elite_count]]
            next_population = list(elites)
            while len(next_population) < self.config.population_size:
                parent = elites[(len(next_population) - len(elites)) % len(elites)]
                next_population.append(self._mutate(parent))
            population = next_population

        initial_eval = history[0]["candidates"][0]
        assert best_eval is not None
        return {
            "method": "genetic",
            "config": asdict(self.config),
            "initial": initial_eval,
            "best": best_eval,
            "improvement": float(initial_eval["total_loss"] - best_eval["total_loss"]),
            "candidates_evaluated": candidates_evaluated,
            "history": history,
            "voxel_grid": best_candidate.detach(),
            "binary_grid": (best_candidate > self.config.threshold).float().detach(),
        }

    def _optimize_spsa(self, initial: torch.Tensor, design_spec: Any) -> Dict[str, Any]:
        current = initial.detach().clone()
        history: List[Dict[str, Any]] = []
        initial_eval = self._record_candidate(0, 0, current, design_spec)
        best_candidate = current.detach().clone()
        best_eval = dict(initial_eval)
        candidates_evaluated = 1

        for step in range(1, self.config.spsa_steps + 1):
            raw = torch.randint(
                0,
                2,
                tuple(current.shape),
                generator=self._rng,
                dtype=torch.int64,
            ).to(current.device)
            delta = raw.float().mul_(2.0).sub_(1.0)
            plus = (current + self.config.spsa_perturbation * delta).clamp(0.0, 1.0)
            minus = (current - self.config.spsa_perturbation * delta).clamp(0.0, 1.0)

            plus_eval = self._record_candidate(step, 0, plus, design_spec)
            minus_eval = self._record_candidate(step, 1, minus, design_spec)
            candidates_evaluated += 2

            gradient_scale = (
                (float(plus_eval["total_loss"]) - float(minus_eval["total_loss"]))
                / (2.0 * self.config.spsa_perturbation)
            )
            update = self.config.spsa_learning_rate * gradient_scale * delta
            if self.config.spsa_max_update > 0.0:
                update = update.clamp(-self.config.spsa_max_update, self.config.spsa_max_update)
            current = (current - update).clamp(0.0, 1.0)
            if self.config.symmetry_blend > 0.0:
                current = self._apply_symmetry_blend(current)

            current_eval = self._record_candidate(step, 2, current, design_spec)
            candidates_evaluated += 1
            history.append(
                {
                    "generation": step,
                    "candidates": [plus_eval, minus_eval, current_eval],
                    "spsa_gradient_scale": float(gradient_scale),
                }
            )
            if float(current_eval["total_loss"]) < float(best_eval["total_loss"]):
                best_candidate = current.detach().clone()
                best_eval = dict(current_eval)

        return {
            "method": "spsa",
            "config": asdict(self.config),
            "initial": initial_eval,
            "best": best_eval,
            "improvement": float(initial_eval["total_loss"] - best_eval["total_loss"]),
            "candidates_evaluated": candidates_evaluated,
            "history": [{"generation": 0, "candidates": [initial_eval]}, *history],
            "voxel_grid": best_candidate.detach(),
            "binary_grid": (best_candidate > self.config.threshold).float().detach(),
        }
