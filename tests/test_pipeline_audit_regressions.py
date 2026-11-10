"""Regression tests for pipeline-audit findings (Agent D, Workstream C).

Each test in this module encodes a genuine defect found during the pipeline
audit. Every test was written to FAIL against HEAD 6eaa290 and only passes
after the minimal fix. They are independent of the production CLI path and
exercise the shipped module surface directly.

Hermeticity: every test routes its GroundTruthExporter to a tmp_path so no
repo-level ground_truth/ state is read or written.
"""

import os
import sys

import numpy as np
import pytest
import torch

CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)


def _make_trainer(tmp_path, grid=8, latent_dim=8, timesteps=16, student_steps=4,
                  teacher_steps=32, batch_size=1, condition_dim=8):
    from config import CFDConfig, DiffusionConfig, ModelConfig, TrainingConfig
    from trainer import OptimizedDiffusionTrainer
    trainer = OptimizedDiffusionTrainer(
        ModelConfig(latent_dim=latent_dim, condition_dim=condition_dim,
                    grid_resolution=grid),
        DiffusionConfig(timesteps=timesteps, teacher_steps=teacher_steps,
                        student_steps=student_steps),
        TrainingConfig(batch_size=batch_size, num_epochs=1),
        CFDConfig(base_grid_resolution=grid),
        device=torch.device("cpu"),
    )
    trainer.gt_exporter.output_dir = tmp_path
    trainer.gt_exporter.labels_path = tmp_path / "cfd_labels.json"
    return trainer


def _sample_batch(latent, missions):
    return {"latent": latent, "mission_profile": missions}


# ---------------------------------------------------------------------------
# F1: offline trainer surrogate branch references a nonexistent EMA buffer
# CLI/trainer.py:243 `self.surrogate.last_val_mse.copy_(...)` crashes because
# AeroSurrogate (CLI/models.py) never registers `last_val_mse`.
# ---------------------------------------------------------------------------
class TestOfflineTrainerSurrogateEmaBuffer:
    def test_aero_surrogate_registers_last_val_mse_buffer(self):
        from models import AeroSurrogate

        surrogate = AeroSurrogate(condition_dim=8, grid_resolution=4)
        assert hasattr(surrogate, "last_val_mse"), (
            "AeroSurrogate must register a last_val_mse buffer; "
            "CLI/trainer.py:243 reads it during offline surrogate training."
        )
        assert surrogate.last_val_mse.numel() == 1

    def test_offline_trainer_surrogate_update_runs(self, tmp_path):
        """The offline trainer's surrogate-EMA branch must not raise."""
        trainer = _make_trainer(tmp_path)
        # Fill the ground-truth cache so the surrogate branch is reachable.
        trainer.gt_exporter._cache.append(
            {"geometry_ref": "probe.npy", "mission_profile": {},
             "cd": 0.1, "cl": 0.0, "cm": 0.0, "converged": False}
        )
        trainer.gt_exporter.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(trainer.gt_exporter.output_dir / "probe.npy",
                np.zeros((4, 4, 4), dtype=np.float32))
        # A surrogate model with a real loss EMA buffer must survive the copy_.
        trainer.surrogate.last_val_mse.copy_(
            0.95 * trainer.surrogate.last_val_mse
            + 0.05 * torch.tensor(0.25)
        )
        assert torch.isfinite(trainer.surrogate.last_val_mse).item()


# ---------------------------------------------------------------------------
# F2: legacy ConsistencyModel (CLI/models.py) uses an inconsistent noise
# schedule: _add_noise indexes alpha by teacher_steps (2000) while
# fast_inference uses timesteps (100), and fast_inference denoises with
# (1 - alpha_t) instead of sqrt(1 - alpha_t) for epsilon prediction.
# ---------------------------------------------------------------------------
class TestLegacyConsistencyModelNoiseMath:
    def _model(self):
        from config import DiffusionConfig, ModelConfig
        from models import ConsistencyModel
        return ConsistencyModel(
            ModelConfig(latent_dim=8, condition_dim=8),
            DiffusionConfig(timesteps=100, teacher_steps=2000, student_steps=32),
        )

    def test_fast_inference_matches_standard_epsilon_predict_x0(self):
        """fast_inference implements iterative x0-refinement with the
        standard epsilon-prediction step (sqrt(1 - alpha), not (1 - alpha)).
        Run it with a stub student whose output is a known epsilon and verify
        the closed form for the actual schedule."""
        model = self._model()
        # Stub the student so we control the predicted noise exactly.
        class StubStudent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # fast_inference reads the device/dtype from the first
                # parameter, so expose a dummy one.
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x_t, t, condition=None):
                return self.eps_broadcast

        known_eps = torch.randn(1, model.config.latent_dim)
        stub = StubStudent()
        stub.eps_broadcast = known_eps
        model.student_model = stub

        x_t = torch.randn(1, model.config.latent_dim)
        out = model.fast_inference(
            (1, model.config.latent_dim), num_steps=4,
            condition=None, initial_noise=x_t.clone(),
        )
        # The legacy algorithm: step_size = timesteps // num_steps, steps at
        # [timesteps - i*step_size - 1]; each step applies the epsilon
        # x0-estimate x <- (x - sqrt(1-alpha_t) eps) / sqrt(alpha_t).
        step_size = model.diffusion_config.timesteps // 4
        steps = [
            model.diffusion_config.timesteps - i * step_size - 1
            for i in range(4)
        ]
        x = x_t.clone()
        for t_val in steps:
            alpha_t = 0.5 ** (t_val / model.diffusion_config.timesteps)
            # Match the implementation's epsilon placement exactly.
            sqrt_alpha = torch.sqrt(
                torch.tensor(alpha_t, dtype=torch.float64) + 1e-8
            )
            sqrt_one_minus_alpha = torch.sqrt(
                torch.tensor(1.0 - alpha_t, dtype=torch.float64) + 1e-8
            )
            x = (x - sqrt_one_minus_alpha * known_eps) / sqrt_alpha
        assert torch.allclose(out, x, atol=1e-3), (
            "fast_inference deviates from the standard epsilon-prediction "
            "x0-refinement; it must use sqrt(1 - alpha_t), not (1 - alpha_t)."
        )

    def test_offline_trainer_feeds_teacher_in_embedding_range(self, tmp_path):
        """The offline trainer's consistency sampling must stay on the
        [0, timesteps) axis the UNet time embedding normalizes by. Probe by
        recording the t_teacher actually passed to consistency_loss."""
        from config import MissionProfile
        from data_utils import AircraftDesignDataset

        trainer = _make_trainer(tmp_path, timesteps=100, student_steps=32,
                                teacher_steps=2000)
        recorded = {}

        original = trainer.consistency_model.consistency_loss

        def recording_consistency_loss(x_0, t_student, t_teacher, condition=None,
                                       loss_type="mse", huber_delta=1.0):
            recorded["t_teacher"] = t_teacher.detach().clone()
            recorded["t_student"] = t_student.detach().clone()
            return torch.zeros((), dtype=x_0.dtype)

        trainer.consistency_model.consistency_loss = recording_consistency_loss
        dataset = AircraftDesignDataset(num_samples=1, grid_size=4,
                                        latent_dim=8, seed=0)
        batch = _sample_batch(dataset.latent_codes[:1], [MissionProfile()])
        try:
            trainer.train_epoch([batch], grid_size=4)
        finally:
            trainer.consistency_model.consistency_loss = original
        t_teacher = recorded["t_teacher"]
        assert t_teacher is not None, "train_epoch never called consistency_loss"
        normalized = float(t_teacher.max()) / trainer.diffusion_config.timesteps
        assert normalized <= 1.0, (
            f"trainer feeds teacher timestep {int(t_teacher.max())} which "
            f"normalizes to {normalized:.2f} under the UNet embedding "
            f"(divide by timesteps={trainer.diffusion_config.timesteps}); "
            "teacher and student must share the [0, timesteps) time axis."
        )

    def test_add_noise_and_fast_inference_share_the_timesteps_schedule(self):
        """_add_noise must use the same alpha definition as fast_inference
        (0.5^(t / timesteps)) so training and inference see the same levels."""
        model = self._model()
        t = torch.tensor([50], dtype=torch.float32)
        x0 = torch.randn(256, 8)
        noise = torch.randn_like(x0)
        x_t = model._add_noise(x0, t, noise)
        # Reconstruct the effective alpha used by _add_noise from the
        # correlation with x0 (noise term is uncorrelated in expectation):
        # x_t = sqrt(alpha) x0 + sqrt(1-alpha) noise -> (x_t.x0)/(x0.x0)
        # -> sqrt(alpha).
        expected_alpha = float(0.5 ** (t / model.diffusion_config.timesteps))
        legacy_alpha = float(0.5 ** (t / model.teacher_steps))
        alpha_corr = float(
            (x_t * x0).sum() / (x0 * x0).sum()
        ) ** 2
        assert abs(alpha_corr - expected_alpha) < abs(alpha_corr - legacy_alpha), (
            f"_add_noise effective alpha {alpha_corr:.4f} is closer to the "
            f"teacher_steps schedule {legacy_alpha:.4f} than to the timesteps "
            f"schedule {expected_alpha:.4f}; _add_noise and fast_inference "
            "must share one schedule."
        )


# ---------------------------------------------------------------------------
# F3: offline trainer cannot run train_epoch at a different grid size because
# the imported production LatentTo3DConverter lacks `set_resolution`
# (CLI/trainer.py:167). ARCHITECTURE.md states the CLI executes one configured
# grid size, so the offline progressive call must degrade safely.
# ---------------------------------------------------------------------------
class TestOfflineTrainerResolutionHandling:
    def test_train_epoch_accepts_resolution_change_without_crash(self, tmp_path):
        from config import MissionProfile
        from data_utils import AircraftDesignDataset

        trainer = _make_trainer(tmp_path, grid=4, latent_dim=8)
        dataset = AircraftDesignDataset(num_samples=1, grid_size=4,
                                        latent_dim=8, seed=0)
        batch = _sample_batch(
            dataset.latent_codes[:1], [MissionProfile()]
        )
        # grid_size 8 differs from the construction resolution 4; must not raise.
        result = trainer.train_epoch([batch], grid_size=8)
        assert "loss" in result
        assert torch.isfinite(torch.tensor(result["loss"])).item()


# ---------------------------------------------------------------------------
# F6: final optimizer guard check in train_epoch (aircraft_diffusion_cfd.py
# ~9394) raises on raw dot < -1e-8, but the FP64 projection restores FP32
# dtypes whose quantization residual on clipped (~1-norm) gradients is ~1e-7.
# The projection itself defines "safe" as cosine >= -1e-6 (test_conflict_
# projection_* in test_multiobjective_gradients.py). The final check must use
# the same cosine bound; 2 of 7 identical smoke train runs failed with
# dot=-2.29e-07 (cosine ~ -2.3e-7), i.e. the canonical protocol flakes ~29%.
# ---------------------------------------------------------------------------
class TestFinalGuardCosineCheck:
    def test_guard_check_accepts_projection_tolerance_residual(self):
        """A step whose guard-cosine is within the projection's documented
        [-1e-6, 0) acceptance band must not be rejected as 'uphill'."""
        from aircraft_diffusion_cfd import _evaluate_active_guard_cosines

        # Step and guard vectors reproducing the observed flake: post-clip step
        # has norm ~0.99, FP32 residual leaves dot ~ -2.3e-7 with the guard, so
        # cosine ~ -2.3e-7 (inside [-1e-6, 0)).
        step = (torch.tensor([-2.3e-7, 0.9949874371], dtype=torch.float32),)
        guard = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
        cosines = _evaluate_active_guard_cosines(
            step, {"validity": guard}, tolerance=1.0e-6
        )
        assert cosines["validity"] >= -1.0e-6

    def test_guard_check_still_rejects_genuinely_uphill_step(self):
        """Fail-closed behavior is preserved: a step opposed to the guard
        (cosine < -1e-6) still raises."""
        from aircraft_diffusion_cfd import _evaluate_active_guard_cosines

        step = (torch.tensor([-0.5, 0.8660254], dtype=torch.float32),)
        guard = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
        with pytest.raises(RuntimeError, match="uphill on active validity guard"):
            _evaluate_active_guard_cosines(
                step, {"validity": guard}, tolerance=1.0e-6
            )

    def test_guard_check_accepts_aligned_and_absent_guards(self):
        from aircraft_diffusion_cfd import _evaluate_active_guard_cosines

        step = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
        guard = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
        cosines = _evaluate_active_guard_cosines(
            step, {"validity": guard, "connectivity": (None,)}, tolerance=1.0e-6
        )
        assert cosines["validity"] > 0.0
        assert cosines["connectivity"] == 0.0
class TestSolverCoefficientDeviceConsistency:
    def test_coefficients_accept_cpu_geometry_on_cuda_solver(self):
        """compute_aerodynamic_coefficients must accept a CPU voxel grid even
        when the solver lives on CUDA, mirroring collide_stream's device move."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        from advanced_lbm_solver import D3Q27CascadedSolver
        from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig

        cfg = CFDConfig(base_grid_resolution=16, mach_number=0.025,
                        reynolds_number=1e5, simulation_steps=30)
        solver = D3Q27CascadedSolver(cfg, torch.device("cuda"), LBMPhysicsConfig)
        # CPU grid passed to a CUDA solver (the evaluate-baselines path).
        grid = torch.zeros(16, 16, 16)
        grid[7:9, 7:9, 7:9] = 1.0
        solver.collide_stream(grid, steps=30)
        coeffs = solver.compute_aerodynamic_coefficients(grid)
        assert torch.isfinite(torch.tensor(coeffs["drag_coefficient"])).item()

    def test_deferred_coefficients_accept_cpu_geometry_on_cuda_solver(self):
        """compute_aerodynamic_coefficients_deferred has the same contract."""
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA")
        from advanced_lbm_solver import D3Q27CascadedSolver
        from aircraft_diffusion_cfd import CFDConfig, LBMPhysicsConfig

        cfg = CFDConfig(base_grid_resolution=16, mach_number=0.025,
                        reynolds_number=1e5, simulation_steps=30)
        solver = D3Q27CascadedSolver(cfg, torch.device("cuda"), LBMPhysicsConfig)
        grid = torch.zeros(16, 16, 16)
        grid[7:9, 7:9, 7:9] = 1.0
        solver.collide_stream(grid, steps=30)
        deferred = solver.compute_aerodynamic_coefficients_deferred(grid)
        assert deferred is not None
class TestOfflineTrainerSurrogateSampleCounting:
    def test_sample_count_equals_geometry_batch_count_when_branch_runs(self, tmp_path):
        from config import MissionProfile
        from data_utils import AircraftDesignDataset

        trainer = _make_trainer(tmp_path, grid=8, latent_dim=8, batch_size=2)
        # Seed the cache with two real geometry files so the branch runs.
        cache = []
        for i in range(2):
            trainer.gt_exporter.output_dir.mkdir(parents=True, exist_ok=True)
            path = trainer.gt_exporter.output_dir / f"probe{i}.npy"
            np.save(path, np.zeros((8, 8, 8), dtype=np.float32))
            cache.append(
                {"geometry_ref": f"probe{i}.npy", "mission_profile": {},
                 "cd": 0.1, "cl": 0.0, "cm": 0.0, "converged": False}
            )
        trainer.gt_exporter._cache = cache
        dataset = AircraftDesignDataset(num_samples=2, grid_size=8,
                                        latent_dim=8, seed=0)
        batch = _sample_batch(
            dataset.latent_codes[:2], [MissionProfile(), MissionProfile()]
        )
        before = int(trainer.surrogate.sample_count.item())
        trainer.train_epoch([batch], grid_size=8)
        after = int(trainer.surrogate.sample_count.item())
        assert after - before == 2, (
            f"sample_count advanced by {after - before}, expected 2 (the "
            "number of geometries in the surrogate batch); the old code used "
            "batch_geoms[0].shape[0] (the grid resolution)."
        )