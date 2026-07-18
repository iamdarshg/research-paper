from pathlib import Path

from aircraft_diffusion_cfd import (
    OptimizedDiffusionTrainer,
    TrainingConfig,
    evaluate_overfit_stop,
    evaluate_geometry_promotion_gate,
    validate_direct_solver_iteration_coverage,
    validate_solver_integrated_training_config,
)


def test_overfit_stop_waits_for_minimum_epochs():
    config = TrainingConfig(
        overfit_stop_enabled=True,
        overfit_min_epochs=3,
        overfit_loss_floor=0.1,
    )
    history = [
        {"epoch": 1, "optimization_loss": 0.05},
        {"epoch": 2, "optimization_loss": 0.04},
    ]

    assert evaluate_overfit_stop(history, config) is None


def test_overfit_stop_triggers_on_implausibly_low_training_loss():
    config = TrainingConfig(
        overfit_stop_enabled=True,
        overfit_min_epochs=3,
        overfit_loss_floor=0.01,
    )
    history = [
        {"epoch": 1, "optimization_loss": 0.5},
        {"epoch": 2, "optimization_loss": 0.2},
        {"epoch": 3, "optimization_loss": 0.009},
    ]

    decision = evaluate_overfit_stop(history, config)

    assert decision is not None
    assert decision["reason"] == "loss_floor"
    assert decision["epoch"] == 3
    assert decision["metric_value"] == 0.009


def test_overfit_stop_triggers_when_loss_stops_learning():
    config = TrainingConfig(
        overfit_stop_enabled=True,
        overfit_min_epochs=2,
        overfit_patience=2,
        overfit_min_delta=0.01,
        overfit_relative_delta=0.0,
        overfit_loss_floor=0.000001,
    )
    history = [
        {"epoch": 1, "optimization_loss": 1.0},
        {"epoch": 2, "optimization_loss": 0.5},
        {"epoch": 3, "optimization_loss": 0.495},
        {"epoch": 4, "optimization_loss": 0.494},
    ]

    decision = evaluate_overfit_stop(history, config)

    assert decision is not None
    assert decision["reason"] == "plateau"
    assert decision["best_epoch"] == 2
    assert decision["epochs_since_improvement"] == 2


def test_trainer_loop_stops_when_overfit_policy_triggers():
    trainer = OptimizedDiffusionTrainer.__new__(OptimizedDiffusionTrainer)
    trainer.model_config = type("Config", (), {"grid_resolution": 8})()
    trainer.training_config = TrainingConfig(
        num_epochs=100,
        overfit_stop_enabled=True,
        overfit_min_epochs=2,
        overfit_loss_floor=0.01,
        overfit_geometry_gate_enabled=True,
    )
    trainer.converter = type("Converter", (), {"decoder_mode": "dense"})()
    trainer.training_history = []
    trainer.stop_decision = None
    trainer.scheduler = type("Scheduler", (), {"step": lambda self: None})()
    losses = iter([0.5, 0.009, 0.008])

    def fake_train_epoch(train_loader, grid_size):
        value = next(losses)
        return {
            "loss": value,
            "optimization_loss": value,
            "diagnostic_total": value,
            "mse": value,
            "geometry_reconstruction": 0.0,
            "generation_reconstruction": 0.0,
            "consistency": 0.0,
            "direct_solver_loss": 0.0,
            "direct_solver_eval_loss": 0.0,
            "direct_solver_eval_count": 0.0,
            "connectivity": 0.0,
            "aerodynamic": 0.0,
        }

    trainer.train_epoch = fake_train_epoch
    trainer.validate_epoch = lambda *args, **kwargs: None
    trainer.save_checkpoint = lambda *args, **kwargs: None
    trainer._run_progressive_distillation = lambda *args, **kwargs: None
    trainer.evaluate_geometry_promotion_gate = lambda *args, **kwargs: {
        "status": "pass",
        "reconstruction_topk_recall": 0.75,
        "generated_aircraft_valid_fraction": 0.5,
    }

    history = OptimizedDiffusionTrainer.train(trainer, train_loader=[object()])

    assert len(history) == 2
    assert trainer.stop_decision["reason"] == "loss_floor"
    assert history[-1]["stop_decision"]["reason"] == "loss_floor"
    assert history[-1]["geometry_promotion_gate"]["status"] == "pass"


def test_geometry_promotion_gate_rejects_collapsed_geometry():
    config = TrainingConfig(
        overfit_geometry_gate_enabled=True,
        overfit_min_reconstruction_topk_recall=0.2,
        overfit_min_generated_aircraft_valid_fraction=0.125,
    )

    decision = evaluate_geometry_promotion_gate(
        {
            "materialization_mode": "fixed_global_threshold",
            "geometry_threshold_calibrated": True,
            "reconstruction_recall": 0.0,
            "generated_aircraft_valid_fraction": 0.0,
            "sample_count": 8,
        },
        config,
    )

    assert decision["status"] == "fail"
    assert decision["failed_checks"] == [
        "reconstruction_recall",
        "generated_aircraft_valid_fraction",
        "generated_unique_fraction",
        "generated_mean_largest_component_fraction",
        "generated_mean_normalization_boundary_fraction",
        "generated_minimum_mean_occupied_fraction",
    ]


def test_solver_integrated_training_requires_cfd_and_connectivity_every_iteration():
    config = TrainingConfig(
        require_direct_solver_every_iteration=True,
        direct_solver_loss_weight=0.2,
        direct_solver_interval=64,
        direct_solver_steps=5,
        direct_connectivity_weight=1.0,
    )

    try:
        validate_solver_integrated_training_config(config)
    except ValueError as exc:
        assert "direct_solver_interval must be 1" in str(exc)
    else:
        raise AssertionError("Expected sparse CFD scheduling to be rejected")


def test_solver_iteration_coverage_fails_when_any_batch_skips_cfd():
    config = TrainingConfig(require_direct_solver_every_iteration=True)

    try:
        validate_direct_solver_iteration_coverage(
            evaluated_iterations=7,
            optimizer_iterations=8,
            training_config=config,
        )
    except RuntimeError as exc:
        assert "7/8" in str(exc)
    else:
        raise AssertionError("Expected incomplete CFD coverage to be rejected")


def test_trainer_continues_after_failed_geometry_gate():
    trainer = OptimizedDiffusionTrainer.__new__(OptimizedDiffusionTrainer)
    trainer.model_config = type("Config", (), {"grid_resolution": 8})()
    trainer.training_config = TrainingConfig(
        num_epochs=100,
        overfit_stop_enabled=True,
        overfit_min_epochs=1,
        overfit_loss_floor=0.01,
        overfit_geometry_gate_enabled=True,
    )
    trainer.converter = type("Converter", (), {"decoder_mode": "dense"})()
    trainer.training_history = []
    trainer.stop_decision = None
    trainer.geometry_promotion_gate = None
    trainer.scheduler = type("Scheduler", (), {"step": lambda self: None})()
    gate_results = iter(
        [
            {
                "status": "fail",
                "reconstruction_topk_recall": 0.0,
                "generated_aircraft_valid_fraction": 0.0,
            },
            {
                "status": "pass",
                "reconstruction_topk_recall": 0.5,
                "generated_aircraft_valid_fraction": 0.25,
            },
        ]
    )

    trainer.train_epoch = lambda *args, **kwargs: {
        "loss": 0.009,
        "optimization_loss": 0.009,
        "diagnostic_total": 0.009,
        "mse": 0.0,
        "geometry_reconstruction": 0.0,
        "generation_reconstruction": 0.0,
        "consistency": 0.0,
        "direct_solver_loss": 0.0,
        "direct_solver_eval_loss": 0.0,
        "direct_solver_eval_count": 0.0,
        "connectivity": 0.0,
        "aerodynamic": 0.0,
    }
    trainer.evaluate_geometry_promotion_gate = lambda *args, **kwargs: next(gate_results)
    trainer.validate_epoch = lambda *args, **kwargs: None
    trainer.save_checkpoint = lambda *args, **kwargs: None
    trainer._run_progressive_distillation = lambda *args, **kwargs: None

    history = OptimizedDiffusionTrainer.train(trainer, train_loader=[object()])

    assert len(history) == 2
    assert history[0]["geometry_promotion_gate"]["status"] == "fail"
    assert "stop_decision" not in history[0]
    assert history[1]["geometry_promotion_gate"]["status"] == "pass"
    assert history[1]["stop_decision"]["reason"] == "loss_floor"


def test_trainer_loop_saves_periodic_checkpoints_under_configured_dir():
    trainer = OptimizedDiffusionTrainer.__new__(OptimizedDiffusionTrainer)
    trainer.model_config = type("Config", (), {"grid_resolution": 8})()
    trainer.training_config = TrainingConfig(
        num_epochs=1,
        save_interval=1,
        checkpoint_dir="checkpoints/run-a",
    )
    trainer.converter = type("Converter", (), {"decoder_mode": "dense"})()
    trainer.training_history = []
    trainer.stop_decision = None
    trainer.scheduler = type("Scheduler", (), {"step": lambda self: None})()

    def fake_train_epoch(train_loader, grid_size):
        return {
            "loss": 0.5,
            "optimization_loss": 0.5,
            "diagnostic_total": 0.5,
            "mse": 0.5,
            "geometry_reconstruction": 0.0,
            "generation_reconstruction": 0.0,
            "consistency": 0.0,
            "direct_solver_loss": 0.0,
            "direct_solver_eval_loss": 0.0,
            "direct_solver_eval_count": 0.0,
            "connectivity": 0.0,
            "aerodynamic": 0.0,
        }

    saved_paths = []
    trainer.train_epoch = fake_train_epoch
    trainer.validate_epoch = lambda *args, **kwargs: None
    trainer.save_checkpoint = lambda path: saved_paths.append(path)
    trainer._run_progressive_distillation = lambda *args, **kwargs: None

    OptimizedDiffusionTrainer.train(trainer, train_loader=[object()])

    assert saved_paths == [
        str(Path("checkpoints/run-a") / "checkpoint_optimized_grid8_ep1.pt")
    ]
