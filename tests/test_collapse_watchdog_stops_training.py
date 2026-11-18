"""A model-collapse watchdog pause must stop ``OptimizedDiffusionTrainer.train``.

Context (2026-11-12 audit).  ``train_epoch`` fails closed when the collapse
watchdog fires: it saves ``collapse_paused_step<N>.pt``, records
``stop_decision = {"kind": "model_collapse", ...}`` and breaks out of the
epoch.  The outer ``train()`` loop ignored that decision and started the next
epoch.  Because the paused optimizer transaction was never committed, the
converter step counter then lagged and every later epoch re-tripped the
watchdog after a single batch (``coordinate_converter_optimizer_step_stalled``)
until ``num_epochs`` ran out.  The CLI then printed "Training complete" and
wrote ``final_optimized_model.pt`` -- a paused run reported as a finished one.
Measured on 2026-11-12: a 14-epoch request paused at global step 269, then
logged eight one-step "epochs" (steps 436-443) and reported completion.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

CLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from aircraft_diffusion_cfd import OptimizedDiffusionTrainer  # noqa: E402


class _NullScheduler:
    def step(self) -> None:
        pass


def _stub_trainer(tmp_path, *, num_epochs: int, collapse_on_epoch: int | None):
    trainer = object.__new__(OptimizedDiffusionTrainer)
    trainer.model_config = SimpleNamespace(grid_resolution=8)
    trainer.training_config = SimpleNamespace(
        num_epochs=num_epochs,
        overfit_stop_enabled=False,
        overfit_geometry_gate_enabled=False,
        val_interval=1_000_000,
        save_interval=1_000_000,
        checkpoint_dir=str(tmp_path),
        latent_standardization=False,
    )
    trainer.training_history = []
    trainer.scheduler = _NullScheduler()
    trainer.scheduler_step_per_update = True
    trainer.converter = SimpleNamespace(decoder_mode="dense")
    trainer.stop_decision = None
    trainer.latent_standardization_mean = None
    trainer.latent_standardization_scale = None
    calls = []

    def train_epoch(_loader, grid_size):
        calls.append(grid_size)
        if collapse_on_epoch is not None and len(calls) == collapse_on_epoch:
            trainer.stop_decision = {
                "kind": "model_collapse",
                "reason": "model_collapse_watchdog",
                "global_step": 269,
                "reason_codes": ["representation_effective_rank_collapsed"],
            }
        return {"loss": 1.0, "optimization_loss": 1.0}

    trainer.train_epoch = train_epoch
    return trainer, calls


def test_collapse_pause_stops_the_epoch_loop(tmp_path):
    trainer, calls = _stub_trainer(tmp_path, num_epochs=14, collapse_on_epoch=4)
    history = trainer.train(train_loader=[])
    assert len(calls) == 4, "training continued after a fail-closed watchdog pause"
    assert len(history) == 4
    assert history[-1]["collapse_watchdog_stop"]["global_step"] == 269
    assert trainer.stop_decision["kind"] == "model_collapse"


def test_healthy_run_completes_all_epochs(tmp_path):
    trainer, calls = _stub_trainer(tmp_path, num_epochs=3, collapse_on_epoch=None)
    history = trainer.train(train_loader=[])
    assert len(calls) == 3 and len(history) == 3
    assert "collapse_watchdog_stop" not in history[-1]


def test_cli_completion_message_distinguishes_pause():
    from aircraft_diffusion_cfd import training_completion_status

    assert training_completion_status(None) == "complete"
    assert training_completion_status({"kind": "overfit"}) == "complete"
    assert training_completion_status({"kind": "model_collapse"}) == "paused_model_collapse"
