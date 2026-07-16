import json

from watch_training_progress import snapshot
from training_tui import _latest_live_batch


def test_snapshot_labels_completed_epoch_and_loss_trend(tmp_path):
    history = tmp_path / "history.json"
    telemetry = tmp_path / "telemetry.jsonl"
    history.write_text(json.dumps({"history": [
        {"epoch": 1, "optimization_loss": 2.0},
        {"epoch": 2, "optimization_loss": 1.0, "direct_solver_iteration_coverage": 1.0,
         "promotion_gate_passed": 0.0},
    ]}), encoding="utf-8")
    telemetry.write_text('{"elapsed_s": 10}\n', encoding="utf-8")

    item = snapshot(history, telemetry, trainer_pid=None)

    assert item["labels"] == {
        "epoch_state": "completed",
        "loss_trend": "improving",
        "solver_coverage": "complete",
        "promotion": "failed",
        "trainer_state": "not_checked",
    }
    assert item["epoch_metrics"]["epoch"] == 2
    assert item["resource_telemetry"]["elapsed_s"] == 10


def test_snapshot_labels_missing_history_without_inventing_loss(tmp_path):
    item = snapshot(tmp_path / "missing.json", tmp_path / "missing.jsonl", trainer_pid=None)

    assert item["epoch_metrics"] is None
    assert item["labels"]["epoch_state"] == "waiting_for_first_completed_epoch"
    assert item["labels"]["loss_trend"] == "unavailable"


def test_live_batch_parser_reads_tqdm_loss_postfix(tmp_path):
    console = tmp_path / "training.log"
    console.write_text(
        "loading checkpoint\r"
        "Training: 12%|### | 91/758 [1:02:03<7:35:00, 40.9s/it, "
        "opt_loss=12.5, mse=0.25, clean_geom=1.1, gen_geom=2.2, "
        "consistency=3.3, latent_recon=0.4, direct_solver=0.75]\r",
        encoding="utf-8",
    )

    live = _latest_live_batch(console)

    assert live is not None
    assert live["done"] == 91
    assert live["total"] == 758
    assert live["metrics"]["opt_loss"] == 12.5
    assert live["metrics"]["direct_solver"] == 0.75
