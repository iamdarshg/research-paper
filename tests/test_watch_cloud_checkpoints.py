import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from watch_cloud_checkpoints import (
    CheckpointWatcher,
    WatchError,
    parse_manifest,
    render_inference_command,
    verified_atomic_download,
)


class FakeBackend:
    def __init__(self, manifest, objects):
        self.manifest = manifest
        self.objects = objects
        self.reads = []
        self.downloads = []

    def read_object(self, uri):
        self.reads.append(uri)
        return self.manifest

    def download(self, uri, destination):
        self.downloads.append((uri, Path(destination)))
        Path(destination).write_bytes(self.objects[uri])


def _manifest_for(uri, payload, *, generation="g-100", step=100):
    return json.dumps(
        {
            "schema_version": 1,
            "checkpoints": [
                {
                    "generation": generation,
                    "global_step": step,
                    "object": uri,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_size": len(payload),
                }
            ],
        }
    ).encode("utf-8")


def test_parse_manifest_selects_latest_and_resolves_relative_object():
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    record = parse_manifest(
        {
            "checkpoints": [
                {"generation": "g99", "global_update": 99, "object": "old.pt", "sha256": digest, "bytes": len(payload)},
                {"generation": "g100", "global_update": 100, "object": "new.pt", "sha256": digest, "bytes": len(payload)},
            ]
        },
        run_prefix="gs://bucket/run",
        manifest_uri="gs://bucket/run/checkpoint_manifest.json",
    )

    assert record.generation == "g100"
    assert record.global_step == 100
    assert record.object_uri == "gs://bucket/run/new.pt"
    assert record.sha256 == digest


def test_verified_atomic_download_accepts_exact_bytes_and_leaves_no_partial(tmp_path):
    payload = b"verified checkpoint bytes"
    uri = "gs://bucket/run/checkpoint.pt"
    backend = FakeBackend(b"unused", {uri: payload})
    destination = tmp_path / "checkpoint.g100.pt"

    result = verified_atomic_download(
        backend,
        uri,
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_byte_size=len(payload),
    )

    assert destination.read_bytes() == payload
    assert not Path(str(destination) + ".partial").exists()
    assert result["byte_size"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_verified_atomic_download_rejects_bad_digest_and_cleans_quarantine(tmp_path):
    payload = b"bad checkpoint"
    uri = "gs://bucket/run/checkpoint.pt"
    backend = FakeBackend(b"unused", {uri: payload})
    destination = tmp_path / "checkpoint.pt"

    with pytest.raises(WatchError, match="SHA-256 mismatch"):
        verified_atomic_download(
            backend,
            uri,
            destination,
            expected_sha256=hashlib.sha256(b"different").hexdigest(),
            expected_byte_size=len(payload),
        )

    assert not destination.exists()
    assert not Path(str(destination) + ".partial").exists()


def test_render_command_is_argv_only_and_requires_three_artifact_placeholders(tmp_path):
    command = render_inference_command(
        ["python", "probe.py", "--checkpoint", "{checkpoint}", "--output", "{output}", "--profile", "{profile}"],
        checkpoint=tmp_path / "model with spaces.pt",
        output=tmp_path / "output dir",
        profile=tmp_path / "f16 profile.json",
        profile_name="f16_like",
        generation="g/100",
        global_step=100,
    )

    assert command[0:2] == ["python", "probe.py"]
    assert command[3] == str((tmp_path / "model with spaces.pt").resolve())
    assert command[5] == str((tmp_path / "output dir").resolve())
    assert command[7] == str((tmp_path / "f16 profile.json").resolve())
    assert all("{checkpoint}" not in value for value in command)
    with pytest.raises(WatchError, match=r"\{profile\}"):
        render_inference_command(
            ["python", "probe.py", "{checkpoint}", "{output}"],
            checkpoint=tmp_path / "model.pt",
            output=tmp_path / "out",
            profile=tmp_path / "profile.json",
            profile_name="f16_like",
            generation="g1",
            global_step=100,
        )


def test_watcher_processes_boundary_once_and_runs_both_profiles(tmp_path):
    payload = b"checkpoint at update 100"
    object_uri = "gs://bucket/run/checkpoint.pt"
    backend = FakeBackend(_manifest_for(object_uri, payload), {object_uri: payload})
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    watcher = CheckpointWatcher(
        run_prefix="gs://bucket/run",
        download_dir=tmp_path / "downloads",
        inference_command=["fake-infer", "--checkpoint", "{checkpoint}", "--output", "{output}", "--profile", "{profile}"],
        f16_profile=tmp_path / "f16.json",
        cessna_profile=tmp_path / "cessna.json",
        backend=backend,
        process_runner=fake_runner,
        evidence_path=tmp_path / "evidence.jsonl",
        state_path=tmp_path / "state.json",
    )

    first = watcher.poll_once()
    second = watcher.poll_once()

    assert first["kind"] == "checkpoint_processed"
    assert first["inference_failures"] == 0
    assert second["kind"] == "checkpoint_already_processed"
    assert len(calls) == 2
    assert calls[0][0][0] == "fake-infer"
    assert calls[0][0][1] == "--checkpoint"
    assert calls[0][0][2].endswith("checkpoint.gg-100.pt")
    assert {call[0][-1] for call in calls} == {
        str((tmp_path / "f16.json").resolve()),
        str((tmp_path / "cessna.json").resolve()),
    }
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["processed_generations"] == ["g-100"]
    events = [json.loads(line) for line in (tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["kind"] for event in events] == [
        "checkpoint_accepted",
        "inference_succeeded",
        "inference_succeeded",
        "checkpoint_processed",
        "checkpoint_already_processed",
    ]


def test_watcher_records_inference_failure_but_still_invokes_second_profile(tmp_path):
    payload = b"checkpoint"
    object_uri = "gs://bucket/run/checkpoint.pt"
    backend = FakeBackend(_manifest_for(object_uri, payload), {object_uri: payload})
    calls = []

    def fake_runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 7, stdout="", stderr="broken")

    watcher = CheckpointWatcher(
        run_prefix="gs://bucket/run",
        download_dir=tmp_path / "downloads",
        inference_command=["probe", "{checkpoint}", "{output}", "{profile}"],
        backend=backend,
        process_runner=fake_runner,
        evidence_path=tmp_path / "evidence.jsonl",
        state_path=tmp_path / "state.json",
    )

    result = watcher.poll_once()

    assert result["inference_failures"] == 2
    assert len(calls) == 2
    events = [json.loads(line) for line in (tmp_path / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["kind"] for event in events].count("inference_failed") == 2
    assert all(event["returncode"] == 7 for event in events if event["kind"] == "inference_failed")


def test_watcher_waits_for_global_update_boundary_without_downloading(tmp_path):
    payload = b"checkpoint at update 101"
    object_uri = "gs://bucket/run/checkpoint.pt"
    backend = FakeBackend(_manifest_for(object_uri, payload, generation="g101", step=101), {object_uri: payload})
    watcher = CheckpointWatcher(
        run_prefix="gs://bucket/run",
        download_dir=tmp_path / "downloads",
        inference_command=["probe", "{checkpoint}", "{output}", "{profile}"],
        backend=backend,
        evidence_path=tmp_path / "evidence.jsonl",
        state_path=tmp_path / "state.json",
    )

    result = watcher.poll_once()

    assert result["kind"] == "checkpoint_waiting_for_update_boundary"
    assert backend.downloads == []


def test_watcher_removes_previous_payload_before_next_download(tmp_path):
    first_payload = b"checkpoint 100"
    second_payload = b"checkpoint 200"
    first_uri = "gs://bucket/run/step-100.pt"
    second_uri = "gs://bucket/run/step-200.pt"
    backend = FakeBackend(
        _manifest_for(first_uri, first_payload, generation="g100", step=100),
        {first_uri: first_payload, second_uri: second_payload},
    )
    observed_checkpoint_counts = []
    original_download = backend.download

    def recording_download(uri, destination):
        observed_checkpoint_counts.append(len(list((tmp_path / "downloads").glob("*.pt"))))
        original_download(uri, destination)

    backend.download = recording_download
    watcher = CheckpointWatcher(
        run_prefix="gs://bucket/run",
        download_dir=tmp_path / "downloads",
        inference_command=["probe", "{checkpoint}", "{output}", "{profile}"],
        backend=backend,
        process_runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
        evidence_path=tmp_path / "evidence.jsonl",
        state_path=tmp_path / "state.json",
    )

    watcher.poll_once()
    backend.manifest = _manifest_for(second_uri, second_payload, generation="g200", step=200)
    watcher.poll_once()

    assert observed_checkpoint_counts == [0, 0]
    checkpoints = list((tmp_path / "downloads").glob("*.pt"))
    assert len(checkpoints) == 1
    assert checkpoints[0].read_bytes() == second_payload
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert len(state["accepted"]) == 1
    assert state["accepted"][0]["generation"] == "g200"
