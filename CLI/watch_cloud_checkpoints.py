#!/usr/bin/env python3
"""Stream verified GCS checkpoints to a Windows machine and probe them.

The watcher deliberately has no Google Cloud Python dependency.  Production
uses the installed ``gcloud`` executable; tests can inject a tiny backend and
process runner.  A checkpoint is accepted only after its manifest-declared
byte count and SHA-256 have both been verified.  Downloads use a quarantine
``.partial`` file and are atomically renamed only after verification.

The manifest is expected at ``<run-prefix>/checkpoint_manifest.json`` by
default.  It may contain one checkpoint object or a ``checkpoints`` list.  A
checkpoint object must provide a published generation, global update/step,
object URI (or object name), SHA-256, and byte size.  Common aliases are
accepted to keep this boundary stable as the trainer's publication surface
evolves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse


DEFAULT_MANIFEST_NAME = "checkpoint_manifest.json"
DEFAULT_F16_PROFILE = Path(
    "build/mission_profile_current_best_20260831/f16_like_mission.json"
)
DEFAULT_CESSNA_PROFILE = Path(
    "build/mission_profile_current_best_20260831/cessna_like_mission.json"
)
_SHA256_LENGTH = 64
_PARTIAL_SUFFIX = ".partial"


class WatchError(RuntimeError):
    """A fail-closed watcher or artifact validation error."""


@dataclass(frozen=True)
class CheckpointRecord:
    generation: str
    global_step: int
    object_uri: str
    sha256: str
    byte_size: int
    manifest_uri: str


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        value = mapping.get(name)
        if value is not None and value != "":
            return value
    return None


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise WatchError(f"manifest {field} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise WatchError(f"manifest {field} must be an integer") from exc
    if converted < 0:
        raise WatchError(f"manifest {field} must be non-negative")
    return converted


def _as_sha256(value: Any) -> str:
    digest = str(value).strip().lower() if value is not None else ""
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise WatchError("manifest sha256 must be a 64-character hexadecimal digest")
    return digest


def _join_gcs(prefix: str, object_name: str) -> str:
    if object_name.startswith("gs://"):
        return object_name
    return f"{prefix.rstrip('/')}/{object_name.lstrip('/')}"


def _manifest_candidates(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        listed = payload.get("checkpoints")
        if listed is not None:
            if not isinstance(listed, list) or not all(
                isinstance(item, Mapping) for item in listed
            ):
                raise WatchError("manifest checkpoints must be a list of objects")
            return list(listed)
        nested = payload.get("checkpoint")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            for key in (
                "generation",
                "manifest_generation",
                "global_step",
                "global_update",
                "update",
                "object",
                "object_uri",
                "checkpoint_object",
                "sha256",
                "byte_size",
                "bytes",
                "size",
            ):
                if key in payload and key not in merged:
                    merged[key] = payload[key]
            return [merged]
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
        return list(payload)
    raise WatchError("checkpoint manifest must be a JSON object or list")


def parse_manifest(
    payload: bytes | str | Mapping[str, Any] | list[Any],
    *,
    run_prefix: str,
    manifest_uri: str,
) -> CheckpointRecord:
    """Parse one published checkpoint from a JSON or JSONL manifest.

    For a multi-checkpoint manifest the greatest global step is selected.  A
    generation is mandatory in meaning, but if a publisher omitted a GCS
    generation this function derives a stable content identity from the
    declared digest, size, and update.  That fallback is recorded by callers
    as a non-GCS-generation identity and still prevents duplicate work.
    """

    if isinstance(payload, (bytes, str)):
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        try:
            decoded: Any = json.loads(text)
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise WatchError("checkpoint manifest is not valid JSON/JSONL") from exc
                    rows.append(row)
            decoded = rows
    else:
        decoded = payload

    candidates = _manifest_candidates(decoded)
    parsed: list[CheckpointRecord] = []
    for item in candidates:
        nested = item.get("checkpoint") if isinstance(item.get("checkpoint"), Mapping) else {}
        source = {**nested, **item}
        global_step_value = _first(
            source,
            ("global_step", "global_update", "update", "updates", "step"),
        )
        object_name = _first(
            source,
            ("object_uri", "checkpoint_object", "checkpoint_uri", "object", "path"),
        )
        digest = _first(source, ("sha256", "checkpoint_sha256", "sha"))
        byte_size = _first(source, ("byte_size", "bytes", "size", "checkpoint_bytes"))
        if global_step_value is None or object_name is None or digest is None or byte_size is None:
            raise WatchError(
                "manifest checkpoint requires global_step, object, sha256, and byte_size"
            )
        global_step = _as_int(global_step_value, "global_step")
        digest = _as_sha256(digest)
        byte_size = _as_int(byte_size, "byte_size")
        generation_value = _first(
            source,
            (
                "generation",
                "checkpoint_generation",
                "object_generation",
                "manifest_generation",
            ),
        )
        generation = (
            str(generation_value).strip()
            if generation_value is not None
            else f"sha256-{digest}-bytes-{byte_size}-step-{global_step}"
        )
        if not generation:
            raise WatchError("manifest generation must not be empty")
        parsed.append(
            CheckpointRecord(
                generation=generation,
                global_step=global_step,
                object_uri=_join_gcs(run_prefix, str(object_name).strip()),
                sha256=digest,
                byte_size=byte_size,
                manifest_uri=manifest_uri,
            )
        )
    if not parsed:
        raise WatchError("checkpoint manifest contains no checkpoints")
    return max(parsed, key=lambda record: (record.global_step, record.generation))


def sha256_and_size(path: Path, *, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verified_atomic_download(
    backend: Any,
    object_uri: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_byte_size: int,
) -> dict[str, Any]:
    """Download, verify, and atomically publish one checkpoint.

    ``backend.download`` must write to the supplied path.  Any failed or
    mismatching transfer remains quarantined only for the duration of this
    call and is removed before the error is raised.
    """

    expected_sha256 = _as_sha256(expected_sha256)
    expected_byte_size = _as_int(expected_byte_size, "expected_byte_size")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + _PARTIAL_SUFFIX)
    try:
        partial.unlink(missing_ok=True)
        backend.download(object_uri, partial)
        if not partial.is_file():
            raise WatchError("download backend did not create the partial file")
        actual_sha256, actual_byte_size = sha256_and_size(partial)
        if actual_byte_size != expected_byte_size:
            raise WatchError(
                f"checkpoint byte-size mismatch: expected {expected_byte_size}, "
                f"got {actual_byte_size}"
            )
        if actual_sha256 != expected_sha256:
            raise WatchError(
                f"checkpoint SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        os.replace(partial, destination)
        return {
            "path": str(destination.resolve()),
            "sha256": actual_sha256,
            "byte_size": actual_byte_size,
        }
    except Exception:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_durable_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed_generations": [], "accepted": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WatchError(f"state file is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("processed_generations", []), list):
        raise WatchError("state file has an invalid schema")
    payload.setdefault("accepted", [])
    return payload


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_." else "_" for character in value)
    return result.strip(".") or "checkpoint"


def _replace_placeholders(argv: Sequence[str], values: Mapping[str, str]) -> list[str]:
    placeholders = {f"{{{key}}}": value for key, value in values.items()}
    rendered = []
    for argument in argv:
        result = str(argument)
        for placeholder, value in placeholders.items():
            result = result.replace(placeholder, value)
        rendered.append(result)
    return rendered


def render_inference_command(
    command: Sequence[str],
    *,
    checkpoint: Path,
    output: Path,
    profile: Path,
    profile_name: str,
    generation: str,
    global_step: int,
) -> list[str]:
    """Render an argv template without ever invoking a shell."""

    required = ("{checkpoint}", "{output}", "{profile}")
    missing = [placeholder for placeholder in required if not any(placeholder in arg for arg in command)]
    if missing:
        raise WatchError(
            "inference command must include placeholders: " + ", ".join(missing)
        )
    return _replace_placeholders(
        command,
        {
            "checkpoint": str(checkpoint.resolve()),
            "output": str(output.resolve()),
            "profile": str(profile.resolve()),
            "profile_name": profile_name,
            "generation": generation,
            "global_step": str(global_step),
        },
    )


class GCloudStorageBackend:
    """Minimal GCS object backend using argv-only gcloud subprocesses."""

    def __init__(self, executable: str = "gcloud", *, timeout: float = 120.0) -> None:
        self.executable = executable
        self.timeout = timeout

    def read_object(self, object_uri: str) -> bytes:
        completed = subprocess.run(
            [self.executable, "storage", "cat", object_uri],
            capture_output=True,
            check=False,
            timeout=self.timeout,
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise WatchError(f"gcloud manifest read failed ({completed.returncode}): {detail}")
        return completed.stdout

    def download(self, object_uri: str, destination: Path) -> None:
        completed = subprocess.run(
            [self.executable, "storage", "cp", "-q", object_uri, str(destination)],
            capture_output=True,
            check=False,
            timeout=self.timeout,
        )
        if completed.returncode:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
            raise WatchError(f"gcloud checkpoint download failed ({completed.returncode}): {detail}")


class CheckpointWatcher:
    def __init__(
        self,
        *,
        run_prefix: str,
        download_dir: Path,
        inference_command: Optional[Sequence[str]],
        f16_profile: Path = DEFAULT_F16_PROFILE,
        cessna_profile: Path = DEFAULT_CESSNA_PROFILE,
        manifest_object: Optional[str] = None,
        update_interval: int = 100,
        evidence_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
        inference_timeout: float = 900.0,
        backend: Any = None,
        process_runner: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not run_prefix.startswith("gs://"):
            raise ValueError("run_prefix must start with gs://")
        if update_interval <= 0:
            raise ValueError("update_interval must be positive")
        self.run_prefix = run_prefix.rstrip("/")
        self.download_dir = Path(download_dir)
        self.manifest_uri = _join_gcs(
            self.run_prefix,
            manifest_object or DEFAULT_MANIFEST_NAME,
        )
        self.inference_command = list(inference_command) if inference_command else None
        self.profiles = (
            ("f16_like", Path(f16_profile)),
            ("cessna_like", Path(cessna_profile)),
        )
        self.update_interval = update_interval
        self.evidence_path = evidence_path or self.download_dir / "watcher_evidence.jsonl"
        self.state_path = state_path or self.download_dir / "watcher_state.json"
        self.inference_timeout = inference_timeout
        self.backend = backend or GCloudStorageBackend()
        self.process_runner = process_runner or subprocess.run

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"kind": kind, "timestamp": utc_now(), "run_prefix": self.run_prefix, **fields}
        _append_durable_jsonl(self.evidence_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        return event

    def _checkpoint_destination(self, record: CheckpointRecord) -> Path:
        parsed = urlparse(record.object_uri)
        basename = Path(parsed.path).name or "checkpoint.pt"
        stem = _safe_name(Path(basename).stem)
        suffix = Path(basename).suffix or ".pt"
        return self.download_dir / f"{stem}.g{_safe_name(record.generation)}{suffix}"

    def _remove_superseded_local_checkpoints(
        self,
        state: Mapping[str, Any],
        *,
        keep: Optional[Path] = None,
    ) -> list[dict[str, Any]]:
        """Remove only checkpoint files previously accepted by this watcher.

        This deliberately does not glob the download directory.  Provenance,
        evidence, inference products, and unrelated user checkpoints are never
        cleanup candidates.
        """

        removed: list[dict[str, Any]] = []
        keep_resolved = keep.resolve() if keep is not None else None
        root = self.download_dir.resolve()
        for accepted in state.get("accepted", []):
            if not isinstance(accepted, Mapping) or not accepted.get("path"):
                continue
            candidate = Path(str(accepted["path"])).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                continue
            if keep_resolved is not None and candidate == keep_resolved:
                continue
            if candidate.is_file():
                size = candidate.stat().st_size
                candidate.unlink()
                removed.append({"path": str(candidate), "byte_size": size})
        return removed

    def _run_profile_inference(
        self,
        record: CheckpointRecord,
        checkpoint_path: Path,
        profile_name: str,
        profile_path: Path,
    ) -> dict[str, Any]:
        output = self.download_dir / "inference" / (
            f"step-{record.global_step}-g{_safe_name(record.generation)}-{profile_name}"
        )
        output.mkdir(parents=True, exist_ok=True)
        if not self.inference_command:
            return self._event(
                "inference_failed",
                generation=record.generation,
                global_step=record.global_step,
                profile=profile_name,
                checkpoint=str(checkpoint_path.resolve()),
                output=str(output.resolve()),
                error="inference command is not configured",
            )
        try:
            argv = render_inference_command(
                self.inference_command,
                checkpoint=checkpoint_path,
                output=output,
                profile=profile_path,
                profile_name=profile_name,
                generation=record.generation,
                global_step=record.global_step,
            )
            completed = self.process_runner(
                argv,
                capture_output=True,
                check=False,
                timeout=self.inference_timeout,
                text=True,
            )
            result = CommandResult(
                returncode=int(getattr(completed, "returncode", 1)),
                stdout=str(getattr(completed, "stdout", "") or "")[-4000:],
                stderr=str(getattr(completed, "stderr", "") or "")[-4000:],
            )
            kind = "inference_succeeded" if result.returncode == 0 else "inference_failed"
            return self._event(
                kind,
                generation=record.generation,
                global_step=record.global_step,
                profile=profile_name,
                checkpoint=str(checkpoint_path.resolve()),
                output=str(output.resolve()),
                argv=argv,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return self._event(
                "inference_failed",
                generation=record.generation,
                global_step=record.global_step,
                profile=profile_name,
                checkpoint=str(checkpoint_path.resolve()),
                output=str(output.resolve()),
                error=f"inference timed out after {self.inference_timeout}s: {exc}",
            )
        except (OSError, WatchError, ValueError, TypeError) as exc:
            return self._event(
                "inference_failed",
                generation=record.generation,
                global_step=record.global_step,
                profile=profile_name,
                checkpoint=str(checkpoint_path.resolve()),
                output=str(output.resolve()),
                error=str(exc),
            )

    def poll_once(self) -> dict[str, Any]:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest_payload = self.backend.read_object(self.manifest_uri)
            record = parse_manifest(
                manifest_payload,
                run_prefix=self.run_prefix,
                manifest_uri=self.manifest_uri,
            )
        except Exception as exc:
            return self._event("manifest_unavailable", manifest_uri=self.manifest_uri, error=str(exc))

        if record.global_step == 0 or record.global_step % self.update_interval:
            return self._event(
                "checkpoint_waiting_for_update_boundary",
                manifest_uri=self.manifest_uri,
                generation=record.generation,
                global_step=record.global_step,
                update_interval=self.update_interval,
            )

        try:
            state = _read_state(self.state_path)
        except WatchError as exc:
            return self._event("watcher_state_error", state_path=str(self.state_path), error=str(exc))
        processed = {str(value) for value in state.get("processed_generations", [])}
        if record.generation in processed:
            return self._event(
                "checkpoint_already_processed",
                generation=record.generation,
                global_step=record.global_step,
                state_path=str(self.state_path),
            )

        destination = self._checkpoint_destination(record)
        # Enforce a one-checkpoint SSD budget.  The old accepted payload is
        # removed before the new .partial download begins, so two full local
        # checkpoints never coexist even transiently.
        try:
            removed = self._remove_superseded_local_checkpoints(state)
            if removed:
                self._event(
                    "superseded_local_checkpoints_removed",
                    generation=record.generation,
                    global_step=record.global_step,
                    removed=removed,
                )
        except OSError as exc:
            return self._event(
                "checkpoint_retention_error",
                generation=record.generation,
                global_step=record.global_step,
                error=str(exc),
            )
        try:
            verification = verified_atomic_download(
                self.backend,
                record.object_uri,
                destination,
                expected_sha256=record.sha256,
                expected_byte_size=record.byte_size,
            )
        except Exception as exc:
            return self._event(
                "checkpoint_rejected",
                generation=record.generation,
                global_step=record.global_step,
                object_uri=record.object_uri,
                error=str(exc),
                partial_path=str(destination) + _PARTIAL_SUFFIX,
            )

        accepted = {
            **asdict(record),
            **verification,
            "accepted_at": utc_now(),
        }
        processed.add(record.generation)
        state["processed_generations"] = sorted(processed)
        # Preserve generation IDs for deduplication, but retain only the
        # current payload path in mutable state.  Durable history stays JSONL.
        state["accepted"] = [accepted]
        _atomic_json_write(self.state_path, state)
        self._event("checkpoint_accepted", **accepted)

        inference_events = [
            self._run_profile_inference(record, destination, profile_name, profile_path)
            for profile_name, profile_path in self.profiles
        ]
        failed = sum(event.get("kind") == "inference_failed" for event in inference_events)
        return self._event(
            "checkpoint_processed",
            generation=record.generation,
            global_step=record.global_step,
            checkpoint=str(destination.resolve()),
            inference_failures=failed,
            inference_profiles=[event.get("profile") for event in inference_events],
        )


def _parse_command(command: Optional[str], command_argv: Optional[Sequence[str]]) -> Optional[list[str]]:
    if command_argv:
        return list(command_argv)
    if command:
        # This is only configuration parsing.  The resulting argv is passed
        # directly to subprocess.run; no shell is involved.
        return shlex.split(command, posix=False)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", required=True, help="GCS prefix, for example gs://bucket/run-id")
    parser.add_argument("--manifest-object", default=None, help="Manifest object name or full gs:// URI")
    parser.add_argument("--download-dir", type=Path, default=Path("build/checkpoint_stream"))
    parser.add_argument("--evidence", type=Path, default=None, help="Durable JSONL evidence path")
    parser.add_argument("--state", type=Path, default=None, help="Atomic processed-generation state path")
    parser.add_argument("--update-interval", type=int, default=100, help="Process only these global-step boundaries")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    parser.add_argument("--gcloud", default="gcloud", help="gcloud executable")
    parser.add_argument("--gcloud-timeout", type=float, default=120.0)
    parser.add_argument("--inference-command", default=None, help="Quoted argv template with {checkpoint} {output} {profile}")
    parser.add_argument("--inference-arg", action="append", dest="inference_args", help="One argv token; repeat for a safe template")
    parser.add_argument("--f16-profile", type=Path, default=DEFAULT_F16_PROFILE)
    parser.add_argument("--cessna-profile", type=Path, default=DEFAULT_CESSNA_PROFILE)
    parser.add_argument("--inference-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    if args.inference_command and args.inference_args:
        parser.error("use --inference-command or repeated --inference-arg, not both")
    command = _parse_command(args.inference_command, args.inference_args)
    watcher = CheckpointWatcher(
        run_prefix=args.run_prefix,
        download_dir=args.download_dir,
        inference_command=command,
        f16_profile=args.f16_profile,
        cessna_profile=args.cessna_profile,
        manifest_object=args.manifest_object,
        update_interval=args.update_interval,
        evidence_path=args.evidence,
        state_path=args.state,
        inference_timeout=args.inference_timeout,
        backend=GCloudStorageBackend(args.gcloud, timeout=args.gcloud_timeout),
    )
    while True:
        watcher.poll_once()
        if args.once:
            return 0
        time.sleep(max(0.1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
