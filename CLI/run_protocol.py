#!/usr/bin/env python3
"""Run checked-in smoke/final protocols for the conditioned generator workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from report_metadata import file_sha256


def load_protocol_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a mapping")
    payload["_config_path"] = str(config_path)
    payload["_config_dir"] = str(config_path.parent)
    return payload


def _resolve_path(config: Dict[str, Any], value: str | None) -> str | None:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(config["_config_dir"]) / path).resolve())


def _add_option(command: List[str], flag: str, value: Any) -> None:
    if value is None or value is False:
        return
    if value is True:
        command.append(flag)
        return
    command.extend([flag, str(value)])


def _default_checkpoint(config: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    save_dir = _resolve_path(config, train_cfg.get("save_dir", "./checkpoints"))
    return str((Path(save_dir) / "final_optimized_model.pt").resolve())


def _protocol_run_id(config: Dict[str, Any]) -> str:
    metadata = dict(config.get("run_metadata", {}))
    if metadata.get("run_id"):
        return str(metadata["run_id"])
    protocol_hash = file_sha256(config["_config_path"]) or "unknown"
    return f"protocol-{protocol_hash[:12]}"


def build_protocol_commands(config: Dict[str, Any]) -> List[List[str]]:
    config_path = Path(config["_config_path"]).resolve()
    cli_dir = config_path.parent.parent
    python_exe = sys.executable
    cli_script = str((cli_dir / "aircraft_diffusion_cfd.py").resolve())
    multi_seed_script = str((cli_dir / "multi_seed_eval.py").resolve())
    manifest_validator_script = str((cli_dir / "validate_manifest.py").resolve())
    condition_benchmark_script = str((cli_dir / "run_condition_benchmark.py").resolve())
    aircraft_validity_script = str((cli_dir / "aircraft_validity.py").resolve())
    aircraft_validity_input_builder_script = str((cli_dir / "build_aircraft_validity_inputs.py").resolve())
    manufacturing_constraints_script = str((cli_dir / "condition_feasibility.py").resolve())
    final_evidence_script = str((cli_dir / "final_evidence.py").resolve())

    commands: List[List[str]] = []
    run_id = _protocol_run_id(config)
    protocol_config_path = str(config_path)
    train_cfg = dict(config.get("train", {}))
    manifest_cfg = dict(config.get("validate_manifest", {}))
    baseline_cfg = dict(config.get("evaluate_baselines", {}))
    condition_cfg = dict(config.get("validate_conditions", {}))
    condition_benchmark_cfg = dict(config.get("condition_benchmark", {}))
    manufacturing_cfg = dict(config.get("manufacturing_constraints", {}))
    aircraft_validity_input_cfg = dict(config.get("prepare_aircraft_validity_inputs", {}))
    aircraft_validity_cfg = dict(config.get("aircraft_validity", {}))
    multi_seed_cfg = dict(config.get("multi_seed_eval", {}))
    final_evidence_cfg = dict(config.get("final_evidence", {}))
    checkpoint = _resolve_path(config, config.get("checkpoint")) or _resolve_path(config, train_cfg.get("checkpoint"))
    if not checkpoint:
        checkpoint = _default_checkpoint(config, train_cfg)

    if manifest_cfg.get("enabled"):
        manifest_path = manifest_cfg.get("manifest") or train_cfg.get("dataset_manifest")
        manifest_path = _resolve_path(config, manifest_path)
        if not manifest_path:
            raise ValueError("validate_manifest.enabled requires a manifest path or train.dataset_manifest")

        manifest_cmd = [python_exe, manifest_validator_script, "--manifest", manifest_path]
        _add_option(manifest_cmd, "--level", manifest_cfg.get("level", "basic"))
        output_path = _resolve_path(config, manifest_cfg.get("output"))
        _add_option(manifest_cmd, "--output", output_path)
        _add_option(manifest_cmd, "--run-id", run_id)
        _add_option(manifest_cmd, "--checkpoint", checkpoint)
        _add_option(manifest_cmd, "--protocol-config", protocol_config_path)
        commands.append(manifest_cmd)

    if train_cfg.get("enabled", True):
        train_cmd = [python_exe, cli_script, "train"]
        for flag, key in (
            ("--num-epochs", "num_epochs"),
            ("--batch-size", "batch_size"),
            ("--learning-rate", "learning_rate"),
            ("--latent-dim", "latent_dim"),
            ("--grid-size", "grid_size"),
            ("--precision", "precision"),
            ("--disconnection-penalty", "disconnection_penalty"),
            ("--num-samples", "num_samples"),
            ("--run-class", "run_class"),
            ("--solver", "solver"),
        ):
            _add_option(train_cmd, flag, train_cfg.get(key))
        for flag, key in (
            ("--dataset-artifact", "dataset_artifact"),
            ("--dataset-manifest", "dataset_manifest"),
            ("--resume-from", "resume_from"),
            ("--save-dir", "save_dir"),
            ("--baseline-config", "baseline_config"),
            ("--claim-gates", "claim_gates"),
        ):
            value = train_cfg.get(key)
            if key in {"dataset_artifact", "dataset_manifest", "resume_from", "save_dir", "baseline_config", "claim_gates"}:
                value = _resolve_path(config, value)
            _add_option(train_cmd, flag, value)
        for flag, key in (
            ("--enable-consistency", "enable_consistency"),
            ("--enable-pipeline", "enable_pipeline"),
            ("--enable-checkpointing", "enable_checkpointing"),
            ("--enable-compile", "enable_compile"),
        ):
            if key in train_cfg:
                _add_option(train_cmd, flag, bool(train_cfg[key]))
        commands.append(train_cmd)

    if baseline_cfg.get("enabled"):
        baseline_cmd = [python_exe, cli_script, "evaluate-baselines"]
        for flag, key in (
            ("--solver", "solver"),
            ("--grid-size", "grid_size"),
            ("--steps", "steps"),
            ("--output", "output"),
        ):
            value = baseline_cfg.get(key)
            if key == "output":
                value = _resolve_path(config, value)
            _add_option(baseline_cmd, flag, value)
        _add_option(baseline_cmd, "--baseline-config", _resolve_path(config, train_cfg.get("baseline_config")))
        _add_option(baseline_cmd, "--manifest", _resolve_path(config, train_cfg.get("dataset_manifest")))
        _add_option(baseline_cmd, "--checkpoint", checkpoint)
        _add_option(baseline_cmd, "--run-id", run_id)
        _add_option(baseline_cmd, "--protocol-config", protocol_config_path)
        commands.append(baseline_cmd)

    if condition_cfg.get("enabled"):
        condition_cmd = [python_exe, cli_script, "validate-conditions", "--checkpoint", checkpoint]
        for flag, key in (
            ("--num-seeds", "num_seeds"),
            ("--grid-size", "grid_size"),
            ("--output", "output"),
        ):
            value = condition_cfg.get(key)
            if key == "output":
                value = _resolve_path(config, value)
            _add_option(condition_cmd, flag, value)
        commands.append(condition_cmd)

    if condition_benchmark_cfg.get("enabled"):
        manifest_path = condition_benchmark_cfg.get("manifest") or train_cfg.get("dataset_manifest")
        manifest_path = _resolve_path(config, manifest_path)
        if not manifest_path:
            raise ValueError("condition_benchmark.enabled requires a manifest path or train.dataset_manifest")

        benchmark_cmd = [
            python_exe,
            condition_benchmark_script,
            "--checkpoint",
            checkpoint,
            "--manifest",
            manifest_path,
        ]
        output_path = _resolve_path(config, condition_benchmark_cfg.get("output"))
        _add_option(benchmark_cmd, "--output", output_path)
        if condition_benchmark_cfg.get("seeds"):
            _add_option(benchmark_cmd, "--seeds", condition_benchmark_cfg.get("seeds"))
        elif condition_benchmark_cfg.get("num_seeds"):
            num_seeds = int(condition_benchmark_cfg["num_seeds"])
            _add_option(benchmark_cmd, "--seeds", f"0-{max(0, num_seeds - 1)}")
        _add_option(benchmark_cmd, "--min-grounded-records", condition_benchmark_cfg.get("min_grounded_records"))
        _add_option(benchmark_cmd, "--min-effect", condition_benchmark_cfg.get("min_effect"))
        _add_option(benchmark_cmd, "--run-id", run_id)
        _add_option(benchmark_cmd, "--protocol-config", protocol_config_path)
        commands.append(benchmark_cmd)

    if manufacturing_cfg.get("enabled"):
        manifest_path = manufacturing_cfg.get("manifest") or train_cfg.get("dataset_manifest")
        manifest_path = _resolve_path(config, manifest_path)
        manufacturing_cmd = [python_exe, manufacturing_constraints_script]
        _add_option(manufacturing_cmd, "--manifest", manifest_path)
        _add_option(manufacturing_cmd, "--payload-json", manufacturing_cfg.get("payload_json"))
        output_path = _resolve_path(config, manufacturing_cfg.get("output"))
        _add_option(manufacturing_cmd, "--output", output_path)
        _add_option(manufacturing_cmd, "--run-id", run_id)
        _add_option(manufacturing_cmd, "--checkpoint", checkpoint)
        _add_option(manufacturing_cmd, "--protocol-config", protocol_config_path)
        commands.append(manufacturing_cmd)

    if aircraft_validity_input_cfg.get("enabled"):
        builder_cmd = [python_exe, aircraft_validity_input_builder_script]
        output_dir = _resolve_path(config, aircraft_validity_input_cfg.get("output_dir"))
        metadata_path = _resolve_path(config, aircraft_validity_input_cfg.get("metadata"))
        _add_option(builder_cmd, "--output-dir", output_dir)
        _add_option(builder_cmd, "--metadata", metadata_path)
        _add_option(builder_cmd, "--num-samples", aircraft_validity_input_cfg.get("num_samples"))
        _add_option(builder_cmd, "--grid-size", aircraft_validity_input_cfg.get("grid_size"))
        _add_option(builder_cmd, "--seed-start", aircraft_validity_input_cfg.get("seed_start"))
        _add_option(builder_cmd, "--max-attempts", aircraft_validity_input_cfg.get("max_attempts"))
        commands.append(builder_cmd)

    if aircraft_validity_cfg.get("enabled"):
        validity_cmd = [python_exe, aircraft_validity_script]
        for input_path in aircraft_validity_cfg.get("inputs", []) or []:
            _add_option(validity_cmd, "--input", _resolve_path(config, input_path))
        _add_option(validity_cmd, "--input-dir", _resolve_path(config, aircraft_validity_cfg.get("input_dir")))
        _add_option(validity_cmd, "--output", _resolve_path(config, aircraft_validity_cfg.get("output")))
        _add_option(validity_cmd, "--manifest", _resolve_path(config, train_cfg.get("dataset_manifest")))
        _add_option(validity_cmd, "--checkpoint", checkpoint)
        _add_option(validity_cmd, "--run-id", run_id)
        _add_option(validity_cmd, "--protocol-config", protocol_config_path)
        commands.append(validity_cmd)

    if multi_seed_cfg.get("enabled"):
        multi_cmd = [python_exe, multi_seed_script, "--checkpoint", checkpoint]
        for flag, key in (
            ("--num-seeds", "num_seeds"),
            ("--grid-size", "grid_size"),
            ("--output-dir", "output_dir"),
        ):
            value = multi_seed_cfg.get(key)
            if key == "output_dir":
                value = _resolve_path(config, value)
            _add_option(multi_cmd, flag, value)
        _add_option(multi_cmd, "--baseline-config", _resolve_path(config, train_cfg.get("baseline_config")))
        _add_option(multi_cmd, "--baseline-report", _resolve_path(config, baseline_cfg.get("output")))
        _add_option(multi_cmd, "--validation-report", _resolve_path(config, condition_cfg.get("output")))
        _add_option(multi_cmd, "--output-report", _resolve_path(config, final_evidence_cfg.get("baseline_statistics")))
        _add_option(multi_cmd, "--manifest", _resolve_path(config, train_cfg.get("dataset_manifest")))
        _add_option(multi_cmd, "--protocol-config", protocol_config_path)
        _add_option(multi_cmd, "--run-id", run_id)
        commands.append(multi_cmd)

    if manifest_cfg.get("enabled") and final_evidence_cfg.get("enabled"):
        final_manifest_cmd = [python_exe, manifest_validator_script, "--manifest", _resolve_path(config, manifest_cfg.get("manifest") or train_cfg.get("dataset_manifest"))]
        _add_option(final_manifest_cmd, "--level", manifest_cfg.get("level", "basic"))
        _add_option(final_manifest_cmd, "--output", _resolve_path(config, manifest_cfg.get("output")))
        _add_option(final_manifest_cmd, "--run-id", run_id)
        _add_option(final_manifest_cmd, "--checkpoint", checkpoint)
        _add_option(final_manifest_cmd, "--protocol-config", protocol_config_path)
        commands.append(final_manifest_cmd)

    if final_evidence_cfg.get("enabled"):
        evidence_cmd = [python_exe, final_evidence_script]
        report_paths = {
            "manifest_validation": manifest_cfg.get("output"),
            "aircraft_validity": aircraft_validity_cfg.get("output"),
            "condition_benchmark": condition_benchmark_cfg.get("output"),
            "manufacturing_constraints": manufacturing_cfg.get("output"),
            "baseline_statistics": final_evidence_cfg.get("baseline_statistics"),
        }
        for gate_id, report_path in report_paths.items():
            _add_option(evidence_cmd, f"--{gate_id.replace('_', '-')}", _resolve_path(config, report_path))
        if final_evidence_cfg.get("require_run_consistency"):
            _add_option(evidence_cmd, "--require-run-consistency", True)
        _add_option(evidence_cmd, "--output", _resolve_path(config, final_evidence_cfg.get("output")))
        commands.append(evidence_cmd)

    return commands


def run_commands(commands: Iterable[List[str]], *, dry_run: bool = False) -> None:
    for command in commands:
        print("$ " + " ".join(shlex.quote(part) for part in command))
        if dry_run:
            continue
        subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a checked-in smoke/final evaluation protocol.")
    parser.add_argument("--config", required=True, help="Path to a protocol YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    config = load_protocol_config(args.config)
    commands = build_protocol_commands(config)
    run_commands(commands, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
