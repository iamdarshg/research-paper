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


def _add_dual_flag(command: List[str], enable_flag: str, disable_flag: str, value: Any) -> None:
    if value is None:
        return
    command.append(enable_flag if value else disable_flag)


def _default_checkpoint(config: Dict[str, Any], train_cfg: Dict[str, Any]) -> str:
    save_dir = _resolve_path(config, train_cfg.get("save_dir", "./checkpoints"))
    return str((Path(save_dir) / "final_optimized_model.pt").resolve())


def build_protocol_commands(config: Dict[str, Any]) -> List[List[str]]:
    config_path = Path(config["_config_path"]).resolve()
    cli_dir = config_path.parent.parent
    python_exe = sys.executable
    cli_script = str((cli_dir / "aircraft_diffusion_cfd.py").resolve())
    multi_seed_script = str((cli_dir / "multi_seed_eval.py").resolve())

    commands: List[List[str]] = []
    train_cfg = dict(config.get("train", {}))
    if train_cfg.get("enabled", True):
        train_cmd = [python_exe, cli_script, "train"]
        for flag, key in (
            ("--num-epochs", "num_epochs"),
            ("--batch-size", "batch_size"),
            ("--learning-rate", "learning_rate"),
            ("--latent-dim", "latent_dim"),
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
        for enable_flag, disable_flag, key in (
            ("--enable-consistency", "--disable-consistency", "enable_consistency"),
            ("--enable-pipeline", "--disable-pipeline", "enable_pipeline"),
            ("--enable-checkpointing", "--disable-checkpointing", "enable_checkpointing"),
        ):
            if key in train_cfg:
                _add_dual_flag(train_cmd, enable_flag, disable_flag, bool(train_cfg[key]))
        if "enable_compile" in train_cfg:
            _add_option(train_cmd, "--enable-compile", bool(train_cfg["enable_compile"]))
        commands.append(train_cmd)

    checkpoint = _resolve_path(config, config.get("checkpoint")) or _resolve_path(config, train_cfg.get("checkpoint"))
    if not checkpoint:
        checkpoint = _default_checkpoint(config, train_cfg)

    baseline_cfg = dict(config.get("evaluate_baselines", {}))
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
        commands.append(baseline_cmd)

    condition_cfg = dict(config.get("validate_conditions", {}))
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

    multi_seed_cfg = dict(config.get("multi_seed_eval", {}))
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
        commands.append(multi_cmd)

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
