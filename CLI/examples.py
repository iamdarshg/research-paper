#!/usr/bin/env python3
"""Runnable examples for the current aircraft_diffusion_cfd CLI.

This file intentionally focuses on the public CLI surface instead of importing
internal training/generation classes directly. The implementation in
aircraft_diffusion_cfd.py is still a proof of concept, so command recipes are a
safer example surface than a pseudo-SDK.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


ROOT = Path(__file__).resolve().parent
CLI_SCRIPT = ROOT / "aircraft_diffusion_cfd.py"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "final_optimized_model.pt"


@dataclass(frozen=True)
class Example:
    name: str
    kind: str
    summary: str
    command_builder: Optional[Callable[[argparse.Namespace], list[str]]] = None
    notes: tuple[str, ...] = ()
    pseudocode: Optional[str] = None


def cli_command(*args: str) -> list[str]:
    return [sys.executable, str(CLI_SCRIPT), *args]


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    return DEFAULT_CHECKPOINT


def require_checkpoint(path: Path) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        "Checkpoint not found. Pass --checkpoint or create one first, for "
        f"example with: {CLI_SCRIPT.name} train --save-dir ./checkpoints"
    )


def build_info(_: argparse.Namespace) -> list[str]:
    return cli_command("info")


def build_benchmark(_: argparse.Namespace) -> list[str]:
    return cli_command("performance-benchmark")


def build_smoke_train(args: argparse.Namespace) -> list[str]:
    return cli_command(
        "train",
        "--num-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
        "--save-dir",
        args.save_dir,
    )


def build_generate(args: argparse.Namespace) -> list[str]:
    checkpoint = resolve_checkpoint(args)
    return cli_command(
        "generate",
        "--checkpoint",
        str(checkpoint),
        "--output",
        args.output,
        "--target-speed",
        str(args.target_speed),
        "--num-steps",
        str(args.num_steps),
    )


def build_batch_generate(args: argparse.Namespace) -> list[str]:
    checkpoint = resolve_checkpoint(args)
    return cli_command(
        "batch-generate",
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        args.output_dir,
        "--num-designs",
        str(args.num_designs),
    )


def build_resume_train(args: argparse.Namespace) -> list[str]:
    checkpoint = resolve_checkpoint(args)
    return cli_command(
        "train",
        "--num-epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-samples",
        str(args.num_samples),
        "--resume-from",
        str(checkpoint),
        "--save-dir",
        args.save_dir,
    )


EXAMPLES: dict[str, Example] = {
    "info": Example(
        name="info",
        kind="runnable",
        summary="Print environment and optimization status.",
        command_builder=build_info,
        notes=(
            "Fast smoke check.",
            "Does not require a checkpoint.",
        ),
    ),
    "benchmark": Example(
        name="benchmark",
        kind="runnable",
        summary="Print the static benchmark/status report.",
        command_builder=build_benchmark,
        notes=(
            "Fast smoke check.",
            "Reports compiled-in claims; it does not validate a training run.",
        ),
    ),
    "smoke-train": Example(
        name="smoke-train",
        kind="runnable",
        summary="Run a minimal training invocation against the current CLI.",
        command_builder=build_smoke_train,
        notes=(
            "Writes the final checkpoint as <save-dir>/final_optimized_model.pt.",
            "Uses the current CLI defaults for everything not explicitly overridden.",
        ),
    ),
    "generate": Example(
        name="generate",
        kind="runnable",
        summary="Generate one STL from an existing checkpoint.",
        command_builder=build_generate,
        notes=(
            "Requires a checkpoint file.",
            "The public CLI exposes a partial structured conditioning subset here, not the full scientific workflow.",
        ),
    ),
    "batch-generate": Example(
        name="batch-generate",
        kind="runnable",
        summary="Generate multiple STL files from an existing checkpoint.",
        command_builder=build_batch_generate,
        notes=(
            "Requires a checkpoint file.",
            "The current CLI records a condition manifest per STL and still uses fixed num_steps=4.",
        ),
    ),
    "resume-train": Example(
        name="resume-train",
        kind="runnable",
        summary="Resume training from an existing checkpoint.",
        command_builder=build_resume_train,
        notes=(
            "Requires a checkpoint produced by the current train command.",
            "Useful when you want to exercise --resume-from without editing the CLI.",
        ),
    ),
    "designspec-pseudocode": Example(
        name="designspec-pseudocode",
        kind="pseudocode",
        summary="Sketch of direct Python usage for custom DesignSpec weights.",
        notes=(
            "This is intentionally labeled pseudocode.",
            "The public CLI still does not expose the full schema or claim-bearing evaluation workflow.",
            "Internal Python classes are not treated as a stable API.",
        ),
        pseudocode="""from aircraft_diffusion_cfd import DesignSpec, OptimizedAircraftGenerator

generator = OptimizedAircraftGenerator("checkpoints/final_optimized_model.pt")
spec = DesignSpec(
    target_speed=25.0,
    space_weight=0.20,
    drag_weight=0.50,
    lift_weight=0.30,
)
voxels = generator.generate(spec, num_steps=4)
generator.voxels_to_stl(voxels, "custom_design.stl")
""",
    ),
}


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def print_example_list() -> None:
    print("Available examples:\n")
    for name, example in EXAMPLES.items():
        print(f"- {name}: {example.summary} [{example.kind}]")
    print("\nUse `python examples.py <name>` to inspect one example.")
    print("Add `--run` to execute runnable examples.")


def print_example(example: Example, args: argparse.Namespace) -> None:
    print(f"Example: {example.name}")
    print(f"Kind: {example.kind}")
    print(f"Summary: {example.summary}")

    if example.notes:
        print("\nNotes:")
        for note in example.notes:
            print(f"- {note}")

    if example.command_builder is not None:
        try:
            command = example.command_builder(args)
        except FileNotFoundError as exc:
            print(f"\nCommand: unavailable")
            print(f"Reason: {exc}")
            return
        print("\nCommand:")
        print(format_command(command))

    if example.pseudocode:
        print("\nPseudocode:")
        print(example.pseudocode.rstrip())


def run_example(example: Example, args: argparse.Namespace) -> int:
    if example.command_builder is None:
        print("This example is pseudocode only and cannot be executed directly.")
        return 2

    try:
        if example.name in {"generate", "batch-generate", "resume-train"}:
            require_checkpoint(resolve_checkpoint(args))
        command = example.command_builder(args)
    except FileNotFoundError as exc:
        print(exc)
        return 2

    print(format_command(command))
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or run current CLI examples for aircraft_diffusion_cfd.py."
    )
    parser.add_argument("example", nargs="?", help="Example name. Omit to list examples.")
    parser.add_argument("--run", action="store_true", help="Execute the example if it is runnable.")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint path for examples that need one. Defaults to ./checkpoints/final_optimized_model.pt.",
    )
    parser.add_argument("--save-dir", default="./checkpoints_smoke", help="Save directory for training examples.")
    parser.add_argument("--output", default="./artifacts/example_design.stl", help="Output STL path for generate.")
    parser.add_argument(
        "--output-dir",
        default="./generations_optimized_example",
        help="Output directory for batch-generate.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count for training examples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training examples.")
    parser.add_argument("--num-samples", type=int, default=8, help="Synthetic sample count for training examples.")
    parser.add_argument("--num-designs", type=int, default=2, help="Design count for batch-generate.")
    parser.add_argument("--target-speed", type=float, default=7.0, help="Target speed for generate.")
    parser.add_argument("--num-steps", type=int, default=4, help="Diffusion steps for generate.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.example:
        print_example_list()
        return 0

    example = EXAMPLES.get(args.example)
    if example is None:
        print(f"Unknown example: {args.example}\n")
        print_example_list()
        return 2

    if args.run:
        return run_example(example, args)

    print_example(example, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
