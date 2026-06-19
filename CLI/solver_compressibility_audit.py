from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditRow:
    assumption: str
    location: str
    physical_consequence: str
    valid_regime: str
    required_fix_or_gate: str


AUDIT_ROWS = [
    AuditRow(
        "Second-order isothermal D3Q27 equilibrium",
        "CLI/advanced_lbm_solver.py:151; CLI/advanced_lbm_solver.py:274; CLI/cascaded_lbm.py:206; CLI/lbm_utils.py:145",
        "No energy distribution and no temperature evolution; density-pressure relation is isothermal.",
        "Low-Mach weakly compressible flows where density fluctuations stay small.",
        "Gate internal results above Mach 0.3 as experimental until a thermal/compressible model is implemented and validated.",
    ),
    AuditRow(
        "Raw/tensor-product moment basis with fixed isothermal cs2",
        "CLI/advanced_lbm_solver.py:73; CLI/advanced_lbm_solver.py:156; CLI/advanced_lbm_solver.py:398; CLI/cascaded_lbm.py:50",
        "Collision relaxes moments toward isothermal raw-moment equilibria, not compressible energy moments.",
        "Low-Mach weakly compressible/isothermal D3Q27 operation.",
        "Add thermal moments or a coupled energy distribution before making compressible LBM claims.",
    ),
    AuditRow(
        "Pressure model p = rho / 3 in lattice units",
        "CLI/advanced_lbm_solver.py:467; CLI/advanced_lbm_solver.py:592; CLI/advanced_lbm_solver.py:634; CLI/cascaded_lbm.py:379",
        "Pressure lacks perfect-gas p=rho R T coupling and cannot model compressible thermodynamics.",
        "Isothermal LBM with fixed lattice sound speed.",
        "Document as weakly compressible only; external compressible OpenFOAM is required for high-Mach reference data.",
    ),
    AuditRow(
        "Speed of sound fixed by D3Q27 isothermal lattice",
        "CLI/lbm_utils.py:9; CLI/advanced_lbm_solver.py:23; CLI/advanced_lbm_solver.py:31; CLI/lbm_utils.py:96",
        "Internal acoustic scaling is fixed; there is no temperature-dependent sound speed.",
        "Current low-Mach isothermal model.",
        "Implement thermal state and gas EOS before claiming validated compressible acoustics.",
    ),
    AuditRow(
        "Mach-to-lattice mapping u_lattice = Ma / sqrt(3)",
        "CLI/lbm_utils.py:15; CLI/advanced_lbm_solver.py:569; CLI/cascaded_lbm.py:175",
        "The internal solver preserves the requested Mach only while stability clipping does not reduce u_lattice.",
        "Claim-grade only through Mach 0.3 and current low-Mach tests.",
        "Emit u_lattice and lattice_mach in every output so clipping is visible.",
    ),
    AuditRow(
        "Viscosity from low-Mach LBM relaxation relation",
        "CLI/advanced_lbm_solver.py:551; CLI/advanced_lbm_solver.py:602; CLI/cascaded_lbm.py:183; CLI/cascaded_lbm.py:297",
        "Relaxation time sets lattice viscosity for the isothermal solver; it does not encode high-speed gas transport.",
        "Under-resolved low-Mach sanity/regression cases.",
        "For compressible work, add Mach/Re nondimensionalization with temperature-dependent viscosity or state an explicit gate.",
    ),
    AuditRow(
        "Momentum-exchange wall force",
        "CLI/advanced_lbm_solver.py:180; CLI/advanced_lbm_solver.py:218; CLI/advanced_lbm_solver.py:453; CLI/cascaded_lbm.py:340",
        "Force is raw bounce-back/BFL momentum exchange, not a compressible pressure/shear surface integration.",
        "Low-Mach voxelized wall-force sanity checks.",
        "Keep raw force separate from calibrated/surrogate force and require external compressible validation for high-Mach force claims.",
    ),
    AuditRow(
        "Far-field and wall boundaries are low-Mach/simple bounce-back style",
        "CLI/advanced_lbm_solver.py:258; CLI/advanced_lbm_solver.py:290; CLI/advanced_lbm_solver.py:430; CLI/d3q27_kernels.py:48",
        "No subsonic/supersonic inlet/outlet characteristic treatment and no shock-aware boundary handling.",
        "Low-Mach exploratory internal runs.",
        "Gate transonic/supersonic internal runs as experimental until compressible-aware boundaries exist.",
    ),
    AuditRow(
        "LES/turbulence diagnostics are not a validated compressible turbulence closure",
        "CLI/advanced_lbm_solver.py:654; CLI/advanced_lbm_solver.py:674; CLI/advanced_lbm_solver.py:842; CLI/config.py:82",
        "Smagorinsky-like diagnostics may stabilize/diagnose but do not validate compressible turbulent physics.",
        "Qualitative low-Mach diagnostics.",
        "Keep turbulence outputs as diagnostics and do not use them to upgrade claim grade.",
    ),
    AuditRow(
        "Convergence gate uses finite fields and force stability",
        "CLI/advanced_lbm_solver.py:617; CLI/advanced_lbm_solver.py:644; CLI/advanced_lbm_solver.py:828",
        "A numerically stable internal run can still be physically invalid for high Mach.",
        "Internal low-Mach sanity gate only.",
        "Emit lbm_converged separately from validity_regime and claim_grade.",
    ),
    AuditRow(
        "Training label source selection is tiered",
        "CLI/advanced_lbm_solver.py:848; CLI/cfd_simulator.py:95; CLI/cfd_simulator.py:119; CLI/data_utils.py:253",
        "Internal LBM labels can train surrogates, but cannot be silently treated as external ground truth.",
        "Low-Mach raw LBM for internal/surrogate training; external PDE for PINN-ready labels.",
        "High-Mach internal LBM must set training_drag_source to none_high_mach_internal_lbm_unvalidated.",
    ),
]


def render_markdown() -> str:
    lines = [
        "# Solver Compressibility Audit",
        "",
        "This audit documents where the current internal D3Q27 LBM path remains low-Mach, weakly compressible, and isothermal.",
        "",
        "| assumption | location | physical consequence | valid regime | required fix or gate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in AUDIT_ROWS:
        lines.append(
            f"| {row.assumption} | `{row.location}` | {row.physical_consequence} | "
            f"{row.valid_regime} | {row.required_fix_or_gate} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Internal D3Q27 low-Mach raw outputs are bounded sanity/regression evidence, not production CFD.",
            "- Internal D3Q27 Mach > 0.3 outputs are executable exploratory runs labeled `experimental_high_mach_unvalidated`.",
            "- Compressible/high-Mach claim support must come from external compressible OpenFOAM evidence or a future thermal/compressible LBM implementation with validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "build" / "solver_diagnostics" / "compressibility_audit_20260612",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "solver_compressibility_audit.md"
    output.write_text(render_markdown(), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
