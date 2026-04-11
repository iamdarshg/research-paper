# Jules' Research and Implementation Log

## Changes implemented on 2024-05-22

### Summary of Physics and Numerics Improvements
To address the 17–20% discrepancy in the drag coefficient ($C_d$) between the internal D3Q27 LBM solver and OpenFOAM results, the following surgical changes were made:

1.  **Refactored Mach/Velocity Mapping**:
    - The `mach_number` parameter in `CFDConfig` is now strictly interpreted as the physical Mach number.
    - The lattice velocity $u_{lattice}$ is derived using the standard relation $u_{lattice} = \text{Mach} \cdot c_s$, where $c_s = 1/\sqrt{3}$ for the D3Q27 lattice. This corrected a previous O(15%) offset caused by using a `/3.0` scaling factor.
    - Reference length for viscosity calculation now defaults to the physical object extent (e.g., cube edge length) rather than the entire domain size, ensuring the simulated Reynolds number matches the validation intent.

2.  **Lattice-Native Aerodynamic Coefficients**:
    - Implemented a "direct" Cd computation in `D3Q27CascadedSolver`. Instead of scaling through multiple physical units (which accumulates floating-point and scaling errors), the solver now derives $C_d$ and $C_l$ directly in lattice units: $C_d = F_{raw} / (0.5 \cdot \rho_0 \cdot u_{inf, lat}^2 \cdot A_{ref, lat})$.
    - The far-field lattice velocity $u_{inf, lat}$ is measured dynamically from the domain boundaries each time coefficients are computed, further reducing normalization bias.

3.  **Modular Boundary Conditions**:
    - Introduced a modular BC system for the D3Q27 solver.
    - Added support for **Equilibrium Inlet** (on the $x=0$ face) and **Neumann Outlet** (on the $x=L-1$ face).
    - This allows for an "apples-to-apples" comparison with OpenFOAM's inlet/outlet setup, moving away from the previous fully periodic domain which introduced recirculation artifacts.

4.  **Benchmark Infrastructure Enhancements**:
    - Updated `run_internal_benchmark.py` to automatically detect and use canonical reference areas and lengths for validation objects (e.g., the 1.0 unit cube).
    - Added an `--run-averaging-sweep` flag to help identify the optimal simulation window that minimizes transient effects.
    - Implemented robust JSON serialization for `torch.Tensor` and `numpy` types in benchmark reports.

5.  **Diagnostic Logging**:
    - Created `CLI/lbm_logger.py` to capture detailed physics parameters (viscosity, Mach, lattice velocity, forces) into `lbm_debug.log`.

### Citations
- [1] Krüger, T., et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer International Publishing. (Lattice scaling and normalization).
- [2] Guo, Z., & Shu, C. (2013). *Lattice Boltzmann Method and Its Applications in Engineering*. World Scientific. (Boundary condition implementations).
