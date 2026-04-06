# Repository State and Future Directions - Jules

This document outlines the current state of the Aircraft Structural Design via Diffusion Models + FluidX3D CFD project and provides recommendations for future development.

## Current State

### Repository Structure
The repository is organized into several key areas:
- `CLI/`: Contains the main command-line interface and the improved D3Q27 cascaded LBM solver.
- `paper/`: Contains the LaTeX source for the research paper.
- `.github/workflows/`: Contains CI/CD pipelines for building the paper PDF and running internal benchmarks.

### CFD Implementation
The CFD section now features a robust D3Q27 cascaded LBM solver. This solver is more accurate than the previous D3Q19 implementation, especially for complex flows, by using tensor-product raw moments and cascaded relaxation.

#### Error and Compute Time Requirements
- **Error Percentage:** Currently targeting < 5.5% deviation from OpenFOAM sonicFoam for standard validation objects (e.g., centered unit cube).
- **Execution Speed:** The GPU-accelerated implementation is optimized for 8-13GB VRAM. It can achieve hundreds of iterations per second (it/s) on modern hardware (e.g., A100 or RTX 3090), which is critical for the integrated training loop.
- **Progressive Refinement:** The pipeline uses grid sizes from 16³ to 32³, allowing for fast initial training and high-fidelity final generation.

## Future Ideas for ML (Diffusion Models)

1.  **Consistency Distillation Enhancements:**
    - Further reduce sampling steps from 4 to 1 or 2 while maintaining geometry fidelity using more advanced distillation techniques like Latent Consistency Models (LCM).
    - Explore "Trajectory Consistency" to ensure generated designs follow realistic aerodynamic evolutions.

2.  **Multimodal Conditioning:**
    - Condition the diffusion model on a wider range of design specifications, such as target lift-to-drag ratios, payload requirements, or structural weight constraints.
    - Integrate text-based conditioning for high-level design descriptions (e.g., "high-altitude long-endurance wing structure").

3.  **Hierarchical Voxel Generation:**
    - Move beyond fixed-resolution grids to octree-based or sparse voxel representations. This would allow the model to focus compute on complex surface details while using coarse voxels for internal volumes.

4.  **Physics-Informed Latent Space:**
    - Train the autoencoder to map geometries to a latent space that is organized by aerodynamic properties. This could make the diffusion process more efficient by navigating a "physics-aware" manifold.

5.  **Connectivity and Structural Integrity:**
    - Develop a more sophisticated connectivity loss that uses graph-based methods to ensure all parts of the aircraft are properly joined and can withstand aerodynamic loads.
    - Integrate a basic FEA (Finite Element Analysis) step in the latent space to score structural viability during generation.
