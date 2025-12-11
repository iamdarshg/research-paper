# Jules's Documentation

This file documents the changes made by Jules, an AI software engineer, to the Aircraft Structural Design via Diffusion Models + FluidX3D CFD project.

## Changes Made

1.  **Added D3Q27 Cascaded LBM Solver**: The `D3Q27CascadedSolver` has been integrated into the CFD simulation pipeline, providing a more accurate (though computationally intensive) alternative to the existing D3Q19 solver.
2.  **Solver Selection via CLI**: The `train` and `generate` commands now include a `--solver` option, allowing the user to select between the "D3Q19" and "D3Q27" solvers.
3.  **Memory-Constrained Training Strategy**: The training pipeline has been updated to use the fast D3Q19 solver for training iterations and the more accurate D3Q27 solver for validation. This allows for faster training without sacrificing validation accuracy.
4.  **Adaptive Mesh Refinement (AMR)**: The `AdvancedCFDSimulator` now supports AMR, which can be enabled via the `use_amr` flag in the `CFDConfig`.

## How to Use the New Solver Option

To use the D3Q27 solver for training, run the `train` command with the `--solver` option:

```bash
python CLI/aircraft_diffusion_cfd.py train --solver D3Q27
```

To use the D3Q27 solver for generating a design, run the `generate` command with the `--solver` option:

```bash
python CLI/aircraft_diffusion_cfd.py generate --checkpoint <path_to_checkpoint> --solver D3Q27
```

## Citations

[1] M. Thompson, et al., "trimesh: a Python library for working with triangular meshes," *Journal of Open Source Software*, vol. 4, no. 37, p. 1124, 2019. [Online]. Available: https://doi.org/10.21105/joss.01124

[2] S. van der Walt, et al., "scikit-image: image processing in Python," *PeerJ*, vol. 2, p. e453, 2014. [Online]. Available: https://doi.org/10.7717/peerj.453

[3] P. Virtanen, et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020. [Online]. Available: https://doi.org/10.1038/s41592-019-0686-2

[4] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. In *Advances in Neural Information Processing Systems* (Vol. 33). Curran Associates, Inc. Retrieved from https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

[5] Lorensen, W. E., & Cline, H. E. (1987). Marching cubes: A high resolution 3d surface construction algorithm. *ACM SIGGRAPH Computer Graphics*, 21(4), 163–169. https://doi.org/10.1145/37402.37422
