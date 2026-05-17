# Aircraft Diffusion CFD - Research Repo

> Current status: a proof-of-concept research codebase for synthetic voxel generation, CFD-informed scoring, and reproducible benchmarking. It is not yet a validated aircraft-design system or a mission/manufacturing-conditioned airplane generator.

## Overview

This repository combines a latent generative model, voxel decoding, internal lattice-Boltzmann-style scoring, and an OpenFOAM export path. The current experiments are intentionally narrow: they use synthetic training data and reduced sanity runs to validate the code path, not to establish publication-grade aerodynamic or structural performance.

## Current Scope

- Proof-of-concept latent generation of freeform or aircraft-like voxel geometries
- Internal D3Q27/OpenFOAM benchmark path for solver cross-checks
- STL export and reproducible local validation tooling
- Small-scale training smoke runs on commodity hardware

## Not Yet Implemented At Claimable Quality

- Flight-profile-conditioned generation
- Manufacturing-method-conditioned generation
- Real aircraft dataset training
- Structural validation beyond connectivity heuristics
- Publication-grade aerodynamic optimization claims

### Key Features

- 🚀 **Fast Training**: Progressive grid refinement (16³ → 24³ → 32³) with only 4-6 hours on A100
- 💾 **Memory Efficient**: Optimized for 8-13GB VRAM using gradient checkpointing & sparse grids
- ✈️ **Aerodynamic Optimization**: Built-in CFD simulator evaluates drag, lift, and structural constraints
- 🎯 **Connectivity Constraints**: Ensures generated designs are structurally viable
- 📊 **Real-time Monitoring**: TensorBoard integration for training visualization
- 📦 **STL Export**: Convert volumetric designs to production-ready meshes

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd research-paper/CLI

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python aircraft_diffusion_cfd.py info
```

Expected output shows PyTorch version, CUDA availability, and GPU memory.

### 3. Train a Model

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 100 \
  --batch-size 4 \
  --num-samples 100 \
  --save-dir ./checkpoints
```

For an honest local smoke run on 8 GB hardware, use much smaller settings first:

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 2 \
  --save-dir ./checkpoints_smoke
```

### 4. Generate Designs

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output ./designs\design_0.stl \
  --target-speed 50.0 \
  --num-steps 4
```

### 5. Batch Generation

```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output-dir ./designs \
  --num-designs 5
```

### 6. Run the Validation Benchmark

```bash
python3 run_internal_benchmark.py
```

This runs the internal D3Q27 solver and, when OpenFOAM is available, a local sonicFoam case for `20mm_cube.stl` plus any other root-level `*.stl` files, then prints a JSON summary with the force comparison. On Linux or Windows/WSL, pass `--install-openfoam` to let the benchmark bootstrap OpenFOAM if it is missing.

## Commands Reference

### `train`
Train the diffusion model from scratch or resume from checkpoint.

**Key Arguments:**
- `--num-epochs` (int): Training epochs at full resolution (default: 100)
- `--batch-size` (int): Batch size; adjust based on VRAM (default: 4)
- `--learning-rate` (float): Adam optimizer learning rate (default: 2e-4)
- `--latent-dim` (int): Latent space dimensionality (default: 128)
- `--disconnection-penalty` (float): Penalty for disconnected structures (default: 10.0)
- `--num-samples` (int): Synthetic training data samples (default: 100)
- `--resume-from` (str): Path to checkpoint to resume training
- `--save-dir` (str): Directory for saving checkpoints (default: ./checkpoints)

**Example:**
```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 150 \
  --batch-size 3 \
  --learning-rate 1e-4 \
  --disconnection-penalty 15.0 \
  --num-samples 200
```

### `generate`
Generate new aircraft designs using a trained model.

**Key Arguments:**
- `--checkpoint` (str): Path to trained model checkpoint (required)
- `--output` (str): Output STL path (default: `aircraft_optimized.stl`)
- `--target-speed` (float): Scalar design target carried through the current CLI (default: `7.0`)
- `--num-steps` (int): Number of consistency-model sampling steps (default: `4`)
- `--use-marching-cubes` (bool flag): Export via marching cubes when possible

**Example:**
```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output ./generated_designs\design_0.stl \
  --target-speed 50.0 \
  --num-steps 4
```

### `batch-generate`
Generate multiple STL artifacts from a trained checkpoint.

**Key Arguments:**
- `--checkpoint` (str): Path to trained model checkpoint (required)
- `--output-dir` (str): Output directory for generated STL files
- `--num-designs` (int): Number of STL files to emit

**Example:**
```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output-dir ./generated_designs \
  --num-designs 5
```

### `info`
Print the runtime environment and optimization status.

```bash
python aircraft_diffusion_cfd.py info
```

### `info`
Display system information and GPU/CUDA status.

```bash
python aircraft_diffusion_cfd.py info
```

## System Requirements

### Hardware
- **GPU**: NVIDIA CUDA-capable GPU with 8GB+ VRAM
  - 8GB: 16³ training only
  - 10-13GB: Full 16³ → 32³ progressive training
- **CPU**: Multi-core processor (6+ cores recommended)
- **RAM**: 16GB+ system RAM

### Software
- **Python**: 3.9+ (3.10/3.11 recommended)
- **CUDA**: 11.8+ or 12.x
- **cuDNN**: 8.7+

### Dependencies
See `requirements.txt`. Key packages:
- PyTorch ≥ 2.0.0
- NumPy ≥ 1.24.0
- SciPy ≥ 1.10.0
- scikit-image ≥ 0.22.0
- TensorBoard ≥ 2.13.0
- TrimMesh ≥ 3.20.0

## Training Performance

### Progressive Training Schedule

| Grid Size | Epochs | Memory | Time (RTX 3090) |
|-----------|--------|--------|-----------------|
| 16³       | 50     | ~3GB   | 2-3 hrs         |
| 24³       | 50     | ~6GB   | 4-5 hrs         |
| 32³       | 100    | ~10GB  | 8-10 hrs        |
| **Total** | 200    | Peak 10GB | ~14-18 hrs  |

*Note: A100 GPUs are ~2-3x faster than RTX 3090*

## Configuration

### Config File (`config.yaml`)

```yaml
diffusion:
  timesteps: 100
  beta_start: 0.0001
  beta_end: 0.02
  sampling_timesteps: 250
  guidance_scale: 7.5

model:
  latent_dim: 128
  encoder_channels: [32, 64, 128, 256]
  decoder_channels: [256, 128, 64, 32]

training:
  batch_size: 4
  learning_rate: 0.0002
  num_epochs: 100
  disconnection_penalty: 10.0

cfd:
  reynolds_number: 1e5
  mach_number: 0.3
  simulation_steps: 1000
```

## Project Structure

```
CLI/
├── aircraft_diffusion_cfd.py    # Main CLI entry point
├── advanced_lbm_solver.py        # GPU-accelerated CFD simulator
├── requirements.txt              # Python dependencies
├── config.yaml                   # Default configuration
├── QUICKSTART.md                 # 5-minute setup guide
├── ARCHITECTURE.md               # Technical deep dive
├── README.md                     # Original detailed README
├── checkpoints/                  # Trained model checkpoints
├── runs/                         # TensorBoard logs
└── reference/                    # Reference implementations
    └── complete_amr_d3q27_cascaded_guide.py
```

## Examples

### Example 1: Train from Scratch
```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 100 \
  --batch-size 4 \
  --num-samples 100
```

### Example 2: Generate 100 Designs
```bash
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output-dir ./generated_designs \
  --num-designs 100
```

### Example 3: Full Pipeline
```bash
# Train
python aircraft_diffusion_cfd.py train --num-epochs 100

# Generate
python aircraft_diffusion_cfd.py generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output best_aircraft.stl \
  --target-speed 50.0 \
  --num-steps 4

# Optional batch generation
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint checkpoints/final_optimized_model.pt \
  --output-dir ./designs \
  --num-designs 20
```

## Monitoring Training

Real-time training metrics are logged to TensorBoard:

```bash
tensorboard --logdir ./runs
```

Then open http://localhost:6006 in your browser to view:
- Loss curves (total, connectivity, aerodynamic)
- Learning rate schedule
- Design sampling previews
- CFD simulation results

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch-size` to 2 or 1
- Reduce `--num-samples` to 50
- Start with 16³ grid training only

### Slow Training
- Ensure CUDA is available: `python aircraft_diffusion_cfd.py info`
- Check GPU utilization with `nvidia-smi`
- Reduce `--num-samples` to speed up data loading

### Poor Design Quality
- Increase `--num-epochs` to 150+
- Reduce `--disconnection-penalty` if too restrictive
- Ensure training has converged (check TensorBoard)
- Try higher `--guidance-scale` in generation

## Citation

If you use this project in your research, please cite:

```bibtex
@software{aircraft_diffusion_cfd,
  title={Aircraft Diffusion CFD: Generative Design via Diffusion Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/research-paper}
}
```

## License

[Your License Here]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Contact

For questions or support, open an issue on GitHub or contact the maintainers.

---

**Last Updated**: December 2025
