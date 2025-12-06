# AI-Optimized Paper Airplane Folding

Research workflow using RL (Keras) to optimize folds on parametric A4 sheet for max range via surrogate model with optional **FluidX3D CFD** integration.

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Install FluidX3D for high-fidelity CFD:
   ```
   # Download from https://www.fluidx3d.com/
   # Windows installer adds to PATH automatically
   ```

3. Customize `config.yaml` (goals, params).

4. Launch GUI (recommended):
   ```
   python launch_gui.py
   # Or manually:
   python -m streamlit run src/gui/app.py
   ```

5. Run training via GUI or command-line:
   ```
   python src/trainer/train.py
   ```

## Project Structure

```
src/
├── folding/     # Sheet mesh + fold simulation → 3D STL
├── surrogate/   # Physics-based aerodynamic model + FluidX3D integration
├── cfd/         # FluidX3D runner (GPU LBM CFD solver)
├── rl_agent/    # DDPG Agent + Gym environment
├── trainer/     # Recursive GNN + DDPG training
└── gui/         # Streamlit web interface + real-time visualization
```

## Features

✅ **GPU-Accelerated**:
- Parallel batch processing on GPU
- Real-time progress monitoring
- Automatic batch-size detection

✅ **Dual Training Methods**:
- **DDPG Agent**: Reinforcement learning direct optimization
- **Recursive GNN**: Graph neural networks for pattern learning (TRM/ARC-inspired)

✅ **Aerodynamic Analysis**:
- Physics-based surrogate model (fast)
- **FluidX3D CFD** integration (high-fidelity, optional)
- Automatic fallback if FluidX3D unavailable

✅ **Interactive GUI**:
- 4-tab Streamlit interface
- Example analyses with performance graphs
- Real-time training visualization
- GPU device selector
- Batch evaluation tools

## GPU Support

- **Automatic Detection**: All NVIDIA GPUs detected automatically
- **Device Selector**: Choose GPU in Streamlit sidebar
- **Auto-Batching**: Batch size optimized per device VRAM
- **Memory Efficient**: ~1-4GB per training method

## Training Methods

### DDPG (Reinforcement Learning)
- Deep Deterministic Policy Gradient
- Direct optimization of fold sequences
- Time: 5-10 min for 100 episodes on GPU
- Best for: Single-objective optimization

### Recursive GNN (Pattern Recognition)
- 3-level hierarchical graph neural network
- Multi-head attention mechanisms
- ARC/TRM-inspired architecture
- Time: 2-5 min for 50 epochs on GPU
- Best for: Pattern recognition, transfer learning

- CFD: OpenFOAM10 Docker, blockMesh → snappyHex
- RL: Continuous actions (fold points), reward=range

See `paper/` for LaTeX manuscript.
