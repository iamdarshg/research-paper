# AI-Optimized Paper Airplane Folding

Research workflow using RL (Keras) to optimize folds on parametric A4 sheet for max range via surrogate + OpenFOAM CFD.

## Quick Start

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Customize `config.yaml` (goals, params).

3. Run training:
   ```
   python src/trainer/train.py
   ```

4. Launch GUI:
   ```
   streamlit run src/gui/app.py
   ```

## Project Structure

```
src/
├── folding/     # Sheet mesh + fold sim -> 3D STL
├── surrogate/   # Physics-based aero approx (panel method)
├── cfd/         # Docker OpenFOAM runner, adaptive mesh
├── rl_agent/    # Keras DDPG/PPO + Gym env
├── trainer/     # Multi-fidelity RL training
└── gui/         # Streamlit + Plotly/PyVista viz
```

## Features

- Customizable goals (range, speed, AoA via config)
- Multi-fidelity: Surrogate → high-res CFD
- Plotly 3D viz of folds, meshes, flow fields
- Live CFD iteration monitoring

## Usage

Edit `config.yaml` for experiments. Training logs/results in `data/`.

## Development

- Folding: Rigid origami kinematics (Trimesh/Shapely)
- Surrogate: Vortex lattice / panel method
- CFD: OpenFOAM10 Docker, blockMesh → snappyHex
- RL: Continuous actions (fold points), reward=range

See `paper/` for LaTeX manuscript.
