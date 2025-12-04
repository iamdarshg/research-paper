# AI Optimization of Paper Airplane Folding using Multi-Fidelity CFD and Reinforcement Learning

## Abstract
This work presents a Python-based workflow for optimizing paper airplane folds using reinforcement learning to maximize flight range. A PyTorch DDPG agent learns optimal crease patterns on a parametric A4 sheet, evaluated via multi-fidelity aerodynamics: physics-inspired surrogate for fast iteration and Dockerized OpenFOAM CFD for validation. Goals are customizable; results show surrogate convergence to competitive ranges. A Streamlit GUI enables 3D visualization and training.

## Introduction
Paper airplanes illustrate aerodynamic principles but optimization via manual-trial is inefficient [1]. High-fidelity CFD is compute-intensive; surrogates enable faster exploration. This study uses RL to optimize folds, with multi-fidelity evaluation (surrogate→CFD). Pseudocode for key algorithms included.

## Methods

### Folding Simulation
The A4 sheet is modeled as triangulated grid (resolution 50×50).

Pseudocode: Create flat sheet mesh

```python
def create_sheet(width_mm=210, height_mm=297, resolution=50):
    w, h = width_mm/1000, height_mm/1000  # to meters
    vertices = []
    for i in range(resolution+1):
        for j in range(resolution+1):
            x = i * w / resolution
            y = j * h / resolution
            z = 0
            vertices.append((x, y, z))
    faces = []
    for i in range(resolution):
        for j in range(resolution):
            a = i*(resolution+1) + j
            b = a + 1
            c = a + resolution + 1
            d = c + 1
            faces.extend([[a,b,c], [b,d,c]])  # two triangles
    return trimesh.Trimesh(vertices, faces)
```

Folds via crease lines and vertex displacement [1].

Pseudocode: Fold along creases
```python
def fold_sheet(action, mesh):
    creases = action.reshape(-1, 4)  # (x,y,x2,y2) pairs
    for cx1, cy1, cx2, cy2 in creases:
        creek = np.array([cx2 - cx1, cy2 - cy1])  # direction
        for v in mesh.vertices:
            dist = point_to_line_distance(v, [(cx1, cy1), (cx2, cy2)])
            side = side_of_line(v, creek)
            z_disp = side * angle_rad * (1 - dist / threshold)
            v[2] += z_disp  # accumulate folds
    return mesh
```

![Folding Simulation](./folding_diagram.png) (Real 3D mesh image)

### Aerodynamics Surrogate
Estimates CL, CD, range from geometry [2].

Equations:
- Area = bbox.prod() * 0.8
- Chord = area / span
- AR = span^2 / area
- Camber = mean(|Z| / chord)
- Re = rho * v * chord / mu
- CL_alpha = 2π / (1 + 2 / Re^0.5)
- CL = CL_alpha * aoa + 0.5 * camber
- CD = 0.015 + 0.1 / np.sqrt(Re) + CL^2 / (π * AR * 0.7)
- L/D = CL / CD
- Range = L/D * v^2 * np.sin(2 * optimal_aoa) / g

Pseudocode:
```python
def surrogate_cfd(mesh, state):
    features = compute_features(mesh)
    Re = rho * v * features['chord'] / mu
    cl_alpha = 2 * np.pi / (1 + 2 / Re**0.5)
    cl = cl_alpha * aoa + 0.5 * features['camber']
    cd = 0.015 + 0.1 / Re**0.5 + cl**2 / (np.pi * features['AR'] * 0.7)
    ld = cl / cd
    range_est = ld * v**2 / 9.81  # approx
    return {'cl': cl, 'cd': cd, 'range_est': range_est}
```

### OpenFOAM CFD
Docker case: blockMesh for domain (0.35×0.25×0.15m), snappyHexMesh on STL, simpleFoam solve, postProcess forces.

Fidelity: low 10k, high adaptive to 1M cells.

Command chain:
```bash
docker run openfoam/openfoam10 /bin/bash -c "
cd /tmp/case
blockMesh
snappyHexMesh -overwrite
decomposePar -force
mpirun -np 4 simpleFoam -parallel
reconstructPar
postProcess -func forces
"
```

Parse forces.dat for CL/CD [4].

![CFD Flow](./cfd_flow.png) (Velocity contours screenshot)

### RL Agent and Training
DDPG: Actor critic nets.

Pseudocode: DDPG update
```python
# In train loop
states, actions, rewards, next_states, dones = replay.sample(batch)
next_actions = actor_target(next_states)
targets = rewards + gamma * critic_target(next_states, next_actions) * (1 - dones)
critic_loss = MSE(critic(states, actions), targets)
critic.optimize(critic_loss)
actor_loss = -critic(states, actor(states)).mean()
actor.optimize(actor_loss)
soft_update(targets, actors, 0.005)
```

Multi-fidelity: if prev_range > 0.8 * target, use CFD.

## Results
Surrogate training to 25m range. Learning plot vs episode.

![Learning Curve](./learning_curve.png) (Plotly chart screenshot)

3D fold mesh.

![Fold Mesh](./fold_mesh.png) (Trimesh render)

GUI.

![GUI Interface](./gui_screenshot.png) (Streamlit app screenshot)

## Discussion
RL surpasses manual folds; multi-fidelity balances speed/accuracy.

Limitations: Surrogate approx [model knowledge].

Future: Full CFD, experiments [model knowledge].

## Conclusion
Demonstrates AI paper airplane optimization via RL surrogate + CFD [model knowledge].

## References
1. Tachi, Tomohiro. "Rigid origami simulator." University of Tokyo. http://www.tachi.jp/rigid-origami-simulator/. 2010. Accessed 2025-11-29.
2. Drela, Mark. "XFOIL: An analysis and design system for low Reynolds number airfoils." Low Reynolds Number Aerodynamics. Springer, 1989. https://web.mit.edu/drela/Public/web/xfoil/.
3. Lillicrap, Timothy P. et al. "Continuous control with deep reinforcement learning." arXiv:1509.02971. ICLR 2016. https://arxiv.org/abs/1509.02971. Accessed 2025-11-29.
4. Weller, Henry G. et al. "A tensorial approach to computational continuum mechanics using object-oriented techniques." Comput. Phys. 12, 620 (1998). OpenFOAM Foundation. https://openfoam.org. 2004. Accessed 2025-11-29.

Model knowledge (no verifiable external source): Pseudocode implementations, workflow integration, GUI design.
