# Quick Start Guide

## 5-Minute Setup

### 1. Install

```bash
# Clone or download the repository
git clone <your-repo-url>
cd aircraft-diffusion-cfd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Check Your GPU

```bash
python aircraft_diffusion_cfd.py info
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3090
CUDA capability: (8, 6)
Total GPU memory: 23.70 GB
```

### 3. Train (Start Small)

```bash
python aircraft_diffusion_cfd.py train \
  --num-epochs 50 \
  --batch-size 4 \
  --num-samples 50
```

This runs for ~2-3 hours on RTX 3090 and generates a basic trained model.

### 4. Generate Design

```bash
python aircraft_diffusion_cfd.py generate \
  --checkpoint checkpoints/final_model.pt \
  --output my_aircraft.stl
```

### 5. Open in CAD Software

- **Free**: FreeCAD, Blender, Fusion 360 (educational)
- **Commercial**: Solidworks, CATIA
- **CAM**: Fusion 360, FreeCAD

---

## Understanding the Output

### What You Get

1. **STL File** (`aircraft.stl`)
   - 3D mesh ready for 3D printing or CAD analysis
   - ~5-50MB depending on resolution and marching cubes
   - Binary format (80-byte header + triangle data)

2. **Voxel Grid** (optional numpy export)
   - Raw 32×32×32 (or custom) occupancy grid
   - Each voxel: 0 (air) to 1 (solid)

3. **Training Logs** (TensorBoard)
   - View loss curves: `tensorboard --logdir ./runs`
   - Real-time metrics on loss components

### Visualizing Results

**Using Python:**
```python
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# Load voxel grid
voxels = np.load('voxel_grid.npy')

# Plot slice
plt.imshow(voxels[:, :, 16], cmap='gray')
plt.colorbar()
plt.title('Vertical Slice of Aircraft')
plt.show()

# Connectivity analysis
from scipy.ndimage import label
binary = (voxels > 0.5).astype(int)
labeled, num_components = label(binary)
print(f"Connected components: {num_components}")
```

**Using FreeCAD:**
1. File → Open → `aircraft.stl`
2. View → Fit All (or press Home)
3. Change shading: View → Standard View → Flat Lines

---

## Common Workflows

### Workflow 1: Quick Test (10 minutes)

```bash
# Train on small dataset
python aircraft_diffusion_cfd.py train \
  --num-epochs 10 \
  --batch-size 2 \
  --num-samples 20

# Generate one design
python aircraft_diffusion_cfd.py generate \
  --checkpoint checkpoints/final_model.pt

# You'll have: aircraft.stl
```

### Workflow 2: Production Training (4-6 hours)

```bash
# Full training
python aircraft_diffusion_cfd.py train \
  --num-epochs 100 \
  --batch-size 4 \
  --num-samples 100 \
  --latent-dim 128 \
  --disconnection-penalty 10.0

# Generate batch
python aircraft_diffusion_cfd.py batch-generate \
  --checkpoint checkpoints/final_model.pt \
  --num-designs 10 \
  --output-dir ./aircraft_designs
```

### Workflow 3: Custom Objective

```python
# examples.py - modify as needed
from aircraft_diffusion_cfd import DesignSpec

# High-speed fighter
fighter_spec = DesignSpec(
    target_speed=200.0,
    space_weight=0.1,   # Compact
    drag_weight=0.7,    # Aerodynamic
    lift_weight=0.2
)

# Your generation code...
```

### Workflow 4: Memory-Constrained (8GB GPU)

```bash
python aircraft_diffusion_cfd.py train \
  --batch-size 2 \
  --latent-dim 64 \
  --num-epochs 50 \
  --num-samples 50

# May take longer but fits in 8GB VRAM
```

---

## Key Parameters Explained

### Training Parameters

| Parameter | Default | 8GB GPU | 10GB GPU | 12GB+ GPU |
|-----------|---------|---------|----------|-----------|
| `batch-size` | 4 | 2 | 3 | 4 |
| `latent-dim` | 128 | 64 | 96 | 128 |
| `num-samples` | 100 | 50 | 75 | 100 |
| `num-epochs` | 100 | 50 | 75 | 100 |
| Memory Usage | ~12GB | ~7GB | ~9GB | ~12GB |

### Generation Parameters

| Parameter | Effect |
|-----------|--------|
| `--num-steps` | 50 = fast/rough, 250 = slow/smooth |
| `--target-speed` | Design for specific speed (m/s) |
| `--use-marching-cubes` | Smooth surface (True) vs blocky (False) |

**Rule of thumb:**
- **num-steps**: More steps = better quality but slower (250 is good)
- **batch-size**: Smaller batch = less memory but slower training
- **latent-dim**: Larger = more detail but more memory

---

## Troubleshooting

### "CUDA out of memory"

**Solution 1: Reduce batch size**
```bash
python aircraft_diffusion_cfd.py train --batch-size 1
```

**Solution 2: Reduce latent dimension**
```bash
python aircraft_diffusion_cfd.py train --latent-dim 64
```

**Solution 3: Reduce num-samples**
```bash
python aircraft_diffusion_cfd.py train --num-samples 25
```

### "Marching cubes failed"

The system falls back to voxel export automatically. This is fine—the STL is still valid, just looks more blocky.

To debug:
```python
from scipy.ndimage import label
import numpy as np

voxels = np.load('voxel_grid.npy')
binary = (voxels > 0.5).astype(int)
labeled, num = label(binary)
print(f"Connected: {num == 1}")  # Should be True
```

### "FileNotFoundError: checkpoint not found"

Check the path:
```bash
ls -la checkpoints/
python aircraft_diffusion_cfd.py generate \
  --checkpoint ./checkpoints/final_model.pt
```

### "Module 'aircraft_diffusion_cfd' not found"

Ensure you're in the right directory:
```bash
pwd  # Should show: .../aircraft-diffusion-cfd/

# If not:
cd /path/to/aircraft-diffusion-cfd
python aircraft_diffusion_cfd.py info
```

---

## Next Steps

1. **Understand the architecture**: Read README.md Core Architecture section
2. **Customize objectives**: Modify `DesignSpec` in examples.py
3. **Analyze results**: Use `example_analyze_geometry()` in examples.py
4. **Fine-tune**: Use `example_finetune_on_custom_data()` for custom aircraft
5. **Integrate CFD**: Connect to external CFD solvers (ANSYS, OpenFOAM)

---

## Performance Tips

### Speed Up Training

1. **Use larger batch size** (if VRAM allows)
   ```bash
   python aircraft_diffusion_cfd.py train --batch-size 8
   ```

2. **Reduce disconnection penalty** (trains faster but less connected)
   ```bash
   python aircraft_diffusion_cfd.py train --disconnection-penalty 5.0
   ```

3. **Use smaller latent dimension**
   ```bash
   python aircraft_diffusion_cfd.py train --latent-dim 64
   ```

### Speed Up Inference

1. **Fewer diffusion steps**
   ```bash
   python aircraft_diffusion_cfd.py generate --num-steps 100
   ```

2. **Skip marching cubes** (but output is blockier)
   ```bash
   python aircraft_diffusion_cfd.py generate --no-marching-cubes
   ```

### Monitor Progress

```bash
# Watch training in real-time
tensorboard --logdir ./runs --port 6006
# Open: http://localhost:6006
```

---

## File Structure

```
aircraft-diffusion-cfd/
├── aircraft_diffusion_cfd.py   # Main application (monolithic)
├── examples.py                 # 9 example workflows
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── QUICKSTART.md              # This file
├── checkpoints/               # Saved models (created after training)
│   ├── checkpoint_grid16_ep*.pt
│   ├── checkpoint_grid24_ep*.pt
│   └── final_model.pt
├── runs/                      # TensorBoard logs (created during training)
│   └── events.out.tfevents*
└── generated_aircraft/        # Generated STL files (created during generation)
    ├── aircraft_001.stl
    ├── aircraft_002.stl
    └── ...
```

---

## Getting Help

**Check log files:**
```bash
tail -f runs/*.log
```

**Run diagnostics:**
```bash
python aircraft_diffusion_cfd.py info

# Also check:
python -c "import torch; print(torch.cuda.memory_summary())"
```

**Review training config:**
```python
from aircraft_diffusion_cfd import TrainingConfig
config = TrainingConfig()
print(config)  # Print all defaults
```

---

## Hardware Recommendations

| GPU | 16³ | 24³ | 32³ | Notes |
|-----|-----|-----|-----|-------|
| RTX 3060 (12GB) | ✅ | ✅ | ⚠️ (batch=2) | Good entry-level |
| RTX 3090 (24GB) | ✅ | ✅ | ✅ | Excellent |
| RTX 4090 (24GB) | ✅ | ✅ | ✅ | Best consumer |
| A100 (40GB) | ✅ | ✅ | ✅ | Best overall |
| H100 (80GB) | ✅ | ✅ | ✅ | Enterprise-grade |

---

**Last Updated**: December 2025  
**Status**: Ready to use
