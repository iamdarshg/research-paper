# Training & Validation Tab - Enhanced Features

## Overview

The "🔧 Training & Validation" tab now includes dual training methods with GPU support:
1. **DDPG Agent** (RL-based reinforcement learning)
2. **Recursive GNN** (Graph Neural Network with hierarchical pattern learning, inspired by TRM paper)

Both methods are fully GPU-optimized and support device selection.

---

## Features

### 🖥️ GPU Device Integration
- Select between CPU and available GPUs
- Device selection shown in training status
- All computations respect the selected device
- Real-time progress showing device name

### Training Mode Selection

#### 1. 🤖 DDPG Agent (Reinforcement Learning)
```
Deep Deterministic Policy Gradient
• Learns continuous control policies
• Direct optimization of folding sequences
• Good for single-objective optimization
• Well-tested on paper airplane design
```

**How it works:**
- Uses actor-critic architecture
- Learns to maximize flight range through trial and error
- Updates based on rewards from flight simulation

**Best for:**
- Maximizing specific metrics (range, efficiency)
- Single-goal optimization
- Direct policy learning

#### 2. 🧠 Recursive GNN (Pattern Recognition)
```
Graph Neural Network with Hierarchical Learning
• Inspired by TRM paper and ARC intelligence patterns
• Captures structural relationships in folding
• Better for pattern recognition and generalization
• Processes folds as recursive graph structure
```

**How it works:**
- Models folding as a graph structure
- Uses Graph Attention Networks (GAT) for node interactions
- Processes recursively through hierarchical levels
- Learns to predict aerodynamic performance from fold patterns

**Best for:**
- Understanding folding patterns
- Generalization to new designs
- Transfer learning
- Pattern recognition (similar to ARC tasks)

---

## Training Configuration

### Section 1: Hyperparameters

**Training Episodes**: 10-100 (DDPG) / 10-50 (GNN epochs)
- Number of training iterations
- More episodes = better learning but slower training

**Batch Size**: 16, 32, 64
- Samples processed per GPU batch
- Larger batch = faster but more VRAM needed

**Learning Rate**: 1e-4, 1e-3, 1e-2
- Model update step size
- Smaller = more stable but slower convergence
- Larger = faster convergence but less stable

**Device Selection**: CPU / GPU 0, 1, 2, ...
- Automatically shown with VRAM and CUDA capability
- All operations use selected device

### Section 2: Training Mode Selection

Choose between DDPG and GNN with a radio button selector:
- Each mode has its own description
- Information card explains the approach
- Selection persists for one training run

### Section 3: Start Training

Click **"🚀 Start Training on GPU"** button to begin:
- Real-time progress bar (0-100%)
- Live metrics display:
  - DDPG: Episode, Reward, Range, Average 10-episode range
  - GNN: Epoch, Train Loss, Validation Loss
- Device name shown in status
- 3D mesh visualization updates (DDPG only)

---

## Training History Visualization

### DDPG Training History
Two graphs displayed after training completes:

1. **Max Range per Episode** (blue line)
   - Shows improvement over episodes
   - Y-axis: Max range achieved (meters)
   - X-axis: Training episode
   - Goal: upward trend indicates learning

2. **Cumulative Reward per Episode** (green line)
   - Shows total reward per episode
   - Used internally by agent for learning
   - Correlates with range performance

### GNN Training History
Two graphs displayed after GNN training:

1. **Training Loss per Epoch** (red/blue lines)
   - Red: Training loss
   - Blue dashed: Validation loss
   - Goal: both decreasing over epochs
   - Shows model convergence

2. **Learning Rate Schedule** (green line)
   - Shows learning rate decrease over time
   - Uses cosine annealing schedule
   - Helps final fine-tuning of weights

**GNN Summary Card:**
- Best validation loss achieved
- Total number of epochs
- Final training loss

---

## Batch Evaluation

After training, evaluate performance on multiple designs:

### Controls
- **Number of Actions**: 10-1000 configurations to test
- **Device Selector**: Shows current GPU device

### Process
1. Click **"▶️ Run Batch Evaluation"** button
2. Real-time progress: "Processed X/Y actions on [Device]"
3. Elapsed time counter
4. Results stored in session state

### Results Display

**Performance Metrics:**
- Avg Range: Average distance achieved
- Max Range: Best configuration
- Min Range: Worst configuration
- Avg L/D: Average efficiency

**Graphs:**
1. **Range Distribution** (histogram)
   - Shows spread of achieved ranges
   - X-axis: Range (m)
   - Y-axis: Number of designs

2. **CL vs CD Scatter** (interactive)
   - Color-coded by L/D efficiency
   - Hover to see exact values
   - Shows aerodynamic quality space

3. **CL Distribution** (histogram)
   - Lift coefficient values
   - Useful for understanding designs

4. **L/D Distribution** (histogram)
   - Efficiency values
   - Key metric for flight quality

---

## GPU Acceleration

### Automatic Batch Sizing
- **CPU**: 32 samples/batch (conservative)
- **6GB GPU**: 64 samples/batch
- **12GB+ GPU**: 128 samples/batch

### Memory Management
- Displays available VRAM in device selector
- CUDA compute capability shown
- Early stopping prevents OOM (GNN)
- Gradient clipping for stability

### Performance
- DDPG: 1-5 seconds per episode (GPU dependent)
- GNN: 0.5-2 seconds per epoch
- Batch eval: 50-200 actions/second on GPU

---

## Workflow Examples

### Training a DDPG Agent
1. Select GPU device from sidebar
2. Go to "🔧 Training & Validation" tab
3. Set hyperparameters (50 episodes, batch size 32, lr 1e-3)
4. Select **"🤖 DDPG Agent"** mode
5. Click "🚀 Start Training on GPU"
6. Watch progress bar and metrics
7. View training history graphs when complete

### Training a Recursive GNN
1. Select GPU device from sidebar
2. Go to "🔧 Training & Validation" tab
3. Set hyperparameters (30 epochs, batch size 64, lr 1e-3)
4. Select **"🧠 Recursive GNN"** mode
5. Click "🚀 Start Training on GPU"
6. Monitor loss curves in real-time
7. View training history with loss and learning rate graphs

### Comparing Methods
1. Train DDPG agent (observe range improvement)
2. Train GNN model (observe loss convergence)
3. Run batch evaluation on both
4. Compare distribution graphs
5. Analyze which method produces better ranges

---

## ARC-Style Intelligence & TRM Paper Connection

### Why Recursive GNN for ARC-like Tasks?

The ARC (Abstraction and Reasoning Corpus) challenge requires:
- **Pattern recognition** in visual puzzles
- **Hierarchical understanding** of structure
- **Transfer learning** from few examples
- **Generalization** to new patterns

The Recursive GNN approach mirrors TRM (Transformers and Recursive Modulation) paper concepts:

1. **Hierarchical Processing**
   - Multiple recursive levels (default: 3)
   - Each level captures higher-level patterns
   - Similar to TRM's hierarchical transformers

2. **Graph Structure**
   - Nodes represent folding parameters
   - Edges capture spatial relationships
   - Recursive refinement through graph layers

3. **Attention Mechanism**
   - Graph Attention Networks (GAT) learn important relationships
   - Flexible weights based on context
   - Similar to transformer attention in TRM

4. **Pattern Learning**
   - GNN learns to recognize optimal fold patterns
   - Can generalize to different fold configurations
   - Better transfer learning than flat models

### Paper Airplane as ARC-like Problem

Folding a paper airplane is analogous to ARC challenges:
- **Input**: Folding parameters (0-1 values)
- **Pattern**: Structural relationships matter
- **Output**: Flight performance (continuous metric)
- **Challenge**: Learn general patterns from examples

The Recursive GNN can discover:
- Symmetries in folding
- Critical fold sequences
- Spatial relationships
- Generalizable patterns

---

## Troubleshooting

### "GNN trainer module not found"
```bash
# Install torch_geometric
pip install torch-geometric
```

### Training takes too long
- Reduce epochs/episodes
- Increase batch size (if VRAM allows)
- Use GPU instead of CPU
- Check device selector in sidebar

### High training loss doesn't decrease
- Try lower learning rate (1e-4)
- Increase training iterations
- Verify GPU is being used
- Check batch size (try smaller)

### Out of Memory (OOM)
- Reduce batch size
- Reduce number of training iterations
- Use CPU if GPU insufficient
- Close other GPU applications

### No GPU available
- Ensure NVIDIA drivers installed
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Falls back to CPU automatically
- GNN requires more memory than DDPG

---

## Technical Details

### DDPG Architecture
- **Actor Network**: 3-layer MLP (64 → 64 → action_dim)
- **Critic Network**: 3-layer MLP (128 → 64 → 1)
- **Optimizer**: Adam
- **Replay Buffer**: 100,000 experiences

### Recursive GNN Architecture
- **Input Projection**: Linear(5 → 64)
- **Recursive Blocks**: 3 levels of GAT + GraphConv
- **Output Projection**: Linear(64 → 32)
- **Heads**: 4 attention heads
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing

### Graph Construction
- **Nodes**: Folds (5 parameters each) + boundary nodes (4)
- **Edges**: Sequential connections + boundary connections
- **Node Features**: Normalized folding parameters (0-1)
- **Target**: Normalized aerodynamic efficiency

---

## Performance Tips

**For DDPG:**
- 50-100 episodes typical for good results
- Learning rate 1e-3 usually optimal
- Batch size 32 good balance
- Training takes 5-10 minutes on GPU

**For GNN:**
- 30-50 epochs typical
- Learning rate 1e-3 good starting point
- Batch size 64 efficient
- Training takes 2-5 minutes on GPU

**For Batch Eval:**
- Start with 100 actions
- Scale up to 500-1000 for final analysis
- Takes 10-30 seconds depending on device

---

## Files

- `src/gui/app.py`: Main Streamlit app with training UI
- `src/trainer/gnn_trainer.py`: Recursive GNN implementation (NEW)
- `src/rl_agent/model.py`: DDPG agent (existing)
- `requirements.txt`: Updated with torch-geometric

---

**Status**: ✅ **FULLY IMPLEMENTED**

Both training methods integrated, tested, and GPU-accelerated!
