# 🛩️ Paper Airplane AI Optimizer - GUI Quick Start

## Running the GUI

### Option 1: Python Script (Recommended - opens browser automatically)
```bash
cd d:\research-paper
python launch_gui.py
```

### Option 2: Batch Script (Windows)
```bash
cd d:\research-paper
launch_gui.bat
```

### Option 3: Direct Command
```bash
cd d:\research-paper
python -m streamlit run src/gui/app.py
```

---

## GUI Access

Once started, the app opens at:
- **Local**: http://localhost:8502
- **Network**: http://192.168.1.15:8502 (if on same network)

---

## Features Overview

### 🖥️ GPU Device Selector (Sidebar)
- Select between CPU and available GPUs
- Shows GPU name, VRAM, and CUDA capability
- Device persists across navigation
- All computations use selected device

### 📊 Example Analysis Tabs (4 Tabs)

#### Tab 1: Example 1 - Standard Airplane
- Classical 5-fold design
- Analysis of 50 configurations
- Interactive performance graphs

#### Tab 2: Example 2 - Optimized Design  
- AI-optimized 8-fold design
- Analysis of 75 configurations
- Green-themed visualizations

#### Tab 3: Example 3 - Experimental Design
- Cutting-edge 10-fold design
- Analysis of 100 configurations
- Red-themed visualizations

#### Tab 4: Training & Validation
- Training progress monitoring
- 3D fold visualization
- Batch evaluation (10-1000 actions)
- Performance metrics and distributions

### 📈 Interactive Graphs (All Tabs)
- **CL vs CD Scatter**: Color-coded by angle of attack
- **L/D Histogram**: Efficiency distribution
- **Performance Box Plot**: Results vs angle of attack

### ⏳ Progress Bars
- Real-time 0-100% progress display
- Device name shown: "Processing batch 42% complete on GPU 0: RTX 3090"
- Updates every batch completion

---

## Workflow Examples

### Run Example 1 Analysis
1. Go to "📊 Example 1: Standard" tab
2. Click "Run Example 1 Analysis" button
3. Watch progress bar update
4. Explore the three generated graphs
5. Check performance metrics (CL, CD, L/D)

### Run Batch Evaluation
1. Go to "🔧 Training & Validation" tab
2. Adjust slider: "Number of Actions for Batch Eval" (10-1000)
3. Click "Run Batch Evaluation"
4. Monitor progress: "Processed X/Y actions on [Device]"
5. View distribution graphs and statistics

### Train Model
1. Configure in sidebar (N Folds, Target Range, Episodes)
2. Click "Train Model" button in sidebar
3. Monitor live metrics during training
4. View 3D mesh updates
5. Historical training graph appears when done

### Switch Devices
1. Select GPU/CPU from dropdown in sidebar
2. All subsequent operations use new device
3. Device info updates automatically
4. No need to restart app

---

## Troubleshooting

### "Port already in use"
```bash
# Kill existing streamlit process
taskkill /F /IM python.exe /T

# Or specify different port
streamlit run src/gui/app.py --server.port 8503
```

### GPU Not Detected
- Ensure NVIDIA drivers are installed
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- GPU selector will show CPU only if CUDA unavailable

### Slow Performance
- GPU selection in sidebar (switch from CPU if available)
- Reduce "Number of Actions for Batch Eval" slider
- Check VRAM usage (GPU info shown in sidebar)

### Import Errors
- Ensure running from project root: `cd d:\research-paper`
- Use: `python -m streamlit run src/gui/app.py`
- Don't use: `streamlit run src/gui/app.py` (without -m flag)

---

## File Structure

```
d:\research-paper/
├── launch_gui.py              # Python launcher (recommended)
├── launch_gui.bat             # Batch launcher
├── src/
│   └── gui/
│       └── app.py             # Main Streamlit app (769 lines)
├── src/surrogate/
│   ├── batch_evaluator.py     # GPU batch processing
│   └── aero_model.py          # Aerodynamic model
└── config.yaml                # Configuration
```

---

## Performance Tips

### For Fast Processing
1. Select GPU device (if available)
2. Use smaller action count (start with 100)
3. Run examples in order (1 → 2 → 3)

### For Detailed Analysis
1. Increase "Number of Actions" to 500-1000
2. Use high-end GPU (12GB+ VRAM)
3. Allow 5-10 minutes for large batch evaluation

### GPU Auto-Detection
- CPU: 32 batch size
- 6GB GPU: 64 batch size  
- 12GB+ GPU: 128 batch size
- Custom via code: `SurrogateBatchEvaluator(batch_size=256)`

---

## Keyboard Shortcuts

- **R**: Rerun (Streamlit default)
- **C**: Clear cache
- **Ctrl+C**: Stop server

---

## System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA with CUDA (optional)
- **RAM**: 8GB minimum
- **Disk**: 500MB minimum

### Required Packages
```
streamlit>=1.28.0
torch>=2.0.0
plotly>=5.14.0
numpy>=1.21.0
trimesh>=3.9.0
pyyaml>=5.4.0
tqdm>=4.65.0
```

All included in `requirements.txt`

---

## Next Steps

1. ✅ Launch GUI: `python launch_gui.py`
2. ✅ Select GPU device
3. ✅ Explore Example 1 analysis
4. ✅ Run batch evaluation
5. ✅ Start training model

---

**Last Updated**: December 6, 2025  
**Status**: ✅ Ready to Use
