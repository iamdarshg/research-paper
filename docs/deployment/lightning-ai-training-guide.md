# Training on Lightning AI — Setup & Optimizations Guide

## 1. Spin Up a Lightning AI GPU Instance

1. Go to [lightning.ai](https://lightning.ai) → Create a new Studio
2. Select a GPU tier:
   - **Free (80h/month)**: T4/P100 (16 GB VRAM) — enough for 96³ training
   - **Paid**: A100 40/80 GB — for 128³+ or batch size experiments
3. Clone the repo from within the Studio terminal:
   ```bash
   git clone <your-repo-url>
   cd research-paper
   ```

## 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install -r CLI/requirements.txt
pip install -r requirements-dev.txt
```

Verify CUDA + Triton:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0)); import triton; print('Triton:', triton.__version__)"
```

## 3. Run the Regression Gate (First Priority)

This test validates the fused stream/BFL kernel against the reference PyTorch path over **50 LBM steps** — the new default. If this passes, the kernel is safe to enable permanently.

```bash
python -m pytest tests/test_d3q27_kernel_parity.py::test_regression_gate_long_run_parity -v
```

**Expected output:**
```
PASSED  test_regression_gate_long_run_parity
```

**If it fails**, check:
- Max field diff in the assertion message — should be ≤ `4 × 4e-6 = 1.6e-5`
- Late-stage diffs growing? That means FMA contraction drift is diverging
- Take a screenshot and we'll investigate

**If it passes**, flip the fused backend to default-on in `config.yaml`:
```yaml
# In CLI/config.yaml under the solver section
use_fused_stream_bfl: true
```

Or just keep passing `--lbm-stream-bfl-backend fused_stream_bfl` at runtime — dealer's choice.

## 4. Verify the Optimizations Are Active

```bash
# Check the config defaults
python -c "
import yaml
with open('CLI/config.yaml') as f:
    cfg = yaml.safe_load(f)
t = cfg['training']
print(f'LBM steps: {t[\"direct_solver_steps\"]} (target: 50)')
print(f'SPSA directions: {t[\"direct_solver_directions\"]} (target: 32)')
print(f'PCGrad enabled: {t[\"enable_pcgrad\"]}')
print(f'Adaptive balancing: {t[\"enable_adaptive_balancing\"]}')
print(f'Fused stream/BFL at startup: {t.get(\"use_fused_stream_bfl\", \"not set\")}')
"
```

All should show the non-default values.

## 5. Run a Short Smoke Test (Optional but Recommended)

Before a full training run, verify one optimizer update works end-to-end:

```bash
python CLI/aircraft_diffusion_cfd.py train \
  --num-epochs 1 \
  --batch-size 1 \
  --num-samples 8 \
  --grid-size 32 \
  --solver D3Q27 \
  --lbm-stream-bfl-backend fused_stream_bfl \
  --save-dir ./checkpoints_smoke
```

This runs 1 epoch on 8 synthetic samples at 32³. Expected: finishes without NaN errors, outputs a checkpoint.

## 6. Run a Monitored Stage-A Probe (40 Updates)

The staged recovery plan from the stabilization doc:

```bash
python CLI/run_monitored_training.py \
  --manifest build/grounded_combined_1k_20260716/manifest.jsonl \
  --num-epochs 1 \
  --batch-size 1 \
  --grid-size 96 \
  --cpu-threads 4 \
  --lbm-stream-bfl-backend fused_stream_bfl \
  --save-dir build/lightning_stage_a/checkpoints \
  --history-output build/lightning_stage_a/history.json \
  --save-every 5 \
  --max-samples-per-epoch 40 \
  --stop-on-promotion-pass
```

This runs 40 updates (~65 solver calls each) and then evaluates the checkpoint against the promotion gate.

### What to watch during the run:

1. **Gradient cosines** (logged every update):
   - `data_direct`: should rarely go below -0.5 (PCGrad is fixing this)
   - `data_consistency`: should be stable, not wildly negative
   - If cosines are near-zero or positive, PCGrad has nothing to do — good

2. **Branch gradient norms**:
   - Data: typically 0.1–0.5
   - Consistency: should be ≤ 2× data (adaptive scaler enforces this)
   - Direct: should be ≤ 2× data (adaptive scaler also enforces this)
   - If consistency OR direct is consistently > 2× data, check the scaler's `adaptive_scale` telemetry

3. **Consistency raw MSE**:
   - Should NOT spike to 1e11 anymore (Huber + adaptive scaler + teacher-from-EMA)
   - If it does exceed `consistency_raw_mse_fail_threshold` (1e6), the run fails closed

4. **Solver convergence**:
   - `solver_lbm_converged`: 0 (expected at 50 steps — 50 is still unconverged but much better than 5)
   - `solver_drag_sign_reversed`: should be rare (was common at 5 steps)

### Health check commands during the run:

```bash
# Watch the live update stream
tail -f build/lightning_stage_a/history.json | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    print(f'Update {r.get(\"global_step\",\"?\")}: '
          f'loss={r.get(\"optimization_loss\",\"?\"):.4f}, '
          f'data/direct cos={r.get(\"gradient_cosines\",{}).get(\"data_direct\",\"?\"):.4f}'
          if isinstance(r.get(\"gradient_cosines\",{}).get(\"data_direct\",\"?\"), float) else line)
"
```

## 7. Full Epoch (Stage C)

```bash
python CLI/run_monitored_training.py \
  --manifest build/grounded_combined_1k_20260716/manifest.jsonl \
  --num-epochs 1 \
  --batch-size 1 \
  --grid-size 96 \
  --cpu-threads 4 \
  --lbm-stream-bfl-backend fused_stream_bfl \
  --save-dir build/lightning_stage_c/checkpoints \
  --history-output build/lightning_stage_c/history.json \
  --save-every 10 \
  --stop-on-promotion-pass
```

Expected wall time with 65 solver calls × 50 steps each:
- **Fused BFL enabled**: ~35 ms/step → ~7h for 758 records
- **This fits well in Lightning AI's 80h/month free tier**

## 8. Interpreting Results

### If the run promotes successfully:
The checkpoint beats the source baseline on `best_geometry_model.pt` for:
- Generated aircraft valid fraction (target: >0.5)
- Generated unique fraction (target: >0.5)
- Generated mean top-k recall (target: no regression)
- Occupancy error (target: <0.005)

This means the model is learning physically meaningful features instead of filling the grid with solid.

### If the run fails to promote:
Check these in order:

1. **Check gradient cosines** in the history JSON:
   ```bash
   python -c "
   import json
   with open('build/lightning_stage_c/history.json') as f:
       for line in f:
           r = json.loads(line)
           cos = r.get('branch_telemetry', {}).get('direct', {}).get('anchor_cosine_before', 0)
           if cos < -0.5:
               print(f'Update {r[\"global_step\"]}: direct vs data cos={cos:.3f}')
   "
   ```
   If many updates show cosine < -0.5 even after PCGrad, the SPSA gradient is still too noisy — increase directions to 64.

2. **Check occupancy drift**:
   ```bash
   python -c "
   import json
   occs = []
   with open('build/lightning_stage_c/history.json') as f:
       for line in f:
           r = json.loads(line)
           occ = r.get('solver_components', {}).get('occupancy', 0)
           if occ: occs.append(occ)
   print(f'Occupancy: min={min(occs):.4f} max={max(occs):.4f} mean={sum(occs)/len(occs):.4f}')
   "
   ```
   If occupancy is consistently > 0.5 (50% of grid full), the occupancy loss is too weak — increase `space_weight` in the design spec.

3. **Check consistency loss**: If raw MSE is consistently above 1e4, the teacher and student are drifting. The teacher-from-EMA fix should handle this, but the `consistency_interval` can be reduced from 10 to 5.

## 9. Config Tuning Knobs

All in `CLI/config.yaml` under `training:`:

| Key | Default | When to tweak |
|-----|---------|---------------|
| `direct_solver_steps` | 50 | Increase to 100+ if drag sign is still reversed |
| `direct_solver_directions` | 32 | Increase to 64 if SPSA variance is high |
| `adaptive_max_ratio` | 2.0 | Lower to 1.5 if consistency still dominates |
| `adaptive_ema_decay` | 0.9 | Higher = smoother EMA, slower to react |
| `consistency_interval` | 10 | Lower = more frequent distillation (more stable but slower) |
| `student_direct_gradient_max_norm` | 0.25 | Increase if direct gradient is too weak |
| `consistency_gradient_max_norm` | 0.25 | Increase if consistency needs more influence |

## 10. What's Still Not Implemented (Future Work)

These optimizations exist as plans but haven't been coded yet:

| Optimization | Est. gain | Effort | Blocked by |
|-------------|-----------|--------|------------|
| GPU-resident EDT (CuPy + DLPack) | ~11% wall time | 2-3 days | CuPy install on Lightning AI |
| Fused MRT collision kernel | ~20% of solver time | 1 week | Triton expertise |
| Batch-two antithetic SPSA | ~30% fewer solver launches | 2 days | Solver batch dimension |
| CPU/GPU EDT pipeline | overlap EDT + solve | 2 days | Two-workspace EDT pool |
| CUDA graph capture | ~5-10% launch overhead | 1 day | Stable tensor addresses |
| GPU-resident AdamW | ~1% | 1 day | VRAM peak measurement |
