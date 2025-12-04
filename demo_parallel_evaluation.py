#!/usr/bin/env python3
"""
Demo script showcasing GPU-optimized surrogate model with parallel evaluation.
Shows timing comparisons between serial and parallel batch evaluation.
"""
import numpy as np
import torch
import time
from pathlib import Path
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.surrogate.batch_evaluator import SurrogateBatchEvaluator, _autodetect_batch_size
from src.folding.sheet import load_config

def generate_random_actions(n_actions):
    """Generate random folding action vectors."""
    config = load_config()
    n_folds = config['project']['n_folds']
    action_dim = n_folds * 5  # x1, y1, x2, y2, angle
    return np.random.uniform(0, 1, (n_actions, action_dim))

def main():
    print("=" * 70)
    print("GPU-Optimized Surrogate Model Parallel Evaluator")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📊 Device: {device}")
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
    
    # Create state dict
    config = load_config()
    state = {
        'angle_of_attack_deg': config['goals'].get('angle_of_attack_deg', 5),
        'air_density_kgm3': config['environment'].get('air_density_kgm3', 1.225),
        'air_viscosity_pas': config['environment'].get('air_viscosity_pas', 1.8e-5),
        'throw_speed_mps': config['goals'].get('throw_speed_mps', 10)
    }
    
    print(f"\n🌬️  Aerodynamic State:")
    print(f"   AoA: {state['angle_of_attack_deg']}°")
    print(f"   Speed: {state['throw_speed_mps']} m/s")
    print(f"   Air Density: {state['air_density_kgm3']} kg/m³")
    
    # Test different population sizes
    test_sizes = [4, 16, 64, 256]
    
    print(f"\n{'Population Size':<20} {'Batch GPU Time':<20} {'Speedup':<15}")
    print("-" * 55)
    
    baseline_time = None
    
    for n_actions in test_sizes:
        print(f"Generating {n_actions} random actions...", end='', flush=True)
        actions = generate_random_actions(n_actions)
        print(" ✓")
        
        # Evaluate batch
        autodetected_batch_size = _autodetect_batch_size(device)
        evaluator = SurrogateBatchEvaluator(device=device, max_workers=4)
        
        start_time = time.time()
        results = evaluator.evaluate_batch(
            actions, 
            state, 
            show_progress=True,
            batch_size=autodetected_batch_size # Use auto-detected batch size
        )
        elapsed = time.time() - start_time
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        avg_time_per_action = elapsed / n_actions * 1000  # ms
        
        print(f"{n_actions:<20} {elapsed:>6.3f}s ({avg_time_per_action:>5.2f}ms/action) {f'{speedup:.2f}x':<15}")
        
        # Show sample results
        print(f"\n   Sample Results (first 3 of {n_actions}):")
        for i in range(min(3, n_actions)):
            print(f"   [{i}] Range: {results['range_est'][i]:>7.3f}m | "
                  f"CL: {results['cl'][i]:>6.3f} | "
                  f"CD: {results['cd'][i]:>6.4f} | "
                  f"L/D: {results['ld'][i]:>6.2f}")
        
        print()
    
    # Batch size demonstration with auto-detected batch size
    print("\n" + "=" * 70)
    print("Batch Size Demonstration (Auto-detected for Population: 256)")
    print("=" * 70)
    
    actions = generate_random_actions(256)
    autodetected_batch_size = _autodetect_batch_size(device)
    evaluator = SurrogateBatchEvaluator(device=device, max_workers=4)
    
    start_time = time.time()
    results = evaluator.evaluate_batch(
        actions, 
        state, 
        show_progress=False,
        batch_size=autodetected_batch_size
    )
    elapsed = time.time() - start_time
    per_action = elapsed / 256 * 1000
    
    print(f"\nAuto-detected Batch Size: {autodetected_batch_size}")
    print(f"{'Total Time':<15} {'Per Action (ms)':<15}")
    print("-" * 30)
    print(f"{elapsed:>8.3f}s      {per_action:>10.2f}ms")
    
    print("\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\nKey Optimizations:")
    print("  ✓ GPU tensor operations for aerodynamic calculations")
    print("  ✓ Parallel mesh generation with ThreadPoolExecutor")
    print("  ✓ Batch GPU evaluation (vectorized computations)")
    print("  ✓ Progress tracking with tqdm")
    print("  ✓ Automatic device selection (CUDA if available)")

if __name__ == "__main__":
    main()
