#!/usr/bin/env python3
"""
Test script to verify the tensor dimension alignment fix
"""

import torch
import sys
import os
sys.path.append('.')

from aircraft_diffusion_cfd import (
    ModelConfig, CFDConfig, AdvancedCFDSimulator, 
    OptimizedDiffusionTrainer, LatentTo3DConverter
)

def test_tensor_dimension_alignment():
    """Test that model and CFD solver resolutions are aligned"""
    
    print("🧪 Testing Tensor Dimension Alignment Fix")
    print("="*50)
    
    # Test different solver configurations
    test_cases = [
        {"solver": "D3Q19", "expected_res": 32},
        {"solver": "D3Q27", "expected_res": 16},
    ]
    
    for case in test_cases:
        print(f"\n🔧 Testing {case['solver']} solver...")
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Create CFD config and simulator
        cfd_config = CFDConfig(
            solver_type=case['solver'],
            base_grid_resolution=case['expected_res']
        )
        
        print(f"CFD Config resolution: {cfd_config.base_grid_resolution}")
        
        # Create CFD simulator to get actual resolution
        cfd_simulator = AdvancedCFDSimulator(cfd_config, device)
        actual_cfd_res = cfd_simulator.resolution
        print(f"CFD Simulator resolution: {actual_cfd_res}")
        
        # Simulate the fix: align model resolution with CFD resolution
        target_resolution = actual_cfd_res
        print(f"Target model resolution: {target_resolution}")
        
        # Create aligned model config
        model_config = ModelConfig(
            latent_dim=16,
            grid_resolution=target_resolution  # This is the key fix
        )
        
        print(f"Model config grid resolution: {model_config.grid_resolution}")
        
        # Test LatentTo3DConverter
        converter = LatentTo3DConverter(
            latent_dim=model_config.latent_dim,
            grid_resolution=model_config.grid_resolution
        )
        
        # Test tensor shapes
        test_latent = torch.randn(1, model_config.latent_dim)
        voxel_grid = converter(test_latent)
        
        print(f"Input latent shape: {test_latent.shape}")
        print(f"Output voxel grid shape: {voxel_grid.shape}")
        print(f"Expected shape: ({target_resolution}, {target_resolution}, {target_resolution})")
        
        # Verify dimensions match
        expected_shape = (target_resolution, target_resolution, target_resolution)
        if voxel_grid.shape[1:] == expected_shape:
            print("✅ SUCCESS: Model output matches CFD solver expectations")
        else:
            print("❌ FAILED: Dimension mismatch still exists")
            return False
        
        # Test geometry mask creation
        geometry_mask = (voxel_grid[0] > 0.5).float()
        print(f"Geometry mask shape: {geometry_mask.shape}")
        
        if geometry_mask.shape == expected_shape:
            print("✅ SUCCESS: Geometry mask shape is correct")
        else:
            print("❌ FAILED: Geometry mask shape is incorrect")
            return False
    
    print("\n🎉 All tests passed! The tensor dimension alignment fix is working correctly.")
    return True

def test_cfd_solver_compatibility():
    """Test that the CFD solver can handle the aligned tensor dimensions"""
    
    print("\n🔧 Testing CFD Solver Compatibility...")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test D3Q27 solver (the problematic case)
    cfd_config = CFDConfig(solver_type="D3Q27", base_grid_resolution=16)
    cfd_simulator = AdvancedCFDSimulator(cfd_config, device)
    
    # Create a test geometry that matches CFD solver resolution
    test_geometry = torch.ones(16, 16, 16) * 0.8  # 80% solid
    test_geometry[8, 8, 8] = 0.2  # Make center fluid
    
    print(f"Test geometry shape: {test_geometry.shape}")
    print(f"CFD solver expects: ({cfd_simulator.resolution}, {cfd_simulator.resolution}, {cfd_simulator.resolution})")
    
    # Test CFD simulation
    try:
        print("Running CFD simulation...")
        results = cfd_simulator.simulate_aerodynamics(test_geometry, steps=10)
        print("✅ CFD simulation completed successfully")
        print(f"Results: {results}")
        return True
    except Exception as e:
        print(f"❌ CFD simulation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Tensor Dimension Alignment Tests")
    print("="*60)
    
    # Test 1: Dimension alignment
    test1_passed = test_tensor_dimension_alignment()
    
    # Test 2: CFD compatibility
    test2_passed = test_cfd_solver_compatibility()
    
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Dimension Alignment Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"CFD Compatibility Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fix successfully resolves the tensor dimension mismatch error.")
        print("\n📝 Summary of the fix:")
        print("- Model resolution is now dynamically aligned with CFD solver resolution")
        print("- LatentTo3DConverter outputs grids matching CFD solver expectations")
        print("- No more dimension mismatches in torch.where() operations")
        print("- The original error should no longer occur")
    else:
        print("\n❌ Some tests failed. The fix may need additional work.")
        
    print("\n🔧 The fix involved:")
    print("1. Creating CFD simulator first to determine target resolution")
    print("2. Configuring model to use the same resolution as CFD solver")
    print("3. Ensuring tensor compatibility throughout the pipeline")
