#!/usr/bin/env python3
"""
Example configuration and usage patterns for aircraft diffusion model.
Shows how to customize training and inference.
"""

import torch
import numpy as np
from pathlib import Path
from aircraft_diffusion_cfd import (
    DiffusionConfig, ModelConfig, TrainingConfig, CFDConfig, DesignSpec,
    DiffusionTrainer, AircraftGenerator, AircraftDesignDataset, NoiseSchedule
)
from torch.utils.data import DataLoader

# ============================================================================
# EXAMPLE 1: Basic Training Setup
# ============================================================================

def example_basic_training():
    """Train with default configuration"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    model_config = ModelConfig(latent_dim=128)
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig(
        num_epochs=100,
        batch_size=4,
        learning_rate=2e-4,
        disconnection_penalty=10.0
    )
    cfd_config = CFDConfig(resolution=16)
    
    # Dataset
    dataset = AircraftDesignDataset(num_samples=100)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Trainer
    trainer = DiffusionTrainer(
        model_config, diffusion_config, training_config, cfd_config, device=device
    )
    
    # Train
    trainer.train(train_loader)
    trainer.save_checkpoint('aircraft_model.pt')

# ============================================================================
# EXAMPLE 2: Memory-Optimized Training (8GB VRAM)
# ============================================================================

def example_memory_optimized():
    """Training optimized for 8GB VRAM"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = ModelConfig(
        latent_dim=64,  # Reduced from 128
        encoder_channels=[32, 64, 128],  # Smaller channels
        decoder_channels=[128, 64, 32]
    )
    
    training_config = TrainingConfig(
        batch_size=2,  # Reduced from 4
        num_epochs=50,  # Fewer total epochs (but scale up grid later)
        learning_rate=1e-4,  # Slightly lower LR for stability
        disconnection_penalty=8.0
    )
    
    cfd_config = CFDConfig(resolution=16)
    diffusion_config = DiffusionConfig()
    
    dataset = AircraftDesignDataset(num_samples=50)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    trainer = DiffusionTrainer(
        model_config, diffusion_config, training_config, cfd_config, device=device
    )
    
    trainer.train(train_loader)
    trainer.save_checkpoint('aircraft_8gb.pt')

# ============================================================================
# EXAMPLE 3: Custom Design Specification
# ============================================================================

def example_custom_design_spec():
    """Define custom design objectives"""
    
    # High-speed fighter (emphasize drag reduction)
    fighter_spec = DesignSpec(
        target_speed=200.0,  # m/s (supersonic)
        space_weight=0.1,    # Less concern for space
        drag_weight=0.7,     # Maximize drag efficiency
        lift_weight=0.2,
        bounding_box=(64, 64, 64)
    )
    
    # Cargo aircraft (emphasize volume)
    cargo_spec = DesignSpec(
        target_speed=100.0,  # m/s (subsonic)
        space_weight=0.6,    # Maximize internal volume
        drag_weight=0.2,     # Some aerodynamic concern
        lift_weight=0.2,
        bounding_box=(128, 64, 64)  # Elongated fuselage
    )
    
    # Racing drone (balanced performance)
    drone_spec = DesignSpec(
        target_speed=50.0,   # m/s
        space_weight=0.33,   # Equal weights
        drag_weight=0.33,
        lift_weight=0.34,
        bounding_box=(32, 32, 32)  # Compact
    )
    
    return fighter_spec, cargo_spec, drone_spec

# ============================================================================
# EXAMPLE 4: Inference with Custom Specifications
# ============================================================================

def example_inference():
    """Generate designs with different specifications"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load trained model
    generator = AircraftGenerator('./aircraft_model.pt', device=device)
    
    # Generate high-performance fighter
    fighter_spec = DesignSpec(
        target_speed=200.0,
        space_weight=0.1,
        drag_weight=0.7,
        lift_weight=0.2
    )
    
    print("Generating fighter aircraft...")
    fighter_voxels = generator.generate(fighter_spec, num_steps=250)
    generator.voxels_to_stl(fighter_voxels, 'fighter.stl', use_marching_cubes=True)
    
    # Generate cargo aircraft
    cargo_spec = DesignSpec(
        target_speed=100.0,
        space_weight=0.6,
        drag_weight=0.2,
        lift_weight=0.2
    )
    
    print("Generating cargo aircraft...")
    cargo_voxels = generator.generate(cargo_spec, num_steps=250)
    generator.voxels_to_stl(cargo_voxels, 'cargo.stl', use_marching_cubes=True)

# ============================================================================
# EXAMPLE 5: Resume Training from Checkpoint
# ============================================================================

def example_resume_training():
    """Resume training from a saved checkpoint"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_config = ModelConfig()
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig(num_epochs=50)  # Additional epochs
    cfd_config = CFDConfig()
    
    trainer = DiffusionTrainer(
        model_config, diffusion_config, training_config, cfd_config, device=device
    )
    
    # Load previous checkpoint
    trainer.load_checkpoint('aircraft_model_checkpoint.pt')
    
    # Continue training
    dataset = AircraftDesignDataset(num_samples=100)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    trainer.train(train_loader)
    trainer.save_checkpoint('aircraft_model_resumed.pt')

# ============================================================================
# EXAMPLE 6: Analyze Voxel Grid Properties
# ============================================================================

def example_analyze_geometry():
    """Analyze generated geometry properties"""
    
    import scipy.ndimage as ndi
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = AircraftGenerator('./aircraft_model.pt', device=device)
    
    spec = DesignSpec(target_speed=50.0)
    voxels = generator.generate(spec, num_steps=250)
    
    voxel_np = voxels.cpu().numpy()
    binary = (voxel_np > 0.5).astype(int)
    
    # Statistics
    total_voxels = np.prod(binary.shape)
    occupied = binary.sum()
    occupancy_rate = occupied / total_voxels * 100
    
    print(f"Total voxels: {total_voxels}")
    print(f"Occupied voxels: {occupied}")
    print(f"Occupancy rate: {occupancy_rate:.2f}%")
    
    # Connectivity analysis
    labeled, num_components = ndi.label(binary)
    component_sizes = np.bincount(labeled.flatten())
    
    print(f"\nConnected components: {num_components}")
    print(f"Largest component size: {component_sizes[1:].max() if num_components > 1 else occupied}")
    print(f"Component size distribution: {component_sizes[:10]}")
    
    # Bounding box
    coords = np.where(binary)
    if len(coords[0]) > 0:
        bounds = [
            (coords[0].min(), coords[0].max()),
            (coords[1].min(), coords[1].max()),
            (coords[2].min(), coords[2].max())
        ]
        print(f"\nBounding box: {bounds}")
        print(f"Effective dimensions: {[b[1]-b[0]+1 for b in bounds]}")
    
    return binary

# ============================================================================
# EXAMPLE 7: Batch Processing with Monitoring
# ============================================================================

def example_batch_generation_with_monitoring():
    """Generate multiple designs with quality metrics"""
    
    from pathlib import Path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = AircraftGenerator('./aircraft_model.pt', device=device)
    
    output_dir = Path('./generated_aircraft_batch')
    output_dir.mkdir(exist_ok=True)
    
    specs = [
        DesignSpec(target_speed=50.0, space_weight=0.5, drag_weight=0.3, lift_weight=0.2),
        DesignSpec(target_speed=100.0, space_weight=0.2, drag_weight=0.6, lift_weight=0.2),
        DesignSpec(target_speed=150.0, space_weight=0.1, drag_weight=0.7, lift_weight=0.2),
    ]
    
    for idx, spec in enumerate(specs):
        print(f"\nGenerating design {idx+1}/{len(specs)}")
        print(f"  Target speed: {spec.target_speed} m/s")
        print(f"  Space weight: {spec.space_weight}")
        print(f"  Drag weight: {spec.drag_weight}")
        
        voxels = generator.generate(spec, num_steps=250)
        
        # Save STL
        stl_path = output_dir / f'design_{idx+1:03d}_speed{int(spec.target_speed):03d}.stl'
        generator.voxels_to_stl(voxels, str(stl_path), use_marching_cubes=True)
        
        # Save voxel grid as numpy
        np_path = output_dir / f'design_{idx+1:03d}_voxels.npy'
        np.save(np_path, voxels.cpu().numpy())
        
        print(f"  Saved to {stl_path}")

# ============================================================================
# EXAMPLE 8: Fine-tuning on Custom Data
# ============================================================================

def example_finetune_on_custom_data():
    """Fine-tune model on your own aircraft designs"""
    
    # Assume you have custom voxel grids in a directory
    custom_voxels_dir = Path('./my_aircraft_designs')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained model
    model_config = ModelConfig()
    diffusion_config = DiffusionConfig()
    training_config = TrainingConfig(
        num_epochs=20,  # Fine-tune for fewer epochs
        learning_rate=5e-5  # Lower learning rate
    )
    cfd_config = CFDConfig()
    
    trainer = DiffusionTrainer(
        model_config, diffusion_config, training_config, cfd_config, device=device
    )
    
    # Load pre-trained weights
    checkpoint = torch.load('./aircraft_model.pt')
    trainer.diffusion_model.load_state_dict(checkpoint['diffusion_model'])
    trainer.converter.load_state_dict(checkpoint['converter'])
    
    # Use custom dataset (you would implement custom dataset loading here)
    dataset = AircraftDesignDataset(num_samples=50)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fine-tune
    trainer.train(train_loader)
    trainer.save_checkpoint('aircraft_finetuned.pt')

# ============================================================================
# EXAMPLE 9: Export and Visualization
# ============================================================================

def example_export_workflow():
    """Complete workflow from inference to export"""
    
    from pathlib import Path
    import json
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = AircraftGenerator('./aircraft_model.pt', device=device)
    
    # Design specification
    design_spec = DesignSpec(
        target_speed=75.0,
        space_weight=0.3,
        drag_weight=0.4,
        lift_weight=0.3
    )
    
    # Generate
    print("Generating aircraft...")
    voxels = generator.generate(design_spec, num_steps=250)
    
    # Create output directory
    output_dir = Path('./aircraft_export')
    output_dir.mkdir(exist_ok=True)
    
    # Export STL
    stl_path = output_dir / 'aircraft.stl'
    generator.voxels_to_stl(voxels, str(stl_path), use_marching_cubes=True)
    
    # Export voxel grid
    voxel_path = output_dir / 'voxel_grid.npy'
    np.save(voxel_path, voxels.cpu().numpy())
    
    # Save design specification as JSON
    spec_path = output_dir / 'design_spec.json'
    with open(spec_path, 'w') as f:
        json.dump({
            'target_speed': design_spec.target_speed,
            'space_weight': design_spec.space_weight,
            'drag_weight': design_spec.drag_weight,
            'lift_weight': design_spec.lift_weight,
            'bounding_box': design_spec.bounding_box
        }, f, indent=2)
    
    print(f"\nExport complete!")
    print(f"  STL: {stl_path}")
    print(f"  Voxels: {voxel_path}")
    print(f"  Spec: {spec_path}")

# ============================================================================
# MAIN: Run Examples
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Basic Training', example_basic_training),
        '2': ('Memory-Optimized (8GB)', example_memory_optimized),
        '3': ('Custom Design Specs', example_custom_design_spec),
        '4': ('Inference', example_inference),
        '5': ('Resume Training', example_resume_training),
        '6': ('Analyze Geometry', example_analyze_geometry),
        '7': ('Batch Generation', example_batch_generation_with_monitoring),
        '8': ('Fine-tuning', example_finetune_on_custom_data),
        '9': ('Export Workflow', example_export_workflow),
    }
    
    print("\n" + "="*60)
    print("Aircraft Diffusion Model - Example Scripts")
    print("="*60 + "\n")
    
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    print("\nUsage: python examples.py <number>")
    print("Example: python examples.py 1")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            name, func = examples[choice]
            print(f"\nRunning: {name}")
            print("="*60 + "\n")
            func()
        else:
            print(f"Invalid choice: {choice}")
    else:
        print("\nRun with argument to execute example.")
