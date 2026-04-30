#!/usr/bin/env python3
"""
Aircraft Structural Design via Diffusion Models + FluidX3D CFD
(Split module version)
"""

import os
import sys
import click
import torch
import numpy as np
import trimesh
from pathlib import Path
from torch.utils.data import DataLoader

from config import (
    ModelConfig, DiffusionConfig, TrainingConfig, CFDConfig, DesignSpec,
    LBMPhysicsConfig, OPENFOAM_AVAILABLE, OPENFOAM_ROOT, OPENFOAM_BIN
)
from models import (
    GroupedQueryAttention, GradientCheckpointingWrapper, SpatialAttention,
    ResidualBlock3D, LatentDiffusionUNet, ConsistencyModel, LatentTo3DConverter,
    NoiseSchedule
)
from data_utils import (
    AircraftDesignDataset, ConnectivityLoss, GroundTruthExporter, AerodynamicLoss
)
from cfd_simulator import AdvancedCFDSimulator
from trainer import OptimizedDiffusionTrainer
from generator import OptimizedAircraftGenerator
from constraints import ConstraintReport
from mesh_utils import normalize_stl_mesh
from utils import get_vram_limit_resolution, get_stl_adaptive_resolution

@click.group()
def cli():
    """Aircraft Structural Design via Diffusion Models + CFD (Fully Optimized)"""
    print("🚀 TRM/HRM Recursive Style Implementation (Modular)")
    pass

@cli.command()
@click.option('--num-epochs', default=100)
@click.option('--batch-size', default=4)
@click.option('--learning-rate', default=2e-4)
@click.option('--latent-dim', default=16)
@click.option('--precision', default='float32')
@click.option('--disconnection-penalty', default=30.0)
@click.option('--num-samples', default=500)
@click.option('--resume-from', default=None)
@click.option('--save-dir', default='./checkpoints')
@click.option('--enable-consistency', is_flag=True, default=True)
@click.option('--enable-pipeline', is_flag=True, default=True)
@click.option('--enable-checkpointing', is_flag=True, default=True)
@click.option('--enable-compile', is_flag=True, default=False)
@click.option('--solver', default='D3Q27')
def train(num_epochs, batch_size, learning_rate, latent_dim, precision, disconnection_penalty, 
          num_samples, resume_from, save_dir, enable_consistency, enable_pipeline, 
          enable_checkpointing, enable_compile, solver):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    model_config = ModelConfig(latent_dim=latent_dim, attention_groups=4, enable_gradient_checkpointing=enable_checkpointing, use_torch_compile=enable_compile)
    diffusion_config = DiffusionConfig(teacher_steps=1000, student_steps=4)
    training_config = TrainingConfig(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, disconnection_penalty=disconnection_penalty, precision=precision, enable_pipeline_parallelism=enable_pipeline)
    cfd_config = CFDConfig(base_grid_resolution=16 if solver == "D3Q27" else 32, solver_type=solver)

    dataset = AircraftDesignDataset(num_samples=num_samples, grid_size=32, latent_dim=model_config.latent_dim, target_grid_size=1024)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    trainer = OptimizedDiffusionTrainer(model_config, diffusion_config, training_config, cfd_config, device=device)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.train(train_loader)
    trainer.save_checkpoint(os.path.join(save_dir, 'final_optimized_model.pt'))

@cli.command()
@click.option('--design', required=True)
@click.option('--cfd-steps', default=500)
@click.option('--solver', default='D3Q27')
def evaluate(design, cfd_steps, solver):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if design.endswith('.stl'):
        res = get_stl_adaptive_resolution(design)
        cfd_config = CFDConfig(base_grid_resolution=res, solver_type=solver)
        simulator = AdvancedCFDSimulator(cfd_config, device)
        mesh = trimesh.load(design)
        mesh = normalize_stl_mesh(mesh)
        voxel_grid_np = mesh.voxelized(1.0 / res).matrix
        geometry = torch.zeros((res, res, res), device=device)
        d, h, w = min(res, voxel_grid_np.shape[0]), min(res, voxel_grid_np.shape[1]), min(res, voxel_grid_np.shape[2])
        geometry[:d, :h, :w] = torch.from_numpy(voxel_grid_np[:d, :h, :w].astype(np.float32)).to(device)
    else:
        voxel_grid_np = np.load(design)
        res = voxel_grid_np.shape[0]
        geometry = torch.from_numpy(voxel_grid_np).to(device)
        cfd_config = CFDConfig(base_grid_resolution=res, solver_type=solver)
        simulator = AdvancedCFDSimulator(cfd_config, device)
    results = simulator.simulate_aerodynamics(geometry, steps=cfd_steps)
    for k, v in results.items():
        if isinstance(v, (int, float)): print(f"  • {k}: {v:.6f}")

@cli.command()
@click.option('--stl', required=True)
@click.option('--steps', default=500)
def accuracy_benchmark(stl, steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 CANONICAL ACCURACY BENCHMARK: {stl}")
    for corr in [True, False]:
        print(f"\n📊 Running {'with' if corr else 'WITHOUT'} Shape-Drag Correction...")
        conf = CFDConfig(base_grid_resolution=64)
        conf.lbm_config.use_shape_drag_correction = corr
        sim = AdvancedCFDSimulator(conf, device)
        mesh = trimesh.load(stl)
        mesh = normalize_stl_mesh(mesh)
        vg = mesh.voxelized(1.0/64).matrix
        geom = torch.zeros((64, 64, 64), device=device)
        d, h, w = min(64, vg.shape[0]), min(64, vg.shape[1]), min(64, vg.shape[2])
        geom[:d, :h, :w] = torch.from_numpy(vg[:d, :h, :w].astype(np.float32)).to(device)
        res = sim.simulate_aerodynamics(geom, steps=steps)
        print(f"  Drag Coeff: {res['drag_coefficient']:.6f}, PINN Ready: {res.get('pinn_ready')}")

@cli.command()
@click.option('--checkpoint', required=True)
@click.option('--output', default='aircraft_optimized.stl')
@click.option('--target-speed', default=7.0)
@click.option('--num-steps', default=4)
@click.option('--use-marching-cubes', is_flag=True, default=True)
@click.option('--solver', default='D3Q27')
@click.option('--num-candidates', default=1, help='Number of candidates to sample for surrogate ranking (Issue #15)')
@click.option('--top-k', default=1, help='Number of top candidates to validate with D3Q27')
@click.option('--external-validation', is_flag=True, help='Force external PDE validation for the final design')
@click.option('--surrogate-checkpoint', default=None, help='Load a standalone surrogate model checkpoint')
def generate(checkpoint, output, target_speed, num_steps, use_marching_cubes, solver, num_candidates, top_k, external_validation, surrogate_checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = OptimizedAircraftGenerator(checkpoint, device=device)
    if surrogate_checkpoint:
        generator.load_surrogate(surrogate_checkpoint)
    mission = DesignSpec(target_speed=target_speed).to_mission_profile()
    mission.force_external_validation = external_validation

    # Request typed geometry to preserve semantic info for feasibility checks (Issue #16)
    report = ConstraintReport()

    # Issue #15: Multi-fidelity ranking loop
    # Return both geometry and results to avoid redundant simulation (Review Feedback)
    typed_geom, results = generator.generate(
        mission,
        num_steps=num_steps,
        return_typed=True,
        existing_report=report,
        num_candidates=num_candidates,
        top_k=top_k,
        return_results=True
    )

    # Save with watertightness check
    generator.save_stl(typed_geom, output, use_marching_cubes=use_marching_cubes, report=report)

    if results:
        print(f"Drag: {results['drag_coefficient']:.6f}, Lift: {results['lift_coefficient']:.6f}")
        if 'feasibility' in results:
            print(f"Feasibility: Lift/Weight: {results['feasibility']['lift_ratio']:.2f}")

    final_report = report.to_dict()
    print(f"Repaired: {final_report['repaired']}, Violations: {len(final_report['violations'])}")
    print(f"Export Status: {final_report['export_status']}")

@cli.command()
@click.option('--checkpoint', required=True)
@click.option('--output-dir', default='./generations_optimized')
@click.option('--num-designs', default=5)
def batch_generate(checkpoint, output_dir, num_designs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generator = OptimizedAircraftGenerator(checkpoint, device=device)
    for i in range(num_designs):
        voxel_grid = generator.generate(DesignSpec(target_speed=50.0))
        generator.save_stl(voxel_grid, os.path.join(output_dir, f'aircraft_{i+1:03d}.stl'))

@cli.command()
@click.option('--labels-dir', default='./ground_truth')
@click.option('--epochs', default=50)
@click.option('--lr', default=1e-3)
@click.option('--batch-size', default=16)
def train_surrogate(labels_dir, epochs, lr, batch_size):
    """Train the AeroSurrogate model on collected CFD labels (Issue #15)"""
    from models import AeroSurrogate, MissionEncoder
    from data_utils import CFDLabelDataset
    from torch.utils.data import DataLoader
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = CFDLabelDataset(labels_dir=labels_dir)
    if len(dataset) == 0:
        print(f"❌ No labels found in {labels_dir}. Run some 'generate' or 'train' steps first.")
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    model = AeroSurrogate(condition_dim=32).to(device)
    encoder = MissionEncoder(condition_dim=32).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=lr)

    print(f"🚀 Training AeroSurrogate on {len(dataset)} samples...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for geoms, targets, mission_dicts in loader:
            geoms = geoms.to(device)
            # targets is a dict of tensors
            targets = {k: v.to(device) for k, v in targets.items()}

            # Convert back to MissionProfile list for encoder compatibility
            # (DataLoader collates dict of tensors, we need list of objects or handle dict in encoder)
            # encoder.forward expects Union[MissionProfile, List[MissionProfile]]
            from config import MissionProfile
            batch_size = geoms.shape[0]
            profiles = []
            for i in range(batch_size):
                kwargs = {k: (v[i].item() if torch.is_tensor(v) else v[i]) for k, v in mission_dicts.items()}
                profiles.append(MissionProfile(**kwargs))

            # Encode mission profiles
            cond = encoder(profiles)

            loss = model.train_step(geoms, targets, cond, optimizer)
            epoch_loss += loss

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    # Save surrogate
    save_path = Path(labels_dir) / "aero_surrogate.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': encoder.state_dict()
    }, save_path)
    print(f"✅ Saved trained surrogate to {save_path}")

@cli.command()
def info():
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

if __name__ == '__main__':
    cli()
