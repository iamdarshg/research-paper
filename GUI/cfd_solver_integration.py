# UPDATED: Fixed mesh expansion + mixed precision compatibility
import numpy as np
import trimesh
from PyQt5.QtCore import QObject, pyqtSignal
import sys
import os
import traceback
sys.path.append(r"D:\research-paper\CLI")

try:
    from aircraft_diffusion_cfd import AdvancedCFDSimulator, CFDConfig, LBMPhysicsConfig
    from advanced_lbm_solver import GPULBMSolver, D3Q27CascadedSolver
    IMPORTS_SUCCESSFUL = True
except Exception as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

class CFDSolverWorker(QObject):
    update_progress = pyqtSignal(int, str)
    simulation_finished = pyqtSignal(dict)
    simulation_error = pyqtSignal(str)

    def __init__(self, stl_path, reynolds, mach, steps, solver_type="d3q27_cascaded", 
                 grid_resolution=32, cfd_domain_params=None, use_mixed_precision=False):
        super().__init__()
        self.stl_path = stl_path
        self.reynolds = reynolds
        self.mach = mach
        self.steps = steps
        self.solver_type = solver_type
        self.grid_resolution = grid_resolution
        self.cfd_domain_params = cfd_domain_params or {}
        self.use_mixed_precision = use_mixed_precision
        self._is_interrupted = False

    def run_simulation(self):
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.update_progress.emit(0, "Loading STL...")
            mesh = trimesh.load_mesh(self.stl_path)
            
            domain_size = self.cfd_domain_params.get('domain_size', [1.0, 1.0, 1.0])
            body_size = self.cfd_domain_params.get('body_size', 1.0)
            offset = self.cfd_domain_params.get('offset', [0, 0, 0])
            
            grid_resolution = self.grid_resolution
            grid_spacing_x = domain_size[0] / grid_resolution
            grid_spacing_y = domain_size[1] / grid_resolution
            grid_spacing_z = domain_size[2] / grid_resolution
            
            print(f"Domain: {domain_size}, Body: {body_size}, Grid: {grid_resolution}^3")
            
            # FIX: Keep mesh at original physical size, don't expand to fill domain
            bounds = mesh.bounds
            mesh_extent = bounds[1] - bounds[0]
            max_mesh_extent = np.max(mesh_extent)
            
            # Scale to physical size only
            if max_mesh_extent > 1e-6:
                scale_factor = body_size / max_mesh_extent
                mesh.vertices = (mesh.vertices - bounds[0]) * scale_factor
            
            # Center in domain
            mesh_center = np.mean(mesh.vertices, axis=0)
            domain_center = np.array(domain_size) / 2.0
            mesh.vertices = mesh.vertices - mesh_center + domain_center + np.array(offset)
            
            # Voxelize at appropriate resolution
            voxel_pitch = min(grid_spacing_x, grid_spacing_y, grid_spacing_z)
            self.update_progress.emit(5, f"Voxelizing (pitch={voxel_pitch:.4f})...")
            
            try:
                voxel_grid = mesh.voxelized(voxel_pitch).fill()
                voxel_np = voxel_grid.matrix.view(np.ndarray)
            except:
                voxel_pitch *= 2
                voxel_grid = mesh.voxelized(voxel_pitch).fill()
                voxel_np = voxel_grid.matrix.view(np.ndarray)
            
            # Resize to target grid
            from scipy.ndimage import zoom
            target_shape = (grid_resolution, grid_resolution, grid_resolution)
            zoom_factors = np.array(target_shape) / np.array(voxel_np.shape)
            resized_voxel = zoom(voxel_np.astype(np.float32), zoom_factors, order=1)
            resized_voxel = (resized_voxel > 0.5).astype(np.float32)
            
            voxel_tensor = torch.from_numpy(resized_voxel).float().to(device)
            
            # Setup CFD
            cfd_config = CFDConfig(
                base_grid_resolution=grid_resolution,
                reynolds_number=self.reynolds,
                mach_number=self.mach,
                simulation_steps=self.steps
            )
            
            cfd_config.lbm_config = LBMPhysicsConfig()
            cfd_config.lbm_config.physical_length_scale = body_size
            cfd_config.lbm_config.grid_spacing = grid_spacing_x
            cfd_config.lbm_config.compute_q_criterion = True
            cfd_config.lbm_config.use_vorticity_confinement = True
            
            if self.solver_type == "d3q27_cascaded":
                lbm_solver = D3Q27CascadedSolver(cfd_config, device, LBMPhysicsConfig)
            else:
                simulator = AdvancedCFDSimulator(cfd_config, device)
                lbm_solver = simulator.lbm_solver
            
            # Apply mixed precision if enabled
            if self.use_mixed_precision and torch.cuda.is_available():
                try:
                    from mixed_precision_solver import wrap_solver_mixed_precision
                    lbm_solver = wrap_solver_mixed_precision(lbm_solver, enable_fp16=True)
                    print("Mixed precision enabled")
                except ImportError:
                    print("Mixed precision module not found, using FP32")
            
            self.update_progress.emit(10, "Running CFD...")
            geometry_mask = (voxel_tensor > 0.5).float()
            
            # Run simulation
            for step in range(self.steps):
                if self._is_interrupted:
                    return
                
                lbm_solver.collide_stream(geometry_mask, steps=1)
                
                if step % max(1, self.steps // 10) == 0:
                    progress = int(10 + (step / self.steps) * 90)
                    self.update_progress.emit(progress, f"Step {step}/{self.steps}")
            
            # Extract results - FIXED to handle both wrapped and unwrapped solvers
            if hasattr(lbm_solver, 'compute_macroscopic'):
                # Has compute_macroscopic method (works with wrapper too via __getattr__)
                rho, u = lbm_solver.compute_macroscopic()
            else:
                # Fallback: compute directly from f
                f_data = lbm_solver.f if hasattr(lbm_solver, 'f') else lbm_solver.solver.f
                rho = torch.sum(f_data, dim=0)
                u = None
            
            # Get velocity components
            velocity_x = lbm_solver.velocity_x.cpu().numpy()
            velocity_y = lbm_solver.velocity_y.cpu().numpy()
            velocity_z = lbm_solver.velocity_z.cpu().numpy()
            
            results = {
                "Pressure": lbm_solver.pressure.cpu().numpy() if hasattr(lbm_solver, 'pressure') else rho.cpu().numpy(),
                "Velocity_X": velocity_x,
                "Velocity_Y": velocity_y,
                "Velocity_Z": velocity_z,
                "Velocity Magnitude": np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2),
                "Density": rho.cpu().numpy(),
                "domain_size": domain_size,
                "grid_spacing": [grid_spacing_x, grid_spacing_y, grid_spacing_z],
                "geometry_mask": resized_voxel
            }
            
            # Add optional fields if available
            if hasattr(lbm_solver, 'vorticity'):
                vorticity_mag = torch.sqrt(torch.sum(lbm_solver.vorticity**2, dim=0))
                results["Vorticity Magnitude"] = vorticity_mag.cpu().numpy()
            
            if hasattr(lbm_solver, 'q_criterion'):
                results["Q-Criterion"] = lbm_solver.q_criterion.cpu().numpy()
            
            self.simulation_finished.emit(results)
            
        except Exception as e:
            self.simulation_error.emit(f"Error: {e}\n{traceback.format_exc()}")
    
    def requestInterruption(self):
        self._is_interrupted = True
