import numpy as np
import trimesh
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import sys
import os
import traceback
# Add the parent directory to the sys.path to allow importing CLI modules
sys.path.append(r"D:\research-paper\CLI")

try:
    from aircraft_diffusion_cfd import AdvancedCFDSimulator, CFDConfig, LBMPhysicsConfig
    from advanced_lbm_solver import GPULBMSolver, D3Q27CascadedSolver # Direct import of the solvers
    IMPORTS_SUCCESSFUL = True
except Exception as e:
    print(f"Warning: Could not import CFD modules: {e}")
    print(traceback.format_exc())
    IMPORTS_SUCCESSFUL = False

class CFDSolverWorker(QObject):
    """
    Worker class to run CFD simulation in a separate thread.
    Emits signals for progress, completion, and errors.
    """
    update_progress = pyqtSignal(int, str)
    simulation_finished = pyqtSignal(dict) # Dictionary of all CFD results
    simulation_error = pyqtSignal(str)

    def __init__(self, stl_path, reynolds, mach, steps, solver_type="d3q19_mrt", 
                 grid_resolution=32, cfd_domain_params=None):
        super().__init__()
        self.stl_path = stl_path
        self.reynolds = reynolds
        self.mach = mach
        self.steps = steps
        self.solver_type = solver_type
        self.grid_resolution = grid_resolution
        self.cfd_domain_params = cfd_domain_params or {}
        self._is_interrupted = False

    def run_simulation(self):
        try:
            import torch
            self._is_interrupted = False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.update_progress.emit(0, "Loading STL geometry...")
            # Convert STL to voxel grid
            mesh = trimesh.load_mesh(self.stl_path)
            
            # Get CFD domain parameters from GUI
            domain_size = self.cfd_domain_params.get('domain_size', [1.0, 1.0, 1.0])
            body_size = self.cfd_domain_params.get('body_size', 1.0)
            padding_front = self.cfd_domain_params.get('padding_front', 1.0)
            padding_back = self.cfd_domain_params.get('padding_back', 2.0)
            padding_sides = self.cfd_domain_params.get('padding_sides', 1.0)
            padding_vertical = self.cfd_domain_params.get('padding_vertical', 1.0)
            offset = self.cfd_domain_params.get('offset', [0, 0, 0])
            
            # Use grid_resolution from GUI
            grid_resolution = self.grid_resolution
            
            # Calculate grid spacing based on domain size
            grid_spacing_x = domain_size[0] / grid_resolution
            grid_spacing_y = domain_size[1] / grid_resolution
            grid_spacing_z = domain_size[2] / grid_resolution
            
            print(f"CFD Domain: {domain_size}, Body Size: {body_size}, Grid: {grid_resolution}^3")
            print(f"Grid Spacing: [{grid_spacing_x:.4f}, {grid_spacing_y:.4f}, {grid_spacing_z:.4f}]")
            
            # Estimate bounding box and voxelize
            bounds = mesh.bounds
            mesh_extent = bounds[1] - bounds[0]
            max_mesh_extent = np.max(mesh_extent)
            
            # Scale mesh to physical body size
            if max_mesh_extent > 1e-6:
                scale_factor = body_size / max_mesh_extent
                mesh.vertices = (mesh.vertices - bounds[0]) * scale_factor
            
            # Center the mesh in the domain
            mesh_center = np.mean(mesh.vertices, axis=0)
            domain_center = np.array(domain_size) / 2
            mesh.vertices = mesh.vertices - mesh_center + domain_center + np.array(offset)
            
            # Calculate appropriate voxel pitch based on grid resolution
            voxel_pitch = min(grid_spacing_x, grid_spacing_y, grid_spacing_z)
            
            self.update_progress.emit(5, f"Voxelizing mesh (pitch={voxel_pitch:.4f})...")
            # Voxelize with calculated pitch
            try:
                voxel_grid_trimesh = mesh.voxelized(voxel_pitch).fill()
                voxel_grid_np = voxel_grid_trimesh.matrix.view(np.ndarray)
            except Exception as e:
                print(f"Voxelization error: {e}, using coarser pitch")
                voxel_pitch *= 2
                voxel_grid_trimesh = mesh.voxelized(voxel_pitch).fill()
                voxel_grid_np = voxel_grid_trimesh.matrix.view(np.ndarray)

            # Resize voxel grid to target resolution
            current_shape = voxel_grid_np.shape
            target_shape = (grid_resolution, grid_resolution, grid_resolution)
            
            self.update_progress.emit(8, f"Resizing voxel grid: {current_shape} -> {target_shape}...")
            
            # Use scipy for better interpolation
            from scipy.ndimage import zoom
            zoom_factors = np.array(target_shape) / np.array(current_shape)
            resized_voxel_grid_np = zoom(voxel_grid_np.astype(np.float32), zoom_factors, order=1)
            
            # Ensure binary mask
            resized_voxel_grid_np = (resized_voxel_grid_np > 0.5).astype(np.float32)

            # Convert to torch tensor
            voxel_grid_tensor = torch.from_numpy(resized_voxel_grid_np).float().to(device)

            # Setup CFD configuration with proper physical scaling
            cfd_config = CFDConfig(
                base_grid_resolution=grid_resolution,
                reynolds_number=self.reynolds,
                mach_number=self.mach,
                simulation_steps=self.steps
            )
            
            # Initialize LBM physics config with proper grid spacing
            cfd_config.lbm_config = LBMPhysicsConfig()
            cfd_config.lbm_config.physical_length_scale = body_size
            cfd_config.lbm_config.grid_spacing = grid_spacing_x  # Use actual calculated spacing
            cfd_config.lbm_config.compute_q_criterion = True
            cfd_config.lbm_config.use_vorticity_confinement = True
            
            # Choose solver based on type
            if self.solver_type == "d3q27_cascaded":
                self.update_progress.emit(10, "Using D3Q27 Cascaded solver...")
                lbm_solver = D3Q27CascadedSolver(cfd_config, device, LBMPhysicsConfig)
            else:
                simulator = AdvancedCFDSimulator(cfd_config, device)
                lbm_solver: GPULBMSolver = simulator.lbm_solver

            self.update_progress.emit(10, "Running CFD simulation...")
            geometry_mask = (voxel_grid_tensor > 0.5).float()
            
            # Tracking for dynamic range
            max_pressure = -float("inf")
            min_pressure = float("inf")
            max_vel_mag = -float("inf")
            min_vel_mag = float("inf")
            max_density = -float("inf")
            min_density = float("inf")
            max_vort_mag = -float("inf")
            min_vort_mag = float("inf")
            max_q_crit = -float("inf")
            min_q_crit = float("inf")

            # Data collection
            all_pressure_data = []
            all_velocity_x = []
            all_velocity_y = []
            all_velocity_z = []
            all_density_data = []
            all_vorticity_magnitude_data = []
            all_q_criterion_data = []

            output_data_interval = max(1, self.steps // 10)

            for step in range(self.steps):
                if self._is_interrupted:
                    self.update_progress.emit(0, "Simulation interrupted.")
                    return

                lbm_solver.collide_stream(geometry_mask, steps=1)

                # Extract data
                rho = torch.sum(lbm_solver.f, dim=0)
                ux, uy, uz = lbm_solver.velocity_x, lbm_solver.velocity_y, lbm_solver.velocity_z
                pressure = lbm_solver.pressure
                vorticity_mag = torch.sqrt(torch.sum(lbm_solver.vorticity**2, dim=0))
                q_criterion = lbm_solver.q_criterion
                
                # Update ranges
                max_pressure = max(max_pressure, pressure.max().item())
                min_pressure = min(min_pressure, pressure.min().item())
                
                velocity_magnitude = torch.sqrt(ux**2 + uy**2 + uz**2)
                max_vel_mag = max(max_vel_mag, velocity_magnitude.max().item())
                min_vel_mag = min(min_vel_mag, velocity_magnitude.min().item())
                
                max_density = max(max_density, rho.max().item())
                min_density = min(min_density, rho.min().item())
                
                max_vort_mag = max(max_vort_mag, vorticity_mag.max().item())
                min_vort_mag = min(min_vort_mag, vorticity_mag.min().item())
                
                max_q_crit = max(max_q_crit, q_criterion.max().item())
                min_q_crit = min(min_q_crit, q_criterion.min().item())

                if step % output_data_interval == 0 or step == self.steps - 1:
                    all_pressure_data.append(pressure.cpu().numpy())
                    all_velocity_x.append(ux.cpu().numpy())
                    all_velocity_y.append(uy.cpu().numpy())
                    all_velocity_z.append(uz.cpu().numpy())
                    all_density_data.append(rho.cpu().numpy())
                    all_vorticity_magnitude_data.append(vorticity_mag.cpu().numpy())
                    all_q_criterion_data.append(q_criterion.cpu().numpy())

                progress_percent = int(10 + (step + 1) / self.steps * 90)
                self.update_progress.emit(progress_percent, f"CFD: {step+1}/{self.steps} steps.")
            
            # Final data
            final_pressure = all_pressure_data[-1] if all_pressure_data else np.zeros(target_shape)
            final_velocity_x = all_velocity_x[-1] if all_velocity_x else np.zeros(target_shape)
            final_velocity_y = all_velocity_y[-1] if all_velocity_y else np.zeros(target_shape)
            final_velocity_z = all_velocity_z[-1] if all_velocity_z else np.zeros(target_shape)
            final_density = all_density_data[-1] if all_density_data else np.zeros(target_shape)
            final_vorticity_magnitude = all_vorticity_magnitude_data[-1] if all_vorticity_magnitude_data else np.zeros(target_shape)
            final_q_criterion = all_q_criterion_data[-1] if all_q_criterion_data else np.zeros(target_shape)
            
            final_velocity_magnitude = np.sqrt(final_velocity_x**2 + final_velocity_y**2 + final_velocity_z**2)

            results = {
                "Pressure": final_pressure,
                "Velocity Magnitude": final_velocity_magnitude,
                "Velocity_X": final_velocity_x,
                "Velocity_Y": final_velocity_y,
                "Velocity_Z": final_velocity_z,
                "Density": final_density,
                "Vorticity Magnitude": final_vorticity_magnitude,
                "Q-Criterion": final_q_criterion,
                "Pressure_MinMax": (min_pressure, max_pressure),
                "VelocityMagnitude_MinMax": (min_vel_mag, max_vel_mag),
                "Density_MinMax": (min_density, max_density),
                "VorticityMagnitude_MinMax": (min_vort_mag, max_vort_mag),
                "Q_Criterion_MinMax": (min_q_crit, max_q_crit),
                "domain_size": domain_size,
                "grid_spacing": [grid_spacing_x, grid_spacing_y, grid_spacing_z],
                "geometry_mask": resized_voxel_grid_np
            }
            self.simulation_finished.emit(results)

        except Exception as e:
            import traceback
            error_msg = f"CFD Simulation Error: {e}\n{traceback.format_exc()}"
            self.simulation_error.emit(error_msg)
            print(error_msg)

    def requestInterruption(self):
        self._is_interrupted = True
