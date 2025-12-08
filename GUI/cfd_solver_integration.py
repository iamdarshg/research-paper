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
    from advanced_lbm_solver import GPULBMSolver # Direct import of the solver
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

    def __init__(self, stl_path, reynolds, mach, steps):
        super().__init__()
        self.stl_path = stl_path
        self.reynolds = reynolds
        self.mach = mach
        self.steps = steps
        self._is_interrupted = False

    def run_simulation(self):
        try:
            import torch
            self._is_interrupted = False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            self.update_progress.emit(0, "Loading STL geometry...")
            # Convert STL to voxel grid
            mesh = trimesh.load_mesh(self.stl_path)
            
            # Estimate bounding box and voxelize
            # For simplicity, we'll assume a fixed grid size for the solver
            # A more advanced approach would dynamically size or pad the grid
            grid_resolution = 32 # Consistent with the CFDConfig default
            
            # Scale mesh to fit within a 0-1 unit cube for voxelization, then scale to grid_resolution
            # Find the max dimension of the mesh
            bounds = mesh.bounds
            max_extent = np.max(bounds[1] - bounds[0])
            
            # Scale and translate mesh to fit in a unit cube from 0 to 1
            if max_extent > 1e-6: # Avoid division by zero
                scale_factor = 1.0 / max_extent
                mesh.vertices = (mesh.vertices - bounds[0]) * scale_factor
            
            # Voxelize the mesh
            # Use `fill_distance` to create a solid object
            voxel_grid_trimesh = mesh.voxelized(0.01).fill() # Smaller pitch for better resolution
            voxel_grid_np = voxel_grid_trimesh.matrix.view(np.ndarray)

            # The voxel_grid_np will have its own dimensions based on the mesh and pitch.
            # We need to resize it to a fixed resolution for the LBM solver (e.g., 32x32x32)
            # Pad or crop and then interpolate to the target resolution
            current_shape = voxel_grid_np.shape
            target_shape = (grid_resolution, grid_resolution, grid_resolution)
            
            # Create an empty target grid
            resized_voxel_grid_np = np.zeros(target_shape, dtype=np.float32)

            # Calculate scaling factors
            scale_factors = np.array(target_shape) / np.array(current_shape)
            
            # Simple nearest-neighbor resize for demonstration, could use more advanced interpolation
            for x in range(target_shape[0]):
                for y in range(target_shape[1]):
                    for z in range(target_shape[2]):
                        src_x = int(x / scale_factors[0])
                        src_y = int(y / scale_factors[1])
                        src_z = int(z / scale_factors[2])
                        
                        if 0 <= src_x < current_shape[0] and \
                           0 <= src_y < current_shape[1] and \
                           0 <= src_z < current_shape[2]:
                            resized_voxel_grid_np[x, y, z] = voxel_grid_np[src_x, src_y, src_z]

            # Convert to torch tensor, add batch and channel dimensions as expected by AdvancedCFDSimulator
            voxel_grid_tensor = torch.from_numpy(resized_voxel_grid_np).float().to(device)
            # The AdvancedCFDSimulator expects [D, H, W] for the geometry parameter
            # The voxel_grid in _voxel_to_stl_path is also [D, H, W]
            # So no unsqueeze needed here, just make sure it's 3D

            # Setup CFD configuration
            cfd_config = CFDConfig(
                base_grid_resolution=grid_resolution,
                reynolds_number=self.reynolds,
                mach_number=self.mach,
                simulation_steps=self.steps
            )
            # Ensure LBMPhysicsConfig is properly initialized with new grid spacing
            cfd_config.lbm_config = LBMPhysicsConfig()
            cfd_config.lbm_config.grid_spacing = cfd_config.lbm_config.physical_length_scale / cfd_config.base_grid_resolution

            simulator = AdvancedCFDSimulator(cfd_config, device)

            self.update_progress.emit(10, "Running CFD simulation...")

            # Run simulation steps
            # The simulate_aerodynamics method only returns final coefficients.
            # We need to modify it or the solver to get intermediate field data.
            # For now, I'll directly interact with the LBM solver to extract data.
            
            # Initialize LBM solver within simulator
            # The simulator already has an lbm_solver instance
            lbm_solver: GPULBMSolver = simulator.lbm_solver
            geometry_mask = (voxel_grid_tensor > 0.5).float() # Binary mask for solid
            
            # Create placeholders to store max/min of fields for dynamic range adjustment
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

            # Run steps and collect data
            all_pressure_data = []
            all_velocity_x = []
            all_velocity_y = []
            all_velocity_z = []
            all_density_data = [] # Rho is density
            all_vorticity_magnitude_data = []
            all_q_criterion_data = []

            output_data_interval = max(1, self.steps // 10) # Collect data ~10 times during simulation

            for step in range(self.steps):
                if self._is_interrupted:
                    self.update_progress.emit(0, "Simulation interrupted.")
                    return

                lbm_solver.collide_stream(geometry_mask, steps=1) # Run one step

                # Extract data after each step
                rho = torch.sum(lbm_solver.f, dim=0) # Rho is sum of populations
                ux, uy, uz = lbm_solver.velocity_x, lbm_solver.velocity_y, lbm_solver.velocity_z
                pressure = lbm_solver.pressure
                vorticity_mag = torch.sqrt(torch.sum(lbm_solver.vorticity**2, dim=0))
                q_criterion = lbm_solver.q_criterion
                
                # Update max/min for dynamic range
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

                progress_percent = int((step + 1) / self.steps * 100)
                self.update_progress.emit(progress_percent, f"CFD progress: {step+1}/{self.steps} steps.")
            
            # Take the last collected data for final visualization
            final_pressure = all_pressure_data[-1] if all_pressure_data else np.zeros(target_shape)
            final_velocity_x = all_velocity_x[-1] if all_velocity_x else np.zeros(target_shape)
            final_velocity_y = all_velocity_y[-1] if all_velocity_y else np.zeros(target_shape)
            final_velocity_z = all_velocity_z[-1] if all_velocity_z else np.zeros(target_shape)
            final_density = all_density_data[-1] if all_density_data else np.zeros(target_shape)
            final_vorticity_magnitude = all_vorticity_magnitude_data[-1] if all_vorticity_magnitude_data else np.zeros(target_shape)
            final_q_criterion = all_q_criterion_data[-1] if all_q_criterion_data else np.zeros(target_shape)
            
            # Calculate final velocity magnitude from its components
            final_velocity_magnitude = np.sqrt(final_velocity_x**2 + final_velocity_y**2 + final_velocity_z**2)

            results = {
                "Pressure": final_pressure,
                "Velocity Magnitude": final_velocity_magnitude,
                "Velocity_X": final_velocity_x, # For streamlines
                "Velocity_Y": final_velocity_y, # For streamlines
                "Velocity_Z": final_velocity_z, # For streamlines
                "Density": final_density,
                "Vorticity Magnitude": final_vorticity_magnitude,
                "Q-Criterion": final_q_criterion,
                "Pressure_MinMax": (min_pressure, max_pressure),
                "VelocityMagnitude_MinMax": (min_vel_mag, max_vel_mag),
                "Density_MinMax": (min_density, max_density),
                "VorticityMagnitude_MinMax": (min_vort_mag, max_vort_mag),
                "Q_Criterion_MinMax": (min_q_crit, max_q_crit)
            }
            self.simulation_finished.emit(results)

        except Exception as e:
            import traceback
            error_msg = f"CFD Simulation Error: {e}\n{traceback.format_exc()}"
            self.simulation_error.emit(error_msg)
            print(error_msg)

    def requestInterruption(self):
        self._is_interrupted = True
