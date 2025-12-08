import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QSlider, QComboBox,
    QProgressBar, QGroupBox, QRadioButton, QTabWidget, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QVector3D

import pyqtgraph.opengl as gl
import numpy as np
import trimesh
import os
import traceback
from cfd_solver_integration import CFDSolverWorker


class CFDVisualizationWidget(gl.GLViewWidget):
    """
    A PyQtGraph GLViewWidget for displaying 3D CFD visualizations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['distance'] = 100
        self.opts['elevation'] = 20  # Look down from slightly above
        self.opts['azimuth'] = 45    # Rotate 45 degrees
        self.addItem(gl.GLGridItem())
        self.mesh_item = None
        self.volume_item = None
        self.streamline_item = None
        self.threshold = 0.0

    def display_stl(self, vertices, faces):
        """
        Displays an STL mesh with proper centering and orientation.
        """
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
        
        # Center the mesh at origin
        center = np.mean(vertices, axis=0)
        centered_vertices = vertices - center
        
        # Compute bounding box for scaling
        bbox_min = np.min(centered_vertices, axis=0)
        bbox_max = np.max(centered_vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        max_extent = np.max(bbox_size)
        
        # Scale to fit within [-25, 25] range for good visibility
        if max_extent > 0:
            scale_factor = 50.0 / max_extent
            normalized_vertices = centered_vertices * scale_factor
        else:
            normalized_vertices = centered_vertices
        
        # Create mesh item with proper orientation
        # Most STLs are in Z-up convention, but PyQtGraph uses Y-up
        # Rotate -90 degrees around X to convert Z-up to Y-up
        rotation_matrix = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ])
        rotated_vertices = normalized_vertices @ rotation_matrix.T

        self.mesh_item = gl.GLMeshItem(
            vertexes=rotated_vertices,
            faces=faces,
            shader='shaded',
            smooth=True,
            color=(0.8, 0.8, 0.8, 0.6),
            glOptions='translucent'
        )
        self.addItem(self.mesh_item)
        self.center_view_on_mesh(rotated_vertices)

    def center_view_on_mesh(self, vertices):
        """
        Adjusts the camera to center the view on the loaded mesh.
        """
        if vertices.size > 0:
            center = np.mean(vertices, axis=0)
            bbox_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            max_extent = np.max(bbox_size)
            
            # Set camera center to mesh center
            self.opts['center'] = QVector3D(
                float(center[0]), 
                float(center[1]), 
                float(center[2])
            )
            
            # Set distance based on mesh size (1.8x for good framing)
            self.opts['distance'] = max(max_extent * 1.8, 50.0)

    def display_volume_data(self, data, color_map='viridis'):
        """
        Displays 3D volumetric data with proper transparency and scaling.
        """
        from matplotlib import cm
        import matplotlib.colors as mcolors
        
        if self.volume_item is not None:
            self.removeItem(self.volume_item)
        
        # Clean data
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check if data is all zeros or garbage
        if np.std(data_clean) < 1e-12:
            print("Warning: Volume data appears to be uniform/empty")
            return
        
        # Robust normalization using percentiles
        vmin = np.percentile(data_clean, 5)   # 5th percentile
        vmax = np.percentile(data_clean, 95)  # 95th percentile
        
        if vmax - vmin < 1e-10:
            print("Warning: Volume data has no variation")
            return
        
        # Normalize to [0, 1]
        data_normalized = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)
        
        # Apply threshold: values below threshold become transparent
        data_thresholded = np.where(
            data_normalized < self.threshold,
            0.0,
            data_normalized
        )
        
        # Apply colormap
        colormap = cm.get_cmap(color_map)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        data_rgba = colormap(norm(data_thresholded))
        
        # CRITICAL: Make alpha proportional to data magnitude
        # This creates proper transparency for low-value regions
        alpha = np.power(data_thresholded, 0.5)  # Gamma correction for better visibility
        alpha = np.clip(alpha, 0.0, 0.8)  # Limit max opacity to 0.8 for translucency
        data_rgba[..., 3] = alpha
        
        # Convert to uint8
        data_rgba_uint8 = (data_rgba * 255).astype(np.uint8)
        
        try:
            self.volume_item = gl.GLVolumeItem(data_rgba_uint8, glOptions='translucent')
            
            # Match coordinate system with mesh
            # If your grid is 32x32x32 and mesh is scaled to fit [-25, 25]:
            mesh_extent = 50.0  # From -25 to +25
            grid_size = max(data.shape)
            voxel_size = mesh_extent / grid_size
            
            # Apply scaling
            self.volume_item.scale(voxel_size, voxel_size, voxel_size)
            
            # Center at origin
            nx, ny, nz = data.shape
            offset_x = -nx * voxel_size / 2.0
            offset_y = -ny * voxel_size / 2.0
            offset_z = -nz * voxel_size / 2.0
            self.volume_item.translate(offset_x, offset_y, offset_z)
            
            self.addItem(self.volume_item)
            
            print(f"Volume displayed: shape={data.shape}, range=[{vmin:.3e}, {vmax:.3e}]")
            
        except Exception as e:
            print(f"Failed to display volume data: {e}")
            traceback.print_exc()



    def generate_streamlines(self, velocity_x, velocity_y, velocity_z, num_seeds=50, step_size=0.5):
        """
        Generates streamlines from 3D velocity fields.
        """
        from scipy.integrate import odeint
        
        def velocity_field(point, t):
            i, j, k = point
            i_int, j_int, k_int = int(i), int(j), int(k)
            
            # Boundary check
            if not (0 <= i_int < velocity_x.shape[0]-1 and 
                    0 <= j_int < velocity_x.shape[1]-1 and 
                    0 <= k_int < velocity_x.shape[2]-1):
                return [0, 0, 0]
            
            dx = i - i_int
            dy = j - j_int
            dz = k - k_int

            # Trilinear interpolation
            vx = (velocity_x[i_int, j_int, k_int] * (1-dx)*(1-dy)*(1-dz) +
                  velocity_x[i_int+1, j_int, k_int] * dx*(1-dy)*(1-dz) +
                  velocity_x[i_int, j_int+1, k_int] * (1-dx)*dy*(1-dz) +
                  velocity_x[i_int+1, j_int+1, k_int] * dx*dy*(1-dz) +
                  velocity_x[i_int, j_int, k_int+1] * (1-dx)*(1-dy)*dz +
                  velocity_x[i_int+1, j_int, k_int+1] * dx*(1-dy)*dz +
                  velocity_x[i_int, j_int+1, k_int+1] * (1-dx)*dy*dz +
                  velocity_x[i_int+1, j_int+1, k_int+1] * dx*dy*dz)

            vy = (velocity_y[i_int, j_int, k_int] * (1-dx)*(1-dy)*(1-dz) +
                  velocity_y[i_int+1, j_int, k_int] * dx*(1-dy)*(1-dz) +
                  velocity_y[i_int, j_int+1, k_int] * (1-dx)*dy*(1-dz) +
                  velocity_y[i_int+1, j_int+1, k_int] * dx*dy*(1-dz) +
                  velocity_y[i_int, j_int, k_int+1] * (1-dx)*(1-dy)*dz +
                  velocity_y[i_int+1, j_int, k_int+1] * dx*(1-dy)*dz +
                  velocity_y[i_int, j_int+1, k_int+1] * (1-dx)*dy*dz +
                  velocity_y[i_int+1, j_int+1, k_int+1] * dx*dy*dz)

            vz = (velocity_z[i_int, j_int, k_int] * (1-dx)*(1-dy)*(1-dz) +
                  velocity_z[i_int+1, j_int, k_int] * dx*(1-dy)*(1-dz) +
                  velocity_z[i_int, j_int+1, k_int] * (1-dx)*dy*(1-dz) +
                  velocity_z[i_int+1, j_int+1, k_int] * dx*dy*(1-dz) +
                  velocity_z[i_int, j_int, k_int+1] * (1-dx)*(1-dy)*dz +
                  velocity_z[i_int+1, j_int, k_int+1] * dx*(1-dy)*dz +
                  velocity_z[i_int, j_int+1, k_int+1] * (1-dx)*dy*dz +
                  velocity_z[i_int+1, j_int+1, k_int+1] * dx*dy*dz)

            return [vx, vy, vz]

        nx, ny, nz = velocity_x.shape
        center_x, center_y, center_z = nx / 2, ny / 2, nz / 2

        # Generate seeds in a plane upstream of the mesh
        seeds = []
        grid_size = int(np.sqrt(num_seeds))
        for y in np.linspace(center_y - 10, center_y + 10, grid_size):
            for z in np.linspace(center_z - 10, center_z + 10, grid_size):
                seeds.append([5, y, z])  # Start from x=5 (upstream)

        streamlines = []
        max_steps = 100

        for seed in seeds[:num_seeds]:
            try:
                t = np.linspace(0, step_size * max_steps, max_steps)
                sol = odeint(velocity_field, seed, t)

                # Filter out-of-bounds points
                mask = ((sol[:, 0] >= 0) & (sol[:, 0] < nx) &
                        (sol[:, 1] >= 0) & (sol[:, 1] < ny) &
                        (sol[:, 2] >= 0) & (sol[:, 2] < nz))

                filtered = sol[mask]
                if len(filtered) > 2:
                    # Center streamline coordinates to match mesh
                    centered = filtered - np.array([nx/2, ny/2, nz/2])
                    streamlines.append(centered)

            except Exception as e:
                print(f"Failed to generate streamline from seed {seed}: {e}")

        return streamlines

    def display_streamlines(self, streamlines):
        """
        Displays streamlines in the 3D view.
        """
        if self.streamline_item is not None:
            for item in self.streamline_item:
                self.removeItem(item)
            self.streamline_item = None

        if not streamlines:
            return

        line_items = []
        for streamline in streamlines[:50]:
            if len(streamline) > 1:
                item = gl.GLLinePlotItem(
                    pos=streamline, 
                    color=(0, 1, 0, 0.8), 
                    width=2, 
                    antialias=True
                )
                line_items.append(item)
                self.addItem(item)

        self.streamline_item = line_items

    def clear_all(self):
        """Clears all displayed items except grid."""
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.volume_item is not None:
            self.removeItem(self.volume_item)
            self.volume_item = None
        if self.streamline_item is not None:
            for item in self.streamline_item:
                self.removeItem(item)
            self.streamline_item = None


class CFD_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CFD Solver GUI")
        self.setGeometry(100, 100, 1400, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.setup_control_panel()
        self.setup_visualization_panel()

        self.current_stl_path = None
        self.cfd_solver_thread = None
        self.stl_mesh = None

    def setup_control_panel(self):
        self.control_panel = QVBoxLayout()
        self.main_layout.addLayout(self.control_panel, 1)

        # --- STL Input ---
        stl_group = QGroupBox("STL Input")
        stl_layout = QVBoxLayout()
        stl_group.setLayout(stl_layout)
        self.control_panel.addWidget(stl_group)

        self.load_stl_button = QPushButton("Load STL File")
        self.load_stl_button.clicked.connect(self.load_stl_file)
        stl_layout.addWidget(self.load_stl_button)

        self.stl_path_label = QLabel("No STL loaded.")
        stl_layout.addWidget(self.stl_path_label)

        # --- CFD Parameters ---
        cfd_params_group = QGroupBox("CFD Configuration")
        cfd_params_layout = QVBoxLayout()
        cfd_params_group.setLayout(cfd_params_layout)
        self.control_panel.addWidget(cfd_params_group)

        self.cfd_tabs = QTabWidget()
        cfd_params_layout.addWidget(self.cfd_tabs)

        # --- Basic Parameters Tab ---
        basic_tab = QWidget()
        basic_layout = QFormLayout()
        basic_tab.setLayout(basic_layout)
        self.cfd_tabs.addTab(basic_tab, "Basic Parameters")

        self.reynolds_input = QDoubleSpinBox()
        self.reynolds_input.setRange(100, 1000000)  # Removed upper bound
        self.reynolds_input.setValue(10000)
        self.reynolds_input.setSingleStep(1000)
        basic_layout.addRow("Reynolds Number:", self.reynolds_input)

        self.mach_input = QDoubleSpinBox()
        self.mach_input.setRange(0.01, 0.5)
        self.mach_input.setValue(0.05)
        self.mach_input.setSingleStep(0.01)
        self.mach_input.setDecimals(3)
        basic_layout.addRow("Mach Number:", self.mach_input)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(10, 100000)  # Increased range
        self.steps_input.setValue(100)
        self.steps_input.setSingleStep(10)
        basic_layout.addRow("Simulation Steps:", self.steps_input)

        # --- Advanced Parameters Tab ---
        advanced_tab = QWidget()
        advanced_layout = QFormLayout()
        advanced_tab.setLayout(advanced_layout)
        self.cfd_tabs.addTab(advanced_tab, "Advanced Parameters")

        self.grid_resolution_input = QSpinBox()
        self.grid_resolution_input.setRange(16, 8192)  # Increased range
        self.grid_resolution_input.setValue(32)
        self.grid_resolution_input.setSingleStep(8)
        advanced_layout.addRow("Grid Resolution:", self.grid_resolution_input)

        self.adaptive_cells_input = QSpinBox()
        self.adaptive_cells_input.setRange(1000, 1000000)  # Increased range
        self.adaptive_cells_input.setValue(5000)
        self.adaptive_cells_input.setSingleStep(1000)
        advanced_layout.addRow("Adaptive Cells Target:", self.adaptive_cells_input)

        self.refinement_levels_input = QSpinBox()
        self.refinement_levels_input.setRange(1, 10)  # Increased range
        self.refinement_levels_input.setValue(3)
        advanced_layout.addRow("Refinement Levels:", self.refinement_levels_input)

        self.output_interval_input = QSpinBox()
        self.output_interval_input.setRange(1, 10000)  # Increased range
        self.output_interval_input.setValue(50)
        advanced_layout.addRow("Output Interval:", self.output_interval_input)

        # Solver selection
        solver_selection_group = QGroupBox("Solver Selection")
        solver_selection_layout = QFormLayout()
        solver_selection_group.setLayout(solver_selection_layout)
        advanced_layout.addRow(solver_selection_group)

        self.solver_selector = QComboBox()
        self.solver_selector.addItems(["d3q19_mrt", "d3q27_cascaded"])
        self.solver_selector.setCurrentText("d3q19_mrt")
        solver_selection_layout.addRow("CFD Solver:", self.solver_selector)

        # LBM Physics parameters
        lbm_group = QGroupBox("LBM Physics Configuration")
        lbm_layout = QFormLayout()
        lbm_group.setLayout(lbm_layout)
        advanced_layout.addRow(lbm_group)

        self.turbulence_model_selector = QComboBox()
        self.turbulence_model_selector.addItems(["none", "smagorinsky", "dynamic_smagorinsky", "wale"])
        self.turbulence_model_selector.setCurrentText("dynamic_smagorinsky")
        lbm_layout.addRow("Turbulence Model:", self.turbulence_model_selector)

        self.smagorinsky_constant_input = QDoubleSpinBox()
        self.smagorinsky_constant_input.setRange(0.01, 1.0)
        self.smagorinsky_constant_input.setValue(0.17)
        self.smagorinsky_constant_input.setDecimals(3)
        self.smagorinsky_constant_input.setSingleStep(0.01)
        lbm_layout.addRow("Smagorinsky Constant:", self.smagorinsky_constant_input)

        self.use_vorticity_confinement = QCheckBox()
        self.use_vorticity_confinement.setChecked(True)
        lbm_layout.addRow("Vorticity Confinement:", self.use_vorticity_confinement)

        self.use_q_criterion = QCheckBox()
        self.use_q_criterion.setChecked(True)
        lbm_layout.addRow("Compute Q-Criterion:", self.use_q_criterion)

        # Relaxation times
        relaxation_group = QGroupBox("Relaxation Parameters")
        relaxation_layout = QFormLayout()
        relaxation_group.setLayout(relaxation_layout)
        advanced_layout.addRow(relaxation_group)

        self.s_bulk_input = QDoubleSpinBox()
        self.s_bulk_input.setRange(0.5, 5.0)  # Increased range
        self.s_bulk_input.setValue(1.0)
        self.s_bulk_input.setSingleStep(0.1)
        self.s_bulk_input.setDecimals(1)
        relaxation_layout.addRow("Bulk Viscosity (s_bulk):", self.s_bulk_input)

        self.s_energy_input = QDoubleSpinBox()
        self.s_energy_input.setRange(1.0, 5.0)  # Increased range
        self.s_energy_input.setValue(1.2)
        self.s_energy_input.setSingleStep(0.1)
        self.s_energy_input.setDecimals(1)
        relaxation_layout.addRow("Energy Mode (s_energy):", self.s_energy_input)

        self.s_higher_input = QDoubleSpinBox()
        self.s_higher_input.setRange(1.0, 5.0)  # Increased range
        self.s_higher_input.setValue(1.4)
        self.s_higher_input.setSingleStep(0.1)
        self.s_higher_input.setDecimals(1)
        relaxation_layout.addRow("Higher Modes (s_higher):", self.s_higher_input)

        self.run_cfd_button = QPushButton("Run CFD Simulation")
        self.run_cfd_button.clicked.connect(self.run_cfd_simulation)
        cfd_params_layout.addWidget(self.run_cfd_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        cfd_params_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready.")
        cfd_params_layout.addWidget(self.status_label)

        # --- Visualization Controls ---
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        vis_group.setLayout(vis_layout)
        self.control_panel.addWidget(vis_group)

        vis_layout.addWidget(QLabel("Display Field:"))
        self.field_selector = QComboBox()
        self.field_selector.addItems(["None", "Pressure", "Velocity Magnitude", "Density", 
                                     "Streamlines", "Vorticity Magnitude", "Q-Criterion"])
        self.field_selector.currentIndexChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.field_selector)

        self.vis_slider_label = QLabel("Threshold (0-1):")
        vis_layout.addWidget(self.vis_slider_label)
        self.vis_slider = QSlider(Qt.Horizontal)
        self.vis_slider.setRange(0, 100)
        self.vis_slider.setValue(10)
        self.vis_slider.valueChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.vis_slider)

        self.control_panel.addStretch(1)

    def setup_visualization_panel(self):
        self.visualization_panel = QVBoxLayout()
        self.main_layout.addLayout(self.visualization_panel, 3)

        self.viewer = CFDVisualizationWidget()
        self.visualization_panel.addWidget(self.viewer)

    def load_stl_file(self):
        file_dialog = QFileDialog()
        filepath, _ = file_dialog.getOpenFileName(self, "Load STL File", "", "STL Files (*.stl)")
        if filepath:
            self.current_stl_path = filepath
            self.stl_path_label.setText(f"Loaded: {os.path.basename(filepath)}")
            self.status_label.setText("STL file loaded. Ready to simulate.")
            
            try:
                mesh = trimesh.load_mesh(filepath)
                if (not hasattr(mesh, 'vertices') or mesh.vertices.size == 0 or
                    not hasattr(mesh, 'faces') or mesh.faces.size == 0):
                    self.status_label.setText("STL file is empty or invalid.")
                    return
                self.stl_mesh = mesh
                self.viewer.display_stl(mesh.vertices, mesh.faces)
            except Exception as e:
                self.status_label.setText(f"Error loading STL: {e}")
                traceback.print_exc()

    def run_cfd_simulation(self):
        if not self.current_stl_path:
            self.status_label.setText("Please load an STL file first.")
            return

        if self.cfd_solver_thread and self.cfd_solver_thread.isRunning():
            self.cfd_solver_thread.requestInterruption()
            self.cfd_solver_thread.wait()

        reynolds = self.reynolds_input.value()
        mach = self.mach_input.value()
        steps = self.steps_input.value()
        solver_type = self.solver_selector.currentText()

        self.status_label.setText("Starting CFD simulation...")
        self.progress_bar.setValue(0)
        self.run_cfd_button.setEnabled(False)
        self.viewer.clear_all()

        self.cfd_thread = QThread()
        self.cfd_solver_worker = CFDSolverWorker(self.current_stl_path, reynolds, mach, steps, solver_type)
        self.cfd_solver_worker.moveToThread(self.cfd_thread)

        self.cfd_solver_worker.update_progress.connect(self.update_progress)
        self.cfd_solver_worker.simulation_finished.connect(self.cfd_simulation_finished)
        self.cfd_solver_worker.simulation_error.connect(self.cfd_simulation_error)
        self.cfd_thread.started.connect(self.cfd_solver_worker.run_simulation)

        self.cfd_thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def cfd_simulation_finished(self, results):
        self.status_label.setText("CFD simulation complete.")
        self.run_cfd_button.setEnabled(True)
        self.cfd_results = results
        
        if "Velocity_X" in results and "Velocity_Y" in results and "Velocity_Z" in results:
            self.cfd_results["Streamlines"] = self.viewer.generate_streamlines(
                results["Velocity_X"], results["Velocity_Y"], results["Velocity_Z"]
            )
        self.update_visualization()
        self.cfd_thread.quit()
        self.cfd_thread.wait()

    def cfd_simulation_error(self, message):
        self.status_label.setText(f"CFD Error: {message}")
        self.run_cfd_button.setEnabled(True)

    def update_visualization(self):
        selected_field = self.field_selector.currentText()
        if not hasattr(self, 'cfd_results') or self.cfd_results is None:
            return

        self.viewer.threshold = self.vis_slider.value() / 100.0
        self.viewer.clear_all()

        if self.stl_mesh:
            self.viewer.display_stl(self.stl_mesh.vertices, self.stl_mesh.faces)

        if selected_field == "None":
            return

        data_to_display = self.cfd_results.get(selected_field)

        if data_to_display is None:
            self.status_label.setText(f"No data available for '{selected_field}'.")
            return

        if selected_field in ["Pressure", "Velocity Magnitude", "Density", "Vorticity Magnitude", "Q-Criterion"]:
            self.viewer.display_volume_data(data_to_display)
            self.status_label.setText(f"Displaying {selected_field}.")
        elif selected_field == "Streamlines":
            if data_to_display and len(data_to_display) > 0:
                self.viewer.display_streamlines(data_to_display)
                self.status_label.setText(f"Displaying {selected_field}.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CFD_GUI()
    window.show()
    sys.exit(app.exec_())
