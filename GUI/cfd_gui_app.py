import sys
import os
# Add the script's directory to the path to allow importing local modules
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
# Import the CFD solver integration
from cfd_solver_integration import CFDSolverWorker

class CFDVisualizationWidget(gl.GLViewWidget):
    """
    A PyQtGraph GLViewWidget for displaying 3D CFD visualizations.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.opts['distance'] = 200 # Initial camera distance
        self.addItem(gl.GLGridItem())
        self.mesh_item = None
        self.volume_item = None
        self.streamline_item = None
        self.threshold = 0.0  # Threshold for debugging (0-1)

    def display_stl(self, vertices, faces):
        """
        Displays an STL mesh.
        """
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
        
        # Normalize vertices to fit within a reasonable range for visualization
        # This is a simplified normalization; a more robust one might center the mesh.
        max_dim = np.max(vertices)
        min_dim = np.min(vertices)
        scale_factor = 50.0 / (max_dim - min_dim) if (max_dim - min_dim) != 0 else 1.0
        normalized_vertices = (vertices - min_dim) * scale_factor - 25.0 # Center around origin

        self.mesh_item = gl.GLMeshItem(
            vertexes=normalized_vertices,
            faces=faces,
            shader='shaded',
            color=(1, 1, 1, 0.5)
        )
        self.addItem(self.mesh_item)
        self.center_view_on_mesh(normalized_vertices)

    def center_view_on_mesh(self, vertices):
        """
        Adjusts the camera to center the view on the loaded mesh.
        """
        if vertices.size > 0:
            center = np.mean(vertices, axis=0)
            # Adjust 'center' option directly if it exists, or set a target for camera
            # PyQtGraph's GLViewWidget doesn't have a direct setCenter, but `opts['center']` can influence
            # the pivot point for rotation. Distance is more effective for initial view.
            
            self.opts['center'] = QVector3D(float(center[0]), float(center[1]), float(center[2]))
            self.opts['distance'] = max(np.max(vertices) - np.min(vertices), 1e-12) * 1.5 # Adjust distance based on mesh size

    def display_volume_data(self, data, color_map='viridis'):
        """
        Displays 3D volumetric data (e.g., pressure, density, vorticity).
        `data` is a 3D numpy array.
        Debug mode shows values below threshold or NaNs at threshold level.
        """
        from matplotlib import cm
        if self.volume_item is not None:
            self.removeItem(self.volume_item)
        
        # Normalize data to [0, 1] for visualization if not already
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-9)

        # Apply debug threshold: show low values and NaNs at threshold level
        data_debug = np.where(
            (data_normalized < self.threshold) | np.isnan(data_normalized),
            self.threshold,
            data_normalized
        )
        # Apply colormap to create RGBA values
        import matplotlib.colors as mcolors

        colormap = cm.get_cmap(color_map)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        data_rgba = colormap(norm(data_debug))
        data_debug = (data_rgba[..., :3] * 255).astype(np.uint8)
        try:
            self.volume_item = gl.GLVolumeItem(data_debug, glOptions='translucent')
            self.addItem(self.volume_item)
            # Center the view on the volume data
            center_x, center_y, center_z = data.shape[0] / 2, data.shape[1] / 2, data.shape[2] / 2
            self.opts['center'] = QVector3D(center_x, center_y, center_z)
            self.opts['distance'] = max(data.shape) * 1.2
        except Exception as e:
            print(f"Failed to display volume data: {e}")

    def generate_streamlines(self, velocity_x, velocity_y, velocity_z, num_seeds=50, step_size=0.5):
        """
        Generates streamlines from 3D velocity fields using a simple integration method.
        Returns a list of streamline points (list of numpy arrays).
        """
        from scipy.integrate import odeint
        

        def velocity_field(point, t):
            # Interpolate velocity at point (bilinear for simplicity)
            i, j, k = point
            i_int, j_int, k_int = int(i), int(j), int(k)
            dx = i - i_int
            dy = j - j_int
            dz = k - k_int

            # Clamp indices
            i_int = np.clip(i_int, 0, velocity_x.shape[0] - 2)
            j_int = np.clip(j_int, 0, velocity_x.shape[1] - 2)
            k_int = np.clip(k_int, 0, velocity_x.shape[2] - 2)

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

        # Generate seed points in a grid pattern around the center
        seeds = []
        grid_size = int(np.sqrt(num_seeds))
        for x in np.linspace(center_x - 5, center_x + 5, grid_size):
            for y in np.linspace(center_y - 5, center_y + 5, grid_size):
                seeds.append([x, y, center_z])

        streamlines = []
        max_steps = 50  # Maximum integration steps

        for seed in seeds[:50]:  # Increase to 50 seeds for more streamlines
            try:
                # Integrate forward
                t_forward = np.arange(0, step_size * max_steps, step_size)
                sol_forward = odeint(velocity_field, seed, t_forward)

                # Integrate backward
                t_backward = np.arange(0, -step_size * max_steps, -step_size)
                sol_backward = odeint(velocity_field, seed, t_backward)[::-1]  # Reverse to connect smoothly

                # Combine forward and backward
                streamline = np.concatenate([sol_backward[:-1], sol_forward], axis=0)

                # Filter out points outside the domain
                mask = ((streamline[:, 0] >= 0) & (streamline[:, 0] < nx) &
                        (streamline[:, 1] >= 0) & (streamline[:, 1] < ny) &
                        (streamline[:, 2] >= 0) & (streamline[:, 2] < nz))

                filtered_streamline = streamline[mask]
                if len(filtered_streamline) > 1:
                    streamlines.append(filtered_streamline)

            except Exception as e:
                print(f"Failed to generate streamline from seed {seed}: {e}")

        return streamlines

    def display_streamlines(self, streamlines):
        """
        Displays streamlines in the 3D view.
        """
        if self.streamline_item is not None:
            self.removeItem(self.streamline_item)

        if not streamlines:
            return

        # Create a list of gl.GLLinePlotItem for each streamline
        line_items = []
        for streamline in streamlines[:50]:  # Limit for performance
            if len(streamline) > 1:
                item = gl.GLLinePlotItem(pos=streamline, color=(0, 1, 0, 1), width=2, antialias=True)
                line_items.append(item)
                self.addItem(item)

        self.streamline_item = line_items  # Store the list
        # Center the view on the streamlines
        all_points = np.concatenate(streamlines, axis=0)
        if len(all_points) > 0:
            center = np.mean(all_points, axis=0)
            min_bounds = np.min(all_points, axis=0)
            max_bounds = np.max(all_points, axis=0)
            size = np.max(max_bounds - min_bounds)
            self.opts['center'] = QVector3D(center[0], center[1], center[2])
            self.opts['distance'] = size * 1.5 if size > 0 else 50
        else:
            self.opts['center'] = QVector3D(16, 16, 16)
            self.opts['distance'] = 50

    def clear_all(self):
        """Clears all displayed items."""
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
        self.setGeometry(100, 100, 1200, 800)

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
        self.main_layout.addLayout(self.control_panel, 1) # Occupy 1/3 of width

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

        # Create tab widget for CFD configuration
        self.cfd_tabs = QTabWidget()
        cfd_params_layout.addWidget(self.cfd_tabs)

        # --- Basic Parameters Tab ---
        basic_tab = QWidget()
        basic_layout = QFormLayout()
        basic_tab.setLayout(basic_layout)
        self.cfd_tabs.addTab(basic_tab, "Basic Parameters")

        self.reynolds_input = QDoubleSpinBox()
        self.reynolds_input.setRange(100, 10000000)
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
        self.steps_input.setRange(10, 10000)
        self.steps_input.setValue(100)
        self.steps_input.setSingleStep(10)
        basic_layout.addRow("Simulation Steps:", self.steps_input)

        # --- Advanced Parameters Tab ---
        advanced_tab = QWidget()
        advanced_layout = QFormLayout()
        advanced_tab.setLayout(advanced_layout)
        self.cfd_tabs.addTab(advanced_tab, "Advanced Parameters")

        # Grid Resolution
        self.grid_resolution_input = QSpinBox()
        self.grid_resolution_input.setRange(16, 128)
        self.grid_resolution_input.setValue(32)
        self.grid_resolution_input.setSingleStep(8)
        advanced_layout.addRow("Grid Resolution:", self.grid_resolution_input)

        # Adaptive mesh settings
        self.adaptive_cells_input = QSpinBox()
        self.adaptive_cells_input.setRange(1000, 100000)
        self.adaptive_cells_input.setValue(5000)
        self.adaptive_cells_input.setSingleStep(1000)
        advanced_layout.addRow("Adaptive Cells Target:", self.adaptive_cells_input)

        self.refinement_levels_input = QSpinBox()
        self.refinement_levels_input.setRange(1, 5)
        self.refinement_levels_input.setValue(3)
        advanced_layout.addRow("Refinement Levels:", self.refinement_levels_input)

        # Simulation settings
        self.output_interval_input = QSpinBox()
        self.output_interval_input.setRange(1, 1000)
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
        self.solver_selector.setToolTip("d3q19_mrt: D3Q19 MRT collision with turbulence models\n"
                                      "d3q27_cascaded: D3Q27 cascaded collision with central moments")
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
        self.s_bulk_input.setRange(0.5, 2.0)
        self.s_bulk_input.setValue(1.0)
        self.s_bulk_input.setSingleStep(0.1)
        self.s_bulk_input.setDecimals(1)
        relaxation_layout.addRow("Bulk Viscosity (s_bulk):", self.s_bulk_input)

        self.s_energy_input = QDoubleSpinBox()
        self.s_energy_input.setRange(1.0, 3.0)
        self.s_energy_input.setValue(1.2)
        self.s_energy_input.setSingleStep(0.1)
        self.s_energy_input.setDecimals(1)
        relaxation_layout.addRow("Energy Mode (s_energy):", self.s_energy_input)

        self.s_higher_input = QDoubleSpinBox()
        self.s_higher_input.setRange(1.0, 3.0)
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
        self.field_selector.addItems(["None", "Pressure", "Velocity Magnitude", "Density", "Streamlines", "Vorticity Magnitude", "Q-Criterion"])
        self.field_selector.currentIndexChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.field_selector)

        # Debug threshold slider for showing low values/NaNs
        self.vis_slider_label = QLabel("Debug Threshold (0-1):")
        vis_layout.addWidget(self.vis_slider_label)
        self.vis_slider = QSlider(Qt.Horizontal)
        self.vis_slider.setRange(0, 100)
        self.vis_slider.setValue(10)  # Default 0.1
        self.vis_slider.valueChanged.connect(self.update_visualization)
        vis_layout.addWidget(self.vis_slider)

        self.control_panel.addStretch(1) # Push everything to the top

    def setup_visualization_panel(self):
        self.visualization_panel = QVBoxLayout()
        self.main_layout.addLayout(self.visualization_panel, 3) # Occupy 2/3 of width

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
                if (not hasattr(mesh, 'vertices') or not isinstance(mesh.vertices, np.ndarray) or mesh.vertices.size == 0 or
                    not hasattr(mesh, 'faces') or not isinstance(mesh.faces, np.ndarray) or mesh.faces.size == 0):
                    self.status_label.setText("STL file is empty or invalid.")
                    return
                self.stl_mesh = mesh
                self.viewer.display_stl(mesh.vertices, mesh.faces)
            except Exception as e:
                self.status_label.setText(f"Error loading STL: {e}")
                print(f"Error loading STL: {e}")
                print(traceback.format_exc())

    def run_cfd_simulation(self):
        if not self.current_stl_path:
            self.status_label.setText("Please load an STL file first.")
            return

        # Ensure previous simulation thread is stopped
        if self.cfd_solver_thread and self.cfd_solver_thread.isRunning():
            self.cfd_solver_thread.requestInterruption()
            self.cfd_solver_thread.wait() # Wait for it to finish

        # Get parameters from GUI
        reynolds = self.reynolds_input.value()
        mach = self.mach_input.value()
        steps = self.steps_input.value()
        solver_type = self.solver_selector.currentText()

        self.status_label.setText("Starting CFD simulation...")
        self.progress_bar.setValue(0)
        self.run_cfd_button.setEnabled(False)
        self.viewer.clear_all() # Clear previous visualization

        # Create the worker and thread for real CFD simulation
        self.cfd_thread = QThread()
        self.cfd_solver_worker = CFDSolverWorker(self.current_stl_path, reynolds, mach, steps, solver_type)
        self.cfd_solver_worker.moveToThread(self.cfd_thread)

        # Connect signals
        self.cfd_solver_worker.update_progress.connect(self.update_progress)
        self.cfd_solver_worker.simulation_finished.connect(self.cfd_simulation_finished)
        self.cfd_solver_worker.simulation_error.connect(self.cfd_simulation_error)
        self.cfd_thread.started.connect(self.cfd_solver_worker.run_simulation)

        # Start the thread
        self.cfd_thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def cfd_simulation_finished(self, results):
        self.status_label.setText("CFD simulation complete.")
        self.run_cfd_button.setEnabled(True)
        # Store results for visualization
        self.cfd_results = results
        # Generate streamlines from velocity data if available
        if "Velocity_X" in results and "Velocity_Y" in results and "Velocity_Z" in results:
            self.cfd_results["Streamlines"] = self.viewer.generate_streamlines(
                results["Velocity_X"], results["Velocity_Y"], results["Velocity_Z"]
            )
        self.update_visualization() # Automatically update display with new data
        # Quit and wait for the thread
        self.cfd_thread.quit()
        self.cfd_thread.wait()

    def cfd_simulation_error(self, message):
        self.status_label.setText(f"CFD Error: {message}")
        self.run_cfd_button.setEnabled(True)

    def update_visualization(self):
        selected_field = self.field_selector.currentText()
        if not hasattr(self, 'cfd_results') or self.cfd_results is None:
            self.status_label.setText("No CFD results to visualize. Run simulation first.")
            return

        # Update debug threshold from slider
        self.viewer.threshold = self.vis_slider.value() / 100.0

        self.viewer.clear_all() # Clear volume and streamlines, but STL will be re-added

        # Always display STL as background if available
        if self.stl_mesh:
            self.viewer.display_stl(self.stl_mesh.vertices, self.stl_mesh.faces)

        if selected_field == "None":
            self.status_label.setText("Displaying STL geometry.")
            return

        data_to_display = self.cfd_results.get(selected_field)

        if data_to_display is None:
            self.status_label.setText(f"No data available for '{selected_field}'.")
            return

        if selected_field in ["Pressure", "Velocity Magnitude", "Density", "Vorticity Magnitude", "Q-Criterion"]:
            # Display volumetric data
            self.viewer.display_volume_data(data_to_display)
            self.status_label.setText(f"Displaying {selected_field}.")
        elif selected_field == "Streamlines":
            if data_to_display and len(data_to_display) > 0:
                self.viewer.display_streamlines(data_to_display)
                self.status_label.setText(f"Displaying {selected_field}.")
            else:
                self.status_label.setText("No streamlines to display.")
        else:
            self.status_label.setText(f"Unknown visualization field: {selected_field}.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CFD_GUI()
    window.show()
    sys.exit(app.exec_())
