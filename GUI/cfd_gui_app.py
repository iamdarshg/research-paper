import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QComboBox,
    QProgressBar, QGroupBox, QTabWidget, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QVector3D

import pyqtgraph.opengl as gl
import numpy as np
import trimesh
import traceback
from cfd_solver_integration import CFDSolverWorker
from scipy.ndimage import gaussian_filter


class CFDVisualizationWidget(gl.GLViewWidget):
    """GPU-accelerated PyQtGraph GLViewWidget for 3D CFD visualizations"""
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set minimum size to prevent collapse
        self.setMinimumSize(800, 600)
        
        # Camera setup
        self.opts['distance'] = 100
        self.opts['elevation'] = 20
        self.opts['azimuth'] = 45
        self.opts['fov'] = 60
        
        # Background color
        self.setBackgroundColor((20, 20, 30))
        
        # Grid
        self.grid_item = gl.GLGridItem()
        self.grid_item.setSize(20, 20, 1)
        self.grid_item.setSpacing(1, 1, 1)
        self.addItem(self.grid_item)
        
        # Items
        self.mesh_item = None
        self.volume_item = None
        self.streamline_items = []
        
        # Parameters
        self.threshold = 0.1
        self.brightness = 1.0
        self.contrast = 1.0
        self.streamline_count = 50
        self.streamline_length = 100
        
        # Mesh transform
        self.mesh_rotation = [0, 0, 0]
        self.mesh_scale_factor = 1.0
        self.mesh_physical_size = 1.0
        
        # CFD domain
        self.cfd_padding_front = 1.0
        self.cfd_padding_back = 2.0
        self.cfd_padding_sides = 1.0
        self.cfd_padding_vertical = 1.0
        self.cfd_domain_size = [1.0, 1.0, 1.0]
        self.cfd_offset = [0, 0, 0]
        
        # Timer for smooth rendering
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update)
        self.render_timer.start(33)  # 30 FPS

    def set_streamline_params(self, count, length):
        self.streamline_count = count
        self.streamline_length = length

    def set_cfd_padding(self, front, back, sides, vertical):
        self.cfd_padding_front = front
        self.cfd_padding_back = back
        self.cfd_padding_sides = sides
        self.cfd_padding_vertical = vertical

    def set_mesh_rotation(self, rx, ry, rz):
        self.mesh_rotation = [rx, ry, rz]
        
    def set_brightness_contrast(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast

    def _apply_rotation(self, vertices, rx, ry, rz):
        """Apply rotation to vertices"""
        rx_rad = np.radians(rx)
        ry_rad = np.radians(ry)
        rz_rad = np.radians(rz)
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx_rad), -np.sin(rx_rad)],
                       [0, np.sin(rx_rad), np.cos(rx_rad)]])
        
        Ry = np.array([[np.cos(ry_rad), 0, np.sin(ry_rad)],
                       [0, 1, 0],
                       [-np.sin(ry_rad), 0, np.cos(ry_rad)]])
        
        Rz = np.array([[np.cos(rz_rad), -np.sin(rz_rad), 0],
                       [np.sin(rz_rad), np.cos(rz_rad), 0],
                       [0, 0, 1]])
        
        R = Rz @ Ry @ Rx
        return vertices @ R.T

    def display_stl(self, vertices, faces, physical_size=1.0):
        """Display STL mesh"""
        if self.mesh_item is not None:
            self.removeItem(self.mesh_item)
        
        self.mesh_physical_size = physical_size
        
        # Center mesh
        center = np.mean(vertices, axis=0)
        centered = vertices - center
        
        # Scale to physical units
        bbox = np.max(centered, axis=0) - np.min(centered, axis=0)
        max_extent = np.max(bbox)
        
        if max_extent > 0:
            scale = physical_size / max_extent
            normalized = centered * scale
        else:
            normalized = centered
        
        self.mesh_scale_factor = scale if max_extent > 0 else 1.0
        
        # Apply rotation
        rotated = self._apply_rotation(normalized, *self.mesh_rotation)
        
        # Create mesh item
        self.mesh_item = gl.GLMeshItem(
            vertexes=rotated,
            faces=faces,
            shader='shaded',
            smooth=True,
            color=(0.7, 0.7, 0.7, 0.7),
            glOptions='translucent'
        )
        self.addItem(self.mesh_item)
        
        # Calculate CFD domain
        body_length = physical_size
        self.cfd_domain_size = [
            body_length * (self.cfd_padding_front + 1.0 + self.cfd_padding_back),
            body_length * (2 * self.cfd_padding_sides + 1.0),
            body_length * (2 * self.cfd_padding_vertical + 1.0)
        ]
        
        self.cfd_offset = [
            -body_length * (self.cfd_padding_front - self.cfd_padding_back) / 2,
            0, 0
        ]
        
        # Center view
        self.center_view_on_mesh(rotated)

    def center_view_on_mesh(self, vertices):
        """Center camera on mesh"""
        if vertices.size > 0:
            center = np.mean(vertices, axis=0)
            bbox = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            max_extent = np.max(bbox)
            
            self.opts['center'] = QVector3D(
                float(center[0]), float(center[1]), float(center[2])
            )
            
            domain_extent = max(self.cfd_domain_size)
            self.opts['distance'] = max(domain_extent * 1.5, 3.0)

    def get_cfd_domain_params(self):
        """Return CFD domain parameters"""
        return {
            'domain_size': self.cfd_domain_size,
            'body_size': self.mesh_physical_size,
            'padding_front': self.cfd_padding_front,
            'padding_back': self.cfd_padding_back,
            'padding_sides': self.cfd_padding_sides,
            'padding_vertical': self.cfd_padding_vertical,
            'offset': self.cfd_offset
        }

    def display_volume_data(self, data, color_map='viridis', smooth=True):
        """GPU-accelerated volume rendering"""
        from matplotlib import cm
        import matplotlib.colors as mcolors
        
        if self.volume_item is not None:
            self.removeItem(self.volume_item)
        
        data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.std(data_clean) < 1e-12:
            print("Warning: Volume data uniform/empty")
            return
        
        # GPU processing if available
        if torch.cuda.is_available():
            try:
                data_torch = torch.from_numpy(data_clean).cuda()
                data_flat = data_torch.flatten()
                data_sorted = torch.sort(data_flat)[0]
                n = len(data_sorted)
                vmin = data_sorted[int(n * 0.02)].item()
                vmax = data_sorted[int(n * 0.98)].item()
                
                if vmax - vmin > 1e-10:
                    data_norm = torch.clamp((data_torch - vmin) / (vmax - vmin), 0, 1)
                else:
                    data_norm = torch.zeros_like(data_torch)
                
                data_adj = data_norm * self.contrast + (self.brightness - 1.0)
                data_adj = torch.clamp(data_adj, 0, 1).cpu().numpy()
                
            except Exception as e:
                print(f"GPU failed: {e}")
                vmin = np.percentile(data_clean, 2)
                vmax = np.percentile(data_clean, 98)
                if vmax - vmin < 1e-10:
                    return
                data_norm = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)
                data_adj = np.clip(data_norm * self.contrast + (self.brightness - 1.0), 0, 1)
        else:
            vmin = np.percentile(data_clean, 2)
            vmax = np.percentile(data_clean, 98)
            if vmax - vmin < 1e-10:
                return
            data_norm = np.clip((data_clean - vmin) / (vmax - vmin), 0, 1)
            data_adj = np.clip(data_norm * self.contrast + (self.brightness - 1.0), 0, 1)
        
        if smooth:
            data_adj = gaussian_filter(data_adj, sigma=0.8)
        
        data_thresh = np.where(data_adj < self.threshold, 0.0, data_adj)
        
        colormap = cm.get_cmap(color_map)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        data_rgba = colormap(norm(data_thresh))
        
        alpha = np.power(data_thresh, 0.25)
        alpha = np.clip(alpha * 0.95, 0.0, 0.95)
        data_rgba[..., 3] = alpha
        
        data_rgba_uint8 = (data_rgba * 255).astype(np.uint8)
        
        try:
            self.volume_item = gl.GLVolumeItem(data_rgba_uint8, glOptions='translucent')
            
            nx, ny, nz = data.shape
            voxel_size = [
                self.cfd_domain_size[0] / nx,
                self.cfd_domain_size[1] / ny,
                self.cfd_domain_size[2] / nz
            ]
            
            self.volume_item.scale(*voxel_size)
            
            offset = [
                self.cfd_offset[0] - self.cfd_domain_size[0] / 2,
                self.cfd_offset[1] - self.cfd_domain_size[1] / 2,
                self.cfd_offset[2] - self.cfd_domain_size[2] / 2
            ]
            self.volume_item.translate(*offset)
            
            self.addItem(self.volume_item)
            print(f"Volume: {data.shape}, range=[{vmin:.3e}, {vmax:.3e}]")
            
        except Exception as e:
            print(f"Volume display failed: {e}")
            traceback.print_exc()

    def generate_streamlines(self, vx, vy, vz):
        """Generate streamlines"""
        from scipy.integrate import odeint
        
        def velocity_field(point, t):
            i, j, k = int(point[0]), int(point[1]), int(point[2])
            if not (0 <= i < vx.shape[0]-1 and 
                    0 <= j < vx.shape[1]-1 and 
                    0 <= k < vx.shape[2]-1):
                return [0, 0, 0]
            
            dx, dy, dz = point[0]-i, point[1]-j, point[2]-k
            
            # Trilinear interpolation
            v = [0, 0, 0]
            for vi, vel in enumerate([vx, vy, vz]):
                v[vi] = (
                    vel[i,j,k]*(1-dx)*(1-dy)*(1-dz) +
                    vel[i+1,j,k]*dx*(1-dy)*(1-dz) +
                    vel[i,j+1,k]*(1-dx)*dy*(1-dz) +
                    vel[i+1,j+1,k]*dx*dy*(1-dz) +
                    vel[i,j,k+1]*(1-dx)*(1-dy)*dz +
                    vel[i+1,j,k+1]*dx*(1-dy)*dz +
                    vel[i,j+1,k+1]*(1-dx)*dy*dz +
                    vel[i+1,j+1,k+1]*dx*dy*dz
                )
            return v
        
        nx, ny, nz = vx.shape
        seeds = []
        grid = int(np.sqrt(self.streamline_count))
        for y in np.linspace(ny*0.3, ny*0.7, grid):
            for z in np.linspace(nz*0.3, nz*0.7, grid):
                seeds.append([nx*0.15, y, z])
        
        streamlines = []
        for seed in seeds[:self.streamline_count]:
            try:
                t = np.linspace(0, 0.5*self.streamline_length, self.streamline_length)
                sol = odeint(velocity_field, seed, t)
                
                mask = ((sol[:,0]>=0) & (sol[:,0]<nx) &
                        (sol[:,1]>=0) & (sol[:,1]<ny) &
                        (sol[:,2]>=0) & (sol[:,2]<nz))
                
                filtered = sol[mask]
                if len(filtered) > 2:
                    phys = np.zeros_like(filtered)
                    phys[:,0] = (filtered[:,0]/nx)*self.cfd_domain_size[0] + self.cfd_offset[0] - self.cfd_domain_size[0]/2
                    phys[:,1] = (filtered[:,1]/ny)*self.cfd_domain_size[1] + self.cfd_offset[1] - self.cfd_domain_size[1]/2
                    phys[:,2] = (filtered[:,2]/nz)*self.cfd_domain_size[2] + self.cfd_offset[2] - self.cfd_domain_size[2]/2
                    streamlines.append(phys)
            except:
                pass
        
        return streamlines

    def display_streamlines(self, streamlines):
        """Display streamlines"""
        for item in self.streamline_items:
            self.removeItem(item)
        self.streamline_items = []
        
        if not streamlines:
            return
        
        for streamline in streamlines[:self.streamline_count]:
            if len(streamline) > 1:
                t = np.linspace(0, 1, len(streamline))
                colors = np.zeros((len(streamline), 4))
                colors[:,0] = t
                colors[:,1] = 1-t
                colors[:,2] = 0.2
                colors[:,3] = 0.9
                
                item = gl.GLLinePlotItem(pos=streamline, color=colors, 
                                        width=2.5, antialias=True)
                self.streamline_items.append(item)
                self.addItem(item)

    def clear_all(self):
        """Clear all items"""
        if self.mesh_item:
            self.removeItem(self.mesh_item)
            self.mesh_item = None
        if self.volume_item:
            self.removeItem(self.volume_item)
            self.volume_item = None
        for item in self.streamline_items:
            self.removeItem(item)
        self.streamline_items = []


class CFD_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CFD Solver GUI - GPU Accelerated")
        self.setGeometry(100, 100, 1600, 1000)

        # Main widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.setup_control_panel()
        self.setup_visualization_panel()

        self.current_stl_path = None
        self.cfd_solver_thread = None
        self.stl_mesh = None

    def setup_control_panel(self):
        """Setup scrollable control panel"""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        scroll.setMaximumWidth(450)
        
        # Control widget
        control_widget = QWidget()
        self.control_panel = QVBoxLayout(control_widget)
        self.control_panel.setSpacing(5)
        scroll.setWidget(control_widget)
        
        self.main_layout.addWidget(scroll)

        # === STL Input ===
        stl_group = QGroupBox("STL Input & Geometry")
        stl_layout = QVBoxLayout()
        stl_group.setLayout(stl_layout)
        self.control_panel.addWidget(stl_group)

        self.load_stl_button = QPushButton("Load STL File")
        self.load_stl_button.clicked.connect(self.load_stl_file)
        stl_layout.addWidget(self.load_stl_button)

        self.stl_path_label = QLabel("No STL loaded.")
        self.stl_path_label.setWordWrap(True)
        stl_layout.addWidget(self.stl_path_label)
        
        # Physical size
        form = QFormLayout()
        self.physical_size_input = QDoubleSpinBox()
        self.physical_size_input.setRange(0.001, 1000.0)
        self.physical_size_input.setValue(1.0)
        self.physical_size_input.setDecimals(3)
        self.physical_size_input.setSuffix(" m")
        self.physical_size_input.valueChanged.connect(self.update_mesh_display)
        form.addRow("Physical Length:", self.physical_size_input)
        stl_layout.addLayout(form)
        
        # Padding
        pad_group = QGroupBox("CFD Domain Padding")
        pad_layout = QFormLayout()
        pad_group.setLayout(pad_layout)
        stl_layout.addWidget(pad_group)
        
        self.padding_front_input = QDoubleSpinBox()
        self.padding_front_input.setRange(0.1, 10.0)
        self.padding_front_input.setValue(1.0)
        self.padding_front_input.valueChanged.connect(self.update_cfd_padding)
        pad_layout.addRow("Front (L):", self.padding_front_input)
        
        self.padding_back_input = QDoubleSpinBox()
        self.padding_back_input.setRange(0.1, 10.0)
        self.padding_back_input.setValue(2.0)
        self.padding_back_input.valueChanged.connect(self.update_cfd_padding)
        pad_layout.addRow("Back (L):", self.padding_back_input)
        
        self.padding_sides_input = QDoubleSpinBox()
        self.padding_sides_input.setRange(0.1, 10.0)
        self.padding_sides_input.setValue(1.0)
        self.padding_sides_input.valueChanged.connect(self.update_cfd_padding)
        pad_layout.addRow("Sides (L):", self.padding_sides_input)
        
        self.padding_vertical_input = QDoubleSpinBox()
        self.padding_vertical_input.setRange(0.1, 10.0)
        self.padding_vertical_input.setValue(1.0)
        self.padding_vertical_input.valueChanged.connect(self.update_cfd_padding)
        pad_layout.addRow("Vertical (L):", self.padding_vertical_input)
        
        # Rotation
        rot_group = QGroupBox("Mesh Orientation")
        rot_layout = QFormLayout()
        rot_group.setLayout(rot_layout)
        stl_layout.addWidget(rot_group)
        
        self.rotation_x_slider = QSlider(Qt.Horizontal)
        self.rotation_x_slider.setRange(-180, 180)
        self.rotation_x_slider.setValue(0)
        self.rotation_x_slider.valueChanged.connect(self.update_mesh_rotation)
        self.rotation_x_label = QLabel("0°")
        rx_layout = QHBoxLayout()
        rx_layout.addWidget(self.rotation_x_slider)
        rx_layout.addWidget(self.rotation_x_label)
        rot_layout.addRow("Rotate X:", rx_layout)
        
        self.rotation_y_slider = QSlider(Qt.Horizontal)
        self.rotation_y_slider.setRange(-180, 180)
        self.rotation_y_slider.setValue(0)
        self.rotation_y_slider.valueChanged.connect(self.update_mesh_rotation)
        self.rotation_y_label = QLabel("0°")
        ry_layout = QHBoxLayout()
        ry_layout.addWidget(self.rotation_y_slider)
        ry_layout.addWidget(self.rotation_y_label)
        rot_layout.addRow("Rotate Y:", ry_layout)
        
        self.rotation_z_slider = QSlider(Qt.Horizontal)
        self.rotation_z_slider.setRange(-180, 180)
        self.rotation_z_slider.setValue(0)
        self.rotation_z_slider.valueChanged.connect(self.update_mesh_rotation)
        self.rotation_z_label = QLabel("0°")
        rz_layout = QHBoxLayout()
        rz_layout.addWidget(self.rotation_z_slider)
        rz_layout.addWidget(self.rotation_z_label)
        rot_layout.addRow("Rotate Z:", rz_layout)

        # === CFD Parameters ===
        cfd_group = QGroupBox("CFD Configuration")
        cfd_layout = QVBoxLayout()
        cfd_group.setLayout(cfd_layout)
        self.control_panel.addWidget(cfd_group)

        self.cfd_tabs = QTabWidget()
        cfd_layout.addWidget(self.cfd_tabs)

        # Basic tab
        basic_tab = QWidget()
        basic_form = QFormLayout()
        basic_tab.setLayout(basic_form)
        self.cfd_tabs.addTab(basic_tab, "Basic")

        self.reynolds_input = QDoubleSpinBox()
        self.reynolds_input.setRange(100, 1000000)
        self.reynolds_input.setValue(10000)
        self.reynolds_input.setSingleStep(1000)
        basic_form.addRow("Reynolds:", self.reynolds_input)

        self.mach_input = QDoubleSpinBox()
        self.mach_input.setRange(0.01, 0.5)
        self.mach_input.setValue(0.05)
        self.mach_input.setDecimals(3)
        basic_form.addRow("Mach:", self.mach_input)

        self.steps_input = QSpinBox()
        self.steps_input.setRange(10, 100000)
        self.steps_input.setValue(100)
        basic_form.addRow("Steps:", self.steps_input)

        # Advanced tab
        adv_tab = QWidget()
        adv_form = QFormLayout()
        adv_tab.setLayout(adv_form)
        self.cfd_tabs.addTab(adv_tab, "Advanced")

        self.grid_resolution_input = QSpinBox()
        self.grid_resolution_input.setRange(8, 512)
        self.grid_resolution_input.setValue(32)
        adv_form.addRow("Grid Resolution:", self.grid_resolution_input)

        self.solver_selector = QComboBox()
        self.solver_selector.addItems(["d3q19_mrt", "d3q27_cascaded"])
        adv_form.addRow("Solver:", self.solver_selector)

        self.run_cfd_button = QPushButton("Run CFD Simulation")
        self.run_cfd_button.clicked.connect(self.run_cfd_simulation)
        cfd_layout.addWidget(self.run_cfd_button)
        
        self.progress_bar = QProgressBar()
        cfd_layout.addWidget(self.progress_bar)

        gpu_status = "GPU Available" if torch.cuda.is_available() else "CPU Only"
        self.status_label = QLabel(f"Ready. {gpu_status}")
        self.status_label.setWordWrap(True)
        cfd_layout.addWidget(self.status_label)

        # === Visualization ===
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        vis_group.setLayout(vis_layout)
        self.control_panel.addWidget(vis_group)

        self.field_selector = QComboBox()
        self.field_selector.addItems(["None", "Pressure", "Velocity Magnitude", 
                                     "Density", "Streamlines", "Vorticity Magnitude", "Q-Criterion"])
        self.field_selector.currentIndexChanged.connect(self.update_visualization)
        vis_layout.addWidget(QLabel("Field:"))
        vis_layout.addWidget(self.field_selector)

        self.threshold_label = QLabel("Threshold: 0.10")
        vis_layout.addWidget(self.threshold_label)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        vis_layout.addWidget(self.threshold_slider)
        
        self.brightness_label = QLabel("Brightness: 1.00")
        vis_layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.update_brightness_contrast)
        vis_layout.addWidget(self.brightness_slider)
        
        self.contrast_label = QLabel("Contrast: 1.00")
        vis_layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_brightness_contrast)
        vis_layout.addWidget(self.contrast_slider)
        
        # Streamlines
        stream_group = QGroupBox("Streamlines")
        stream_layout = QVBoxLayout()
        stream_group.setLayout(stream_layout)
        vis_layout.addWidget(stream_group)
        
        self.streamline_count_label = QLabel("Count: 50")
        stream_layout.addWidget(self.streamline_count_label)
        self.streamline_count_slider = QSlider(Qt.Horizontal)
        self.streamline_count_slider.setRange(10, 200)
        self.streamline_count_slider.setValue(50)
        self.streamline_count_slider.valueChanged.connect(self.update_streamline_params)
        stream_layout.addWidget(self.streamline_count_slider)
        
        self.streamline_length_label = QLabel("Length: 100")
        stream_layout.addWidget(self.streamline_length_label)
        self.streamline_length_slider = QSlider(Qt.Horizontal)
        self.streamline_length_slider.setRange(20, 500)
        self.streamline_length_slider.setValue(100)
        self.streamline_length_slider.valueChanged.connect(self.update_streamline_params)
        stream_layout.addWidget(self.streamline_length_slider)

        self.control_panel.addStretch()

    def setup_visualization_panel(self):
        """Setup visualization panel"""
        self.viewer = CFDVisualizationWidget()
        self.main_layout.addWidget(self.viewer, stretch=1)
    
    def update_cfd_padding(self):
        self.viewer.set_cfd_padding(
            self.padding_front_input.value(),
            self.padding_back_input.value(),
            self.padding_sides_input.value(),
            self.padding_vertical_input.value()
        )
        self.update_mesh_display()
    
    def update_mesh_rotation(self):
        rx = self.rotation_x_slider.value()
        ry = self.rotation_y_slider.value()
        rz = self.rotation_z_slider.value()
        
        self.rotation_x_label.setText(f"{rx}°")
        self.rotation_y_label.setText(f"{ry}°")
        self.rotation_z_label.setText(f"{rz}°")
        
        self.viewer.set_mesh_rotation(rx, ry, rz)
        self.update_mesh_display()
    
    def update_mesh_display(self):
        if self.stl_mesh is not None:
            self.viewer.display_stl(
                self.stl_mesh.vertices, 
                self.stl_mesh.faces, 
                self.physical_size_input.value()
            )
    
    def update_threshold(self):
        val = self.threshold_slider.value() / 100.0
        self.threshold_label.setText(f"Threshold: {val:.2f}")
        self.viewer.threshold = val
        self.update_visualization()
    
    def update_brightness_contrast(self):
        brightness = self.brightness_slider.value() / 100.0
        contrast = self.contrast_slider.value() / 100.0
        
        self.brightness_label.setText(f"Brightness: {brightness:.2f}")
        self.contrast_label.setText(f"Contrast: {contrast:.2f}")
        
        self.viewer.set_brightness_contrast(brightness, contrast)
        self.update_visualization()
    
    def update_streamline_params(self):
        count = self.streamline_count_slider.value()
        length = self.streamline_length_slider.value()
        
        self.streamline_count_label.setText(f"Count: {count}")
        self.streamline_length_label.setText(f"Length: {length}")
        
        self.viewer.set_streamline_params(count, length)
        
        if hasattr(self, 'cfd_results') and self.field_selector.currentText() == "Streamlines":
            self.regenerate_streamlines()

    def regenerate_streamlines(self):
        if all(k in self.cfd_results for k in ["Velocity_X", "Velocity_Y", "Velocity_Z"]):
            sl = self.viewer.generate_streamlines(
                self.cfd_results["Velocity_X"],
                self.cfd_results["Velocity_Y"],
                self.cfd_results["Velocity_Z"]
            )
            self.viewer.display_streamlines(sl)

    def load_stl_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load STL", "", "STL Files (*.stl)")
        if filepath:
            self.current_stl_path = filepath
            self.stl_path_label.setText(f"Loaded: {os.path.basename(filepath)}")
            
            try:
                mesh = trimesh.load_mesh(filepath)
                if not hasattr(mesh, 'vertices') or mesh.vertices.size == 0:
                    self.status_label.setText("Invalid STL file")
                    return
                self.stl_mesh = mesh
                self.viewer.display_stl(mesh.vertices, mesh.faces, self.physical_size_input.value())
                self.status_label.setText("STL loaded. Ready to simulate.")
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
                traceback.print_exc()

    def run_cfd_simulation(self):
        if not self.current_stl_path:
            self.status_label.setText("Load STL first!")
            return

        if hasattr(self, 'cfd_thread') and self.cfd_thread.isRunning():
            self.cfd_thread.requestInterruption()
            self.cfd_thread.wait()

        self.status_label.setText("Starting simulation...")
        self.progress_bar.setValue(0)
        self.run_cfd_button.setEnabled(False)
        self.viewer.clear_all()

        self.cfd_thread = QThread()
        self.cfd_solver_worker = CFDSolverWorker(
            self.current_stl_path,
            self.reynolds_input.value(),
            self.mach_input.value(),
            self.steps_input.value(),
            self.solver_selector.currentText()
        )
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
        self.status_label.setText("Simulation complete!")
        self.run_cfd_button.setEnabled(True)
        self.cfd_results = results
        
        print("\n=== Results ===")
        for key, val in results.items():
            if isinstance(val, np.ndarray) and val.ndim == 3:
                print(f"{key}: {val.shape}, [{np.min(val):.3e}, {np.max(val):.3e}]")
        
        self.update_visualization()
        self.cfd_thread.quit()
        self.cfd_thread.wait()

    def cfd_simulation_error(self, message):
        self.status_label.setText(f"Error: {message}")
        self.run_cfd_button.setEnabled(True)

    def update_visualization(self):
        field = self.field_selector.currentText()
        if not hasattr(self, 'cfd_results') or self.cfd_results is None:
            return

        self.viewer.clear_all()

        if self.stl_mesh:
            self.viewer.display_stl(
                self.stl_mesh.vertices,
                self.stl_mesh.faces,
                self.physical_size_input.value()
            )

        if field == "None":
            return

        data = self.cfd_results.get(field)

        if field == "Streamlines" or data is None:
            if field == "Streamlines" and "Velocity_X" in self.cfd_results:
                self.regenerate_streamlines()
            return

        if field in ["Pressure", "Velocity Magnitude", "Density", "Vorticity Magnitude", "Q-Criterion"]:
            self.viewer.display_volume_data(data, smooth=True)
            self.status_label.setText(f"Displaying {field}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent rendering
    window = CFD_GUI()
    window.show()
    sys.exit(app.exec_())
