
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class LabelTier(str, Enum):
    LBM_RAW = "lbm_raw"
    LBM_CALIBRATED = "lbm_calibrated"
    EXTERNAL_PDE = "external_pde"

@dataclass
class DiffusionConfig:
    """Diffusion model hyperparameters with consistency distillation support"""
    timesteps: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.02
    sampling_timesteps: int = 250
    guidance_scale: float = 7.5
    # Consistency distillation settings
    teacher_steps: int = 2000  # Original teacher model steps
    student_steps: int = 32     # Target student model steps
    progressive_distillation: List[int] = None  # 500→250→125→64→32→16→8→4

    def __post_init__(self):
        if self.progressive_distillation is None:
            self.progressive_distillation = [500, 250, 125, 64, 32]

@dataclass
class ModelConfig:
    """Model architecture parameters with grouped-query attention"""
    latent_dim: int = 16
    condition_dim: int = 32
    xyz_dim: int = 3
    encoder_channels: List[int] = None
    decoder_channels: List[int] = None
    # Grouped-query attention instead of multi-head
    attention_groups: int = 8  # 4 groups instead of 8 heads (50% KV-cache reduction)
    attention_kv_groups: int = 8  # Groups for key/value
    num_attention_layers: int = 4
    # Grid resolution - configurable for different lattice sizes
    base_grid_resolution: int = 32  # Consistent grid resolution for voxel, CFD, etc.
    grid_resolution: int = None  # Working grid resolution (defaults to base_grid_resolution if not set)
    target_grid_resolution: int = 1024 # Final high-res target for geometry
    # Memory optimization
    enable_gradient_checkpointing: bool = True  # 60% VRAM savings
    use_torch_compile: bool = False  # Kernel fusion

    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [24, 32, 48]
        if self.decoder_channels is None:
            self.decoder_channels = [48, 32, 24]
        # Set working grid resolution if not specified
        if self.grid_resolution is None:
            self.grid_resolution = self.base_grid_resolution

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 0.99
    ema_decay: float = 0.99
    disconnection_penalty: float = 50.0
    precision: str = 'float32'
    save_interval: int = 5
    val_interval: int = 2
    # Pipeline parallelism
    enable_pipeline_parallelism: bool = True  # Overlap CFD with diffusion
    num_pipeline_stages: int = 8  # CFD + Diffusion stages

@dataclass
class LBMPhysicsConfig:
    """Easy-to-update configuration for LBM physics constants"""
    # Turbulence modeling
    turbulence_model: str = "dynamic_smagorinsky"  # Options: 'smagorinsky', 'dynamic_smagorinsky', 'wale', 'none'
    smagorinsky_constant: float = 0.17  # Cs for LES turbulence model
    wale_constant: float = 0.5  # Cw for WALE model
    use_les_turbulence: bool = True
    physical_length_scale: float = 1.0  # Physical length of the voxel grid (m)
    grid_spacing: float = 0.01  # Physical spacing per grid cell (m) - calculated from physical_length_scale
    time_step: float = 0.001  # Time step size (s)
    test_filter_ratio: float = 2.0  # Ratio for test filter in dynamic Smagorinsky model
    dynamic_cs_clip_min: float = 0.0  # Minimum Cs value for dynamic model
    dynamic_cs_clip_max: float = 0.2  # Maximum Cs value for dynamic model
    use_vorticity_confinement: bool = True  # Enable vorticity confinement
    vc_adaptive_strength: float = 0.1  # Adaptive vorticity confinement strength
    vc_adaptive: bool = True  # Adaptive strength factor
    vorticity_confinement_epsilon: float = 0.1  # Vorticity confinement parameter
    compute_q_criterion: bool = True  # Compute Q-criterion for vortex identification
    q_threshold: float = 0.0  # Threshold for Q-criterion visualization

    # MRT relaxation times (for different moment components)
    # s0-s18 correspond to different moments in MRT collision
    # Lower values = more dissipation, higher stability
    s_nu: float = None  # Set from Reynolds number (kinematic viscosity)
    s_bulk: float = 1.0  # Bulk viscosity relaxation (density fluctuations)
    s_energy: float = 1.2  # Energy mode relaxation
    s_higher: float = 1.4  # Higher order moments relaxation

    # D3Q27 Cascaded LBM relaxation parameters
    s_nu_d3q27: float = 1.0 / 0.6    # Viscosity relaxation
    s_e_d3q27: float = 1.2           # Energy relaxation
    s_h_d3q27: float = 1.6           # Higher order relaxation
    tau_min_d3q27: float = 0.52      # BGK stability floor for under-resolved high-Re runs

    # Boundary conditions
    inlet_velocity_relaxation: float = 0.5  # For Zou-He BC smoothing
    convergence_tolerance: float = 1e-5  # Velocity change threshold
    max_iterations: int = 50000  # Safety limit
    check_convergence_every: int = 250  # Steps between convergence checks

    # Stability
    max_mach: float = 0.3  # Maximum Mach number for stability
    target_lattice_velocity: float = 0.12  # Cap inlet speed in lattice units for high-resolution stability
    cfl_safety_factor: float = 0.8  # CFL condition safety margin

    # Low-Mach corrections
    use_incompressible_correction: bool = True  # Use rho0=1 for incompressible

    # Force computation
    momentum_exchange_correction: bool = True  # Apply momentum-exchange method
    use_triton_streaming: bool = False  # Keep disabled until fused kernel matches physics path exactly
    drag_link_metric_exponent: Optional[float] = None  # Auto D3Q27 face/edge/corner metric correction
    drag_reference_speed: float = 80.0  # Natural-unit reference speed for projected-pressure Cd labels
    drag_speed_normalization_exponent: float = 1.0  # OpenFOAM pressure fallback scales nearly linearly with U_inf
    use_shape_drag_correction: bool = True # Keep True by default for stable training, but solver labels it separately
    shape_drag_correction_coefficients: Tuple[float, ...] = (
        -12.633030612111941, 27.87582461044955, -10.247055184812014,
        22.962648171191816, -17.337224317584685, -3.946645931513679,
        0.08323209768046214, 4.548014973469924, -5.179313884992105,
        -7.623947231425998,
    )
    shape_drag_correction_min: float = 0.1
    shape_drag_correction_max: float = 3.0

@dataclass
class CFDLabel:
    """Comprehensive simulation record for multi-fidelity CFD labels (Issue #15)"""
    geometry_id: str
    geometry_ref: Optional[str] = None # Path to .npy or .stl
    mission_profile: Dict[str, Any] = field(default_factory=dict)
    constraints_profile: Dict[str, Any] = field(default_factory=dict)

    # Aerodynamic metrics
    cd: float = 0.0
    cl: float = 0.0
    cm: Optional[float] = None

    # Field references (Optional paths to stored fields)
    pressure_field_path: Optional[str] = None
    velocity_field_paths: Dict[str, str] = field(default_factory=dict) # ux, uy, uz

    # Multi-fidelity history
    fidelity_history: List[Dict[str, Any]] = field(default_factory=list)

    # Solver metadata
    solver_name: str = "D3Q27"
    grid_resolution: Tuple[int, int, int] = (32, 32, 32)
    num_steps: int = 1000
    converged: bool = False
    convergence_score: float = 0.0
    force_stability: float = 1.0

    # Label metadata
    tier: LabelTier = LabelTier.LBM_RAW
    source: str = "internal"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"

@dataclass
class CFDConfig:
    """FluidX3D simulation parameters with adaptive mesh refinement"""
    solver_type: str = "D3Q27"
    base_grid_resolution: int = 32  # Consistent grid resolution - no resizing needed
    mach_number: float = 0.025
    reynolds_number: float = 1e6
    simulation_steps: int = 1000
    output_interval: int = 50
    device_id: int = 0
    # Adaptive mesh refinement
    use_amr: bool = False
    adaptive_cells_target: int = int(5e3)  # Target ~5k cells for AMR
    refinement_levels: int = 3
    # Intelligent Sampling (Issue #12)
    validation_probability: float = 0.0 # Prob of running external PDE validation (default 0.0)
    # LBM configuration
    lbm_config: LBMPhysicsConfig = None   # LBM parameters
    # Backwards compatibility parameter - default to base_grid_resolution
    resolution: int = None  # If provided, sets base_grid_resolution

    def __post_init__(self):
        # Set default resolution if not provided
        if self.resolution is None:
            self.resolution = self.base_grid_resolution

        # Handle backwards compatibility for resolution parameter
        if self.resolution is not None:
            self.base_grid_resolution = self.resolution

        if self.lbm_config is None:
            # Calculate physical grid spacing: physical_length / resolution
            self.lbm_config = LBMPhysicsConfig()
            self.lbm_config.grid_spacing = self.lbm_config.physical_length_scale / self.base_grid_resolution

OPENFOAM_ROOT = Path(os.environ.get("OPENFOAM_ROOT", "/home/darsh/.openclaw/openfoam/usr/share/openfoam"))
OPENFOAM_BIN = OPENFOAM_ROOT / "bin"
OPENFOAM_AVAILABLE = all((OPENFOAM_BIN / cmd).exists() for cmd in ("blockMesh", "snappyHexMesh", "simpleFoam"))

@dataclass
class MissionProfile:
    """Rich mission profile for conditioned aircraft design (Issue #14)"""
    aircraft_class: str = "uav"  # uav, fast_uav, light_aircraft, airliner, fighter, glider
    payload_kg: float = 10.0
    range_km: float = 100.0
    endurance_hr: float = 2.0
    cruise_speed_mps: float = 30.0
    cruise_altitude_m: float = 1000.0
    max_takeoff_weight_kg: float = 50.0
    stall_speed_mps: float = 15.0
    propulsion_type: str = "electric"  # electric, turboprop, jet, none
    manufacturing_method: str = "3d_print"  # 3d_print, composite, metal_sheet
    max_span_m: float = 2.0
    max_length_m: float = 1.5
    max_height_m: float = 0.5

    # Issue #15 controls
    force_external_validation: bool = False

    def __post_init__(self):
        # Validation
        valid_classes = ["uav", "fast_uav", "light_aircraft", "airliner", "fighter", "glider"]
        if self.aircraft_class not in valid_classes:
            raise ValueError(f"Invalid aircraft_class: {self.aircraft_class}")
        if self.propulsion_type not in ["electric", "turboprop", "jet", "none"]:
            raise ValueError(f"Invalid propulsion_type: {self.propulsion_type}")
        if self.manufacturing_method not in ["3d_print", "composite", "metal_sheet"]:
            raise ValueError(f"Invalid manufacturing_method: {self.manufacturing_method}")

        # Positivity checks
        for field in ["payload_kg", "range_km", "endurance_hr", "cruise_speed_mps", "cruise_altitude_m",
                      "max_takeoff_weight_kg", "stall_speed_mps", "max_span_m", "max_length_m", "max_height_m"]:
            val = getattr(self, field)
            if val <= 0:
                raise ValueError(f"{field} must be positive, got {val}")

        # Physical constraints
        if self.stall_speed_mps >= self.cruise_speed_mps:
            raise ValueError(f"stall_speed ({self.stall_speed_mps}) must be less than cruise_speed ({self.cruise_speed_mps})")

@dataclass
class DesignSpec:
    """Aircraft design specification (Deprecated adapter)"""
    target_speed: float = 7.0  # m/s
    space_weight: float = 0.33*100
    drag_weight: float = 0.33*100
    lift_weight: float = 0.34*100
    bounding_box: Tuple[int, int, int] = (64, 64, 64)
    vital_components: Optional[List] = None

    def to_mission_profile(self) -> MissionProfile:
        """Lossy legacy conversion (Deprecated)"""
        # Mapping 1 voxel to 0.1m for reasonable scale fallback
        cruise = max(1.0, float(self.target_speed))
        return MissionProfile(
            cruise_speed_mps=cruise,
            stall_speed_mps=max(0.1, min(0.5 * cruise, cruise - 1e-3)),
            max_span_m=float(self.bounding_box[0]) * 0.1,
            max_length_m=float(self.bounding_box[1]) * 0.1,
            max_height_m=float(self.bounding_box[2]) * 0.1
        )
