
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from typing import Dict, Any, List, Optional
from geometry import AircraftPart, TypedAircraftGeometry
from config import MissionProfile

class ConstraintViolation:
    def __init__(self, type: str, severity: str, message: str, details: Dict[str, Any] = None):
        self.type = type
        self.severity = severity # 'minor', 'major', 'critical'
        self.message = message
        self.details = details or {}

    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "details": self.details
        }

class ConstraintReport:
    """Accumulates violations across all design and simulation stages (Issue #16)."""
    def __init__(self):
        self.violations: List[ConstraintViolation] = []
        self.repaired = False
        self.metrics = {}
        self.export_status = "pending" # pending, success, repaired, rejected

    def add_violation(self, type: str, severity: str, message: str, details: Dict[str, Any] = None):
        self.violations.append(ConstraintViolation(type, severity, message, details))

    def mark_repaired(self):
        self.repaired = True

    def is_valid(self):
        return not any(v.severity == 'critical' for v in self.violations)

    def to_dict(self):
        return {
            "valid": self.is_valid(),
            "repaired": self.repaired,
            "export_status": self.export_status,
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics
        }

class ConstraintProjector:
    """Deterministic aircraft repair and projection module (Issue #16)."""

    def __init__(self, resolution: int, device: torch.device = 'cpu', existing_report: ConstraintReport = None):
        self.res = resolution
        self.device = device
        self.report = existing_report or ConstraintReport()

    def project(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Run all projection and repair steps sequentially."""

        # 1. Component Cleanup (Remove disconnected noise)
        geometry = self._cleanup_components(geometry)

        # 2. Symmetry Enforcement
        geometry = self._enforce_symmetry(geometry)

        # 3. Bounding Box & Span Constraints
        geometry = self._enforce_bounding_box(geometry, mission)

        # 4. Volume Reservations (Keep-out zones)
        geometry = self._reserve_volumes(geometry, mission)

        # 5. Manufacturing Specific Checks
        geometry = self._check_manufacturing(geometry, mission)

        # 6. Propulsion Constraints
        geometry = self._check_propulsion(geometry, mission)

        # 7. Shell & Spar Thickening
        geometry = self._thicken_structures(geometry)

        # 8. Load Path Verification (Repair connectivity)
        geometry = self._repair_load_paths(geometry)

        # 9. Internal Layout (Ribs/Bulkheads)
        geometry = self._generate_internal_layout(geometry)

        # Final pass: Ensure symmetry after all repairs
        geometry = self._enforce_symmetry(geometry)

        # Update metrics in report
        combined = geometry.get_combined_occupancy()
        self.report.metrics.update({
            "total_voxels": int(torch.sum(combined).item()),
            "occupancy_ratio": float(torch.mean(combined).item()),
            "parts_breakdown": geometry.to_json_metadata()["parts"]
        })

        return geometry

    def check_feasibility(self, geometry: TypedAircraftGeometry, cfd_results: Dict[str, Any], mission: MissionProfile) -> Dict[str, Any]:
        """Perform high-level physics feasibility checks (Issue #16)."""
        phys = geometry.estimate_physical_properties(scale_m=mission.max_span_m)
        weight_n = phys['weight_n']
        lift_n = cfd_results.get('force_z', 0.0)
        drag_n = cfd_results.get('force_x', 0.1)

        # 1. Lift vs Weight Feasibility
        lift_ratio = lift_n / (weight_n + 1e-6)
        if lift_ratio < 1.0:
            self.report.add_violation("insufficient_lift", "major", f"Generated lift ({lift_n:.2f}N) is less than weight ({weight_n:.2f}N).", {"ratio": lift_ratio})

        # 2. Thrust vs Drag (Assuming max thrust based on propulsion type)
        # UAV: 2x weight, Jet: 0.8x weight, etc.
        thrust_factor = 2.0 if mission.propulsion_type == "electric" else 1.2
        max_thrust_n = weight_n * thrust_factor
        thrust_margin = (max_thrust_n - drag_n) / (max_thrust_n + 1e-6)
        if thrust_margin < 0.2:
            self.report.add_violation("excessive_drag", "major", f"Cruise drag ({drag_n:.2f}N) consumes too much thrust margin ({thrust_margin:.2%}).", {"margin": thrust_margin})

        # 3. Structural Strength: Wing Root Bending
        wing_lift = lift_n / 2.0
        moment_arm = mission.max_span_m / 4.0
        root_moment = wing_lift * moment_arm

        res = geometry.res
        mid_y = res // 2
        spar_root_voxels = torch.sum(geometry.get_part_mask(AircraftPart.SPAR)[:, mid_y, :]).item()

        # Allowable moment proxy: 100 Nm per root voxel at 32 res (Carbon composite)
        material_strength = 150.0 if mission.manufacturing_method == "composite" else 100.0
        allowable_moment = spar_root_voxels * material_strength

        if root_moment > allowable_moment:
            self.report.add_violation("spar_overstress", "critical", f"Wing root moment ({root_moment:.1f} Nm) exceeds spar capacity ({allowable_moment:.1f} Nm).", {"moment_n_m": root_moment, "capacity": allowable_moment})

        return {
            "weight_n": weight_n,
            "lift_ratio": lift_ratio,
            "thrust_margin": thrust_margin,
            "root_moment_n_m": root_moment,
            "spar_capacity": allowable_moment
        }

    def _cleanup_components(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Remove tiny disconnected components for each part."""
        new_tensor = geometry.tensor.clone()
        for part in AircraftPart:
            if part == AircraftPart.VOID: continue
            mask = geometry.get_part_mask(part).cpu().numpy()
            binary = (mask > 0.5).astype(np.int32)
            labeled, num = label(binary)
            if num > 1:
                sizes = np.bincount(labeled.flatten())
                largest_label = sizes[1:].argmax() + 1
                new_mask = (labeled == largest_label).astype(np.float32)
                new_tensor[part] = torch.from_numpy(new_mask).to(self.device)
                if sizes[1:].sum() - sizes[largest_label] > 0:
                    self.report.add_violation("fragmented_part", "minor", f"Removed disconnected fragments from {part.name}.", {"part": part.name})
                    self.report.mark_repaired()
        geometry.tensor = new_tensor
        return geometry

    def _enforce_symmetry(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Enforce bilateral symmetry along Y-axis."""
        new_tensor = geometry.tensor.clone()
        for part in AircraftPart:
            if part == AircraftPart.VOID: continue
            mask = geometry.get_part_mask(part)
            flipped = torch.flip(mask, [1])
            sym = torch.maximum(mask, flipped)
            new_tensor[part] = sym
        geometry.tensor = new_tensor
        return geometry

    def _enforce_bounding_box(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Crop geometry to mission-specified bounding box."""
        res = self.res
        new_tensor = geometry.tensor.clone()
        margin = int(res * 0.05)
        for part in AircraftPart:
            if part == AircraftPart.VOID: continue
            mask = geometry.get_part_mask(part)
            mask[:margin, :, :] = 0
            mask[-margin:, :, :] = 0
            mask[:, :margin, :] = 0
            mask[:, -margin:, :] = 0
            mask[:, :, :margin] = 0
            mask[:, :, -margin:] = 0
            new_tensor[part] = mask
        geometry.tensor = new_tensor
        return geometry

    def _reserve_volumes(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Reserve volume for payload, internals, and batteries."""
        res = geometry.res
        z_idx, y_idx, x_idx = torch.meshgrid(torch.arange(res, device=self.device), torch.arange(res, device=self.device), torch.arange(res, device=self.device), indexing='ij')

        # 1. Payload volume
        cx, cy, cz = res // 2, res // 2, res // 2
        r_payload = res // 8 if mission.aircraft_class == "uav" else res // 4
        dist_payload = torch.sqrt((x_idx - cx)**2 + (y_idx - cy)**2 + (z_idx - cz)**2)
        payload_mask = (dist_payload < r_payload).float()
        geometry.set_part_mask(AircraftPart.PAYLOAD, payload_mask)

        # 2. Battery / Fuel volume
        r_energy = r_payload * 0.8
        dist_energy = torch.sqrt((x_idx - cx)**2 + (y_idx - cy)**2 + (z_idx - (cz - r_payload))**2)
        energy_mask = (dist_energy < r_energy).float()
        part_energy = AircraftPart.BATTERY if mission.propulsion_type == "electric" else AircraftPart.FUEL
        geometry.set_part_mask(part_energy, energy_mask)

        # 3. Keep-out zone for propulsion (Front clear)
        prop_keepout = (x_idx < int(res * 0.15)) & (torch.sqrt((y_idx - cy)**2 + (z_idx - cz)**2) < res // 3)
        geometry.set_part_mask(AircraftPart.KEEP_OUT, prop_keepout.float())

        # Clear other parts from payload and energy volumes
        reserved = torch.maximum(payload_mask, energy_mask)
        for part in AircraftPart:
            if part not in (AircraftPart.VOID, AircraftPart.PAYLOAD, AircraftPart.BATTERY, AircraftPart.FUEL, AircraftPart.KEEP_OUT):
                geometry.set_part_mask(part, geometry.get_part_mask(part) * (1.0 - reserved))

        return geometry

    def _check_manufacturing(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Apply manufacturing constraints based on the specific method."""
        method = mission.manufacturing_method
        skin = geometry.get_part_mask(AircraftPart.SKIN).unsqueeze(0).unsqueeze(0)

        if method == '3d_print':
            # Min wall thickness via dilation
            thickened = F.max_pool3d(skin, kernel_size=3, stride=1, padding=1)
            geometry.set_part_mask(AircraftPart.SKIN, thickened.squeeze())
            self.report.add_violation("min_wall_thickness", "minor", "Skin thickened for 3D printing requirements.")
            self.report.mark_repaired()

        elif method == 'composite':
            # Composites require larger radii (no sharp concave corners)
            # We can use a larger kernel for smoothing/dilation
            thickened = F.max_pool3d(skin, kernel_size=5, stride=1, padding=2)
            geometry.set_part_mask(AircraftPart.SKIN, thickened.squeeze())
            self.report.add_violation("composite_radius", "minor", "Applied composite curvature smoothing to skin.")
            self.report.mark_repaired()

        elif method == 'metal_sheet':
            # Metal sheets must have constant thickness (shell-like)
            # For now, we ensure skin exists and is not too thick/solid
            # Repair: enforce single-layer shell via erode/dilate
            eroded = -F.max_pool3d(-skin, kernel_size=3, stride=1, padding=1)
            shell = skin - eroded
            geometry.set_part_mask(AircraftPart.SKIN, shell.squeeze())
            self.report.add_violation("metal_sheet_shell", "minor", "Enforced constant-thickness shell for metal sheet manufacturing.")
            self.report.mark_repaired()

        return geometry

    def _check_propulsion(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Ensure clearance and mounts for propulsion systems."""
        res = geometry.res
        # Front center motor mount
        cx, cy, cz = int(res * 0.1), res // 2, res // 2
        r_prop = res // 3

        z_idx, y_idx, x_idx = torch.meshgrid(torch.arange(res, device=self.device), torch.arange(res, device=self.device), torch.arange(res, device=self.device), indexing='ij')
        dist_yz = torch.sqrt((y_idx - cy)**2 + (z_idx - cz)**2)
        prop_disk = (torch.abs(x_idx - cx) < 3) & (dist_yz < r_prop)
        prop_mask = prop_disk.float()

        # Check clearance
        skin = geometry.get_part_mask(AircraftPart.SKIN)
        if torch.any(skin * prop_mask > 0.5):
            self.report.add_violation("prop_clearance", "major", "Propeller disk intersects with airframe skin.")
            # Repair: clear prop disk
            geometry.set_part_mask(AircraftPart.SKIN, skin * (1.0 - prop_mask))
            self.report.mark_repaired()

        # Ensure hardpoint/mount exists at motor location
        mount_loc = (x_idx == int(res * 0.12)) & (dist_yz < res // 10)
        mount_mask = mount_loc.float()
        geometry.set_part_mask(AircraftPart.HARDPOINT, torch.maximum(geometry.get_part_mask(AircraftPart.HARDPOINT), mount_mask))

        return geometry

    def _thicken_structures(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Ensure load-bearing members meet minimum thickness."""
        for part in [AircraftPart.SPAR, AircraftPart.BULKHEAD, AircraftPart.HARDPOINT]:
            mask = geometry.get_part_mask(part).unsqueeze(0).unsqueeze(0)
            thick = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
            geometry.set_part_mask(part, thick.squeeze())
        return geometry

    def _repair_load_paths(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Verify and repair structural continuity."""
        spar = geometry.get_part_mask(AircraftPart.SPAR)
        fuselage = geometry.get_part_mask(AircraftPart.FUSELAGE)
        wing = geometry.get_part_mask(AircraftPart.WING)
        res = geometry.res
        mid_y = res // 2
        mid_z = res // 2

        # Check if spar intersects fuselage
        mid_x = res // 2
        if torch.sum(spar * (fuselage > 0.5)) < 1.0:
            self.report.add_violation("spar_continuity", "critical", "Main wing spar is not connected to the fuselage load path.")
            # Repair: Create a real bridge through the fuselage
            bridge = torch.zeros_like(spar)
            # Create a centered longitudinal bridge that spans across Y
            bridge[mid_z-1:mid_z+2, :, mid_x-1:mid_x+2] = 1.0

            # Ensure the bridge is constrained to fuselage interior or spans the wing root
            geometry.set_part_mask(AircraftPart.SPAR, torch.maximum(spar, bridge * (fuselage + wing)))
            self.report.mark_repaired()

        return geometry

    def _generate_internal_layout(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Generate ribs and bulkheads for structural integrity."""
        res = geometry.res
        fuselage = geometry.get_part_mask(AircraftPart.FUSELAGE)
        wing = geometry.get_part_mask(AircraftPart.WING)

        # Bulkheads every N voxels in X
        bulkhead_spacing = res // 4
        bulkheads = torch.zeros_like(fuselage)
        for x in range(0, res, bulkhead_spacing):
            bulkheads[:, :, x] = 1.0
        geometry.set_part_mask(AircraftPart.BULKHEAD, torch.maximum(geometry.get_part_mask(AircraftPart.BULKHEAD), bulkheads * fuselage))

        # Ribs every M voxels in Y
        rib_spacing = res // 6
        ribs = torch.zeros_like(wing)
        for y in range(0, res, rib_spacing):
            ribs[:, y, :] = 1.0
        geometry.set_part_mask(AircraftPart.RIB, torch.maximum(geometry.get_part_mask(AircraftPart.RIB), ribs * wing))

        return geometry

    def get_report(self, geometry: TypedAircraftGeometry = None) -> Dict[str, Any]:
        """Return the accumulated violation report."""
        if geometry:
            combined = geometry.get_combined_occupancy()
            self.report.metrics.update({
                "total_voxels": int(torch.sum(combined).item()),
                "occupancy_ratio": float(torch.mean(combined).item()),
                "parts_breakdown": geometry.to_json_metadata()["parts"]
            })
        return self.report.to_dict()
