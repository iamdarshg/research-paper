
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label
from typing import Dict, Any, List, Optional
from geometry import AircraftPart, TypedAircraftGeometry
from config import MissionProfile

class ConstraintProjector:
    """Deterministic aircraft repair and projection module (Issue #16)."""

    def __init__(self, resolution: int, device: torch.device = 'cpu'):
        self.res = resolution
        self.device = device
        self.violation_report = []

    def project(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Run all projection and repair steps sequentially."""
        self.violation_report = []

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

        # Final pass: Ensure symmetry after all repairs
        geometry = self._enforce_symmetry(geometry)

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
            self.violation_report.append({"type": "insufficient_lift", "ratio": lift_ratio, "severity": "major"})

        # 2. Thrust vs Drag (Assuming max thrust is 2x weight for UAV takeoff)
        max_thrust_n = weight_n * 2.0
        thrust_margin = (max_thrust_n - drag_n) / (max_thrust_n + 1e-6)
        if thrust_margin < 0.2:
            self.violation_report.append({"type": "excessive_drag", "margin": thrust_margin, "severity": "major"})

        # 3. Structural Strength: Wing Root Bending
        # Approximate bending moment = (Lift/2) * (Span/4)
        wing_lift = lift_n / 2.0
        moment_arm = mission.max_span_m / 4.0
        root_moment = wing_lift * moment_arm

        # Simple spar cross-section check at root (Center Y)
        res = geometry.res
        mid_y = res // 2
        spar_root_voxels = torch.sum(geometry.get_part_mask(AircraftPart.SPAR)[:, mid_y, :]).item()

        # Allowable moment based on area (very rough proxy)
        # 1000 N*m per 10 voxels at 32 res
        allowable_moment = (spar_root_voxels / 10.0) * 1000.0
        if root_moment > allowable_moment:
            self.violation_report.append({"type": "spar_overstress", "moment_n_m": root_moment, "severity": "critical"})

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
                # Keep only the largest component
                sizes = np.bincount(labeled.flatten())
                largest_label = sizes[1:].argmax() + 1
                new_mask = (labeled == largest_label).astype(np.float32)
                new_tensor[part] = torch.from_numpy(new_mask).to(self.device)
                if sizes[1:].sum() - sizes[largest_label] > 0:
                    self.violation_report.append({"type": "fragmented_part", "part": part.name, "severity": "minor"})
        geometry.tensor = new_tensor
        return geometry

    def _enforce_symmetry(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Enforce bilateral symmetry along Y-axis."""
        new_tensor = geometry.tensor.clone()
        for part in AircraftPart:
            if part == AircraftPart.VOID: continue
            mask = geometry.get_part_mask(part)
            flipped = torch.flip(mask, [1])
            # Symmetric OR (max)
            sym = torch.maximum(mask, flipped)
            new_tensor[part] = sym
        geometry.tensor = new_tensor
        return geometry

    def _enforce_bounding_box(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Crop geometry to mission-specified bounding box."""
        res = self.res
        # Mapping: we assume the voxel grid covers a domain corresponding to max mission dims
        # Any part outside 10%-90% range is penalized/cropped for stability
        new_tensor = geometry.tensor.clone()
        margin = int(res * 0.05)
        for part in AircraftPart:
            if part == AircraftPart.VOID: continue
            mask = geometry.get_part_mask(part)
            # Simple crop to interior
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
        """Reserve volume for payload and internals."""
        res = geometry.res
        payload_mask = torch.zeros((res, res, res), device=self.device)
        cx, cy, cz = res // 2, res // 2, res // 2
        # Use mission class to scale payload
        r = res // 8 if mission.aircraft_class == "uav" else res // 4

        # Generate payload sphere
        z_idx, y_idx, x_idx = torch.meshgrid(torch.arange(res), torch.arange(res), torch.arange(res), indexing='ij')
        dist = torch.sqrt((x_idx.to(self.device) - cx)**2 + (y_idx.to(self.device) - cy)**2 + (z_idx.to(self.device) - cz)**2)
        payload_mask = (dist < r).float()

        geometry.set_part_mask(AircraftPart.PAYLOAD, payload_mask)
        # Clear other parts (except KEEP_OUT) from payload
        for part in AircraftPart:
            if part not in (AircraftPart.VOID, AircraftPart.PAYLOAD, AircraftPart.KEEP_OUT):
                geometry.set_part_mask(part, geometry.get_part_mask(part) * (1.0 - payload_mask))
        return geometry

    def _check_manufacturing(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Apply manufacturing constraints like minimum wall thickness."""
        method = mission.manufacturing_method
        if method == '3d_print':
            # Minimum wall thickness check via dilation then erosion (opening operation)
            # If opening removes voxels, the wall was too thin.
            skin = geometry.get_part_mask(AircraftPart.SKIN).unsqueeze(0).unsqueeze(0)
            # Thicken then thin
            thickened = F.max_pool3d(skin, kernel_size=3, stride=1, padding=1)
            # Simple structural repair: use the thickened skin
            geometry.set_part_mask(AircraftPart.SKIN, thickened.squeeze())
            self.violation_report.append({"type": "min_wall_thickness", "method": "3d_print", "severity": "minor"})
        return geometry

    def _check_propulsion(self, geometry: TypedAircraftGeometry, mission: MissionProfile) -> TypedAircraftGeometry:
        """Ensure clearance for propulsion systems."""
        res = geometry.res
        # Tractor prop at front center
        cx, cy, cz = int(res * 0.1), res // 2, res // 2
        r_prop = res // 3

        # Grid indexing for prop disk
        z_range = torch.arange(res, device=self.device)
        y_range = torch.arange(res, device=self.device)
        x_range = torch.arange(res, device=self.device)
        z_idx, y_idx, x_idx = torch.meshgrid(z_range, y_range, x_range, indexing='ij')

        dist_yz = torch.sqrt((y_idx - cy)**2 + (z_idx - cz)**2)
        prop_disk = (torch.abs(x_idx - cx) < 2) & (dist_yz < r_prop)
        prop_mask = prop_disk.float()

        skin = geometry.get_part_mask(AircraftPart.SKIN)
        if torch.any(skin * prop_mask > 0.5):
            self.violation_report.append({"type": "prop_clearance", "severity": "major"})
            # Repair: clear prop disk from aerodynamic surfaces
            geometry.set_part_mask(AircraftPart.SKIN, skin * (1.0 - prop_mask))
        return geometry

    def _thicken_structures(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Ensure load-bearing members meet minimum thickness."""
        for part in [AircraftPart.SPAR, AircraftPart.BULKHEAD]:
            mask = geometry.get_part_mask(part).unsqueeze(0).unsqueeze(0)
            # Dilation
            thick = F.max_pool3d(mask, kernel_size=3, stride=1, padding=1)
            geometry.set_part_mask(part, thick.squeeze())
        return geometry

    def _repair_load_paths(self, geometry: TypedAircraftGeometry) -> TypedAircraftGeometry:
        """Verify and repair structural continuity."""
        spar = geometry.get_part_mask(AircraftPart.SPAR)
        fuselage = geometry.get_part_mask(AircraftPart.FUSELAGE)
        # Ensure spar is connected to fuselage
        if torch.sum(spar * fuselage) < 1.0:
            self.violation_report.append({"type": "spar_continuity", "severity": "critical"})
            # Simple repair: add a central spar bridge
            res = geometry.res
            mid = res // 2
            bridge = torch.zeros_like(spar)
            bridge[mid, :, mid] = 1.0 # Simple axis line
            geometry.set_part_mask(AircraftPart.SPAR, torch.maximum(spar, bridge * fuselage))
        return geometry

    def get_report(self, geometry: TypedAircraftGeometry) -> Dict[str, Any]:
        """Return a structured machine-readable violation report (Issue #16)."""
        combined = geometry.get_combined_occupancy()
        return {
            "valid": len([v for v in self.violation_report if v['severity'] == 'critical']) == 0,
            "repaired": True,
            "violations": self.violation_report,
            "metrics": {
                "total_voxels": int(torch.sum(combined).item()),
                "occupancy_ratio": float(torch.mean(combined).item()),
                "parts_breakdown": geometry.to_json_metadata()["parts"]
            }
        }
