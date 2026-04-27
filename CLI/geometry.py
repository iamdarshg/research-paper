
import torch
import numpy as np
from enum import IntEnum

class AircraftPart(IntEnum):
    VOID = 0
    SKIN = 1
    FUSELAGE = 2
    WING = 3
    TAIL = 4
    NACELLE = 5
    SPAR = 6
    RIB = 7
    BULKHEAD = 8
    KEEP_OUT = 9
    PAYLOAD = 10
    BATTERY = 11
    FUEL = 12
    LANDING_GEAR = 13
    HARDPOINT = 14

class TypedAircraftGeometry:
    """Typed aircraft geometry representation (Issue #16).

    Encapsulates a multi-channel voxel tensor where each channel corresponds
    to a semantic aircraft part.
    """
    def __init__(self, resolution, device='cpu'):
        self.res = resolution
        self.device = device
        self.num_parts = len(AircraftPart)
        # Channel 0 is VOID (inverted mask of all other parts)
        self.tensor = torch.zeros((self.num_parts, resolution, resolution, resolution), device=device)

    @classmethod
    def from_tensor(cls, tensor):
        """Create from an existing [C, D, H, W] tensor."""
        res = tensor.shape[1]
        obj = cls(res, device=tensor.device)
        obj.tensor = tensor
        return obj

    def get_part_mask(self, part: AircraftPart):
        return self.tensor[part]

    def set_part_mask(self, part: AircraftPart, mask):
        # Allow passing [D, H, W] mask
        if mask.dim() == 3:
            self.tensor[part] = mask.to(self.device).float()
        else:
            self.tensor[part] = mask.squeeze().to(self.device).float()

    def get_combined_occupancy(self):
        """Return a single binary grid of all solid parts (excluding VOID and KEEP_OUT)."""
        solid_parts = [p for p in AircraftPart if p not in (AircraftPart.VOID, AircraftPart.KEEP_OUT)]
        return (torch.sum(self.tensor[solid_parts], dim=0) > 0.5).float()

    def get_skin_surface(self):
        return self.tensor[AircraftPart.SKIN]

    def estimate_physical_properties(self, scale_m: float = 1.0):
        """Estimate mass and CoM based on voxel volume and material density (Issue #16)."""
        res = self.res
        voxel_volume = (scale_m / res)**3

        # Approximate densities (kg/m^3)
        densities = {
            AircraftPart.SKIN: 1500,     # Carbon composite skin
            AircraftPart.FUSELAGE: 50,    # Lightweight internal foam/air
            AircraftPart.WING: 100,
            AircraftPart.SPAR: 2700,      # Aluminum spar
            AircraftPart.PAYLOAD: 500,
            AircraftPart.BATTERY: 2000,
        }

        total_mass = 0.0
        com = torch.zeros(3, device=self.device)

        z_idx, y_idx, x_idx = torch.meshgrid(torch.arange(res), torch.arange(res), torch.arange(res), indexing='ij')
        coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).to(self.device).float()

        for part, rho in densities.items():
            mask = self.get_part_mask(part)
            mass = torch.sum(mask).item() * voxel_volume * rho
            total_mass += mass
            if mass > 1e-6:
                # Weighted average for CoM
                part_com = torch.sum(coords * mask.unsqueeze(-1), dim=(0,1,2)) / (torch.sum(mask) + 1e-8)
                com += part_com * mass

        if total_mass > 1e-6:
            com /= total_mass

        return {
            "total_mass_kg": total_mass,
            "center_of_mass_voxels": com.tolist(),
            "weight_n": total_mass * 9.81
        }

    def to_json_metadata(self):
        return {
            "resolution": self.res,
            "parts": {p.name: int(torch.sum(self.tensor[p] > 0.5).item()) for p in AircraftPart}
        }
