
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import label
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
from datetime import datetime
from config import DesignSpec

class AircraftDesignDataset(Dataset):
    """Synthetic dataset for aircraft structure training with adaptive resolution"""

    def __init__(self, num_samples: int = 10000, grid_size: int = 32, seed: int = None, latent_dim: int = 128, target_grid_size: int = 1024):
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.target_grid_size = target_grid_size

        if seed is None:
            import random
            seed = random.randint(0, 1000000)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.latent_codes = torch.randn(num_samples, latent_dim)
        self.geometries = self._generate_geometries()

    def set_resolution(self, new_grid_size: int):
        """Update dataset resolution for progressive training via interpolation"""
        if new_grid_size == self.grid_size:
            return

        print(f"Dataset: Interpolating geometries from {self.grid_size}^3 to {new_grid_size}^3...")
        new_geometries = []
        for geom in self.geometries:
            geom_reshaped = geom.unsqueeze(0).unsqueeze(0)
            interpolated = F.interpolate(
                geom_reshaped,
                size=(new_grid_size, new_grid_size, new_grid_size),
                mode='trilinear',
                align_corners=False
            )
            new_geometries.append((interpolated.squeeze() > 0.5).float())

        self.geometries = new_geometries
        self.grid_size = new_grid_size

    def _generate_geometries(self) -> List[torch.Tensor]:
        """Generate synthetic aircraft geometries using vectorized ops"""
        geometries = []
        for i in range(self.num_samples):
            geom = torch.zeros(self.grid_size, self.grid_size, self.grid_size)
            scale = self.grid_size / 32.0
            cx, cy, cz = self.grid_size // 2, self.grid_size // 2, self.grid_size // 2

            z_indices, y_indices, x_indices = torch.meshgrid(
                torch.arange(self.grid_size),
                torch.arange(self.grid_size),
                torch.arange(self.grid_size),
                indexing='ij'
            )

            dist_center = torch.sqrt((x_indices - cx)**2 + (z_indices - cz)**2)
            fuselage_mask = (dist_center < 6 * scale) & (y_indices > 10 * scale) & (y_indices < 22 * scale)
            geom[fuselage_mask] = 1.0

            wing_mask = (y_indices > 8 * scale) & (y_indices < 24 * scale) & \
                        ((z_indices < 4 * scale) | (z_indices > self.grid_size - 4 * scale))
            geom[wing_mask] = 1.0

            noise = torch.rand_like(geom)
            geom = (geom + 0.1 * noise > 0.5).float()
            geometries.append(geom)

        return geometries

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'latent': self.latent_codes[idx],
            'geometry': self.geometries[idx],
            'target_speed': torch.tensor(self.grid_size / 32 * 50.0),
            'pinn_ready': torch.tensor(False)
        }

class ConnectivityLoss(nn.Module):
    """Penalize disconnected voxel groups"""

    def __init__(self, penalty: float = 10.0):
        super().__init__()
        self.penalty = penalty

    def forward(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """Compute connectivity penalty for batch of voxel grids"""
        batch_size = voxel_grid.shape[0]
        total_penalty = 0.0

        for b in range(batch_size):
            geom = (voxel_grid[b] > 0.5).int().cpu().numpy()
            labeled, num_components = label(geom)
            if num_components > 1:
                component_sizes = np.bincount(labeled.flatten())
                largest_size = component_sizes[1:].max() if num_components > 1 else 0
                total_size = geom.sum()
                if largest_size > 0:
                    disconnected_fraction = (total_size - largest_size) / (total_size + 1e-6)
                    total_penalty += disconnected_fraction

        result = self.penalty * total_penalty / batch_size if batch_size > 0 else 0.0
        return torch.tensor(result, device=voxel_grid.device, dtype=torch.float32)

class GroundTruthExporter:
    """Exporter for PINN-ready ground truth datasets with field-level data"""

    def __init__(self, output_dir: str = "./ground_truth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sample(self, sample_id: str, geometry: torch.Tensor,
                      velocity_fields: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                      pressure_field: torch.Tensor,
                      metadata: Dict[str, Any]):
        """Export a single simulation sample as ground truth for PINNs"""
        sample_path = self.output_dir / f"sample_{sample_id}"
        sample_path.mkdir(exist_ok=True)

        np.save(sample_path / "geometry.npy", geometry.cpu().numpy())
        ux, uy, uz = velocity_fields
        np.save(sample_path / "velocity_x.npy", ux.cpu().numpy())
        np.save(sample_path / "velocity_y.npy", uy.cpu().numpy())
        np.save(sample_path / "velocity_z.npy", uz.cpu().numpy())
        np.save(sample_path / "pressure.npy", pressure_field.cpu().numpy())

        serializable_meta = {k: v for k, v in metadata.items() if isinstance(v, (int, float, str, bool, list, dict))}
        # Add nondimensionalization metadata
        serializable_meta['units'] = {
            'length': 'm',
            'velocity': 'm/s',
            'pressure': 'Pa',
            'force': 'N',
            'density': 'kg/m^3'
        }
        serializable_meta['manifest_version'] = '1.0'
        serializable_meta['pde_target'] = 'Navier-Stokes (Incompressible/Low-Mach)'

        with open(sample_path / "metadata.json", "w") as f:
            json.dump(serializable_meta, f, indent=2)

        # Create explicit manifest for PINN ingestion
        manifest = {
            'sample_id': sample_id,
            'files': {
                'geometry': 'geometry.npy',
                'ux': 'velocity_x.npy',
                'uy': 'velocity_y.npy',
                'uz': 'velocity_z.npy',
                'p': 'pressure.npy'
            },
            'pinn_ready': metadata.get('pinn_ready', False),
            'label_tier': metadata.get('label_tier', 'lbm_raw'),
            'label_source': metadata.get('label_source', 'lbm_d3q27'),
            'force_stability': metadata.get('force_stability', 1.0),
            'lbm_converged': metadata.get('lbm_converged', False),
            'source': metadata.get('source', metadata.get('label_source', 'lbm_d3q27'))
        }
        with open(sample_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Exported ground truth sample to {sample_path}")

class AerodynamicLoss(nn.Module):
    """Loss based on aerodynamic properties using advanced CFD"""

    def __init__(self):
        super().__init__()

    def forward(self, voxel_grid: torch.Tensor, design_spec: DesignSpec, cfd_simulator: Any,
                gt_exporter: GroundTruthExporter = None, sample_prefix: str = "train") -> torch.Tensor:
        batch_size = voxel_grid.shape[0]
        loss = torch.tensor(0.0, device=voxel_grid.device)

        for b in range(batch_size):
            single_voxel_grid = voxel_grid[b]
            geometry = (single_voxel_grid > 0.5).float()
            cfd_results = cfd_simulator.simulate_aerodynamics(geometry, steps=100)

            if gt_exporter is not None and cfd_results.get('pinn_ready', False):
                gt_exporter.export_sample(
                    f"{sample_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{b}",
                    geometry,
                    cfd_results['velocity_fields'],
                    cfd_results['pressure_field'],
                    cfd_results
                )

            volume = geometry.sum() / np.prod(geometry.shape)
            volume_loss = design_spec.space_weight * volume
            cd = cfd_results.get('drag_coefficient', 0.1)
            drag_loss = design_spec.drag_weight * cd
            cl = abs(cfd_results.get('lift_coefficient', 0.0))
            lift_loss = design_spec.lift_weight * (1.0 - torch.clamp(torch.tensor(cl, device=voxel_grid.device), 0, 1))

            loss += volume_loss + drag_loss + lift_loss

        return loss / batch_size
