
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import label
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import json
from dataclasses import asdict
from datetime import datetime
from config import DesignSpec, LabelTier, CFDLabel, MissionProfile

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
    """Exporter for reusable CFD labels and PINN ground truth (Issue #15)"""

    def __init__(self, output_dir: str = "./ground_truth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.output_dir / "cfd_labels.json"
        self._cache = self._load_labels()

    def _load_labels(self) -> List[Dict[str, Any]]:
        if self.labels_path.exists():
            try:
                with open(self.labels_path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_labels(self):
        with open(self.labels_path, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def _sanitize_metadata(self, obj: Any) -> Any:
        """Recursively sanitize metadata for JSON serialization, stripping Tensors/Tuples"""
        if isinstance(obj, dict):
            return {k: self._sanitize_metadata(v) for k, v in obj.items()
                    if not k.endswith('_fields') and k not in ('pressure_field', 'velocity_fields')}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_metadata(i) for i in obj if not isinstance(i, (torch.Tensor, np.ndarray)) or i.numel() <= 1]
        elif isinstance(obj, list):
            return [self._sanitize_metadata(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else list(obj.shape)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, LabelTier):
            return obj.value
        elif obj is None:
            return None
        return str(obj)

    def export_sample(self, sample_id: str, geometry: torch.Tensor,
                      velocity_fields: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
                      pressure_field: Optional[torch.Tensor] = None,
                      metadata: Dict[str, Any] = None):
        """Export a simulation record as a reusable CFD label (Issue #15)."""
        metadata = metadata or {}
        sample_path = self.output_dir / f"sample_{sample_id}"
        sample_path.mkdir(exist_ok=True)

        geom_path = sample_path / "geometry.npy"
        np.save(geom_path, geometry.cpu().numpy())

        px_path, vel_paths = None, {}
        if velocity_fields is not None:
            ux, uy, uz = velocity_fields
            for name, field_data in zip(['ux', 'uy', 'uz'], [ux, uy, uz]):
                path = sample_path / f"velocity_{name}.npy"
                np.save(path, field_data.cpu().numpy())
                vel_paths[name] = str(path.relative_to(self.output_dir))

        if pressure_field is not None:
            np.save(sample_path / "pressure.npy", pressure_field.cpu().numpy())
            px_path = str((sample_path / "pressure.npy").relative_to(self.output_dir))

        # Create CFDLabel object from metadata
        res = geometry.shape
        label = CFDLabel(
            geometry_id=sample_id,
            geometry_ref=str(geom_path.relative_to(self.output_dir)),
            mission_profile=metadata.get('mission', {}),
            constraints_profile=metadata.get('constraints', {}),
            cd=metadata.get('drag_coefficient', 0.0),
            cl=metadata.get('lift_coefficient', 0.0),
            cm=metadata.get('moment_coefficient'),
            pressure_field_path=px_path,
            velocity_field_paths=vel_paths,
            solver_name=metadata.get('label_source', 'D3Q27'),
            grid_resolution=(res[0], res[1], res[2]),
            num_steps=metadata.get('num_steps', 0),
            converged=metadata.get('lbm_converged', False),
            convergence_score=metadata.get('convergence_score', 0.0),
            force_stability=metadata.get('force_stability', 1.0),
            tier=LabelTier(metadata.get('label_tier', 'lbm_raw')),
            source=metadata.get('source', 'internal')
        )

        label_dict = self._sanitize_metadata(asdict(label))
        label_dict["tier"] = label.tier.value # Clean string serialization

        # Merge extra metadata for backward compatibility (Issue #15)
        for k, v in metadata.items():
            if k not in label_dict and k not in ('velocity_fields', 'pressure_field') and not k.endswith('_fields'):
                label_dict[k] = self._sanitize_metadata(v)

        # Check for existing label to update/promote (Non-lossy promotion Fix 4)
        updated = False
        for i, existing in enumerate(self._cache):
            if existing['geometry_id'] == sample_id:
                # Multi-fidelity preservation: add previous state to fidelity_history
                if label_dict['tier'] != existing['tier']:
                    history = existing.get('fidelity_history', [])
                    # Snapshot current state into history before update
                    snap = {k: v for k, v in existing.items() if k != 'fidelity_history'}
                    history.append(snap)
                    label_dict['fidelity_history'] = history

                # Promotion logic: only update main record if tier is higher or same
                tier_order = {"lbm_raw": 0, "lbm_calibrated": 1, "external_pde": 2}
                if tier_order[label_dict['tier']] >= tier_order.get(existing['tier'], 0):
                    self._cache[i] = label_dict
                updated = True
                break

        if not updated:
            self._cache.append(label_dict)

        self._save_labels()

        # Backward compatibility metadata.json
        with open(sample_path / "metadata.json", "w") as f:
            json.dump(label_dict, f, indent=2)

        # Explicit manifest for PINN ingestion (Fix 3: include all components)
        manifest = {
            'sample_id': sample_id,
            'files': {
                'geometry': 'geometry.npy',
                'ux': 'velocity_ux.npy' if 'ux' in vel_paths else None,
                'uy': 'velocity_uy.npy' if 'uy' in vel_paths else None,
                'uz': 'velocity_uz.npy' if 'uz' in vel_paths else None,
                'p': 'pressure.npy' if px_path else None
            },
            'pinn_ready': metadata.get('pinn_ready', False),
            'label_tier': label_dict['tier'],
            'label_source': label_dict['solver_name'],
            'force_stability': label_dict['force_stability'],
            'lbm_converged': label_dict['converged'],
            'source': label_dict['source']
        }
        with open(sample_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"✅ Exported CFD label {sample_id} ({label_dict['tier']}) to {sample_path}")

class CFDLabelDataset(Dataset):
    """Dataset for training AeroSurrogate from CFD labels (Issue #15)"""
    def __init__(self, labels_dir: str = "./ground_truth"):
        self.labels_dir = Path(labels_dir)
        self.labels_path = self.labels_dir / "cfd_labels.json"
        if not self.labels_path.exists():
            self.labels = []
        else:
            with open(self.labels_path, 'r') as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        record = self.labels[idx]
        geom_path = self.labels_dir / record['geometry_ref']
        geometry = torch.from_numpy(np.load(geom_path)).float()

        # Extract targets
        targets = {
            'Cd': torch.tensor(record.get('cd', 0.1), dtype=torch.float32),
            'Cl': torch.tensor(record.get('cl', 0.0), dtype=torch.float32),
            'Cm': torch.tensor(record.get('cm', 0.0) if record.get('cm') is not None else 0.0, dtype=torch.float32),
            'convergence_score': torch.tensor(record.get('convergence_score', 0.0), dtype=torch.float32),
            'separation_risk': torch.tensor(record.get('separation_risk', 0.0), dtype=torch.float32)
        }

        # Extract mission profile as collatable dict (Issue #15 Review Feedback)
        mission_dict = record.get('mission_profile', {})
        # Ensure all required fields are present for collate
        default_mission = asdict(MissionProfile())
        full_mission_dict = {**default_mission, **mission_dict}

        return geometry, targets, full_mission_dict

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
