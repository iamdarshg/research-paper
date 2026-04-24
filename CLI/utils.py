
import torch
import trimesh
import numpy as np
from mesh_utils import normalize_stl_mesh

def get_vram_limit_resolution(max_usage=0.9, target_res=1024):
    """Estimate max LBM resolution based on available VRAM"""
    if not torch.cuda.is_available():
        return 64

    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory
        usable_vram = total_vram * max_usage

        # D3Q27 memory: ~250 bytes per cell (populations + overhead)
        bytes_per_cell = 250
        max_cells = usable_vram / bytes_per_cell
        max_res = int(max_cells ** (1/3))

        return min(max(max_res, 32), target_res)
    except Exception:
        return 128

def get_stl_adaptive_resolution(stl_path):
    """Determine optimal resolution based on STL complexity"""
    try:
        mesh = trimesh.load(stl_path)
        complexity_res = int((len(mesh.faces) / 2) ** (1/3) * 5)
        vram_res = get_vram_limit_resolution()
        return min(max(complexity_res, 64), vram_res, 512)
    except Exception:
        return get_vram_limit_resolution()
