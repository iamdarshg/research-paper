
import numpy as np
import trimesh
from skimage import measure
from typing import Optional

def voxels_to_stl(voxel_grid, output_path, physical_length_scale=1.0, resolution=32, use_marching_cubes=True):
    """Legacy wrapper for voxels_to_stl_checked."""
    return voxels_to_stl_checked(voxel_grid, output_path, physical_length_scale, resolution, use_marching_cubes)

def voxels_to_stl_checked(voxel_grid, output_path, physical_length_scale=1.0, resolution=32, use_marching_cubes=True, report=None):
    """Convert voxel grid to STL with watertight checks and repair reporting (Issue #16)."""
    voxel_np = voxel_grid.cpu().numpy()
    binary_grid = (voxel_np > 0.5).astype(np.float32)

    mesh = None
    if use_marching_cubes:
        try:
            # Ensure grid is padded to avoid non-watertight holes at boundaries
            padded_grid = np.pad(binary_grid, 1, mode='constant', constant_values=0)
            level = 0.5
            vertices, faces, _, _ = measure.marching_cubes(padded_grid, level=level, spacing=(1.0, 1.0, 1.0))

            # Undo padding shift
            vertices -= 1.0

            h = physical_length_scale / float(resolution)
            vertices = vertices * h - (physical_length_scale * 0.5) + (0.5 * h)

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if len(faces) > 10000:
                try:
                    mesh = mesh.simplify_quadratic_decimation(face_count=min(5000, len(mesh.faces)//2))
                except Exception: pass
        except Exception as e:
            print(f"Marching cubes failed: {e}. Writing voxel representation instead.")
            mesh = _get_voxel_mesh(binary_grid, physical_length_scale, resolution)
    else:
        mesh = _get_voxel_mesh(binary_grid, physical_length_scale, resolution)

    if mesh is not None and len(mesh.faces) > 0:
        # Check watertightness
        is_watertight = mesh.is_watertight
        if not is_watertight:
            if report:
                report.add_violation("non_watertight_mesh", "major", "Generated STL mesh is not watertight. Attempting repair.")

            # Attempt repair
            mesh.fill_holes()
            if mesh.is_watertight:
                if report:
                    report.mark_repaired()
                    report.export_status = "repaired"
            else:
                if report:
                    report.add_violation("repair_failed", "critical", "Could not repair mesh watertightness.")
                    report.export_status = "rejected"
        else:
            if report:
                report.export_status = "success"

        mesh.export(output_path)
        return True
    return False

def _get_voxel_mesh(binary_grid, physical_length_scale, resolution):
    triangles = []
    h = physical_length_scale / float(resolution)
    for x, y, z in np.argwhere(binary_grid > 0.5):
        vertices = np.array([[x,y,z], [x+1,y,z], [x+1,y+1,z], [x,y+1,z], [x,y,z+1], [x+1,y,z+1], [x+1,y+1,z+1], [x,y+1,z+1]], dtype=np.float32)
        vertices = vertices * h - (physical_length_scale * 0.5) + (0.5 * h)
        faces = [[0,1,2], [0,2,3], [4,6,5], [4,7,6], [0,4,5], [0,5,1], [2,6,7], [2,7,3], [0,3,7], [0,7,4], [1,5,6], [1,6,2]]
        for face in faces:
            triangles.append(vertices[face])
    if triangles:
        return trimesh.Trimesh(vertices=np.array(triangles).reshape(-1, 3), faces=np.arange(len(triangles)*3).reshape(-1, 3))
    return None

def normalize_stl_mesh(mesh, padding: float = 0.1):
    extents = mesh.extents
    scale = (1.0 - 2.0 * padding) / max(extents)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(scale)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh
