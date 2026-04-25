
import numpy as np
import trimesh
from skimage import measure

def voxels_to_stl(voxel_grid, output_path, physical_length_scale=1.0, resolution=32, use_marching_cubes=True):
    """Convert voxel grid to STL file using marching cubes with optimizations"""
    voxel_np = voxel_grid.cpu().numpy()
    binary_grid = (voxel_np > 0.5).astype(np.float32)

    if use_marching_cubes:
        try:
            level = (voxel_np.min() + voxel_np.max()) / 2.0
            vertices, faces, _, _ = measure.marching_cubes(binary_grid, level=level, spacing=(1.0, 1.0, 1.0))
            h = physical_length_scale / float(resolution)
            vertices = vertices * h - (physical_length_scale * 0.5) + (0.5 * h)

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if len(faces) > 10000:
                try:
                    mesh = mesh.simplify_quadratic_decimation(face_count=min(5000, len(mesh.faces)//2))
                except Exception: pass
            mesh.export(output_path)
        except Exception as e:
            print(f"Marching cubes failed: {e}. Writing voxel representation instead.")
            _write_voxel_stl(binary_grid, output_path, physical_length_scale, resolution)
    else:
        _write_voxel_stl(binary_grid, output_path, physical_length_scale, resolution)

def _write_voxel_stl(binary_grid, path, physical_length_scale, resolution):
    triangles = []
    h = physical_length_scale / float(resolution)
    for x, y, z in np.argwhere(binary_grid > 0.5):
        vertices = np.array([[x,y,z], [x+1,y,z], [x+1,y+1,z], [x,y+1,z], [x,y,z+1], [x+1,y,z+1], [x+1,y+1,z+1], [x,y+1,z+1]], dtype=np.float32)
        vertices = vertices * h - (physical_length_scale * 0.5) + (0.5 * h)
        faces = [[0,1,2], [0,2,3], [4,6,5], [4,7,6], [0,4,5], [0,5,1], [2,6,7], [2,7,3], [0,3,7], [0,7,4], [1,5,6], [1,6,2]]
        for face in faces:
            triangles.append(vertices[face])
    if triangles:
        mesh = trimesh.Trimesh(vertices=np.array(triangles).reshape(-1, 3), faces=np.arange(len(triangles)*3).reshape(-1, 3))
        mesh.export(path)

def normalize_stl_mesh(mesh, padding: float = 0.1):
    extents = mesh.extents
    scale = (1.0 - 2.0 * padding) / max(extents)
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(scale)
    mesh.apply_translation([0.5, 0.5, 0.5])
    return mesh
