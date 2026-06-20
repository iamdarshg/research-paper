import numpy as np

from CLI.build_airshow_corpus import parse_x3d_indexed_faces, voxelize_mesh


def test_parse_x3d_indexed_faces_triangulates_quad():
    mesh = parse_x3d_indexed_faces(
        """
        <X3D>
          <Scene>
            <Shape>
              <IndexedFaceSet coordIndex="0 1 2 3 -1">
                <Coordinate point="0 0 0 1 0 0 1 1 0 0 1 0" />
              </IndexedFaceSet>
            </Shape>
          </Scene>
        </X3D>
        """
    )

    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (2, 3)


def test_voxelize_mesh_returns_centered_nonempty_grid():
    mesh = parse_x3d_indexed_faces(
        """
        <X3D>
          <Scene>
            <Shape>
              <IndexedFaceSet coordIndex="0 1 2 3 -1 4 7 6 5 -1 0 4 5 1 -1 1 5 6 2 -1 2 6 7 3 -1 3 7 4 0 -1">
                <Coordinate point="0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1" />
              </IndexedFaceSet>
            </Shape>
          </Scene>
        </X3D>
        """
    )

    voxels = voxelize_mesh(mesh, 8)

    assert voxels.shape == (8, 8, 8)
    assert voxels.dtype == np.float32
    assert voxels.sum() > 0
