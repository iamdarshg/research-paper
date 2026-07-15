import os
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

from sdf_utils import _EDT_WORKSPACES, compute_sdf, prepare_edt_workspace


def test_reusable_edt_workspace_preserves_exact_signed_distance():
    geometry = torch.zeros((7, 9, 11), dtype=torch.float32)
    geometry[2:5, 3:7, 4:8] = 1.0
    mask = geometry.numpy() > 0.5
    expected = distance_transform_edt(~mask) - distance_transform_edt(mask)

    first = compute_sdf(geometry)
    workspace_identity = id(_EDT_WORKSPACES[tuple(geometry.shape)][0])
    second = compute_sdf(geometry)

    assert np.allclose(first.numpy(), expected.astype(np.float32))
    assert torch.equal(second, first)
    assert id(_EDT_WORKSPACES[tuple(geometry.shape)][0]) == workspace_identity


def test_workspace_can_be_reserved_before_solver_use():
    shape = (5, 6, 7)

    prepare_edt_workspace(shape)

    assert shape in _EDT_WORKSPACES
