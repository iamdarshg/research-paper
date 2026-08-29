import os
import sys

import numpy as np
import pytest
import torch
from scipy.ndimage import distance_transform_edt


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)

import sdf_utils
from sdf_utils import _edt_workspace, compute_sdf, prepare_edt_workspace


def test_reusable_edt_workspace_preserves_exact_signed_distance():
    geometry = torch.zeros((7, 9, 11), dtype=torch.float32)
    geometry[2:5, 3:7, 4:8] = 1.0
    mask = geometry.numpy() > 0.5
    expected = distance_transform_edt(~mask) - distance_transform_edt(mask)

    first = compute_sdf(geometry)
    workspace_identity = id(_edt_workspace(geometry.shape)[0])
    second = compute_sdf(geometry)

    assert np.allclose(first.numpy(), expected.astype(np.float32))
    assert torch.equal(second, first)
    assert id(_edt_workspace(geometry.shape)[0]) == workspace_identity


def test_workspace_can_be_reserved_before_solver_use():
    shape = (5, 6, 7)

    prepare_edt_workspace(shape)

    workspace = _edt_workspace(shape)
    assert len(workspace) == 3
    assert workspace[0].shape == tuple(shape)
    assert workspace[1].shape == tuple(shape)
    assert workspace[2].shape == (3,) + tuple(shape)


def test_gpu_exact_backend_fails_closed_when_cupy_is_unavailable(monkeypatch):
    monkeypatch.setattr(sdf_utils, "_cupy_available", lambda: False)
    with pytest.raises(RuntimeError, match="gpu_exact"):
        compute_sdf(torch.zeros((8, 8, 8)), backend="gpu_exact")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sdf_utils._cupy_available(),
    reason="CUDA and CuPy are required",
)
def test_gpu_exact_sdf_matches_scipy_reference():
    geometry = torch.zeros((17, 19, 23), dtype=torch.float32, device="cuda")
    geometry[3:14, 5:16, 7:20] = 1.0
    geometry[8:10, 2:18, 10:13] = 1.0

    reference = compute_sdf(geometry, backend="scipy_reference")
    actual = compute_sdf(geometry, backend="gpu_exact")

    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sdf_utils._cupy_available(),
    reason="CUDA and CuPy are required",
)
def test_gpu_exact_sdf_matches_scipy_for_random_thin_geometry():
    generator = torch.Generator(device="cpu").manual_seed(20260829)
    geometry = (torch.rand((19, 17, 21), generator=generator) > 0.96).float()
    geometry[9, 4:14, 5:17] = 1.0
    geometry[5:15, 8, 10] = 1.0
    geometry = geometry.cuda()

    reference = compute_sdf(geometry, backend="scipy_reference")
    actual = compute_sdf(geometry, backend="gpu_exact")

    torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=1e-5, atol=1e-5)
