import builtins
import importlib
import os
import sys

import torch


CLI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "CLI")
if CLI_DIR not in sys.path:
    sys.path.insert(0, CLI_DIR)


def test_tensor_hash_utility_does_not_import_mesh_stack(monkeypatch):
    sys.modules.pop("utils", None)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "trimesh" or name == "mesh_utils":
            raise AssertionError("tensor-only runtime must keep mesh imports lazy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    utils = importlib.import_module("utils")

    digest = utils.compute_tensor_content_hash(torch.zeros((2, 2, 2)))
    assert isinstance(digest, str)
    assert len(digest) == 32
