"""Harness smoke tests.

test_default_dtype_is_float64 fails under `pytest --noconftest`. If conftest.py
ever stops loading, the float64 guarantee every later gate depends on breaks
loudly here instead of silently loosening a 1e-10 comparison elsewhere.
"""

import torch

import frame_potential


def test_package_is_regular_not_namespace():
    # A namespace package (no __init__.py) has __file__ = None and silently merges
    # any same-named directory found on sys.path.
    assert frame_potential.__file__ is not None


def test_default_dtype_is_float64():
    assert torch.get_default_dtype() == torch.float64
    assert torch.zeros(1).dtype == torch.float64
