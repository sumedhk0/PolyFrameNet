"""Shared pytest configuration.

Every test runs in float64. The correctness gates compare to 1e-10 (rotation
invariance, finite-difference forces); float32 resolves only ~1e-7, so in float32
a real symmetry violation is indistinguishable from rounding.
"""

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def _seed_torch():
    """Reseed torch's global RNG before every test, so a random rotation or
    perturbation that breaks a gate reproduces exactly on rerun."""
    torch.manual_seed(0)


@pytest.fixture
def rng():
    """Seeded numpy Generator. Use this instead of the legacy np.random.* globals."""
    return np.random.default_rng(0)
