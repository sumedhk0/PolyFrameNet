"""Generic correctness-gate checkers. Each takes an energy function of a raw structure,
so the same checker works for any model (or for a deliberately broken energy, which is
how the checkers themselves are tested)."""

from __future__ import annotations

import numpy as np
import torch

from .batch import collate
from .model import energy_and_forces
from .topology import topology_from_geometry


def make_energy_fn(model, device="cpu"):
    """numbers, positions, cell -> float energy, rebuilding topology from geometry so that
    a relabelled input gets its own (different) slot assignment and chain direction."""

    def fn(numbers, positions, cell=None):
        topo = topology_from_geometry(numbers, positions, cell)
        b = collate([topo], [positions], None if cell is None else [cell]).to(device)
        with torch.no_grad():
            return float(model(b)[0])

    return fn


def rotation_translation_error(energy_fn, numbers, positions, cell, rng, n=3):
    from .builders import random_rotation

    e0 = energy_fn(numbers, positions, cell)
    worst = 0.0
    for _ in range(n):
        Q = random_rotation(rng)
        t = rng.normal(size=3) * 3
        p = positions @ Q.T + t
        c = None if cell is None else cell @ Q.T
        worst = max(worst, abs(energy_fn(numbers, p, c) - e0))
    return worst


def permutation_error(energy_fn, numbers, positions, cell, rng, n=3):
    """Random relabelling of all atoms: changes hydrogen slots, chain storage direction
    and block order all at once."""
    e0 = energy_fn(numbers, positions, cell)
    worst = 0.0
    for _ in range(n):
        perm = rng.permutation(len(numbers))
        worst = max(worst, abs(energy_fn(numbers[perm], positions[perm], cell) - e0))
    return worst


def wrap_error(energy_fn, numbers, positions, cell, rng):
    """Move random atoms by whole lattice vectors: must not change anything."""
    e0 = energy_fn(numbers, positions, cell)
    k = rng.integers(-2, 3, size=(len(numbers), 3))
    return abs(energy_fn(numbers, positions + k @ cell, cell) - e0)


def finite_difference_error(model, numbers, positions, cell=None, n_atoms=6, h=1e-5, rng=None,
                            device="cpu"):
    """max |F_autodiff - F_fd| over a few atoms/components (central differences)."""
    rng = rng or np.random.default_rng(0)
    topo = topology_from_geometry(numbers, positions, cell)

    def E(p):
        b = collate([topo], [p], None if cell is None else [cell]).to(device)
        with torch.no_grad():
            return float(model(b)[0])

    b = collate([topo], [positions], None if cell is None else [cell]).to(device)
    _, F = energy_and_forces(model, b)
    F = F.detach().cpu().numpy()
    worst = 0.0
    for a in rng.choice(len(numbers), size=min(n_atoms, len(numbers)), replace=False):
        for k in range(3):
            p1 = positions.copy()
            p1[a, k] += h
            p2 = positions.copy()
            p2[a, k] -= h
            fd = -(E(p1) - E(p2)) / (2 * h)
            worst = max(worst, abs(fd - F[a, k]))
    return worst
