"""v2 gate helpers. The symmetry checkers themselves are v1's (frame_potential.gates); they
take any energy function, so they test v2 unchanged."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..model import energy_and_forces
from .batch import collate_v2
from .graph import attach_rows, topology_from_geometry_v2


def make_energy_fn_v2(model, charge=0.0, spin=1.0, topo_hook=None, device="cpu"):
    """numbers, positions, cell -> float energy; topology rebuilt from geometry each call,
    so a relabelled input gets its own variant table."""

    def fn(numbers, positions, cell=None):
        t = topology_from_geometry_v2(numbers, positions, cell)
        if topo_hook is not None:
            t = topo_hook(t)
        b = collate_v2([t], [positions], None if cell is None else [cell],
                       charges=[charge], spins=[spin]).to(device)
        with torch.no_grad():
            return float(model(b)[0])

    return fn


def finite_difference_error_v2(model, numbers, positions, cell=None, n_atoms=6, h=1e-5,
                               rng=None):
    """max |F_autodiff - F_fd| over a few atoms and components (central differences)."""
    rng = rng or np.random.default_rng(0)
    t = topology_from_geometry_v2(numbers, positions, cell)
    cells = None if cell is None else [cell]

    def E(p):
        with torch.no_grad():
            return float(model(collate_v2([t], [p], cells))[0])

    _, F = energy_and_forces(model, collate_v2([t], [positions], cells))
    F = F.detach().numpy()
    worst = 0.0
    for a in rng.choice(len(numbers), size=min(n_atoms, len(numbers)), replace=False):
        for k in range(3):
            p1, p2 = positions.copy(), positions.copy()
            p1[a, k] += h
            p2[a, k] -= h
            worst = max(worst, abs(-(E(p1) - E(p2)) / (2 * h) - F[a, k]))
    return worst


def variant_signatures(numbers, positions):
    """Per block (keyed by its atom set): list of per-variant signatures, each an (n, 4)
    array of the block atoms' (Z, frame coordinates). Clusters only. Compare two
    signature dicts with `signature_mismatch`, which matches rows and variants by optimal
    assignment (sorting by rounded keys mis-pairs near-ties)."""
    t = topology_from_geometry_v2(numbers, positions)
    out = {}
    for k in range(t.n_blocks):
        atoms = [int(a) for a in t.block_atoms[k] if a >= 0]
        X = positions[atoms]
        o = X.mean(0)
        sigs = []
        for r in np.nonzero(t.var_block == k)[0]:
            c, a, b = t.var_cab[r]
            if c < 0:
                Fm = np.zeros((3, 3))
            else:
                e1 = positions[a] - positions[c]
                e1 = e1 / np.linalg.norm(e1)
                v = positions[b] - positions[c]
                e2 = v - (v @ e1) * e1
                e2 = e2 / np.linalg.norm(e2)
                Fm = np.stack([e1, e2, t.var_m[r] * np.cross(e1, e2)], -1)
            sigs.append(np.concatenate([numbers[atoms][:, None], (X - o) @ Fm], 1))
        out[frozenset(atoms)] = sigs
    return out


def _row_mismatch(a, b):
    """Max coordinate error after optimally matching rows (same Z only)."""
    from scipy.optimize import linear_sum_assignment

    cost = np.abs(a[:, None, 1:] - b[None, :, 1:]).max(-1) + 1e6 * (a[:, None, 0] != b[None, :, 0])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max())


def signature_mismatch(sig_a, sig_b) -> float:
    """Worst error over variants after optimally matching variant sets of one block."""
    from scipy.optimize import linear_sum_assignment

    if len(sig_a) != len(sig_b):
        return float("inf")
    cost = np.array([[_row_mismatch(x, y) for y in sig_b] for x in sig_a])
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].max())


def drop_first_variant(topo):
    """Negative control: remove the first variant row of every block with more than one."""
    keep = np.ones(len(topo.var_block), dtype=bool)
    seen = set()
    for r, k in enumerate(topo.var_block.tolist()):
        if k not in seen and (topo.var_block == k).sum() > 1:
            keep[r] = False
        seen.add(k)
    t = replace(topo, var_block=topo.var_block[keep], var_cab=topo.var_cab[keep],
                var_m=topo.var_m[keep])
    return attach_rows(t)
