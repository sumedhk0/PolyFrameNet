"""Correctness gates (CLAUDE.md Testing 1-4) on an untrained model, float64.

An untrained model with random weights is the right subject: training cannot hide a
symmetry violation that is structurally absent, and a violation shows up at O(1)."""

import numpy as np
import pytest
import torch

from frame_potential import gates
from frame_potential.batch import (
    collate,
    radius_graph_brute,
    radius_graph_periodic_brute,
    radius_graph_vesin,
)
from frame_potential.builders import pe_chain, pe_cluster, pe_melt
from frame_potential.model import PolyFrameNet
from frame_potential.topology import END, MID, topology_from_geometry


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1)
    m = PolyFrameNet(cutoff=4.5).double()
    # Random non-trivial reference/scale so they are exercised too.
    m.e0.copy_(torch.tensor([-13.6, -1029.0], dtype=torch.float64))
    return m.eval()


@pytest.fixture(scope="module")
def model_noref(model):
    """Same weights, zero per-element offsets. FD differences of a ~-7e4 eV total lose
    ~1e-6 eV/A to float64 rounding (measured: 3e-6 at h=1e-5 with offsets, 5e-10 without);
    the offsets are constants with zero gradient, so dropping them tests the same forces.
    h = 1e-5: central-difference truncation error is O(h^2); with the valence term's
    0.08 A Gaussians it is 8e-8 at h = 1e-4 and 1.6e-9 at h = 1e-5 (measured)."""
    import copy

    m = copy.deepcopy(model)
    m.e0.zero_()
    return m


@pytest.fixture(scope="module")
def cluster():
    rng = np.random.default_rng(3)
    return pe_cluster(3, 12, rng)


@pytest.fixture(scope="module")
def melt():
    rng = np.random.default_rng(4)
    return pe_melt(4, 16, 0.35, rng)


def test_topology_blocks():
    z, p = pe_chain(20)
    t = topology_from_geometry(z, p)
    assert t.block_atoms[END].shape == (2, 7)
    assert t.block_atoms[MID].shape == (8, 6)
    assert len(t.block_edges) == 9
    # every atom in exactly one block
    allat = np.concatenate([t.block_atoms[MID].ravel(), t.block_atoms[END].ravel()])
    assert sorted(allat) == list(range(len(z)))


def test_radius_graph_vesin_matches_brute(melt):
    z, p, cell = melt
    t = topology_from_geometry(z, p, cell)
    b = collate([t], [p], [cell])
    ref = radius_graph_periodic_brute(b, 4.5)
    got = radius_graph_vesin(b, 4.5)

    def key(e):
        i, j, S = e
        return set(zip(i.tolist(), j.tolist(), map(tuple, S.tolist()), strict=True))

    assert key(ref) == key(got)


def test_radius_graph_cluster_vesin_matches_brute(cluster):
    z, p = cluster
    b = collate([topology_from_geometry(z, p)], [p])
    a = radius_graph_brute(b, 4.5)
    v = radius_graph_vesin(b, 4.5)
    assert set(zip(a[0].tolist(), a[1].tolist(), strict=True)) == set(
        zip(v[0].tolist(), v[1].tolist(), strict=True)
    )


def test_rotation_translation_cluster(model, cluster, rng):
    z, p = cluster
    f = gates.make_energy_fn(model)
    assert gates.rotation_translation_error(f, z, p, None, rng) < 1e-9


def test_rotation_translation_periodic(model, melt, rng):
    z, p, cell = melt
    f = gates.make_energy_fn(model)
    assert gates.rotation_translation_error(f, z, p, cell, rng) < 1e-9


def test_permutation_cluster(model, cluster, rng):
    z, p = cluster
    f = gates.make_energy_fn(model)
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9


def test_permutation_periodic(model, melt, rng):
    z, p, cell = melt
    f = gates.make_energy_fn(model)
    assert gates.permutation_error(f, z, p, cell, rng) < 1e-9


def test_wrap_invariance(model, melt, rng):
    z, p, cell = melt
    f = gates.make_energy_fn(model)
    assert gates.wrap_error(f, z, p, cell, rng) < 1e-9


def test_finite_difference_forces_cluster(model_noref, cluster):
    z, p = cluster
    assert gates.finite_difference_error(model_noref, z, p, h=1e-5) < 1e-8


def test_finite_difference_forces_periodic(model_noref, melt):
    z, p, cell = melt
    assert gates.finite_difference_error(model_noref, z, p, cell, h=1e-5) < 1e-8


def test_energy_smooth_across_cutoff(model):
    """Two chains pulled apart through the cutoff: E(d) must have no jump."""
    z1, p1 = pe_chain(8)
    ds = np.linspace(4.0, 9.0, 400)
    Es = []
    f = gates.make_energy_fn(model)
    for d in ds:
        p2 = p1 + np.array([0.0, 0.0, d])
        Es.append(f(np.concatenate([z1, z1]), np.concatenate([p1, p2])))
    Es = np.asarray(Es)
    assert np.ptp(Es[:80]) > 1e-6  # the two chains do interact at short range
    # A jump at the cutoff shows as an isolated spike in the second difference. Smooth
    # E(d) sampled at spacing h has |second difference| ~ h * |first difference|.
    d1 = np.abs(np.diff(Es))
    d2 = np.abs(np.diff(Es, 2))
    assert d2.max() < 0.2 * d1.max() + 1e-12
    assert np.ptp(Es[-40:]) < 1e-12  # beyond 2 * cutoff along the stack: exactly constant


def test_checkers_catch_broken_energy(cluster, rng):
    """Negative control: the checkers must fail on energies that break each symmetry."""
    z, p = cluster

    def first_atom_x(numbers, positions, cell=None):  # not rotation invariant
        return float(positions[0, 0])

    def label_dependent(numbers, positions, cell=None):  # not permutation invariant
        return float(np.sum(np.arange(len(numbers)) * positions[:, 0]))

    assert gates.rotation_translation_error(first_atom_x, z, p, None, rng) > 1e-3
    assert gates.permutation_error(label_dependent, z, p, None, rng) > 1e-3


def test_automorphism_oracle(model, cluster, rng):
    """Brute-force average over explicit relabellings (H swaps + chain reversal) equals the
    fast path: the fast path is already invariant, so the average of identical numbers is
    that number. Paired with test_checkers_catch_broken_energy as the negative control."""
    z, p = cluster
    f = gates.make_energy_fn(model)
    e = f(z, p)
    es = [f(z[q], p[q]) for q in (rng.permutation(len(z)) for _ in range(8))]
    assert abs(np.mean(es) - e) < 1e-9


def test_radius_graph_multi_structure_batch():
    """The vectorised batch path must never connect atoms of different structures."""
    rng = np.random.default_rng(7)
    systems = [pe_cluster(2, 8, rng) for _ in range(3)]
    b = collate([topology_from_geometry(z, p) for z, p in systems], [p for _, p in systems])
    a = radius_graph_brute(b, 4.5)
    v = radius_graph_vesin(b, 4.5)
    assert set(zip(a[0].tolist(), a[1].tolist(), strict=True)) == set(
        zip(v[0].tolist(), v[1].tolist(), strict=True)
    )
    assert bool((b.struct[a[0]] == b.struct[a[1]]).all())
