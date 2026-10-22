import pathlib

import numpy as np
import pytest

from frame_potential.builders import pe_chain, pe_cluster, pe_melt
from frame_potential.topology import TopologyError, infer_bonds
from frame_potential.v2.graph import (
    build_graph_topology,
    infer_bonds_general,
    topology_from_geometry_v2,
)
from v2_molecules import HETERO, HYDROCARBONS, ether_chain, g2, ion_pair


def blocks_of(t):
    return sorted(sorted(int(a) for a in row if a >= 0) for row in t.block_atoms)


def n_var(t, k):
    return int((t.var_block == k).sum())


PE_CACHE = pathlib.Path("C:/Users/sumedh/polyframenet_data/opoly26_ch/pe_frames.pkl")


def test_bonds_match_v1_on_pe_melt():
    z, p, cell = pe_melt(4, 16, 0.35, np.random.default_rng(4))
    assert np.array_equal(infer_bonds_general(z, p, cell), infer_bonds(z, p, cell))
    # pe_cluster adds 0.05 A Gaussian noise; with seed 0 one C-H stretches to 1.323 A,
    # inside v1's 1.35 cut-off but outside v2's 1.2 * (0.76 + 0.31) = 1.284 A. Noise-free:
    z, p = pe_cluster(3, 12, np.random.default_rng(0), noise=0.0)
    assert np.array_equal(infer_bonds_general(z, p), infer_bonds(z, p))


@pytest.mark.skipif(not PE_CACHE.exists(), reason="v1 PE dataset cache not present")
def test_bonds_match_v1_on_real_pe_frames():
    """The spec's gate: identical bonds on every v1 PE frame (DFT MD at 600 K: longest
    bonded C-H 1.224 A, C-C 1.746 A, both well inside the v2 cut-offs)."""
    import pickle

    frames = pickle.loads(PE_CACHE.read_bytes())
    for f in frames:
        assert np.array_equal(infer_bonds_general(f.numbers, f.positions), f.topo.bonds)


def test_pe_blocks_are_ch2_and_ch3():
    z, p = pe_chain(8)
    t = topology_from_geometry_v2(z, p)
    sizes = sorted(len(b) for b in blocks_of(t))
    assert sizes == [3] * 6 + [4] * 2  # 6 CH2, 2 CH3
    # CH3: e1 to its only C neighbour, e2 over 3 H -> 1 * 3 * 2 mirror = 6.
    # CH2: WL colours separate the two C neighbours (different distance to the chain
    # ends) unless the chain is symmetric about that carbon -> 2, or 4 when they tie.
    for k in range(t.n_blocks):
        size = int((t.block_atoms[k] >= 0).sum())
        assert n_var(t, k) in ((2, 4) if size == 3 else (6,))


def test_benzene_is_one_block_with_24_variants():
    z, p = g2("C6H6")
    t = topology_from_geometry_v2(z, p)
    assert t.n_blocks == 1 and n_var(t, 0) == 24


def test_terminal_and_linear_rules():
    z, p = g2("CH3CN")  # nitrile C is linear: merges with CH3 block; N terminal
    t = topology_from_geometry_v2(z, p)
    assert t.n_blocks == 1 and (t.var_cab[:, 0] >= 0).all()
    z, p = g2("CO2")  # all linear / terminal: one frame-free block
    t = topology_from_geometry_v2(z, p)
    assert t.n_blocks == 1 and t.var_cab.tolist() == [[-1, -1, -1]] and t.var_m.tolist() == [0]
    z, p = ion_pair()  # two isolated ions: two frame-free blocks
    t = topology_from_geometry_v2(z, p)
    assert t.n_blocks == 2 and (t.var_cab[:, 0] < 0).all()


def test_ester_blocks():
    z, p = g2("HCOOCH3")  # H-C(=O)-O-CH3: blocks HC(=O), O, CH3
    t = topology_from_geometry_v2(z, p)
    assert sorted(len(b) for b in blocks_of(t)) == [1, 3, 4]


def test_ether_chain_builds():
    z, p = ether_chain(12)
    t = topology_from_geometry_v2(z, p)
    assert (z == 8).sum() >= 2 and t.n_blocks == (z != 1).sum()


@pytest.mark.parametrize("name", HYDROCARBONS + HETERO)
def test_every_test_molecule_builds_with_valid_variants(name):
    z, p = g2(name)
    t = topology_from_geometry_v2(z, p)
    for k in range(t.n_blocks):
        assert n_var(t, k) >= 1
    framed = t.var_cab[:, 0] >= 0
    c, a, b = t.var_cab[framed].T
    va = p[a] - p[c]
    vb = p[b] - p[c]
    s = np.linalg.norm(np.cross(va, vb), axis=1) / (
        np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1))
    assert (s > 0.5).all()  # frame never near-degenerate


def test_rows_are_consistent():
    z, p = g2("C6H6")
    t = topology_from_geometry_v2(z, p)
    assert len(t.av_atom) == t.n_atoms * 24
    z, p = pe_chain(8)
    t = topology_from_geometry_v2(z, p)
    for e in range(len(t.edge_src)):
        assert int((t.ev_edge == e).sum()) == n_var(t, t.edge_dst[e])
    assert set(t.edge_hop.tolist()) == {0, 1}


def test_valence_pairs():
    z, p = pe_chain(4)  # butane: 13 bonds
    t = topology_from_geometry_v2(z, p)
    assert int((t.val_d == 1).sum()) == 13
    assert set(t.val_d.tolist()) == {1, 2, 3}


def test_rejections():
    z = np.array([1, 6, 1])
    p = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0.3, 0]])  # H bonded to C and H
    with pytest.raises(TopologyError):
        build_graph_topology(z, infer_bonds_general(z, p))
    # 40-carbon ring -> one ring-system block of 40 > 32-atom cap
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    p = np.stack([6.0 * np.cos(ang), 6.0 * np.sin(ang), np.zeros(40)], 1)  # 0.94 A apart
    with pytest.raises(TopologyError):
        topology_from_geometry_v2(np.full(40, 6), p)
