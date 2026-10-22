# PolyFrameNet v2 (general polymer architecture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One element-general, exactly symmetric polymer potential (v2). Its correctness gates run on any chemistry. It ships with a block registry that flags unfamiliar chemistry at inference. For the hackathon it is trained on OPoly26 hydrocarbon frames and benchmarked against MACE, Allegro, DeepMD and UMA.

**Architecture:**
- **Blocks.** Rigid fragments cut by a graph-only rule: each ring system is one block, every other non-terminal atom is its own block, terminal atoms join their neighbour, and linear sp carbons merge with their neighbours.
- **Frames.** One frame per (anchor, e1 atom, e2 atom, mirror) taken from Weisfeiler-Lehman colour classes. Every stage averages over that set, which gives exact O(3) and permutation invariance.
- **Stages:**
  - Stage 2: a shared atom-set encoder.
  - Stage 3: 3 message-passing layers on the block graph.
  - Valence prior.
  - Stage 4: a 4.5 Å moment layer.
  - Stage 5: latent-charge Coulomb (erf for clusters, Ewald for periodic) plus a learned dispersion tail.
  - Readout: per-atom energy plus int64 E0 offsets summed in float64.

**Tech Stack:** Python 3.13, PyTorch 2.11 (CUDA 12.8), numpy, scipy (cKDTree), networkx 3.7, ase (covalent radii, g2 test molecules), vesin-torch, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-hydrocarbon-generalization-design.md`

## Global Constraints

- **Code:**
  - v1 code (`src/frame_potential/{topology,batch,model,data,train,calculator,gates}.py`) is not modified, except that `scripts/md_bench.py` gains a `v2` model choice.
  - New code lives in `src/frame_potential/v2/`.
- **Invariants:**
  - Forces are `-dE/dr` by autodiff.
  - No caching across calls.
  - Frame atoms are chosen by topology (colour classes) only.
  - Raw 3-vectors and 3x3 frames are used; rotations are never compressed.
  - Arbitrary choices are averaged over, never selected.
  - Every radial function goes smoothly to zero at its cutoff.
- **Numerics:**
  - Energy is summed in float64.
  - Per-element offsets are an int64 micro-eV buffer of length `N_ELEM = 100`.
- **Bonds:** `r < 1.2 * (r_cov,i + r_cov,j)`, with Cordero covalent radii (`ase.data.covalent_radii`).
- **Block size cap:** 32 atoms. A larger block raises `TopologyError`.
- **Cutoffs:**
  - Stage 4: 4.5 Å.
  - Long range: 10 Å. The dispersion switch runs from 8 to 10 Å.
  - The Ewald real-space switch runs from 9 to 10 Å.
- **Tests:**
  - Tests run in float64 (`tests/conftest.py` already sets this).
  - Invariance tolerance: |ΔE| < 1e-9 eV.
  - Finite-difference force tolerance: < 1e-8 eV/Å, at h = 1e-5, with E0 = 0.
- **Commits:**
  - Use `git commit -F <scratchpad file>`, never `-m` with `$(`, backticks or a heredoc.
  - End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - Push after each commit.
- **Shell:** never `cd X && <write>` in one Bash call. Use absolute paths.

## Review Focus

1. **Periodic boxes where a block or variant atom straddles the boundary.** Energy must not change under wrap. Pinned by `test_wrap_periodic` (Task 4), using a PE melt with random lattice shifts.
2. **Frame-free blocks** (an ion, CO2, a linear nitrile molecule) mixed with framed blocks in one batch. Expect finite E and F, and all symmetry gates pass. Pinned by `test_frame_free_and_linear` (Task 4).
3. **Charged input Q ≠ 0.** Σq must equal Q, and E must be finite and invariant. Pinned by `test_charge_conservation` (Task 3) and `test_charged_input_invariance` (Task 4).
4. **Batches mixing different block sizes** (benzene plus PE). Batched E must equal single-structure E. Pinned by `test_batch_equals_single` (Task 4).
5. **Topology that cannot be built** (a large fused ring system, H with two neighbours). Expect a clean `TopologyError`, not a crash deep in the model. Pinned by `test_rejections` (Task 1).

---

## File structure

| File | Responsibility |
|---|---|
| `src/frame_potential/v2/__init__.py` | empty |
| `src/frame_potential/v2/graph.py` | Bonds, block rule, WL colours, variant table, block graph, valence pairs, row tables. Pure numpy and networkx. |
| `src/frame_potential/v2/batch.py` | `BatchV2` dataclass and `collate_v2` |
| `src/frame_potential/v2/longrange.py` | `coulomb_energy` (erf cluster and Ewald), `dispersion_energy`, `switch` |
| `src/frame_potential/v2/model.py` | `PolyFrameNetV2` |
| `src/frame_potential/v2/gates.py` | `make_energy_fn_v2`, `finite_difference_error_v2`, `variant_signatures` |
| `src/frame_potential/v2/data.py` | `FrameV2`, data configs, loader, split, reference fit, batching |
| `src/frame_potential/v2/train.py` | Training CLI, `evaluate_v2`, `load_model_v2` |
| `src/frame_potential/v2/calculator.py` | `PolyFrameNetV2Calculator` (ASE) |
| `tests/v2_molecules.py` | Test molecules: g2, PEO-like chain, ion pair, multi-molecule clusters |
| `tests/test_v2_graph.py`, `test_v2_longrange.py`, `test_v2_gates.py` | Tests |
| `src/frame_potential/v2/registry.py` | Block registry: training coverage (seen / similar / novel) per block |
| `scripts/build_v2_dataset.py`, `scripts/export_v2_extxyz.py`, `scripts/export_melts.py` | Data build, baseline exports, benchmark structures |
| `scripts/eval_ase.py`, `scripts/bench_ase.py`, `scripts/eval_v2_all.py`, `scripts/block_coverage.py` | Model-agnostic accuracy, speed and NVE (run inside each baseline's env), results table, coverage CLI |
| `baselines/{allegro,deepmd}/` | Exact baseline configs and commands |

---

### Task 1: Stage 0 graph (`v2/graph.py`)

**Files:**
- Create: `src/frame_potential/v2/__init__.py` (empty), `src/frame_potential/v2/graph.py`, `tests/v2_molecules.py`, `tests/test_v2_graph.py`

**Interfaces:**
- Consumes: `frame_potential.topology.TopologyError`, `_mic`, `infer_bonds` (v1, for the equivalence test); `frame_potential.builders.pe_chain`, `pe_cluster`, `pe_melt`.
- Produces:
  - `infer_bonds_general(numbers, positions, cell=None) -> np.ndarray (E,2) int64`
  - `build_graph_topology(numbers, bonds) -> GraphTopology`
  - `topology_from_geometry_v2(numbers, positions, cell=None) -> GraphTopology`
  - `attach_rows(topo) -> GraphTopology`
  - `GraphTopology` fields:
    - `numbers, bonds, degree, ring, n_ext, colour, block_of_atom, block_atoms (B,n) -1-padded`
    - `var_block (R,), var_cab (R,3) (-1 for frame-free), var_m (R,) int (+1/-1, 0 for frame-free)`
    - `edge_src, edge_dst, edge_hop (Ed,), ev_edge, ev_var (Q,), av_atom, av_var (A,)`
    - `val_pairs (P,2), val_d (P,)`
    - properties `n_atoms`, `n_blocks`

- [ ] **Step 1: Write test molecules helper** `tests/v2_molecules.py`

```python
"""Test molecules for v2 gates: hydrocarbons, heteroatom polymer fragments, ions.

Geometry need not be physical for symmetry gates; it must only be unambiguous for bond
detection (bonded < 1.2 * sum of covalent radii < nonbonded)."""

import numpy as np
from ase.build import molecule

from frame_potential.builders import pe_chain, random_rotation

HYDROCARBONS = ["C6H6", "isobutane", "butadiene", "CH4", "2-butyne", "isobutene"]
# Ether (PEO), amide (nylon), ester (PMMA/PET), C-Cl (PVC), CF4 (PTFE), aromatic N and S
# rings, nitrile (linear sp carbon), CO2 (linear molecule, frame-free), water.
HETERO = ["CH3OCH3", "CH3CONH2", "HCOOCH3", "CH3CH2Cl", "CF4", "C5H5N", "C4H4S",
          "CH3CN", "CO2", "H2O"]


def g2(name):
    a = molecule(name)
    return a.numbers.copy(), a.positions.copy()


def cluster_of(parts, rng, spacing=7.0):
    """Place (numbers, positions) parts on a line, each randomly rotated."""
    zs, ps = [], []
    for k, (z, p) in enumerate(parts):
        q = (p - p.mean(0)) @ random_rotation(rng).T
        zs.append(z)
        ps.append(q + np.array([k * spacing, 0.0, 0.0]))
    return np.concatenate(zs), np.concatenate(ps)


def ether_chain(n_carbons=12):
    """PEO-like: PE chain with every third backbone carbon replaced by O (its H removed)."""
    from frame_potential.v2.graph import infer_bonds_general

    z, p = pe_chain(n_carbons)
    bonds = infer_bonds_general(z, p)
    carbons = [i for i in range(len(z)) if z[i] == 6]
    drop = set()
    z = z.copy()
    for k, c in enumerate(carbons):
        if k % 3 == 1 and 0 < k < len(carbons) - 1:
            z[c] = 8
            for a, b in bonds:
                if c in (a, b):
                    o = b if a == c else a
                    if z[o] == 1:
                        drop.add(o)
    keep = np.array([i for i in range(len(z)) if i not in drop])
    return z[keep], p[keep]


def ion_pair():
    return np.array([11, 17]), np.array([[0.0, 0.0, 0.0], [6.0, 0.3, 0.1]])
```

- [ ] **Step 2: Write failing graph tests** `tests/test_v2_graph.py`

```python
import numpy as np
import pytest

from frame_potential.builders import pe_chain, pe_cluster, pe_melt
from frame_potential.topology import TopologyError, infer_bonds
from frame_potential.v2.graph import (
    build_graph_topology,
    infer_bonds_general,
    topology_from_geometry_v2,
)
from tests.v2_molecules import HETERO, HYDROCARBONS, ether_chain, g2, ion_pair


def blocks_of(t):
    return sorted(sorted(int(a) for a in row if a >= 0) for row in t.block_atoms)


def n_var(t, k):
    return int((t.var_block == k).sum())


def test_bonds_match_v1_on_pe():
    rng = np.random.default_rng(0)
    z, p = pe_cluster(3, 12, rng)
    assert np.array_equal(infer_bonds_general(z, p), infer_bonds(z, p))
    z, p, cell = pe_melt(4, 16, 0.35, np.random.default_rng(4))
    assert np.array_equal(infer_bonds_general(z, p, cell), infer_bonds(z, p, cell))


def test_pe_blocks_are_ch2_and_ch3():
    z, p = pe_chain(8)
    t = topology_from_geometry_v2(z, p)
    sizes = sorted(len(b) for b in blocks_of(t))
    assert sizes == [3] * 6 + [4] * 2  # 6 CH2, 2 CH3
    ch2 = [k for k, b in enumerate(blocks_of(t)) if len(b) == 3]
    # CH2 block: anchor C, a/b classes of size 2 -> 2 * 1 * 2 mirror = 4 variants
    for row_k in range(t.n_blocks):
        size = int((t.block_atoms[row_k] >= 0).sum())
        assert n_var(t, row_k) == (4 if size == 3 else 6)
    assert len(ch2) == 6


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
    # 40-carbon ring system -> exceeds the 32-atom block cap
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    p = np.stack([6.0 * np.cos(ang), 6.0 * np.sin(ang), np.zeros(40)], 1)  # 0.94 A apart
    with pytest.raises(TopologyError):
        topology_from_geometry_v2(np.full(40, 6), p)
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `pytest tests/test_v2_graph.py -q`
Expected: `ModuleNotFoundError: No module named 'frame_potential.v2'`

- [ ] **Step 4: Implement** `src/frame_potential/v2/__init__.py` (empty) and `src/frame_potential/v2/graph.py`

```python
"""v2 Stage 0: element-general bonds, rigid-fragment blocks, WL colours, frame variants.

Integer bookkeeping computed once per system (non-reactive), pure numpy / networkx.

Block rule (graph only, no bond orders):
  1. every ring system (atoms joined by non-bridge bonds) is one block;
  2. every other non-terminal atom is its own block (degree-0 atoms too);
  3. degree-1 atoms join their neighbour's block;
  4. a linear atom (non-ring C with degree 2) merges with all its neighbours' blocks;
  5. blocks larger than MAX_BLOCK atoms are rejected.

Frame variants of a block (Stage 1 uses them; see the spec for the invariance argument):
  anchors c = canonical colour class among block atoms with degree >= 2 that are not linear;
  a = canonical class of N(c); b = canonical class of N(c) minus a; mirror m in {+1, -1}.
  "Canonical class" = the colour class with fewest members, ties broken by colour value:
  a set of atoms chosen by (permutation-invariant) colour, never an individual atom.
  A block without anchors (ion, CO2, H2) gets one frame-free row: cab = -1, m = 0.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
from ase.data import covalent_radii

from ..topology import TopologyError, _mic

BOND_SCALE = 1.2
MAX_BLOCK = 32
# Max bonded neighbours per element; more means bond detection merged a contact.
MAX_VALENCE = {1: 1, 5: 4, 6: 4, 7: 4, 8: 3, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1}


@dataclass
class GraphTopology:
    numbers: np.ndarray
    bonds: np.ndarray
    degree: np.ndarray
    ring: np.ndarray
    n_ext: np.ndarray
    colour: np.ndarray
    block_of_atom: np.ndarray
    block_atoms: np.ndarray
    var_block: np.ndarray
    var_cab: np.ndarray
    var_m: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_hop: np.ndarray
    val_pairs: np.ndarray
    val_d: np.ndarray
    ev_edge: np.ndarray | None = None
    ev_var: np.ndarray | None = None
    av_atom: np.ndarray | None = None
    av_var: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @property
    def n_blocks(self) -> int:
        return len(self.block_atoms)


def infer_bonds_general(numbers, positions, cell=None) -> np.ndarray:
    """Covalent bonds r < 1.2 (r_cov,i + r_cov,j), periodic-aware. Read once at t=0."""
    from scipy.spatial import cKDTree

    numbers = np.asarray(numbers)
    positions = np.asarray(positions, dtype=np.float64)
    n = len(numbers)
    rc = covalent_radii[numbers]
    rmax = BOND_SCALE * 2 * float(rc.max()) if n else 0.0
    if cell is not None and not np.allclose(cell, np.diag(np.diag(cell))):
        d = _mic((positions[None] - positions[:, None]).reshape(-1, 3), cell).reshape(n, n, 3)
        i, j = np.nonzero(np.triu(np.linalg.norm(d, axis=-1) < rmax, 1))
    else:
        if cell is None:
            tree = cKDTree(positions)
        else:
            L = np.abs(np.diag(cell))
            x = positions - L * np.floor(positions / L)
            x[x >= L] = 0.0
            tree = cKDTree(x, boxsize=L)
        pairs = tree.query_pairs(rmax, output_type="ndarray")
        if not len(pairs):
            return np.zeros((0, 2), np.int64)
        i, j = pairs[:, 0], pairs[:, 1]
    r = np.linalg.norm(_mic(positions[j] - positions[i], cell), axis=1)
    keep = r < BOND_SCALE * (rc[i] + rc[j])
    out = np.stack([np.minimum(i, j), np.maximum(i, j)], 1)[keep]
    return out[np.lexsort((out[:, 1], out[:, 0]))].astype(np.int64)


def _h(*xs) -> int:
    """Process-stable 64-bit hash (Python's hash() is salted per process)."""
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(),
                          "little", signed=True)


def wl_colours(numbers, degree, ring, nbrs, max_rounds=20) -> np.ndarray:
    """Weisfeiler-Lehman refinement to a stable partition. Isomorphism-invariant: any
    graph automorphism maps each colour class onto itself."""
    col = [_h(int(z), int(d), bool(r)) for z, d, r in zip(numbers, degree, ring, strict=True)]
    k = len(set(col))
    for _ in range(max_rounds):
        new = [_h(col[i], tuple(sorted(col[j] for j in nbrs[i]))) for i in range(len(col))]
        k_new = len(set(new))
        col = new
        if k_new == k:
            break
        k = k_new
    return np.array(col, dtype=np.int64)


def _canonical_class(atoms, col):
    cnt: dict[int, int] = {}
    for a in atoms:
        cnt[int(col[a])] = cnt.get(int(col[a]), 0) + 1
    best = min(cnt, key=lambda c: (cnt[c], c))
    return [a for a in atoms if int(col[a]) == best]


def build_graph_topology(numbers, bonds) -> GraphTopology:
    numbers = np.asarray(numbers, dtype=np.int64)
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    n = len(numbers)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(map(tuple, bonds.tolist()))
    nbrs = [sorted(g.neighbors(i)) for i in range(n)]
    degree = np.array([len(x) for x in nbrs], dtype=np.int64)
    for i in range(n):
        if numbers[i] == 1 and degree[i] != 1:
            raise TopologyError(f"H atom {i} has {degree[i]} neighbours")
        if degree[i] > MAX_VALENCE.get(int(numbers[i]), 6):
            raise TopologyError(f"atom {i} (Z={numbers[i]}) has {degree[i]} neighbours")

    bridges = {frozenset(e) for e in nx.bridges(g)}
    ring_edges = [e for e in g.edges if frozenset(e) not in bridges]
    ring = np.zeros(n, dtype=bool)
    for a, b in ring_edges:
        ring[a] = ring[b] = True
    linear = (numbers == 6) & (degree == 2) & ~ring

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in ring_edges:  # rule 1
        union(a, b)
    for i in range(n):  # rule 3
        if degree[i] == 1:
            union(i, nbrs[i][0])
    for i in range(n):  # rule 4
        if linear[i]:
            for j in nbrs[i]:
                union(i, j)
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    block_of_atom = np.array([ids[r] for r in roots], dtype=np.int64)
    B = len(ids)
    members: list[list[int]] = [[] for _ in range(B)]
    for i in range(n):
        members[block_of_atom[i]].append(i)
    nmax = max((len(m) for m in members), default=1)
    if nmax > MAX_BLOCK:
        raise TopologyError(f"block of {nmax} atoms exceeds cap {MAX_BLOCK}")
    block_atoms = np.full((B, nmax), -1, dtype=np.int64)
    for k, m in enumerate(members):
        block_atoms[k, : len(m)] = m
    n_ext = np.array([sum(block_of_atom[j] != block_of_atom[i] for j in nbrs[i])
                      for i in range(n)], dtype=np.int64)
    col = wl_colours(numbers, degree, ring, nbrs)

    var_block, var_cab, var_m = [], [], []
    for k, m in enumerate(members):
        cand = [i for i in m if degree[i] >= 2 and not linear[i]]
        if not cand:
            var_block.append(k)
            var_cab.append((-1, -1, -1))
            var_m.append(0)
            continue
        for c in _canonical_class(cand, col):
            for a in _canonical_class(nbrs[c], col):
                for b in _canonical_class([x for x in nbrs[c] if x != a], col):
                    for s in (1, -1):
                        var_block.append(k)
                        var_cab.append((c, a, b))
                        var_m.append(s)

    bb = set()
    for a, b in bonds.tolist():
        I, J = int(block_of_atom[a]), int(block_of_atom[b])
        if I != J:
            bb.add((min(I, J), max(I, J)))
    adj: list[set[int]] = [set() for _ in range(B)]
    for I, J in bb:
        adj[I].add(J)
        adj[J].add(I)
    bb2 = set()
    for K in range(B):
        nb = sorted(adj[K])
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                if (nb[x], nb[y]) not in bb:
                    bb2.add((nb[x], nb[y]))
    e1, e2 = sorted(bb), sorted(bb2)
    src = [J for _, J in e1] + [I for I, _ in e1] + [J for _, J in e2] + [I for I, _ in e2]
    dst = [I for I, _ in e1] + [J for _, J in e1] + [I for I, _ in e2] + [J for _, J in e2]
    hop = [0] * (2 * len(e1)) + [1] * (2 * len(e2))

    vp, vd = [], []
    for i in range(n):
        for j, d in nx.single_source_shortest_path_length(g, i, cutoff=3).items():
            if j > i and d >= 1:
                vp.append((i, j))
                vd.append(d)

    def arr(x, shape=None):
        a = np.asarray(x, dtype=np.int64)
        return a.reshape(shape) if shape else a

    topo = GraphTopology(
        numbers=numbers, bonds=bonds, degree=degree, ring=ring, n_ext=n_ext, colour=col,
        block_of_atom=block_of_atom, block_atoms=block_atoms,
        var_block=arr(var_block), var_cab=arr(var_cab, (-1, 3)), var_m=arr(var_m),
        edge_src=arr(src), edge_dst=arr(dst), edge_hop=arr(hop),
        val_pairs=arr(vp, (-1, 2)), val_d=arr(vd),
    )
    return attach_rows(topo)


def attach_rows(topo: GraphTopology) -> GraphTopology:
    """(edge, variant-of-receiver) rows for Stage 3 and (atom, variant-of-its-block) rows
    for Stage 4, rebuilt from the variant table (tests filter variants and call this)."""
    rows_of: list[list[int]] = [[] for _ in range(topo.n_blocks)]
    for r, k in enumerate(topo.var_block.tolist()):
        rows_of[k].append(r)
    ev_e, ev_v = [], []
    for e, I in enumerate(topo.edge_dst.tolist()):
        ev_e += [e] * len(rows_of[I])
        ev_v += rows_of[I]
    av_a, av_v = [], []
    for i, k in enumerate(topo.block_of_atom.tolist()):
        av_a += [i] * len(rows_of[k])
        av_v += rows_of[k]
    return replace(topo, ev_edge=np.asarray(ev_e, np.int64), ev_var=np.asarray(ev_v, np.int64),
                   av_atom=np.asarray(av_a, np.int64), av_var=np.asarray(av_v, np.int64))


def topology_from_geometry_v2(numbers, positions, cell=None) -> GraphTopology:
    return build_graph_topology(numbers, infer_bonds_general(numbers, positions, cell))
```

- [ ] **Step 5: Run tests, verify they pass**

Run: `pytest tests/test_v2_graph.py -q`
Expected: all PASS.
- If `test_bonds_match_v1_on_pe` fails: print the differing pairs and their distances. Do not loosen the test. The v1 C-C cutoff is 1.85 Å and v2's is 1.824 Å, so a mismatch means some builder geometry has a C-C distance in between. Report it.
- If `test_ester_blocks` fails because carbonyl O formed a ring bond, inspect `ring` and fix the rule (do not adjust the expected value without reason).

- [ ] **Step 6: Lint and commit**

Run: `ruff check src/frame_potential/v2 tests/test_v2_graph.py tests/v2_molecules.py`
Commit message file content:
```
Add v2 Stage 0: element-general bonds, rigid-fragment blocks, frame variants

<what/why per spec section Stage 0a/0b/1; tests listed>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
```

---

### Task 2: Batch and collate (`v2/batch.py`)

**Files:**
- Create: `src/frame_potential/v2/batch.py`
- Test: `tests/test_v2_graph.py` (append)

**Interfaces:**
- Consumes: `GraphTopology` (Task 1).
- Produces:
  - `BatchV2` dataclass. Fields: `numbers, positions, struct, n_struct, cell, charge (S,), spin (S,), degree, ring, n_ext, block_of_atom, block_struct, block_atoms, var_block, var_cab, var_m, edge_src, edge_dst, edge_hop, ev_edge, ev_var, av_atom, av_var, val_pairs, val_d, energy=None, forces=None`. Properties: `n_atoms`, `n_blocks`. Method: `to(device, dtype=None)`.
  - `collate_v2(topos, positions, cells=None, energies=None, forces=None, charges=None, spins=None) -> BatchV2`. Always float64.
  - BatchV2 duck-types v1's `Batch` for `radius_graph` and `edge_vectors`: `positions, struct, n_struct, cell, n_atoms`.

- [ ] **Step 1: Append failing test**

```python
def test_collate_offsets():
    from frame_potential.v2.batch import collate_v2

    z1, p1 = g2("C6H6")
    z2, p2 = pe_chain(6)
    t1, t2 = topology_from_geometry_v2(z1, p1), topology_from_geometry_v2(z2, p2)
    b = collate_v2([t1, t2], [p1, p2])
    assert b.n_atoms == t1.n_atoms + t2.n_atoms and b.n_blocks == t1.n_blocks + t2.n_blocks
    nb1, nv1 = t1.n_blocks, len(t1.var_block)
    assert int(b.var_block[nv1]) == nb1  # second structure's first variant -> its block 0
    assert int(b.var_cab[nv1, 0]) == int(t2.var_cab[0, 0]) + t1.n_atoms
    assert int(b.ev_var.max()) < len(b.var_block) and int(b.av_atom.max()) == b.n_atoms - 1
    assert b.block_atoms.shape[1] == max(t1.block_atoms.shape[1], t2.block_atoms.shape[1])
    assert b.charge.tolist() == [0.0, 0.0] and b.spin.tolist() == [1.0, 1.0]
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_v2_graph.py::test_collate_offsets -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `src/frame_potential/v2/batch.py`

```python
"""v2 batching: concatenate GraphTopology index tables with offsets. Only integers are
prepared here; every vector is rebuilt inside autograd from positions and cell."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch

from .graph import GraphTopology


@dataclass
class BatchV2:
    numbers: torch.Tensor
    positions: torch.Tensor
    struct: torch.Tensor
    n_struct: int
    cell: torch.Tensor | None
    charge: torch.Tensor
    spin: torch.Tensor
    degree: torch.Tensor
    ring: torch.Tensor
    n_ext: torch.Tensor
    block_of_atom: torch.Tensor
    block_struct: torch.Tensor
    block_atoms: torch.Tensor
    var_block: torch.Tensor
    var_cab: torch.Tensor
    var_m: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_hop: torch.Tensor
    ev_edge: torch.Tensor
    ev_var: torch.Tensor
    av_atom: torch.Tensor
    av_var: torch.Tensor
    val_pairs: torch.Tensor
    val_d: torch.Tensor
    energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None

    @property
    def n_atoms(self) -> int:
        return self.numbers.shape[0]

    @property
    def n_blocks(self) -> int:
        return self.block_atoms.shape[0]

    def to(self, device, dtype=None) -> BatchV2:
        out = {}
        for f in fields(self):
            x = getattr(self, f.name)
            if not isinstance(x, torch.Tensor):
                out[f.name] = x
            elif f.name == "energy":  # targets stay float64 (see v1 Batch.to)
                out[f.name] = x.to(device)
            elif torch.is_floating_point(x):
                out[f.name] = x.to(device=device, dtype=dtype or x.dtype)
            else:
                out[f.name] = x.to(device)
        return BatchV2(**out)


def collate_v2(topos: list[GraphTopology], positions, cells=None, energies=None, forces=None,
               charges=None, spins=None) -> BatchV2:
    S = len(topos)
    nmax = max(t.block_atoms.shape[1] for t in topos)
    a_off = b_off = v_off = e_off = 0
    acc: dict[str, list] = {k: [] for k in (
        "numbers", "struct", "degree", "ring", "n_ext", "block_of_atom", "block_struct",
        "block_atoms", "var_block", "var_cab", "var_m", "edge_src", "edge_dst", "edge_hop",
        "ev_edge", "ev_var", "av_atom", "av_var", "val_pairs", "val_d")}
    for s, t in enumerate(topos):
        ba = np.full((t.n_blocks, nmax), -1, np.int64)
        ba[:, : t.block_atoms.shape[1]] = np.where(t.block_atoms >= 0, t.block_atoms + a_off, -1)
        acc["numbers"].append(t.numbers)
        acc["struct"].append(np.full(t.n_atoms, s))
        acc["degree"].append(t.degree)
        acc["ring"].append(t.ring.astype(np.int64))
        acc["n_ext"].append(t.n_ext)
        acc["block_of_atom"].append(t.block_of_atom + b_off)
        acc["block_struct"].append(np.full(t.n_blocks, s))
        acc["block_atoms"].append(ba)
        acc["var_block"].append(t.var_block + b_off)
        acc["var_cab"].append(np.where(t.var_cab >= 0, t.var_cab + a_off, -1))
        acc["var_m"].append(t.var_m)
        acc["edge_src"].append(t.edge_src + b_off)
        acc["edge_dst"].append(t.edge_dst + b_off)
        acc["edge_hop"].append(t.edge_hop)
        acc["ev_edge"].append(t.ev_edge + e_off)
        acc["ev_var"].append(t.ev_var + v_off)
        acc["av_atom"].append(t.av_atom + a_off)
        acc["av_var"].append(t.av_var + v_off)
        acc["val_pairs"].append(t.val_pairs + a_off)
        acc["val_d"].append(t.val_d)
        a_off += t.n_atoms
        b_off += t.n_blocks
        v_off += len(t.var_block)
        e_off += len(t.edge_src)

    def cat(k, shape=None):
        x = torch.as_tensor(np.concatenate(acc[k]), dtype=torch.long)
        return x.reshape(shape) if shape else x

    f64 = torch.float64
    return BatchV2(
        numbers=cat("numbers"),
        positions=torch.as_tensor(np.concatenate([np.asarray(p, np.float64) for p in positions]),
                                  dtype=f64),
        struct=cat("struct"), n_struct=S,
        cell=None if cells is None else torch.as_tensor(np.stack(cells), dtype=f64),
        charge=torch.as_tensor(np.zeros(S) if charges is None else charges, dtype=f64),
        spin=torch.as_tensor(np.ones(S) if spins is None else spins, dtype=f64),
        degree=cat("degree"), ring=cat("ring"), n_ext=cat("n_ext"),
        block_of_atom=cat("block_of_atom"), block_struct=cat("block_struct"),
        block_atoms=cat("block_atoms", (-1, nmax)),
        var_block=cat("var_block"), var_cab=cat("var_cab", (-1, 3)), var_m=cat("var_m"),
        edge_src=cat("edge_src"), edge_dst=cat("edge_dst"), edge_hop=cat("edge_hop"),
        ev_edge=cat("ev_edge"), ev_var=cat("ev_var"), av_atom=cat("av_atom"),
        av_var=cat("av_var"), val_pairs=cat("val_pairs", (-1, 2)), val_d=cat("val_d"),
        energy=None if energies is None else torch.as_tensor(np.asarray(energies), dtype=f64),
        forces=None if forces is None else torch.as_tensor(np.concatenate(forces), dtype=f64),
    )
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_v2_graph.py -q`. Expected: all PASS.
- [ ] **Step 5: Lint and commit** ("Add v2 batch collation").

---

### Task 3: Long-range terms (`v2/longrange.py`)

**Files:**
- Create: `src/frame_potential/v2/longrange.py`, `tests/test_v2_longrange.py`

**Interfaces:**
- Consumes: `BatchV2`, `collate_v2`, `topology_from_geometry_v2`; `frame_potential.batch.edge_vectors`, `radius_graph`.
- Produces:
  - `KE = 14.399645`
  - `switch(r, r_on, r_off) -> Tensor` (C2 smooth 1 → 0)
  - `coulomb_energy(batch, q, lr_edges, sigma, r_c, alpha=None) -> (S,)`
  - `neutralize(q, batch) -> q` (Σq per struct = batch.charge)
  - `dispersion_energy(batch, c6, r0, lr_edges, r_off) -> (S,)`

- [ ] **Step 1: Write failing tests** `tests/test_v2_longrange.py`

```python
import math

import numpy as np
import torch

from frame_potential.batch import radius_graph
from frame_potential.v2.batch import collate_v2
from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.longrange import (
    KE,
    coulomb_energy,
    dispersion_energy,
    neutralize,
    switch,
)
from tests.v2_molecules import g2, ion_pair


def _batch(z, p, cell=None, charge=0.0):
    t = topology_from_geometry_v2(z, p, cell)
    return collate_v2([t], [p], None if cell is None else [cell], charges=[charge])


def test_switch_is_smooth_step():
    r = torch.linspace(7.0, 11.0, 401)
    s = switch(r, 8.0, 10.0)
    assert float(s[0]) == 1.0 and float(s[-1]) == 0.0
    assert float(s.diff().max()) <= 0.0


def test_cluster_coulomb_matches_analytic():
    z, p = ion_pair()
    b = _batch(z, p)
    q = torch.tensor([0.7, -0.7])
    E = coulomb_energy(b, q, None, sigma=1.0, r_c=10.0)
    r = float(np.linalg.norm(p[1] - p[0]))
    assert abs(float(E[0]) - KE * (-0.49) * math.erf(r / 1.0) / r) < 1e-12


def test_ewald_cluster_limit_and_alpha_independence():
    """Neutral pair in a large box: Ewald (tin-foil) - cluster = 2 pi |M|^2 / (3 V) up to
    O(1/L^5); Ewald must not depend on the splitting parameter alpha."""
    z, p = ion_pair()
    p = p + 20.0
    q = torch.tensor([0.5, -0.5])
    L = 40.0
    cell = np.eye(3) * L
    b = _batch(z, p, cell)
    edges = radius_graph(b, 10.0)
    e_a = float(coulomb_energy(b, q, edges, sigma=1.0, r_c=10.0, alpha=0.35)[0])
    e_b = float(coulomb_energy(b, q, edges, sigma=1.0, r_c=10.0, alpha=0.45)[0])
    assert abs(e_a - e_b) < 1e-6
    bc = _batch(z, p)
    e_c = float(coulomb_energy(bc, q, None, sigma=1.0, r_c=10.0)[0])
    M = (q.numpy()[:, None] * p).sum(0)
    corr = KE * 2 * math.pi * float(M @ M) / (3 * L**3)
    assert abs(e_a - (e_c + corr)) < 1e-4


def test_charge_conservation():
    z, p = g2("C5H5N")
    for Q in (-1.0, 0.0, 1.0):
        b = _batch(z, p, charge=Q)
        q = neutralize(torch.randn(b.n_atoms), b)
        assert abs(float(q.sum()) - Q) < 1e-10


def test_dispersion_negative_and_switched():
    z, p = ion_pair()
    for d, expect_zero in ((5.0, False), (10.5, True)):
        pp = np.array([[0.0, 0, 0], [d, 0, 0]])
        b = _batch(z, pp)
        e = dispersion_energy(b, torch.tensor([10.0, 10.0]), torch.tensor([1.5, 1.5]),
                              radius_graph(b, 10.0), r_off=10.0)
        assert (float(e[0]) == 0.0) == expect_zero and float(e[0]) <= 0.0
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_v2_longrange.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `src/frame_potential/v2/longrange.py`

```python
"""Stage 5: latent-charge electrostatics and a learned dispersion tail.

Coulomb kernel phi(r) = erf(r / sigma) / r (Gaussian-screened, finite at r -> 0):
  clusters: E = KE sum_{i<j} q_i q_j phi(r_ij), all pairs;
  periodic: Ewald summation of the same kernel. Real space uses
    phi(r) - erf(alpha r)/r = erfc(alpha r)/r - erfc(r/sigma)/r  (short-ranged),
  multiplied by a C2 switch from r_c - 1 to r_c so the energy is smooth in the positions;
  reciprocal space (2 pi / V) sum_{k != 0} exp(-k^2 / 4 alpha^2) / k^2 |S(k)|^2;
  minus the i = j self term alpha / sqrt(pi) sum q^2, minus the neutralising-background
  term pi Q^2 / (2 V alpha^2) for a charged cell. Tin-foil boundary conditions.
Dispersion: E = -sum_{i<j} sqrt(C6_i C6_j) / (r^6 + R0_ij^6) * switch(r, r_off - 2, r_off).
"""

from __future__ import annotations

import math

import torch

from ..batch import edge_vectors

KE = 14.399645  # e^2 / (4 pi eps0), eV * Angstrom


def switch(r, r_on: float, r_off: float):
    x = ((r - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def neutralize(q, batch):
    n = torch.bincount(batch.struct, minlength=batch.n_struct).to(q.dtype)
    excess = q.new_zeros(batch.n_struct).index_add(0, batch.struct, q) - batch.charge.to(q.dtype)
    return q - (excess / n)[batch.struct]


def coulomb_energy(batch, q, lr_edges, sigma: float, r_c: float, alpha: float | None = None):
    if batch.cell is None:
        return _cluster(batch, q, sigma)
    return _ewald(batch, q, lr_edges, sigma, r_c, alpha)


def _cluster(batch, q, sigma):
    pos = batch.positions
    with torch.no_grad():
        same = batch.struct[:, None] == batch.struct[None, :]
        i, j = torch.nonzero(torch.triu(same, 1), as_tuple=True)
    v = pos[j] - pos[i]
    r = torch.sqrt((v * v).sum(-1))
    e = KE * q[i] * q[j] * torch.erf(r / sigma) / r
    return pos.new_zeros(batch.n_struct).index_add(0, batch.struct[i], e)


def _ewald(batch, q, lr_edges, sigma, r_c, alpha=None, accuracy=1e-6, k_chunk=4096):
    pos = batch.positions
    a = alpha if alpha is not None else math.sqrt(-math.log(accuracy)) / r_c
    i, j, S = lr_edges
    v = edge_vectors(batch, i, j, S)
    r = torch.sqrt((v * v).sum(-1))
    real = 0.5 * KE * q[i] * q[j] * (torch.erf(r / sigma) - torch.erf(a * r)) / r
    real = real * switch(r, r_c - 1.0, r_c)
    E = pos.new_zeros(batch.n_struct).index_add(0, batch.struct[i], real)
    kmax = 2.0 * a * math.sqrt(-math.log(accuracy))
    out = []
    for s in range(batch.n_struct):
        idx = torch.nonzero(batch.struct == s).squeeze(1)
        ps, qs = pos[idx], q[idx]
        cell = batch.cell[s]
        V = torch.abs(torch.det(cell))
        recip = 2 * math.pi * torch.linalg.inv(cell).T  # rows: reciprocal vectors
        with torch.no_grad():
            nk = [int(math.ceil(kmax * float(cell[d].norm()) / (2 * math.pi))) for d in range(3)]
            g = torch.stack(torch.meshgrid(*[torch.arange(-m, m + 1, device=pos.device)
                                             for m in nk], indexing="ij"), -1).reshape(-1, 3)
            # half space: first nonzero component positive
            half = (g[:, 0] > 0) | ((g[:, 0] == 0) & (g[:, 1] > 0)) | (
                (g[:, 0] == 0) & (g[:, 1] == 0) & (g[:, 2] > 0))
            g = g[half].to(pos.dtype)
            k2_ = ((g @ recip.detach()) ** 2).sum(-1)
            g = g[k2_ < kmax**2]
        e_rec = pos.new_zeros(())
        for c0 in range(0, len(g), k_chunk):
            k = g[c0 : c0 + k_chunk] @ recip
            k2 = (k * k).sum(-1)
            ph = ps @ k.T
            sre = (qs[:, None] * torch.cos(ph)).sum(0)
            sim = (qs[:, None] * torch.sin(ph)).sum(0)
            e_rec = e_rec + (torch.exp(-k2 / (4 * a * a)) / k2 * (sre**2 + sim**2)).sum()
        e_rec = KE * (2 * math.pi / V) * 2.0 * e_rec  # x2: half space
        e_self = -KE * a / math.sqrt(math.pi) * (qs * qs).sum()
        Q = qs.sum()
        e_bg = -KE * math.pi * Q * Q / (2 * V * a * a)
        out.append(e_rec + e_self + e_bg)
    return E + torch.stack(out)


def dispersion_energy(batch, c6, r0, lr_edges, r_off: float):
    i, j, S = lr_edges  # full list: each pair twice
    v = edge_vectors(batch, i, j, S)
    r = torch.sqrt((v * v).sum(-1))
    c6ij = torch.sqrt(c6[i] * c6[j] + 1e-12)
    R0 = r0[i] + r0[j]
    e = -0.5 * c6ij / (r**6 + R0**6) * switch(r, r_off - 2.0, r_off)
    return batch.positions.new_zeros(batch.n_struct).index_add(0, batch.struct[i], e)
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_v2_longrange.py -q`.
  - If the cluster-limit test misses by more than 1e-4: first check the sign and size of `corr` by printing e_a, e_c and corr.
  - The O(1/L^5) remainder at L = 40 is about 1e-6, so a larger miss is a bug.
- [ ] **Step 5: Lint and commit** ("Add v2 Stage 5: erf Coulomb, Ewald, dispersion"). The message records the α-independence and cluster-limit checks.

---

### Task 4: Model and full gate suite (`v2/model.py`, `v2/gates.py`)

**Files:**
- Create: `src/frame_potential/v2/model.py`, `src/frame_potential/v2/gates.py`, `tests/test_v2_gates.py`

**Interfaces:**
- Consumes:
  - Tasks 1-3.
  - From v1: `frame_potential.model.{mlp, gaussians, bessel_envelope, _mic_shift, _normalize, energy_and_forces}`, `frame_potential.batch.{radius_graph, edge_vectors}`, `frame_potential.gates.{rotation_translation_error, permutation_error, wrap_error}`.
- Produces:
  - `PolyFrameNetV2(cutoff=4.5, lr_cutoff=10.0, d_atom=64, d_block=128, hidden=128, n_bonded=3, n_rbf=8, n_ch=16, d_nb=64, avg_nbr=20.0, use_coulomb=True, use_dispersion=True)`.
    - `.config` dict.
    - `.set_reference(e0: array of length N_ELEM, eV)`.
    - `.scale` buffer.
    - `.forward(batch, edges=None, lr_edges=None) -> (S,) float64`.
  - `N_ELEM = 100`.
  - `make_energy_fn_v2(model, charge=0.0, spin=1.0, topo_hook=None)`.
  - `finite_difference_error_v2(model, numbers, positions, cell=None, n_atoms=6, h=1e-5)`.
  - `variant_signatures(numbers, positions) -> dict[frozenset, list[np.ndarray]]`.
  - `drop_first_variant(topo) -> GraphTopology`.

- [ ] **Step 1: Write failing gate tests** `tests/test_v2_gates.py`

```python
"""v2 correctness gates on an untrained model (float64), across hydrocarbons, heteroatom
polymer fragments the model is never trained on, frame-free blocks and periodic boxes."""

import copy

import numpy as np
import pytest
import torch

from frame_potential import gates
from frame_potential.builders import pe_chain, pe_cluster, pe_melt
from frame_potential.v2.batch import collate_v2
from frame_potential.v2.gates import (
    drop_first_variant,
    finite_difference_error_v2,
    make_energy_fn_v2,
    variant_signatures,
)
from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.model import N_ELEM, PolyFrameNetV2
from tests.v2_molecules import HETERO, HYDROCARBONS, cluster_of, ether_chain, g2, ion_pair

MIRROR = np.diag([-1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1)
    m = PolyFrameNetV2().double()
    e0 = np.zeros(N_ELEM)
    e0[1], e0[6], e0[8] = -13.6, -1029.0, -2040.0
    m.set_reference(e0)
    return m.eval()


@pytest.fixture(scope="module")
def model_noref(model):
    m = copy.deepcopy(model)
    m.set_reference(np.zeros(N_ELEM))
    return m


@pytest.fixture(scope="module")
def systems():
    rng = np.random.default_rng(5)
    return {
        "hydrocarbons": cluster_of([g2(n) for n in HYDROCARBONS], rng),
        "hetero": cluster_of([g2(n) for n in HETERO], rng),
        "peo_ions": cluster_of([ether_chain(12), ion_pair()], rng, spacing=12.0),
        "pe_cluster": pe_cluster(3, 10, rng),
    }


@pytest.fixture(scope="module")
def melt():
    return pe_melt(4, 16, 0.35, np.random.default_rng(4))


@pytest.mark.parametrize("name", ["hydrocarbons", "hetero", "peo_ions", "pe_cluster"])
def test_rotation_permutation_reflection_cluster(model, systems, name, rng):
    z, p = systems[name]
    f = make_energy_fn_v2(model)
    assert gates.rotation_translation_error(f, z, p, None, rng) < 1e-9
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9
    assert abs(f(z, p @ MIRROR) - f(z, p)) < 1e-9


def test_periodic_rotation_permutation_reflection(model, melt, rng):
    z, p, cell = melt
    f = make_energy_fn_v2(model)
    assert gates.rotation_translation_error(f, z, p, cell, rng) < 1e-9
    assert gates.permutation_error(f, z, p, cell, rng) < 1e-9
    assert abs(f(z, p @ MIRROR, cell @ MIRROR) - f(z, p, cell)) < 1e-9


def test_wrap_periodic(model, melt, rng):
    z, p, cell = melt
    assert gates.wrap_error(make_energy_fn_v2(model), z, p, cell, rng) < 1e-9


@pytest.mark.parametrize("name", ["hetero", "peo_ions"])
def test_finite_difference_cluster(model_noref, systems, name):
    z, p = systems[name]
    assert finite_difference_error_v2(model_noref, z, p, h=1e-5) < 1e-8


def test_finite_difference_periodic(model_noref, melt):
    z, p, cell = melt
    assert finite_difference_error_v2(model_noref, z, p, cell, h=1e-5) < 1e-8


def test_frame_free_and_linear(model, rng):
    z, p = cluster_of([g2("CO2"), g2("CH3CN"), ion_pair(), g2("2-butyne")], rng)
    f = make_energy_fn_v2(model)
    assert np.isfinite(f(z, p))
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9
    assert abs(f(z, p @ MIRROR) - f(z, p)) < 1e-9


def test_charged_input_invariance(model, systems, rng):
    z, p = systems["hetero"]
    f = make_energy_fn_v2(model, charge=1.0, spin=2.0)
    f0 = make_energy_fn_v2(model)
    assert abs(f(z, p) - f0(z, p)) > 1e-6  # conditioning is live
    assert gates.rotation_translation_error(f, z, p, None, rng) < 1e-9
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9


def test_variant_set_oracle(systems, rng):
    """True oracle: the per-block multiset of variant-frame coordinates is unchanged by
    relabelling + rotation + mirror, i.e. the variant set is closed under the symmetry."""
    from frame_potential.builders import random_rotation

    for name in ("hydrocarbons", "hetero"):
        z, p = systems[name]
        ref = variant_signatures(z, p)
        perm = rng.permutation(len(z))
        Q = random_rotation(rng) @ MIRROR
        got = variant_signatures(z[perm], (p @ Q.T)[perm])
        got = {frozenset(int(perm[i]) for i in k): v for k, v in got.items()}
        assert ref.keys() == got.keys()
        for k in ref:
            assert len(ref[k]) == len(got[k])
            for a, b in zip(ref[k], got[k], strict=True):
                assert np.abs(a - b).max() < 1e-9


def test_negative_control_dropped_variant(model, systems, rng):
    z, p = systems["hydrocarbons"]
    f = make_energy_fn_v2(model, topo_hook=drop_first_variant)
    worst = max(gates.permutation_error(f, z, p, None, rng),
                abs(f(z, p @ MIRROR) - f(z, p)))
    assert worst > 1e-6


def test_cutoff_and_switch_smooth(rng):
    torch.manual_seed(2)
    z1, p1 = pe_chain(8)
    for kw, d_far in ((dict(use_coulomb=False, use_dispersion=False), 9.0),
                      (dict(use_coulomb=False, use_dispersion=True), 15.0)):
        m = PolyFrameNetV2(**kw).double().eval()
        f = make_energy_fn_v2(m)
        ds = np.linspace(4.0, d_far + 3.0, 500)
        Es = np.array([f(np.concatenate([z1, z1]), np.concatenate([p1, p1 + [0, 0, d]]))
                       for d in ds])
        d1, d2 = np.abs(np.diff(Es)), np.abs(np.diff(Es, 2))
        assert d2.max() < 0.2 * d1.max() + 1e-12
        assert np.ptp(Es[ds > d_far]) < 1e-12


def test_batch_equals_single(model, systems, melt):
    z1, p1 = systems["hetero"]
    z2, p2 = g2("C6H6")
    t1, t2 = topology_from_geometry_v2(z1, p1), topology_from_geometry_v2(z2, p2)
    s1 = float(model(collate_v2([t1], [p1]))[0])
    s2 = float(model(collate_v2([t2], [p2]))[0])
    both = model(collate_v2([t1, t2], [p1, p2 + 100.0]))
    assert abs(float(both[0]) - s1) < 1e-9 and abs(float(both[1]) - s2) < 1e-9
    z, p, cell = melt
    t = topology_from_geometry_v2(z, p, cell)
    single = float(model(collate_v2([t], [p], [cell]))[0])
    both = model(collate_v2([t, t], [p, p + 0.3], [cell, cell]))
    assert abs(float(both[0]) - single) < 1e-9 and abs(float(both[1]) - single) < 1e-9


def test_float32_matches_float64(model, systems):
    z, p = systems["hetero"]
    t = topology_from_geometry_v2(z, p)
    b = collate_v2([t], [p])
    m32 = copy.deepcopy(model).float()
    e64 = float(model(b)[0])
    out32 = m32(b.to("cpu", torch.float32))
    assert out32.dtype == torch.float64
    assert abs(float(out32[0]) - e64) < 1e-4 * len(z) ** 0.5
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_v2_gates.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `src/frame_potential/v2/model.py`

```python
"""PolyFrameNet v2: one architecture for any non-reactive polymer (spec 2026-10-21).

Stage 1  frames per variant row (c, a, b, m) from batch.var_cab; frame-free rows get F = 0
Stage 2  shared atom-set encoder: pair term (distances, invariant) + per-variant frame
         coordinates; block feature and per-atom features averaged over the block's variants
Stage 3  n_bonded layers on the block graph; message J -> I sees J's atoms in I's frame,
         averaged over I's variants
Valence  learned function of 1-2 / 1-3 / 1-4 distances, conditioned on both atoms' types
Stage 4  one radius-graph pass (4.5 A): Cartesian moments, projected per (atom, variant),
         body-order products, then averaged over variants
Stage 5  latent charges -> erf Coulomb / Ewald; learned C6 dispersion tail (10 A)
Stage 6  E = sum (eps + valence) * scale + E_es + E_disp + sum E0[Z]   (float64 sum)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..batch import edge_vectors, radius_graph
from ..model import _mic_shift, _normalize, bessel_envelope, gaussians, mlp
from .longrange import coulomb_energy, dispersion_energy, neutralize

N_ELEM = 100
_TRIU = (torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 1, 2, 1, 2, 2]))
_F = nn.functional


def _scatter_mean(x, index, n):
    out = x.new_zeros((n,) + tuple(x.shape[1:])).index_add_(0, index, x)
    cnt = torch.bincount(index, minlength=n).clamp(min=1).to(x.dtype)
    return out / cnt.view(-1, *([1] * (x.dim() - 1)))


class PolyFrameNetV2(nn.Module):
    def __init__(self, cutoff=4.5, lr_cutoff=10.0, d_atom=64, d_block=128, hidden=128,
                 n_bonded=3, n_rbf=8, n_ch=16, d_nb=64, avg_nbr=20.0,
                 use_coulomb=True, use_dispersion=True):
        super().__init__()
        self.config = dict(cutoff=cutoff, lr_cutoff=lr_cutoff, d_atom=d_atom, d_block=d_block,
                           hidden=hidden, n_bonded=n_bonded, n_rbf=n_rbf, n_ch=n_ch, d_nb=d_nb,
                           avg_nbr=avg_nbr, use_coulomb=use_coulomb,
                           use_dispersion=use_dispersion)
        self.cutoff, self.lr_cutoff, self.avg_nbr = cutoff, lr_cutoff, avg_nbr
        self.use_coulomb, self.use_dispersion = use_coulomb, use_dispersion
        self.n_rbf, self.n_ch = n_rbf, n_ch
        d_u = 32
        self.z_emb = nn.Embedding(N_ELEM, 16)
        self.u_lin = nn.Linear(16 + 7 + 1 + 5, d_u)
        self.qs_lin = nn.Linear(2, d_u)
        # Stage 2
        self.n_pair_rbf = 16
        self.pair_net = mlp([self.n_pair_rbf + 2 * d_u, hidden, d_atom])
        self.atom_net = mlp([3 + d_u + d_atom, hidden, hidden, d_atom])
        self.block_net = mlp([d_atom, hidden, d_block])
        # Stage 3
        d_geo = 32
        self.geo_net = mlp([3 + d_u, hidden, d_geo])
        self.pz = nn.Embedding(N_ELEM, 4)
        self.n_x_rbf = 16
        d_edge = d_geo + 4 * self.n_x_rbf + 2
        self.msg = nn.ModuleList(
            [mlp([2 * d_block + d_edge, hidden, hidden, d_block]) for _ in range(n_bonded)])
        self.upd = nn.ModuleList([mlp([2 * d_block, hidden, d_block]) for _ in range(n_bonded)])
        # valence
        self.val_u = nn.Linear(d_u, 8)
        self.val_net = mlp([48 + 3 + 16, 64, 64, 1])
        # Stage 4 (bias-free radial map keeps every edge term zero at the cutoff)
        d_ctx = d_atom + d_block
        self.nb_pair = nn.Embedding(N_ELEM, 8)
        self.radial = nn.Linear(n_rbf * 8, n_ch, bias=False)
        self.gate = nn.Linear(d_ctx, n_ch)
        na = n_ch * 10
        self.lin1, self.lin2a, self.lin2b = (nn.Linear(na, d_nb) for _ in range(3))
        self.lin3a, self.lin3b, self.lin3c = (nn.Linear(na, d_nb) for _ in range(3))
        # readout + Stage 5 heads
        self.z_small = nn.Embedding(N_ELEM, 16)
        d_f = d_ctx + 3 * d_nb + 16
        self.readout = mlp([d_f, hidden, hidden, 1])
        self.q_net = mlp([d_f, 64, 1])
        self.c6_net = mlp([d_f, 64, 1])
        self.r0_raw = nn.Parameter(torch.full((N_ELEM,), math.log(math.expm1(1.7))))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("c6_scale", torch.tensor(10.0))
        self.register_buffer("e0_micro", torch.zeros(N_ELEM, dtype=torch.long))
        self.register_buffer("scale", torch.ones(()))

    def set_reference(self, e0):
        self.e0_micro.copy_(torch.round(torch.as_tensor(e0, dtype=torch.float64) * 1e6).long())

    def _frames(self, batch, pos):
        cab = batch.var_cab
        free = cab[:, 0] < 0
        c, a, b = (cab[:, k].clamp(min=0) for k in range(3))
        va, vb = pos[a] - pos[c], pos[b] - pos[c]
        if batch.cell is not None:
            cv = batch.cell[batch.block_struct[batch.var_block]]
            va, vb = _mic_shift(va, cv), _mic_shift(vb, cv)
        e1 = _normalize(va)
        e2 = _normalize(vb - (vb * e1).sum(-1, keepdim=True) * e1)
        e3 = batch.var_m.to(pos.dtype)[:, None] * torch.cross(e1, e2, dim=-1)
        F = torch.stack([e1, e2, e3], -1)
        return torch.where(free[:, None, None], torch.zeros_like(F), F)

    def forward(self, batch, edges=None, lr_edges=None):
        pos = batch.positions
        dt = pos.dtype
        N, B = batch.n_atoms, batch.n_blocks
        qs = torch.stack([batch.charge, batch.spin - 1.0], -1).to(dt)
        u = self.u_lin(torch.cat([
            self.z_emb(batch.numbers),
            _F.one_hot(batch.degree.clamp(max=6), 7).to(dt),
            batch.ring.to(dt)[:, None],
            _F.one_hot(batch.n_ext.clamp(max=4), 5).to(dt)], -1)) + self.qs_lin(qs)[batch.struct]

        # Stage 1 ---------------------------------------------------------------------
        ba = batch.block_atoms
        mask = ba >= 0
        X = pos[ba.clamp(min=0)]
        if batch.cell is not None:
            X = X[:, :1] + _mic_shift(X - X[:, :1], batch.cell[batch.block_struct][:, None])
        mf = mask.to(dt)[..., None]
        o = (X * mf).sum(1) / mf.sum(1)
        F = self._frames(batch, pos)

        # Stage 2 ---------------------------------------------------------------------
        n = ba.shape[1]
        uB = u[ba.clamp(min=0)] * mf
        dv = X[:, :, None] - X[:, None]
        r = torch.sqrt((dv * dv).sum(-1) + 1e-12)
        eye = torch.eye(n, dtype=torch.bool, device=pos.device)
        pm = (mask[:, :, None] & mask[:, None, :] & ~eye).to(dt)[..., None]
        pin = torch.cat([gaussians(r, 0.8, 7.0, self.n_pair_rbf),
                         uB[:, :, None].expand(-1, -1, n, -1),
                         uB[:, None].expand(-1, n, -1, -1)], -1)
        P = (self.pair_net(pin) * pm).sum(2)
        vbk = batch.var_block
        coords = torch.einsum("rni,rij->rnj", X[vbk] - o[vbk][:, None], F)
        x = self.atom_net(torch.cat([coords, uB[vbk], P[vbk]], -1)) * mf[vbk]
        H = _scatter_mean(self.block_net(x.sum(1)), vbk, B)
        hb = _scatter_mean(x, vbk, B)
        h_atom = pos.new_zeros(N, hb.shape[-1]).index_put((ba[mask],), hb[mask])

        # Stage 3 ---------------------------------------------------------------------
        if batch.edge_src.numel():
            src, dst = batch.edge_src, batch.edge_dst
            dO = o[src] - o[dst]
            if batch.cell is not None:
                dO = _mic_shift(dO, batch.cell[batch.block_struct[dst]])
            relJ = X[src] - o[src][:, None] + dO[:, None]
            ee, ev = batch.ev_edge, batch.ev_var
            cJ = torch.einsum("qni,qij->qnj", relJ[ee], F[ev])
            g = (self.geo_net(torch.cat([cJ, uB[src][ee]], -1)) * mf[src][ee]).sum(1)
            relI = X[dst] - o[dst][:, None]
            pv = relJ[:, None, :, :] - relI[:, :, None, :]
            rx = torch.sqrt((pv * pv).sum(-1) + 1e-12)
            zI = batch.numbers[ba[dst].clamp(min=0)]
            zJ = batch.numbers[ba[src].clamp(min=0)]
            pe = self.pz(zI)[:, :, None] * self.pz(zJ)[:, None, :]
            pmx = (mask[dst][:, :, None] & mask[src][:, None, :]).to(dt)[..., None]
            pooled = torch.einsum("eabc,eabk->eck", pe * pmx,
                                  gaussians(rx, 1.0, 8.0, self.n_x_rbf)).flatten(1) / 4.0
            efeat = torch.cat([pooled, _F.one_hot(batch.edge_hop, 2).to(dt)], -1)
            for msg, upd in zip(self.msg, self.upd, strict=True):
                z = torch.cat([H[dst][ee], H[src][ee], g, efeat[ee]], -1)
                m = _scatter_mean(msg(z), ee, len(src))
                agg = torch.zeros_like(H).index_add(0, dst, m)
                H = H + upd(torch.cat([H, agg], -1))

        # Valence ---------------------------------------------------------------------
        vi, vj = batch.val_pairs[:, 0], batch.val_pairs[:, 1]
        vv = pos[vj] - pos[vi]
        if batch.cell is not None:
            vv = _mic_shift(vv, batch.cell[batch.struct[vi]])
        rv = torch.sqrt((vv * vv).sum(-1) + 1e-12)
        ui, uj = self.val_u(u[vi]), self.val_u(u[vj])
        ev_ = self.val_net(torch.cat([gaussians(rv, 0.8, 4.4, 48),
                                      _F.one_hot(batch.val_d - 1, 3).to(dt),
                                      ui + uj, ui * uj], -1)).squeeze(-1)
        val_atom = pos.new_zeros(N).index_add(0, vi, 0.5 * ev_).index_add(0, vj, 0.5 * ev_)

        # Stage 4 ---------------------------------------------------------------------
        x_atom = torch.cat([h_atom, H[batch.block_of_atom]], -1)
        i4, j4, S4 = radius_graph(batch, self.cutoff) if edges is None else edges
        vec = edge_vectors(batch, i4, j4, S4)
        r4 = torch.sqrt((vec * vec).sum(-1))
        rhat = vec / r4[:, None]
        rb = bessel_envelope(r4, self.cutoff, self.n_rbf)
        pe4 = self.nb_pair(batch.numbers[i4]) * self.nb_pair(batch.numbers[j4])
        phi = self.radial((rb[:, :, None] * pe4[:, None, :]).flatten(1)) * self.gate(x_atom)[j4]
        s = 1.0 / self.avg_nbr
        M0 = x_atom.new_zeros(N, self.n_ch).index_add(0, i4, phi) * s
        M1 = x_atom.new_zeros(N, self.n_ch, 3).index_add(
            0, i4, phi[:, :, None] * rhat[:, None, :]) * s
        M2 = x_atom.new_zeros(N, self.n_ch, 3, 3).index_add(
            0, i4, phi[:, :, None, None] * (rhat[:, :, None] * rhat[:, None, :])[:, None]) * s
        aa, av = batch.av_atom, batch.av_var
        Fa = F[av]
        P1 = torch.einsum("aci,aij->acj", M1[aa], Fa)
        P2 = torch.einsum("aik,acij,ajl->ackl", Fa, M2[aa], Fa)[:, :, _TRIU[0], _TRIU[1]]
        A = torch.cat([M0[aa][:, :, None], P1, P2], -1).flatten(1)
        Bv = torch.cat([self.lin1(A), self.lin2a(A) * self.lin2b(A),
                        self.lin3a(A) * self.lin3b(A) * self.lin3c(A)], -1)
        Bn = _scatter_mean(Bv, aa, N)

        # Stage 6 + 5 -----------------------------------------------------------------
        xf = torch.cat([x_atom, Bn, self.z_small(batch.numbers)], -1)
        eps = (self.readout(xf).squeeze(-1) + val_atom) * self.scale
        e0 = self.e0_micro.double() * 1e-6
        E = torch.zeros(batch.n_struct, dtype=torch.float64, device=pos.device)
        E = E.index_add(0, batch.struct, eps.double() + e0[batch.numbers])
        if self.use_coulomb or self.use_dispersion:
            need_lr = self.use_dispersion or batch.cell is not None
            if lr_edges is None and need_lr:
                lr_edges = radius_graph(batch, self.lr_cutoff)
            lr = pos.new_zeros(batch.n_struct)
            if self.use_coulomb:
                q = neutralize(0.1 * self.q_net(xf).squeeze(-1), batch)
                lr = lr + coulomb_energy(batch, q, lr_edges, float(self.sigma), self.lr_cutoff)
            if self.use_dispersion:
                c6 = _F.softplus(self.c6_net(xf).squeeze(-1)) * self.c6_scale
                r0 = _F.softplus(self.r0_raw)[batch.numbers]
                lr = lr + dispersion_energy(batch, c6, r0, lr_edges, self.lr_cutoff)
            E = E + lr.double()
        return E
```

- [ ] **Step 4: Implement** `src/frame_potential/v2/gates.py`

```python
"""v2 gate helpers. Symmetry checkers themselves are v1's (frame_potential.gates)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ..model import energy_and_forces
from .batch import collate_v2
from .graph import attach_rows, topology_from_geometry_v2


def make_energy_fn_v2(model, charge=0.0, spin=1.0, topo_hook=None, device="cpu"):
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
    """Per block (keyed by its atom set): sorted list of per-variant signatures, each the
    block atoms' (Z, frame coordinates) sorted lexicographically. Clusters only."""
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
                e1 /= np.linalg.norm(e1)
                v = positions[b] - positions[c]
                e2 = v - (v @ e1) * e1
                e2 /= np.linalg.norm(e2)
                Fm = np.stack([e1, e2, t.var_m[r] * np.cross(e1, e2)], -1)
            rows = np.concatenate([numbers[atoms][:, None], (X - o) @ Fm], 1)
            rows = rows[np.lexsort(np.round(rows, 6).T[::-1])]
            sigs.append(rows.ravel())
        sigs.sort(key=lambda s: tuple(np.round(s, 6)))
        out[frozenset(atoms)] = sigs
    return out


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
```

- [ ] **Step 5: Run gates, verify pass** — `pytest tests/test_v2_gates.py -q`. Expected: all PASS.

  Debugging order if a gate fails:
  1. **Rotation fails:** a projection uses the wrong side of F (check that every einsum contracts `rel @ F`, i.e. F^T r).
  2. **Permutation fails while rotation passes:** a variant set is not closed. Run `test_variant_set_oracle` first; it localises the block.
  3. **Wrap fails:** a vector is missing `_mic_shift`.
  4. **FD fails:** a non-differentiable op is in the path (`cdist`, a `no_grad` region, `torch.where` with NaN branches).

  Never loosen a tolerance to pass.

- [ ] **Step 6: Run the whole suite** — `pytest -q`. Expected: v1 23 + v2 tests all pass.

- [ ] **Step 7: Lint and commit** ("Add PolyFrameNetV2 and gate suite across chemistries").

---

### Task 5: Data pipeline (`v2/data.py`, `scripts/build_v2_dataset.py`)

**Files:**
- Create: `src/frame_potential/v2/data.py`, `scripts/build_v2_dataset.py`

**Interfaces:**
- Consumes:
  - Raw shards `C:/Users/sumedh/polyframenet_data/opoly26_ch/OPoly26-*.npz`, with keys `numbers, positions, forces, offsets, energy, multiplicity, names`.
  - v1 `frame_potential.data.{system_key, load_pe_frames, split_by_system, make_batches}`.
- Produces:
  - `FrameV2(numbers, positions, forces, energy, topo, key, charge, spin)`.
  - `DATA_CONFIGS`.
  - `load_v2_frames(raw, config, seed=0, cache=None) -> list[FrameV2]`.
  - `split_v2(frames, pe_split, seed=0) -> (train, val, test)`.
  - `pe_split_keys(raw) -> (set, set, set)`.
  - `fit_reference_general(frames) -> np.ndarray (N_ELEM,)`.
  - `collate_frames_v2(frames, ids) -> BatchV2`.

- [ ] **Step 1: Implement** `src/frame_potential/v2/data.py`

```python
"""v2 data: config-selected OPoly26 frames, system-level split. The hackathon config is
H/C only; the cluster config is everything. Same code, different config."""

from __future__ import annotations

import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..data import load_pe_frames, split_by_system, system_key
from ..topology import TopologyError
from .batch import collate_v2
from .graph import GraphTopology, topology_from_geometry_v2
from .model import N_ELEM

DATA_CONFIGS = {
    "hydrocarbon": {"elements": {1, 6}, "n_frames": 3000},
    "all": {"elements": None, "n_frames": None},
}


@dataclass
class FrameV2:
    numbers: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    energy: float
    topo: GraphTopology
    key: str
    charge: float = 0.0
    spin: float = 1.0


def _index_shard(args):
    f, elements = args
    d = np.load(f)
    off, z, names = d["offsets"], d["numbers"], d["names"]
    out = []
    for k in range(len(off) - 1):
        name = str(names[k])
        if "reactivity/" in name:
            continue
        if elements is not None and not set(np.unique(z[off[k]: off[k + 1]]).tolist()) <= elements:
            continue
        out.append((str(f), k, system_key(name)))
    return out


def _load_selected(args):
    f, ks = args
    d = np.load(f)
    off, z_all, r_all, f_all = d["offsets"], d["numbers"], d["positions"], d["forces"]
    e_all, mult, names = d["energy"], d["multiplicity"], d["names"]
    out = []
    for k in ks:
        sl = slice(off[k], off[k + 1])
        try:
            topo = topology_from_geometry_v2(z_all[sl], r_all[sl])
        except TopologyError:
            continue
        out.append(FrameV2(z_all[sl], r_all[sl], f_all[sl], float(e_all[k]), topo,
                           system_key(str(names[k])), 0.0, float(mult[k])))
    return out


def pe_split_keys(raw: pathlib.Path):
    pe = load_pe_frames(raw, raw / "pe_frames.pkl")
    return tuple({f.key for f in part} for part in split_by_system(pe, seed=0))


def load_v2_frames(raw: pathlib.Path, config: str, seed: int = 0, cache=None):
    if cache is not None and pathlib.Path(cache).exists():
        return pickle.loads(pathlib.Path(cache).read_bytes())
    cfg = DATA_CONFIGS[config]
    files = sorted(raw.glob("OPoly26-*.npz"))
    with ProcessPoolExecutor(max_workers=6) as ex:
        index = [r for part in ex.map(_index_shard, [(f, cfg["elements"]) for f in files])
                 for r in part]
    by_key: dict[str, list] = {}
    for f, k, key in index:
        by_key.setdefault(key, []).append((f, k))
    pe_keys = set().union(*pe_split_keys(raw))
    chosen = [k for k in sorted(by_key) if k in pe_keys]
    rest = sorted(set(by_key) - pe_keys)
    np.random.default_rng(seed).shuffle(rest)
    budget = cfg["n_frames"]
    total = sum(len(by_key[k]) for k in chosen)
    for k in rest:
        if budget is not None and total >= budget:
            break
        chosen.append(k)
        total += len(by_key[k])
    per_file: dict[str, list[int]] = {}
    for key in chosen:
        for f, k in by_key[key]:
            per_file.setdefault(f, []).append(k)
    with ProcessPoolExecutor(max_workers=6) as ex:
        frames = [fr for part in ex.map(_load_selected, sorted(per_file.items())) for fr in part]
    if cache is not None:
        pathlib.Path(cache).write_bytes(pickle.dumps(frames))
    return frames


def split_v2(frames, pe_split, seed: int = 0, fractions=(0.8, 0.1, 0.1)):
    """v1's PE systems keep their v1 split; every other system is split 80/10/10."""
    pe_tr, pe_va, pe_te = pe_split
    pe_all = pe_tr | pe_va | pe_te
    other = sorted({f.key for f in frames} - pe_all)
    np.random.default_rng(seed).shuffle(other)
    a = int(round(fractions[0] * len(other)))
    b = int(round((fractions[0] + fractions[1]) * len(other)))
    parts = [pe_tr | set(other[:a]), pe_va | set(other[a:b]), pe_te | set(other[b:])]
    return [[f for f in frames if f.key in p] for p in parts]


def fit_reference_general(frames):
    els = sorted({int(z) for f in frames for z in np.unique(f.numbers)})
    A = np.array([[np.sum(f.numbers == z) for z in els] for f in frames], float)
    y = np.array([f.energy for f in frames])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e0 = np.zeros(N_ELEM)
    e0[els] = coef
    return e0


def collate_frames_v2(frames, ids):
    fs = [frames[k] for k in ids]
    return collate_v2([f.topo for f in fs], [f.positions for f in fs], None,
                      [f.energy for f in fs], [f.forces for f in fs],
                      [f.charge for f in fs], [f.spin for f in fs])
```

- [ ] **Step 2: Implement** `scripts/build_v2_dataset.py`

```python
"""Build and cache the v2 dataset for a data config; print its composition."""

import collections
import pathlib
import sys
import time

import numpy as np

from frame_potential.v2.data import load_v2_frames, pe_split_keys, split_v2

if __name__ == "__main__":
    raw, config = pathlib.Path(sys.argv[1]), sys.argv[2]
    t0 = time.time()
    frames = load_v2_frames(raw, config, cache=raw / f"v2_{config}.pkl")
    tr, va, te = split_v2(frames, pe_split_keys(raw))
    n_at = sum(len(f.numbers) for f in frames)
    sizes = collections.Counter(int((row >= 0).sum()) for f in frames for row in f.topo.block_atoms)
    print(f"{len(frames)} frames, {n_at} atoms, {len({f.key for f in frames})} systems, "
          f"{time.time() - t0:.0f} s")
    print(f"split frames: train {len(tr)} val {len(va)} test {len(te)}")
    print("block sizes:", sorted(sizes.items()))
    nv = [len(f.topo.var_block) / f.topo.n_blocks for f in frames]
    print(f"variants per block: mean {np.mean(nv):.1f}")
```

- [ ] **Step 3: Run it**

Run: `python scripts/build_v2_dataset.py C:/Users/sumedh/polyframenet_data/opoly26_ch hydrocarbon`

Expected:
- About 3000 frames, of which at least 631 are PE.
- The v1 test keys appear in the test split.
- Runtime under 15 min.

Check: `python -c "..."` that every v1 PE test key appears among the v2 test frames.

- [ ] **Step 4: Lint and commit** ("Add v2 data pipeline (config-driven), build hydrocarbon set"). The message records the frame, atom and system counts and the block-size histogram.

---

### Task 6: Training, calculator, smoke run (`v2/train.py`, `v2/calculator.py`)

**Files:**
- Create: `src/frame_potential/v2/train.py`, `src/frame_potential/v2/calculator.py`
- Test: `tests/test_v2_gates.py` (append calculator test)

**Interfaces:**
- Consumes: Tasks 1-5; v1 `frame_potential.data.make_batches`, `frame_potential.train._ema_update`.
- Produces:
  - CLI `python -m frame_potential.v2.train --raw R --config hydrocarbon --out O --minutes M ...`, writing `best.pt`, `last.pt`, `history.json` and `results.json` (the last includes `pe_test` metrics).
  - `evaluate_v2(model, frames, device, max_atoms=4000, dtype=float64) -> dict`.
  - `load_model_v2(path, device, dtype) -> PolyFrameNetV2`.
  - `PolyFrameNetV2Calculator(model, device, dtype)`.

- [ ] **Step 1: Append failing calculator test**

```python
def test_calculator_matches_model(model, melt):
    from ase import Atoms

    from frame_potential.model import energy_and_forces
    from frame_potential.v2.calculator import PolyFrameNetV2Calculator

    z, p, cell = melt
    at = Atoms(numbers=z, positions=p, cell=cell, pbc=True)
    at.calc = PolyFrameNetV2Calculator(copy.deepcopy(model), "cpu", torch.float64)
    t = topology_from_geometry_v2(z, p, cell)
    E, F = energy_and_forces(model, collate_v2([t], [p], [cell]))
    assert abs(at.get_potential_energy() - float(E[0])) < 1e-9
    assert np.abs(at.get_forces() - F.detach().numpy()).max() < 1e-9
```

- [ ] **Step 2: Implement** `src/frame_potential/v2/calculator.py`

```python
"""ASE calculator for v2. Topology read once (non-reactive), energy recomputed every call."""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ..model import energy_and_forces
from .batch import collate_v2
from .graph import topology_from_geometry_v2


class PolyFrameNetV2Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, device="cuda", dtype=torch.float32, charge=0.0, spin=1.0, **kw):
        super().__init__(**kw)
        self.model = model.to(device, dtype).eval()
        self.device, self.dtype = device, dtype
        self.charge, self.spin = charge, spin
        self.topo = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        cell = np.asarray(atoms.cell) if atoms.pbc.any() else None
        if self.topo is None or not np.array_equal(self.topo.numbers, atoms.numbers):
            self.topo = topology_from_geometry_v2(atoms.numbers, pos, cell)
        b = collate_v2([self.topo], [pos], None if cell is None else [cell],
                       charges=[self.charge], spins=[self.spin]).to(self.device, self.dtype)
        E, F = energy_and_forces(self.model, b)
        self.results["energy"] = float(E.detach().sum())
        self.results["forces"] = F.detach().double().cpu().numpy()
```

- [ ] **Step 3: Implement** `src/frame_potential/v2/train.py`

```python
"""v2 training loop: same objective, optimiser and schedule as v1 (frame_potential.train),
v2 model and data. --config selects the data; nothing else changes between the hackathon
run and a full-OPoly26 run."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import time

import numpy as np
import torch

from ..batch import radius_graph
from ..data import make_batches
from ..model import energy_and_forces
from ..train import _ema_update
from .data import (
    collate_frames_v2,
    fit_reference_general,
    load_v2_frames,
    pe_split_keys,
    split_v2,
)
from .model import PolyFrameNetV2


def evaluate_v2(model, frames, device, max_atoms=4000, dtype=torch.float64):
    m = copy.deepcopy(model).to(dtype).eval()
    e_err, f_err = [], []
    for ids in make_batches(frames, max_atoms):
        b = collate_frames_v2(frames, ids).to(device, dtype)
        E, F = energy_and_forces(m, b)
        n = torch.bincount(b.struct, minlength=b.n_struct).double()
        e_err.append(((E.detach() - b.energy) / n).cpu().numpy())
        f_err.append((F.detach() - b.forces).cpu().numpy().ravel())
    e = np.concatenate(e_err) * 1000
    f = np.concatenate(f_err) * 1000
    return {"e_mae_meV_atom": float(np.abs(e).mean()),
            "e_rmse_meV_atom": float(np.sqrt((e**2).mean())),
            "f_mae_meV_A": float(np.abs(f).mean()),
            "f_rmse_meV_A": float(np.sqrt((f**2).mean())), "n_frames": len(frames)}


def load_model_v2(path, device="cpu", dtype=torch.float64):
    ck = torch.load(path, map_location=device)
    m = PolyFrameNetV2(**ck["config"])
    m.load_state_dict(ck["state"])
    return m.to(device, dtype).eval()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=pathlib.Path, required=True)
    p.add_argument("--config", default="hydrocarbon")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--max-epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--max-atoms", type=int, default=1000)
    p.add_argument("--w-energy", type=float, default=20.0)
    p.add_argument("--w-forces", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--max-train-frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; refusing to overwrite a finished run")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = load_v2_frames(a.raw, a.config, cache=a.raw / f"v2_{a.config}.pkl")
    pe_split = pe_split_keys(a.raw)
    train, val, test = split_v2(frames, pe_split, seed=a.seed)
    if a.max_train_frames:
        train = train[: a.max_train_frames]
    pe_test = [f for f in test if f.key in pe_split[2]]
    print(f"frames {len(frames)}: train {len(train)} val {len(val)} test {len(test)} "
          f"(PE test {len(pe_test)})", flush=True)

    model = PolyFrameNetV2()
    model.set_reference(fit_reference_general(train))
    f_rms = float(np.sqrt(np.mean(np.concatenate([f.forces.ravel() for f in train]) ** 2)))
    model.scale.fill_(f_rms)
    tot_e = tot_a = 0
    for f in train[:50]:
        b = collate_frames_v2([f], [0])
        tot_e += len(radius_graph(b, model.cutoff)[0])
        tot_a += b.n_atoms
    model.avg_nbr = model.config["avg_nbr"] = max(tot_e / tot_a, 1.0)
    model.to(device, torch.float32)
    ema = copy.deepcopy(model)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"params {n_params}, force rms {f_rms:.3f}, avg nbr {model.avg_nbr:.1f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6, amsgrad=True)
    rng = np.random.default_rng(a.seed)
    t0, budget, best, history, step = time.time(), a.minutes * 60, math.inf, [], 0
    for epoch in range(a.max_epochs):
        model.train()
        frac = min((time.time() - t0) / budget, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        batches = make_batches(train, a.max_atoms, rng)
        tl = 0.0
        for ids in batches:
            b = collate_frames_v2(train, ids).to(device, torch.float32)
            E, F = energy_and_forces(model, b, create_graph=True)
            n = torch.bincount(b.struct, minlength=b.n_struct).double()
            loss = (a.w_energy * (((E - b.energy) / n) ** 2).mean().float()
                    + a.w_forces * ((F - b.forces) ** 2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            _ema_update(ema, model, a.ema)
            tl += float(loss)
            step += 1
            if time.time() - t0 > budget:
                break
        elapsed = time.time() - t0
        if epoch % 2 == 0 or elapsed > budget:
            mtr = evaluate_v2(ema, val, device)
            mtr.update(epoch=epoch, step=step, minutes=elapsed / 60, train_loss=tl / len(batches))
            history.append(mtr)
            score = mtr["e_rmse_meV_atom"] + mtr["f_rmse_meV_A"] / 20
            tag = ""
            if score < best:
                best = score
                torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "best.pt")
                tag = " *"
            print(f"ep {epoch:4d} {elapsed / 60:6.1f} min loss {tl / len(batches):.3e} "
                  f"val E {mtr['e_mae_meV_atom']:.2f} F {mtr['f_mae_meV_A']:.1f}{tag}", flush=True)
            (a.out / "history.json").write_text(json.dumps(history, indent=1))
        if elapsed > budget:
            break
    torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "last.pt")
    bm = load_model_v2(a.out / "best.pt", device)
    res = {"val": evaluate_v2(bm, val, device), "test": evaluate_v2(bm, test, device),
           "pe_test": evaluate_v2(bm, pe_test, device), "n_params": n_params,
           "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
```

The evaluation interval is every 2 epochs because v2 epochs are long.

- [ ] **Step 4: Run tests** — `pytest -q`. Expected: all pass, including the calculator test.

- [ ] **Step 5: Smoke run (3 min)**

Run: `python -m frame_potential.v2.train --raw C:/Users/sumedh/polyframenet_data/opoly26_ch --out C:/Users/sumedh/polyframenet_data/runs/v2_smoke --minutes 3 --max-train-frames 200`

Expected:
- The loss decreases.
- No NaN.
- `results.json` is written.
- The seconds per epoch on 200 frames is recorded and used to choose `--max-atoms` and confirm the 90-min budget gives at least 20 epochs on the full train set.
- If fewer than 20 epochs: lower `n_frames` in `DATA_CONFIGS["hydrocarbon"]`. Record the decision in the commit message.

- [ ] **Step 6: Lint and commit** ("Add v2 training loop and ASE calculator"). The message records smoke timing.

---

### Task 6b: Block registry (training-coverage dictionary)

**Files:**
- Create: `src/frame_potential/v2/registry.py`, `scripts/block_coverage.py`, `tests/test_v2_registry.py`
- Modify:
  - `src/frame_potential/v2/model.py`: `forward(self, batch, edges=None, lr_edges=None, capture=None)`. Right after Stage 2 computes `H`, add `if capture is not None: capture["H2"] = H.detach()`.
  - `src/frame_potential/v2/train.py`: after training, build the registry from `train` and save it as `registry.json`.
  - `src/frame_potential/v2/calculator.py`: optional `registry=` argument. On a new topology it warns, listing any `novel` blocks.

**Interfaces:**
- Consumes: `GraphTopology`, `_h` (graph.py), `PolyFrameNetV2(capture=...)`, `collate_v2`, v1 `make_batches`.
- Produces:
  - `block_hashes(topo) -> np.ndarray (B, 3) int64` (L0, L1, L2);
  - `block_label(topo, k) -> str`;
  - `BlockRegistry.build(model, frames, device) -> BlockRegistry`;
  - `.report(model, topo, positions, cell=None, device="cpu") -> list[dict]`, with keys `block, label, level, count, verdict, nearest, distance`;
  - `.save(path)`, `BlockRegistry.load(path)`.

- [ ] **Step 1: Failing tests** `tests/test_v2_registry.py`

```python
import numpy as np
import torch

from frame_potential.builders import pe_chain, pe_cluster
from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.model import PolyFrameNetV2
from frame_potential.v2.registry import BlockRegistry, block_hashes
from tests.v2_molecules import g2


class _F:  # minimal frame for registry building
    def __init__(self, z, p):
        self.numbers, self.positions, self.topo = z, p, topology_from_geometry_v2(z, p)
        self.forces, self.energy, self.charge, self.spin = np.zeros_like(p), 0.0, 0.0, 1.0


def test_hashes_permutation_invariant(rng):
    z, p = g2("isobutane")
    t = topology_from_geometry_v2(z, p)
    perm = rng.permutation(len(z))
    t2 = topology_from_geometry_v2(z[perm], p[perm])
    assert sorted(map(tuple, block_hashes(t).tolist())) == sorted(
        map(tuple, block_hashes(t2).tolist()))


def test_registry_seen_and_novel(tmp_path):
    torch.manual_seed(0)
    m = PolyFrameNetV2().double().eval()
    rng = np.random.default_rng(0)
    frames = [_F(*pe_cluster(2, 8, rng)) for _ in range(3)]
    reg = BlockRegistry.build(m, frames, "cpu")
    reg.save(tmp_path / "r.json")
    reg = BlockRegistry.load(tmp_path / "r.json")
    z, p = pe_chain(6)
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    assert {r["verdict"] for r in rep} == {"seen"}
    z, p = g2("isobutane")  # 3 CH3 (seen in PE: C + 3H, one external C) and 1 CH (3 ext C)
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    verdicts = sorted(r["verdict"] for r in rep)
    assert verdicts.count("seen") == 3 and any(v != "seen" for v in verdicts)
    z, p = g2("C6H6")
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    assert rep[0]["level"] is None and rep[0]["verdict"] != "seen"
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_v2_registry.py -q`. Expected: ImportError.

- [ ] **Step 3: Implement** `src/frame_potential/v2/registry.py`

```python
"""Block registry: which blocks the training data contained, for flagging unfamiliar
chemistry at inference. Diagnostic only: the model never reads it (shared weights, never
shared values). Block identity is computed inside the block, independent of context:
  L0 composition  multiset of (Z, sorted Z of external neighbours)
  L1 local        one WL round over the block's internal bonds
  L2 exact        WL over internal bonds refined to stability
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import torch
from ase.data import chemical_symbols

from ..data import make_batches
from .batch import collate_v2
from .graph import _h

LEVELS = ("L0", "L1", "L2")
_KEEP = 200  # embeddings kept per exact class for spread statistics


def _adjacency(topo):
    nb = [[] for _ in range(topo.n_atoms)]
    for a, b in topo.bonds.tolist():
        nb[a].append(b)
        nb[b].append(a)
    return nb


def block_hashes(topo) -> np.ndarray:
    nb = _adjacency(topo)
    out = np.zeros((topo.n_blocks, 3), dtype=np.int64)
    for k in range(topo.n_blocks):
        atoms = [int(a) for a in topo.block_atoms[k] if a >= 0]
        inside = set(atoms)
        col = {i: _h(int(topo.numbers[i]),
                     tuple(sorted(int(topo.numbers[j]) for j in nb[i] if j not in inside)))
               for i in atoms}
        out[k, 0] = _h(tuple(sorted(col.values())))
        n_cls = len(set(col.values()))
        for rnd in range(20):
            col = {i: _h(col[i], tuple(sorted(col[j] for j in nb[i] if j in inside)))
                   for i in atoms}
            if rnd == 0:
                out[k, 1] = _h(tuple(sorted(col.values())))
            k_new = len(set(col.values()))
            if rnd > 0 and k_new == n_cls:
                break
            n_cls = k_new
        out[k, 2] = _h(tuple(sorted(col.values())))
    return out


def block_label(topo, k) -> str:
    nb = _adjacency(topo)
    atoms = [int(a) for a in topo.block_atoms[k] if a >= 0]
    inside = set(atoms)
    comp = Counter(chemical_symbols[int(topo.numbers[i])] for i in atoms)
    ext = Counter(chemical_symbols[int(topo.numbers[j])] for i in atoms for j in nb[i]
                  if j not in inside)
    f = "".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(comp.items()))
    e = ",".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(ext.items()))
    return f"{f}[{e}]" if e else f


def _embeddings(model, topos, positions, device, cells=None):
    b = collate_v2(topos, positions, cells).to(device, torch.float64)
    cap = {}
    m = model.to(device, torch.float64).eval()
    with torch.no_grad():
        m(b, capture=cap)
    return cap["H2"].cpu().numpy()


class BlockRegistry:
    def __init__(self, counts, labels, means, spread, threshold):
        self.counts, self.labels, self.means = counts, labels, means
        self.spread, self.threshold = spread, threshold

    @classmethod
    def build(cls, model, frames, device, max_atoms=4000):
        counts = {lv: Counter() for lv in LEVELS}
        labels, samples = {}, {}
        for ids in make_batches(frames, max_atoms):
            fs = [frames[i] for i in ids]
            H = _embeddings(model, [f.topo for f in fs], [f.positions for f in fs], device)
            row = 0
            for f in fs:
                hs = block_hashes(f.topo)
                for k in range(f.topo.n_blocks):
                    for li, lv in enumerate(LEVELS):
                        counts[lv][str(hs[k, li])] += 1
                    key = str(hs[k, 2])
                    labels.setdefault(key, block_label(f.topo, k))
                    s = samples.setdefault(key, [])
                    if len(s) < _KEEP:
                        s.append(H[row + k])
                row += f.topo.n_blocks
        means = {k: np.mean(v, 0) for k, v in samples.items()}
        rms = [float(np.sqrt(((np.asarray(v) - means[k]) ** 2).sum(1).mean()))
               for k, v in samples.items() if len(v) > 1]
        spread = float(np.median(rms)) if rms and np.median(rms) > 0 else 1.0
        within = [np.linalg.norm(x - means[k]) / spread for k, v in samples.items() for x in v]
        threshold = float(np.percentile(within, 95)) if within else 1.0
        return cls({lv: dict(c) for lv, c in counts.items()}, labels, means, spread, threshold)

    def report(self, model, topo, positions, cell=None, device="cpu"):
        H = _embeddings(model, [topo], [positions], device, None if cell is None else [cell])
        hs = block_hashes(topo)
        keys = list(self.means)
        M = np.stack([self.means[k] for k in keys]) if keys else np.zeros((0, H.shape[1]))
        out = []
        for k in range(topo.n_blocks):
            level, count = None, 0
            for li in (2, 1, 0):
                c = self.counts[LEVELS[li]].get(str(hs[k, li]), 0)
                if c:
                    level, count = LEVELS[li], c
                    break
            d = np.linalg.norm(M - H[k], axis=1) / self.spread if len(M) else np.array([np.inf])
            j = int(np.argmin(d))
            if level == "L2":
                verdict = "seen"
            elif level == "L1" or float(d[j]) < self.threshold:
                verdict = "similar"
            else:
                verdict = "novel"
            out.append({"block": k, "label": block_label(topo, k), "level": level,
                        "count": count, "verdict": verdict,
                        "nearest": self.labels[keys[j]] if keys else None,
                        "distance": float(d[j])})
        return out

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps({
            "counts": self.counts, "labels": self.labels,
            "means": {k: v.tolist() for k, v in self.means.items()},
            "spread": self.spread, "threshold": self.threshold}))

    @classmethod
    def load(cls, path):
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["counts"], d["labels"], {k: np.asarray(v) for k, v in d["means"].items()},
                   d["spread"], d["threshold"])
```

- [ ] **Step 4: Implement** `scripts/block_coverage.py`

```python
"""Print per-block training coverage for a structure: seen / similar / novel.
Usage: python scripts/block_coverage.py <run dir with best.pt + registry.json> <structure file>"""

import pathlib
import sys

import numpy as np
from ase.io import read

from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.registry import BlockRegistry
from frame_potential.v2.train import load_model_v2

if __name__ == "__main__":
    run, struct = pathlib.Path(sys.argv[1]), sys.argv[2]
    at = read(struct)
    cell = np.asarray(at.cell) if at.pbc.any() else None
    model = load_model_v2(run / "best.pt")
    reg = BlockRegistry.load(run / "registry.json")
    topo = topology_from_geometry_v2(at.numbers, at.positions, cell)
    rows = reg.report(model, topo, at.positions, cell)
    for r in rows:
        print(f"{r['block']:5d} {r['label']:16s} {r['verdict']:8s} level={r['level']} "
              f"n_train={r['count']} nearest={r['nearest']} d={r['distance']:.2f}")
    print({v: sum(r["verdict"] == v for r in rows) for v in ("seen", "similar", "novel")})
```

- [ ] **Step 5: Wire into train and calculator.**
  - In `v2/train.py` `main`, after `bm = load_model_v2(...)`: `BlockRegistry.build(bm, train, device).save(a.out / "registry.json")`.
  - In the calculator `__init__`, add `registry=None`.
  - In `calculate`, when the topology is rebuilt and a registry is set: compute `report`, and if any verdict is `novel`, `warnings.warn(f"{n} novel block types: {labels}")`.

- [ ] **Step 6: Run** `pytest -q`. All pass. Lint. Commit ("Add block registry: training coverage per block at inference").

---

### Task 7: Exports, baseline environments, trainings in two rounds

**Files:**
- Create:
  - `scripts/export_v2_extxyz.py`;
  - `scripts/export_melts.py`;
  - `scripts/eval_ase.py`;
  - `scripts/bench_ase.py`;
  - `baselines/deepmd/input.json`;
  - `baselines/allegro/README.md` (the exact config and commands used).

**Interfaces:**
- `$D = C:/Users/sumedh/polyframenet_data`.
- Produces:
  - `$D/extxyz_v2/{train,val,test,pe_test}.xyz`, `e0.json`;
  - `$D/melts/melt_r{1,2,3}.xyz`, `melt_nve.xyz`;
  - `$D/deepmd_v2/{train,val}/` (dpdata npy);
  - per-model JSON metrics in `$D/runs/eval_v2/`.

- [ ] **Step 1: v2 extxyz export.** `scripts/export_v2_extxyz.py`:

```python
"""Exact v2 split as extxyz for all baselines, plus the shared E0."""

import json
import pathlib
import sys

from frame_potential.data import to_extxyz
from frame_potential.v2.data import fit_reference_general, load_v2_frames, pe_split_keys, split_v2

if __name__ == "__main__":
    raw, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    frames = load_v2_frames(raw, "hydrocarbon", cache=raw / "v2_hydrocarbon.pkl")
    pe_split = pe_split_keys(raw)
    tr, va, te = split_v2(frames, pe_split)
    for name, part in (("train", tr), ("val", va), ("test", te),
                       ("pe_test", [f for f in te if f.key in pe_split[2]])):
        to_extxyz(part, out / f"{name}.xyz")
    e0 = fit_reference_general(tr)
    (out / "e0.json").write_text(json.dumps({"1": e0[1], "6": e0[6]}))
    print({k: len(v) for k, v in (("train", tr), ("val", va), ("test", te))}, e0[1], e0[6])
```

- [ ] **Step 2: Melt export.** `scripts/export_melts.py` writes the md_bench melts as extxyz, so every environment benchmarks identical atoms:

```python
"""Write the benchmark PE melts (md_bench.melt_atoms, tiled 1x/2x/3x) as extxyz."""

import pathlib
import sys

import numpy as np
from ase import Atoms
from ase.io import write

sys.path.insert(0, "scripts")
from md_bench import melt_atoms  # noqa: E402

from frame_potential.builders import tile  # noqa: E402

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base = melt_atoms(8, 40, 0.50, seed=1)
    for r in (1, 2, 3):
        z, p, c = tile(base.numbers, base.positions, np.asarray(base.cell), r)
        write(out / f"melt_r{r}.xyz", Atoms(numbers=z, positions=p, cell=c, pbc=True))
    write(out / "melt_nve.xyz", melt_atoms(8, 40, 0.50, seed=2))
```

- [ ] **Step 3: Generic evaluator.** `scripts/eval_ase.py` depends only on numpy, ase and the model's own package, so it runs inside any baseline environment:

```python
"""Accuracy of any ASE calculator on an extxyz split with REF_energy / REF_forces.
Usage: python scripts/eval_ase.py <kind> <model> <split.xyz> <out.json>
kind: v2 | mace | allegro | deepmd | uma. Runs inside that model's own environment."""

import json
import sys

import numpy as np
from ase.io import read


def make_calc(kind, model):
    if kind == "v2":
        import torch

        from frame_potential.v2.calculator import PolyFrameNetV2Calculator
        from frame_potential.v2.train import load_model_v2
        return PolyFrameNetV2Calculator(load_model_v2(model, "cuda"), "cuda", torch.float64)
    if kind == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=model, device="cuda", default_dtype="float64")
    if kind == "allegro":
        from nequip.ase import NequIPCalculator
        return NequIPCalculator.from_compiled_model(model, device="cuda")
    if kind == "deepmd":
        from deepmd.calculator import DP
        return DP(model=model)
    if kind == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device="cuda"),
                                  task_name="omol")
    raise ValueError(kind)


if __name__ == "__main__":
    kind, model, split, out = sys.argv[1:5]
    calc = make_calc(kind, model)
    e_err, f_err = [], []
    for at in read(split, ":"):
        e_ref, f_ref = at.info["REF_energy"], at.arrays["REF_forces"]
        at.info["charge"], at.info["spin"] = 0, 1  # UMA omol inputs; ignored by others
        at.calc = calc
        e_err.append((at.get_potential_energy() - e_ref) / len(at))
        f_err.append((at.get_forces() - f_ref).ravel())
    e = np.asarray(e_err) * 1000
    f = np.concatenate(f_err) * 1000
    res = {"kind": kind, "model": model, "split": split, "n_frames": len(e_err),
           "e_mae_meV_atom": float(np.abs(e).mean()),
           "e_rmse_meV_atom": float(np.sqrt((e**2).mean())),
           "f_mae_meV_A": float(np.abs(f).mean()), "f_rmse_meV_A": float(np.sqrt((f**2).mean()))}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(res)
```

The `allegro` and `deepmd` constructors must be checked against the installed API in Step 7. If a name differs, fix it there and record the change in `baselines/*/README.md`.

- [ ] **Step 4: Generic speed and NVE benchmark.** `scripts/bench_ase.py` runs inside each environment. It uses the same timing rule as `md_bench.time_calc` (3 warm-ups with 1e-4 Å rattles, then the median of 20) and the same NVE (450 K, velocity Verlet, dt 0.5 fs, 6000 steps):

```python
"""Speed / NVE for any ASE calculator on exported melts, same protocol as md_bench.py.
Usage: python scripts/bench_ase.py <kind> <model> <structure.xyz> <speed|nve> <out.json>"""

import json
import sys
import time

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, "scripts")
from eval_ase import make_calc  # noqa: E402


def speed(calc, atoms):
    atoms.calc = calc
    for _ in range(3):
        atoms.rattle(1e-4)
        atoms.get_forces()
    ts = []
    for _ in range(20):
        atoms.rattle(1e-4)
        t = time.perf_counter()
        atoms.get_forces()
        ts.append(time.perf_counter() - t)
    return {"n_atoms": len(atoms), "s_per_step": float(np.median(ts))}


def nve(calc, atoms, steps=6000, dt_fs=0.5):
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=450, rng=np.random.default_rng(0))
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    t, e = [], []
    for k in range(steps // 10):
        dyn.run(10)
        t.append((k + 1) * 10 * dt_fs / 1000)
        e.append(atoms.get_total_energy() / len(atoms))
    t, e = np.asarray(t), np.asarray(e)
    return {"steps": steps, "drift_meV_atom_ps": float(np.polyfit(t, e, 1)[0] * 1000),
            "e_std_meV_atom": float(np.std(e) * 1000), "trace_t_ps": t.tolist(),
            "trace_e_eV_atom": e.tolist()}


if __name__ == "__main__":
    kind, model, struct, task, out = sys.argv[1:6]
    import torch

    atoms = read(struct)
    calc = make_calc(kind, model)
    torch.cuda.reset_peak_memory_stats()
    try:
        res = speed(calc, atoms) if task == "speed" else nve(calc, atoms)
    except Exception as exc:
        res = {"n_atoms": len(atoms), "s_per_step": None, "error": str(exc)[:300]}
    res.update(kind=kind, peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
    with open(out, "w") as fh:
        json.dump(res, fh)
    print({k: v for k, v in res.items() if not k.startswith("trace")})
```

`make_calc("v2")` builds the calculator in float64 for evaluation. For speed and NVE, `bench_ase.py` uses `kind == "v2"` with float32: pass `torch.float32` there, matching v1 and MACE, which ran in float32. The energy sum stays float64.

- [ ] **Step 5: Run exports and commit before launching anything long.**

```bash
python scripts/export_v2_extxyz.py $D/opoly26_ch $D/extxyz_v2
python scripts/export_melts.py $D/melts
```

Commit ("Add v2 exports and generic ASE eval/bench scripts").

- [ ] **Step 6: Round 1 trainings, in the background, 90 min each.**
  - **v2:** `python -m frame_potential.v2.train --raw $D/opoly26_ch --out $D/runs/v2_main --minutes 90 > $D/runs/v2_main.log 2>&1`
  - **MACE:** run in `$D/runs/mace_v2`. Use the v1 MACE command from the writeup's Reproduce section, with:
    - the `extxyz_v2` files;
    - `--E0s` from `e0.json`;
    - `--max_num_epochs` from a 1-epoch probe, so the wall clock is about 90 min.

- [ ] **Step 7: During round 1, install the baselines.** Each environment is isolated and time-boxed to 30 min:

```powershell
uv venv --python 3.12 $HOME\venvs\bl_allegro;  & $HOME\venvs\bl_allegro\Scripts\python -m pip install nequip-allegro ase
uv venv --python 3.12 $HOME\venvs\bl_deepmd;   & $HOME\venvs\bl_deepmd\Scripts\python -m pip install "deepmd-kit[torch]" dpdata ase
uv venv --python 3.12 $HOME\venvs\bl_uma;      & $HOME\venvs\bl_uma\Scripts\python -m pip install fairchem-core ase
```

After each install, check that it imports and sees the GPU:

```powershell
& $HOME\venvs\bl_<name>\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If torch is CPU-only, reinstall the matching CUDA wheel. On an App Control block, record the exact error and drop that baseline.

- **DeepMD data.** In `bl_deepmd`, convert the extxyz split with dpdata:

```python
import dpdata
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

D = "C:/Users/sumedh/polyframenet_data"
for split in ("train", "val"):
    ms = dpdata.MultiSystems()
    for at in read(f"{D}/extxyz_v2/{split}.xyz", ":"):
        at.calc = SinglePointCalculator(at, energy=at.info["REF_energy"],
                                        forces=at.arrays["REF_forces"])
        ms.append(dpdata.LabeledSystem(at, fmt="ase/structure"))
    ms.to_deepmd_npy(f"{D}/deepmd_v2/{split}")
```

- **DeepMD config.** `baselines/deepmd/input.json`:

```json
{
  "model": {
    "type_map": ["H", "C"],
    "descriptor": {"type": "se_e2_a", "rcut": 6.0, "rcut_smth": 0.5, "sel": [60, 30],
                   "neuron": [25, 50, 100], "axis_neuron": 16},
    "fitting_net": {"neuron": [240, 240, 240]}
  },
  "learning_rate": {"type": "exp", "start_lr": 1e-3, "stop_lr": 3e-8, "decay_steps": 5000},
  "loss": {"type": "ener", "start_pref_e": 0.02, "limit_pref_e": 1, "start_pref_f": 1000,
           "limit_pref_f": 1},
  "training": {
    "training_data": {"systems": "DEEPMD_V2/train", "batch_size": "auto"},
    "validation_data": {"systems": "DEEPMD_V2/val", "batch_size": 1},
    "numb_steps": 400000, "disp_freq": 1000, "save_freq": 10000
  }
}
```

- `DEEPMD_V2` is replaced with `$D/deepmd_v2` at launch. dpdata's MultiSystems writes one subfolder per formula, so `systems` takes the list of those subfolders.
- `numb_steps` is set from a 2-min step-rate probe to fill 90 min.
- Train with `python -m deepmd --pt train input.json`, then freeze with `python -m deepmd --pt freeze -o model.pth`.
- DeepMD's own prefactor loss schedule is used instead of a fixed E 20 / F 1. That is a disclosed deviation, because the schedule is part of DeepMD's recommended recipe.

- **Allegro config.** Start from the tutorial config shipped with nequip-allegro. Locate it from the installed package (`python -c "import allegro, pathlib; print(pathlib.Path(allegro.__file__).parent)"`, then search for `*.yaml`). Set:
  - the data to the `extxyz_v2` files, with key mapping `REF_energy` → total energy and `REF_forces` → forces;
  - chemical symbols [H, C];
  - per-type energy shifts from `e0.json`;
  - r_max 5.0;
  - loss: per-atom energy MSE × 20 + force MSE × 1;
  - a 90-min wall-clock limit.

  Write the final config and the exact train and compile commands into `baselines/allegro/README.md` and commit it. Compile the best checkpoint for the ASE calculator as the package documents, and fix `make_calc("allegro")` to match.

- **UMA access.** The user must have accepted the facebook/UMA license and run `huggingface-cli login`. Verify with `python -c "from fairchem.core import pretrained_mlip; pretrained_mlip.get_predict_unit('uma-s-1p1', device='cuda')"`. If the model name differs in the installed fairchem, use the one listed by `pretrained_mlip.available_models`.

- **Smoke tests.** Allegro and DeepMD get a 2-min smoke train. UMA gets a 5-frame eval with `eval_ase.py` on `pe_test.xyz`.

- [ ] **Step 8: Round 2 trainings (Allegro and DeepMD, 90 min each, concurrent)** after round 1 finishes. Commit the configs first.

---

### Task 8: Evaluation, benchmarks, writeup

**Files:**
- Modify: `docs/writeup.md` (new §8), `README.md` (v2 layout rows, baselines folder)
- Create: `scripts/eval_v2_all.py`, `docs/results/v2/`

- [ ] **Step 1: Accuracy.** For each model, run in its own environment:

```bash
python scripts/eval_ase.py <kind> <model> $D/extxyz_v2/test.xyz    $D/runs/eval_v2/<kind>_test.json
python scripts/eval_ase.py <kind> <model> $D/extxyz_v2/pe_test.xyz $D/runs/eval_v2/<kind>_pe.json
```

- `<kind>` is each of v2, mace, allegro, deepmd and uma.
- Also run v1 (trained on PE only) on its own PE test set with `frame_potential.train.evaluate`. Write it to `$D/runs/eval_v2/v1_pe.json` in the same format.

- [ ] **Step 2: `scripts/eval_v2_all.py`.**

```python
"""Merge per-model eval JSONs into one table (markdown + json)."""

import json
import pathlib
import sys

if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1])
    rows = {}
    for f in sorted(d.glob("*_*.json")):
        kind, split = f.stem.rsplit("_", 1)
        rows.setdefault(kind, {})[split] = json.loads(f.read_text())
    (d / "all.json").write_text(json.dumps(rows, indent=1))
    nan = float("nan")
    print("| model | test E MAE | test F MAE | PE E MAE | PE F MAE |\n|---|---|---|---|---|")
    for k, v in rows.items():
        t, p = v.get("test", {}), v.get("pe", {})
        print(f"| {k} | {t.get('e_mae_meV_atom', nan):.3f} | {t.get('f_mae_meV_A', nan):.2f} | "
              f"{p.get('e_mae_meV_atom', nan):.3f} | {p.get('f_mae_meV_A', nan):.2f} |")
```

- [ ] **Step 3: Speed and NVE.** For each model, run `bench_ase.py` with speed on `melt_r1/2/3.xyz` and with nve on `melt_nve.xyz`.
  - One process per measurement.
  - One measurement on the GPU at a time.
  - Nothing else training at the same time.
  - Also time v2 once with `use_coulomb=False`, to report the Ewald share. Load the checkpoint, set `model.use_coulomb = False`, and time it in a short inline script that reuses `bench_ase.speed`.
  - v1 and MACE speeds are reused from `docs/results/md/`, which used the same protocol.
- [ ] **Step 4: Block registry demo.** Run `scripts/block_coverage.py` on:
  - a PE melt (all seen);
  - an aromatic test cluster;
  - the heteroatom g2 cluster written to xyz (novel blocks flagged).

  Put the outputs in the writeup.
- [ ] **Step 5: Writeup §8.** Cover:
  - the v2 design and data composition;
  - the gate table, including the heteroatom molecules and the oracle;
  - the registry;
  - an accuracy table with all models on test and on PE test;
  - speed and memory per model;
  - NVE per model.

  Disclose, for fairness:
  - the 90-min budgets, with GPU sharing in pairs;
  - single seeds;
  - DeepMD's own loss schedule;
  - UMA zero-shot, with possible contamination;
  - MACE without cuEquivariance;
  - any baseline that failed to install, with its error.

  Run an independent fact-check agent and fix its findings.
- [ ] **Step 6: Commit and push** ("Add v2 results: accuracy, speed, NVE vs MACE, Allegro, DeepMD, UMA").

---

## Self-review notes (done while writing)

- **Spec coverage:**

  | Spec section | Task |
  |---|---|
  | Stage 0a/0b | Task 1 |
  | Stage 1 variants | Task 1 (table) and Task 4 (frames) |
  | Stage 2/3/4/valence/readout | Task 4 |
  | Stage 5 | Task 3 and Task 4 |
  | Charge/spin conditioning | Task 4 (`qs_lin`, test) |
  | Data configs | Task 5 |
  | Training | Task 6 |
  | MACE baseline | Task 7 |
  | Gates incl. oracle and negative control | Task 4 |
  | Block registry | Task 6b |
  | Baselines: Allegro, DeepMD, UMA | Task 7 (environments, configs, two training rounds) and Task 8 (evaluation) |
  | Benchmarks and writeup | Task 8 |

- **Deviations from the spec, all deliberate:**
  - The block-size cap rejects rather than splits. Ring-ring single bonds are bridges, so the block rule already cuts them, and only fused systems can exceed the cap.
  - Training is `v2/train.py`, not a flag on v1's `train.py`, which keeps v1 untouched.
  - The polystyrene MD becomes an aromatic OPoly26 cluster (Task 8).
  - The training budget is 90 min, not 120, so four trained models fit before the deadline in two concurrent pairs.
- **Type consistency:**
  - `var_cab`/`var_m`/`var_block`, `ev_*` and `av_*` names are identical across graph, batch, model and gates.
  - `collate_v2` signature is the same everywhere.
  - `N_ELEM` is imported from `v2.model` in data.
