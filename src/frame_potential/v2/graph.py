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
    numbers: np.ndarray  # (N,)
    bonds: np.ndarray  # (Eb, 2), i < j
    degree: np.ndarray  # (N,)
    ring: np.ndarray  # (N,) bool: atom has a ring bond
    n_ext: np.ndarray  # (N,) bonds leaving the atom's block
    colour: np.ndarray  # (N,) int64 WL colour
    block_of_atom: np.ndarray  # (N,)
    block_atoms: np.ndarray  # (B, n) atom indices, -1 padded
    var_block: np.ndarray  # (R,) block of each frame variant
    var_cab: np.ndarray  # (R, 3) anchor, e1 atom, e2 atom; -1 for frame-free
    var_m: np.ndarray  # (R,) mirror +1 / -1; 0 for frame-free
    edge_src: np.ndarray  # (Ed,) directed block edges J -> I: src J
    edge_dst: np.ndarray  # (Ed,) dst I
    edge_hop: np.ndarray  # (Ed,) 0 = bonded blocks, 1 = share a neighbour block
    val_pairs: np.ndarray  # (P, 2) topological 1-2 / 1-3 / 1-4 pairs
    val_d: np.ndarray  # (P,) topological distance 1..3
    ev_edge: np.ndarray | None = None  # (Q,) edge of each (edge, variant of dst) row
    ev_var: np.ndarray | None = None  # (Q,)
    av_atom: np.ndarray | None = None  # (A,) atom of each (atom, variant of its block) row
    av_var: np.ndarray | None = None  # (A,)

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
