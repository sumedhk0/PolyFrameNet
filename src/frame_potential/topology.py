"""Stage 0, static part: bonds, chains and blocks for linear alkane / polyethylene systems.

Everything here is integer bookkeeping computed once per system (non-reactive: the bond
graph never changes). Nothing here is differentiable and nothing here may be reused
across *different* systems; it is reused across timesteps of the *same* system, which
is allowed because topology is an input like atomic number, not a cached value.

Block dictionary (decision A, see docs/writeup.md):

    PE_mid  -CH2-CH2-   6 atoms, slots [C1, C2, H1a, H1b, H2a, H2b]
    PE_end  CH3-CH2-    7 atoms, slots [Cm, Cc, Hm1, Hm2, Hm3, Hc1, Hc2]

C1 -> C2 runs along the chain's storage direction, which is arbitrary; the model averages
over both directions (chain-direction symmetry is real in PE). For PE_end, Cm is the
methyl carbon and Cc the CH2 carbon, so the end block has no direction ambiguity.

A chain with n_C carbons (n_C even, >= 4) is cut into 2 end blocks and (n_C - 4)/2 mid
blocks. Ethane (n_C = 2), methane and odd chains are rejected: ethane has a three-fold
axis, so every hydrogen-symmetric frame recipe degenerates on it (see writeup).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MID, END = 0, 1
BLOCK_TYPES = {MID: "PE_mid", END: "PE_end"}
BLOCK_SIZE = {MID: 6, END: 7}

# Bond cutoffs in Angstrom. C-C single bond ~1.53, C-H ~1.09. These are only used once,
# at t=0, to read topology off a geometry (the dataset has no explicit bonds).
CC_MAX = 1.85
CH_MAX = 1.35


class TopologyError(ValueError):
    """Structure is not a set of linear saturated alkane chains we can block-factor."""


@dataclass
class Topology:
    numbers: np.ndarray  # (N,)
    bonds: np.ndarray  # (E_b, 2), i < j
    chain_of_atom: np.ndarray  # (N,)
    block_of_atom: np.ndarray  # (N,) global block index
    slot_of_atom: np.ndarray  # (N,)
    block_type: np.ndarray  # (B,)
    block_chain: np.ndarray  # (B,) chain id of each block
    block_pos: np.ndarray  # (B,) position of block along its chain, 0..n_blocks-1
    # Per type: index table [B_t, n_t] of atom indices, and the global block ids [B_t].
    block_atoms: dict[int, np.ndarray] = field(default_factory=dict)
    block_ids: dict[int, np.ndarray] = field(default_factory=dict)
    # Block graph: consecutive blocks along each chain, (E_bb, 2) with I < J along chain.
    block_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))

    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @property
    def n_blocks(self) -> int:
        return len(self.block_type)


def _mic(d: np.ndarray, cell: np.ndarray | None) -> np.ndarray:
    if cell is None:
        return d
    frac = d @ np.linalg.inv(cell)
    return d - np.round(frac) @ cell


def infer_bonds(numbers: np.ndarray, positions: np.ndarray, cell: np.ndarray | None = None):
    """Covalent bonds from distances, read once at t=0. Brute force; N is small at setup."""
    n = len(numbers)
    d = positions[None, :, :] - positions[:, None, :]
    if cell is not None:
        d = _mic(d.reshape(-1, 3), cell).reshape(n, n, 3)
    r = np.linalg.norm(d, axis=-1)
    isC = numbers == 6
    isH = numbers == 1
    cc = isC[:, None] & isC[None, :] & (r < CC_MAX)
    ch = ((isC[:, None] & isH[None, :]) | (isH[:, None] & isC[None, :])) & (r < CH_MAX)
    adj = (cc | ch) & ~np.eye(n, dtype=bool)
    i, j = np.nonzero(np.triu(adj, 1))
    return np.stack([i, j], 1).astype(np.int64)


def build_topology(numbers, bonds) -> Topology:
    """Validate linear alkane chemistry and cut chains into blocks.

    Raises TopologyError for anything else (branches, rings, radicals, unsaturation,
    stray H2, ethane): the caller drops that structure.
    """
    numbers = np.asarray(numbers, dtype=np.int64)
    bonds = np.asarray(bonds, dtype=np.int64).reshape(-1, 2)
    n = len(numbers)
    if not np.all((numbers == 1) | (numbers == 6)):
        raise TopologyError("only C and H supported")
    nbr: list[list[int]] = [[] for _ in range(n)]
    for a, b in bonds:
        nbr[a].append(b)
        nbr[b].append(a)
    cnbr = [[x for x in nbr[i] if numbers[x] == 6] for i in range(n)]
    hnbr = [[x for x in nbr[i] if numbers[x] == 1] for i in range(n)]
    for i in range(n):
        if numbers[i] == 1:
            if len(nbr[i]) != 1 or numbers[nbr[i][0]] != 6:
                raise TopologyError(f"H {i} must have exactly one C neighbour")
        else:
            if len(nbr[i]) != 4:
                raise TopologyError(f"C {i} has {len(nbr[i])} neighbours, need 4 (saturated)")
            if len(cnbr[i]) > 2:
                raise TopologyError(f"C {i} is a branch point")

    # Walk each chain from an end carbon (degree <= 1 in the carbon graph).
    seen = np.zeros(n, dtype=bool)
    chains: list[list[int]] = []
    carbons = [i for i in range(n) if numbers[i] == 6]
    for start in carbons:
        if seen[start] or len(cnbr[start]) > 1:
            continue
        chain = [start]
        seen[start] = True
        prev, cur = -1, start
        while True:
            nxt = [x for x in cnbr[cur] if x != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            if seen[cur]:
                raise TopologyError("cycle in carbon graph")
            seen[cur] = True
            chain.append(cur)
        chains.append(chain)
    if not all(seen[c] for c in carbons):
        # Remaining carbons all have two C neighbours: a ring, or a chain bonded to its own
        # periodic image. Periodic self-bonding is refused (open decision, see writeup).
        raise TopologyError("ring or periodically self-bonded chain")

    chain_of_atom = -np.ones(n, dtype=np.int64)
    block_of_atom = -np.ones(n, dtype=np.int64)
    slot_of_atom = -np.ones(n, dtype=np.int64)
    btype, bchain, bpos, batoms = [], [], [], []
    edges = []
    for ci, ch in enumerate(chains):
        nc = len(ch)
        if nc < 4 or nc % 2:
            raise TopologyError(f"chain of {nc} carbons cannot be cut into C2 blocks")
        pairs = [(ch[k], ch[k + 1]) for k in range(0, nc, 2)]
        first = len(btype)
        for k, (a, b) in enumerate(pairs):
            if k == 0:  # methyl first: a is the chain-end carbon
                atoms = [a, b, *sorted(hnbr[a]), *sorted(hnbr[b])]
                t = END
            elif k == len(pairs) - 1:  # last pair: b is the chain-end carbon
                atoms = [b, a, *sorted(hnbr[b]), *sorted(hnbr[a])]
                t = END
            else:
                atoms = [a, b, *sorted(hnbr[a]), *sorted(hnbr[b])]
                t = MID
            if len(atoms) != BLOCK_SIZE[t]:
                raise TopologyError("block has wrong hydrogen count")
            bid = len(btype)
            for s, at in enumerate(atoms):
                block_of_atom[at] = bid
                slot_of_atom[at] = s
                chain_of_atom[at] = ci
            btype.append(t)
            bchain.append(ci)
            bpos.append(k)
            batoms.append(atoms)
            if k > 0:
                edges.append((bid - 1, bid))
        del first
    if np.any(block_of_atom < 0):
        raise TopologyError("atom not assigned to a block (stray H?)")

    btype = np.asarray(btype, dtype=np.int64)
    topo = Topology(
        numbers=numbers,
        bonds=bonds,
        chain_of_atom=chain_of_atom,
        block_of_atom=block_of_atom,
        slot_of_atom=slot_of_atom,
        block_type=btype,
        block_chain=np.asarray(bchain, dtype=np.int64),
        block_pos=np.asarray(bpos, dtype=np.int64),
        block_edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2),
    )
    for t in (MID, END):
        ids = np.nonzero(btype == t)[0]
        topo.block_ids[t] = ids
        topo.block_atoms[t] = (
            np.asarray([batoms[i] for i in ids], dtype=np.int64).reshape(-1, BLOCK_SIZE[t])
        )
    return topo


def topology_from_geometry(numbers, positions, cell=None) -> Topology:
    return build_topology(numbers, infer_bonds(np.asarray(numbers), np.asarray(positions), cell))


def bond_distances(topo: Topology, max_d: int) -> dict[tuple[int, int], int]:
    """Topological distance for all pairs with d <= max_d (BFS). Used in tests/analysis."""
    n = topo.n_atoms
    nbr: list[list[int]] = [[] for _ in range(n)]
    for a, b in topo.bonds:
        nbr[a].append(int(b))
        nbr[b].append(int(a))
    out = {}
    for s in range(n):
        dist = {s: 0}
        frontier = [s]
        for d in range(1, max_d + 1):
            nxt = []
            for u in frontier:
                for v in nbr[u]:
                    if v not in dist:
                        dist[v] = d
                        nxt.append(v)
            frontier = nxt
        for v, d in dist.items():
            if v > s:
                out[(s, v)] = d
    return out
