"""Batched systems: concatenates topologies of several structures into one set of index
tensors, and builds the radius graph from the current positions.

Only integer index tensors are prepared ahead of time. Every vector the model consumes
(edge vectors, block coordinates, centroid offsets) is rebuilt inside autograd from
`positions` and `cell`, so forces (and, later, stress) are exact derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .topology import END, MID, Topology, valence_pairs


@dataclass
class Batch:
    numbers: torch.Tensor  # (N,)
    positions: torch.Tensor  # (N, 3)
    struct: torch.Tensor  # (N,) structure index of each atom
    n_struct: int
    cell: torch.Tensor | None  # (S, 3, 3) rows are lattice vectors, or None (clusters)
    # Blocks
    block_type: torch.Tensor  # (B,)
    block_struct: torch.Tensor  # (B,)
    block_chain: torch.Tensor  # (B,) globally unique chain id
    block_pos: torch.Tensor  # (B,)
    block_atoms: dict[int, torch.Tensor]  # type -> (B_t, n_t) atom indices
    block_ids: dict[int, torch.Tensor]  # type -> (B_t,) global block ids
    block_edges: torch.Tensor  # (E_bb, 2)
    block_edges2: torch.Tensor  # (E_bb2, 2) second-neighbour block pairs
    val_pairs: torch.Tensor  # (P, 2) topological 1-2 / 1-3 / 1-4 pairs
    val_class: torch.Tensor  # (P,)
    block_of_atom: torch.Tensor  # (N,)
    # Targets (optional)
    energy: torch.Tensor | None = None  # (S,)
    forces: torch.Tensor | None = None  # (N, 3)

    @property
    def n_atoms(self) -> int:
        return self.numbers.shape[0]

    def to(self, device, dtype=None) -> Batch:
        def mv(x):
            if x is None:
                return None
            if torch.is_floating_point(x):
                return x.to(device=device, dtype=dtype or x.dtype)
            return x.to(device)

        return Batch(
            numbers=mv(self.numbers),
            positions=mv(self.positions),
            struct=mv(self.struct),
            n_struct=self.n_struct,
            cell=mv(self.cell),
            block_type=mv(self.block_type),
            block_struct=mv(self.block_struct),
            block_chain=mv(self.block_chain),
            block_pos=mv(self.block_pos),
            block_atoms={k: mv(v) for k, v in self.block_atoms.items()},
            block_ids={k: mv(v) for k, v in self.block_ids.items()},
            block_edges=mv(self.block_edges),
            block_edges2=mv(self.block_edges2),
            val_pairs=mv(self.val_pairs),
            val_class=mv(self.val_class),
            block_of_atom=mv(self.block_of_atom),
            energy=mv(self.energy),
            forces=mv(self.forces),
        )


def collate(
    topos: list[Topology],
    positions: list[np.ndarray],
    cells: list[np.ndarray] | None = None,
    energies: list[float] | None = None,
    forces: list[np.ndarray] | None = None,
) -> Batch:
    a_off = 0
    b_off = 0
    c_off = 0
    numbers, pos, struct = [], [], []
    btype, bstruct, bchain, bpos, bedges, bedges2, boa = [], [], [], [], [], [], []
    vpairs, vcls = [], []
    batoms = {MID: [], END: []}
    bids = {MID: [], END: []}
    for s, (t, r) in enumerate(zip(topos, positions, strict=True)):
        numbers.append(t.numbers)
        pos.append(np.asarray(r, dtype=np.float64))
        struct.append(np.full(t.n_atoms, s))
        btype.append(t.block_type)
        bstruct.append(np.full(t.n_blocks, s))
        bchain.append(t.block_chain + c_off)
        bpos.append(t.block_pos)
        bedges.append(t.block_edges + b_off)
        bedges2.append(t.block_edges2 + b_off)
        vp, vc = valence_pairs(t)
        vpairs.append(vp + a_off)
        vcls.append(vc)
        boa.append(t.block_of_atom + b_off)
        for k in (MID, END):
            batoms[k].append(t.block_atoms[k] + a_off)
            bids[k].append(t.block_ids[k] + b_off)
        a_off += t.n_atoms
        b_off += t.n_blocks
        c_off += int(t.block_chain.max()) + 1 if t.n_blocks else 0

    def cat(x, dtype=torch.long):
        return torch.as_tensor(np.concatenate(x), dtype=dtype)

    return Batch(
        numbers=cat(numbers),
        positions=cat(pos, torch.get_default_dtype()),
        struct=cat(struct),
        n_struct=len(topos),
        cell=None
        if cells is None
        else torch.as_tensor(np.stack(cells), dtype=torch.get_default_dtype()),
        block_type=cat(btype),
        block_struct=cat(bstruct),
        block_chain=cat(bchain),
        block_pos=cat(bpos),
        block_atoms={k: torch.as_tensor(np.concatenate(v), dtype=torch.long).reshape(
            -1, 6 if k == MID else 7) for k, v in batoms.items()},
        block_ids={k: cat(v) for k, v in bids.items()},
        block_edges=torch.as_tensor(np.concatenate(bedges), dtype=torch.long).reshape(-1, 2),
        block_edges2=torch.as_tensor(np.concatenate(bedges2), dtype=torch.long).reshape(-1, 2),
        val_pairs=torch.as_tensor(np.concatenate(vpairs), dtype=torch.long).reshape(-1, 2),
        val_class=cat(vcls),
        block_of_atom=cat(boa),
        energy=None
        if energies is None
        else torch.as_tensor(np.asarray(energies), dtype=torch.get_default_dtype()),
        forces=None if forces is None else cat(forces, torch.get_default_dtype()),
    )


# ----------------------------------------------------------------------------- radius graph


@torch.no_grad()
def radius_graph_brute(batch: Batch, cutoff: float):
    """Exact O(N^2)-per-structure reference / cluster path. Returns (i, j, shift[int]).

    Non-periodic only; shifts are zero. Directed: both (i, j) and (j, i) present.
    """
    assert batch.cell is None
    pos = batch.positions.detach()
    if batch.n_atoms <= 6000:  # one masked all-pairs pass over the whole batch
        d = torch.cdist(pos, pos)
        m = (d < cutoff) & (batch.struct[:, None] == batch.struct[None, :])
        m.fill_diagonal_(False)
        i, j = torch.nonzero(m, as_tuple=True)
        return i, j, torch.zeros(len(i), 3, dtype=torch.long, device=pos.device)
    ii, jj = [], []
    for s in range(batch.n_struct):
        idx = torch.nonzero(batch.struct == s).squeeze(1)
        p = pos[idx]
        d = torch.cdist(p, p)
        m = (d < cutoff) & ~torch.eye(len(idx), dtype=torch.bool, device=p.device)
        a, b = torch.nonzero(m, as_tuple=True)
        ii.append(idx[a])
        jj.append(idx[b])
    i = torch.cat(ii)
    j = torch.cat(jj)
    return i, j, torch.zeros(len(i), 3, dtype=torch.long, device=pos.device)


@torch.no_grad()
def radius_graph_periodic_brute(batch: Batch, cutoff: float):
    """Periodic O(N^2 * 27) reference, used only to test the vesin path."""
    assert batch.cell is not None
    pos = batch.positions.detach()
    out_i, out_j, out_s = [], [], []
    shifts = torch.stack(
        torch.meshgrid(*[torch.arange(-1, 2, device=pos.device)] * 3, indexing="ij"), -1
    ).reshape(-1, 3)
    for s in range(batch.n_struct):
        idx = torch.nonzero(batch.struct == s).squeeze(1)
        p = pos[idx]
        cell = batch.cell[s].detach()
        for S in shifts:
            v = p[None, :, :] + (S.to(cell.dtype) @ cell)[None, None, :] - p[:, None, :]
            d = v.norm(dim=-1)
            m = d < cutoff
            if S.abs().sum() == 0:
                m &= ~torch.eye(len(idx), dtype=torch.bool, device=p.device)
            a, b = torch.nonzero(m, as_tuple=True)
            out_i.append(idx[a])
            out_j.append(idx[b])
            out_s.append(S.expand(len(a), 3))
    return torch.cat(out_i), torch.cat(out_j), torch.cat(out_s).long()


@torch.no_grad()
def radius_graph_vesin(batch: Batch, cutoff: float):
    """Periodic or not, via vesin (the default neighbour-list backend for MD)."""
    from vesin_torch import NeighborList

    nl = NeighborList(cutoff=cutoff, full_list=True)
    pos = batch.positions.detach()
    out_i, out_j, out_s = [], [], []
    for s in range(batch.n_struct):
        idx = torch.nonzero(batch.struct == s).squeeze(1)
        p = pos[idx]
        if batch.cell is None:
            box = torch.zeros(3, 3, dtype=p.dtype, device=p.device)
            periodic = False
        else:
            box = batch.cell[s].detach()
            periodic = True
        i, j, S = nl.compute(points=p, box=box, periodic=periodic, quantities="ijS")
        out_i.append(idx[i.long()])
        out_j.append(idx[j.long()])
        out_s.append(S.long())
    return torch.cat(out_i), torch.cat(out_j), torch.cat(out_s)


def radius_graph(batch: Batch, cutoff: float, backend: str = "auto"):
    if backend == "auto":
        backend = "brute" if batch.cell is None else "vesin"
    if backend == "brute":
        return radius_graph_brute(batch, cutoff)
    if backend == "vesin":
        return radius_graph_vesin(batch, cutoff)
    if backend == "periodic_brute":
        return radius_graph_periodic_brute(batch, cutoff)
    raise ValueError(backend)


def exclusion_keep_mask(batch: Batch, i: torch.Tensor, j: torch.Tensor, max_block_hops: int):
    """Keep-mask for radius edges: drop same-chain pairs whose blocks are <= max_block_hops
    apart along the chain. With max_block_hops = L_b (Stage 3 depth), the non-bonded channel
    never sees a pair the bonded channel can reach: disjoint jobs by construction.
    Block-hop distance, not atom-bond distance: L_b counts block hops (spec flag 1)."""
    bi = batch.block_of_atom[i]
    bj = batch.block_of_atom[j]
    same = batch.block_chain[bi] == batch.block_chain[bj]
    hops = (batch.block_pos[bi] - batch.block_pos[bj]).abs()
    return ~(same & (hops <= max_block_hops))


def edge_vectors(batch: Batch, i, j, shift):
    """r_j - r_i + S @ cell, inside autograd (differentiable in positions and cell)."""
    v = batch.positions[j] - batch.positions[i]
    if batch.cell is not None:
        cell_e = batch.cell[batch.struct[i]]  # (E, 3, 3)
        v = v + torch.einsum("ek,ekl->el", shift.to(v.dtype), cell_e)
    return v
