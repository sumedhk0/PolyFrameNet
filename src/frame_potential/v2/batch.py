"""v2 batching: concatenate GraphTopology index tables with offsets. Only integers are
prepared here; every vector is rebuilt inside autograd from positions and cell.

BatchV2 duck-types v1's Batch (positions, struct, n_struct, cell, n_atoms) so the v1
radius_graph and edge_vectors work on it unchanged."""

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
    charge: torch.Tensor  # (S,) total charge per structure
    spin: torch.Tensor  # (S,) spin multiplicity per structure
    degree: torch.Tensor
    ring: torch.Tensor
    n_ext: torch.Tensor
    block_of_atom: torch.Tensor
    block_struct: torch.Tensor
    block_atoms: torch.Tensor  # (B, n) -1 padded
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


_INT_KEYS = ("numbers", "struct", "degree", "ring", "n_ext", "block_of_atom", "block_struct",
             "block_atoms", "var_block", "var_cab", "var_m", "edge_src", "edge_dst", "edge_hop",
             "ev_edge", "ev_var", "av_atom", "av_var", "val_pairs", "val_d")


def collate_v2(topos: list[GraphTopology], positions, cells=None, energies=None, forces=None,
               charges=None, spins=None) -> BatchV2:
    """Always float64; BatchV2.to(device, dtype) casts for float32 training."""
    S = len(topos)
    nmax = max(t.block_atoms.shape[1] for t in topos)
    a_off = b_off = v_off = e_off = 0
    acc: dict[str, list] = {k: [] for k in _INT_KEYS}
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
        struct=cat("struct"),
        n_struct=S,
        cell=None if cells is None else torch.as_tensor(np.stack(cells), dtype=f64),
        charge=torch.as_tensor(np.zeros(S) if charges is None else np.asarray(charges), dtype=f64),
        spin=torch.as_tensor(np.ones(S) if spins is None else np.asarray(spins), dtype=f64),
        degree=cat("degree"),
        ring=cat("ring"),
        n_ext=cat("n_ext"),
        block_of_atom=cat("block_of_atom"),
        block_struct=cat("block_struct"),
        block_atoms=cat("block_atoms", (-1, nmax)),
        var_block=cat("var_block"),
        var_cab=cat("var_cab", (-1, 3)),
        var_m=cat("var_m"),
        edge_src=cat("edge_src"),
        edge_dst=cat("edge_dst"),
        edge_hop=cat("edge_hop"),
        ev_edge=cat("ev_edge"),
        ev_var=cat("ev_var"),
        av_atom=cat("av_atom"),
        av_var=cat("av_var"),
        val_pairs=cat("val_pairs", (-1, 2)),
        val_d=cat("val_d"),
        energy=None if energies is None else torch.as_tensor(np.asarray(energies), dtype=f64),
        forces=None if forces is None else torch.as_tensor(np.concatenate(forces), dtype=f64),
    )
