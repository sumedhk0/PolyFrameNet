"""PolyFrameNet / BFPP: block-factored polymer potential.

Pipeline (spec stages):
    0  graphs        batch.py  (bond/block graph static; radius graph per call)
    1  frames        block_frames()       one orthonormal frame per block, two variants
    2  encoder       BlockEncoder         dense MLP, Reynolds-averaged over the block group
    3  bonded        BondedLayer          message passing on the block chain, poses as R_IJ
    4  non-bonded    NonBonded            one radius-graph pass, Cartesian moments, nu = 3
    5  long-range    hook only (returns 0)
    6  readout       per-atom MLP, E = sum eps_i + sum E0[Z]; forces = -dE/dr (autodiff)

Symmetry handling, all by averaging (invariant 5), never by selecting:
  * H relabelling inside a CH2 / CH3: frames use a symmetric sum over the hydrogens;
    the encoder averages over the hydrogen permutation group.
  * Chain direction (C1 <-> C2 for every mid block of a chain at once): every stage that
    sees a frame evaluates both frame variants ("fwd", "rev") and averages. End blocks
    (CH3-CH2-) have an intrinsic direction, so their two variants coincide.
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn

from .batch import Batch, edge_vectors, exclusion_keep_mask, radius_graph
from .topology import BLOCK_SIZE, END, MID

# ----------------------------------------------------------------------------- utilities


def mlp(sizes, act=nn.SiLU, last_act=False):
    layers = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        if k < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


def _normalize(v, eps=1e-12):
    # eps only guards an exact zero vector; geometry keeps these far from zero (see writeup).
    return v / torch.sqrt((v * v).sum(-1, keepdim=True) + eps)


def gaussians(r: torch.Tensor, lo: float, hi: float, k: int):
    """Gaussian radial expansion, width = spacing. Used only on topological pairs (fixed set,
    never enters or leaves), so no cutoff envelope is needed."""
    mu = torch.linspace(lo, hi, k, device=r.device, dtype=r.dtype)
    w = (hi - lo) / (k - 1)
    return torch.exp(-0.5 * ((r[..., None] - mu) / w) ** 2)


def _mic_shift(d: torch.Tensor, cell: torch.Tensor | None):
    """Minimum-image correction for short intra-block / intra-chain vectors.

    The integer image count is computed without gradient (it is piecewise constant);
    the correction itself, n @ cell, stays differentiable in the cell."""
    if cell is None:
        return d
    with torch.no_grad():
        n = torch.round(torch.linalg.solve(cell.transpose(-1, -2), d.unsqueeze(-1)).squeeze(-1))
    return d - torch.einsum("...k,...kl->...l", n, cell)


# Hydrogen-slot groups. Mid: [C1, C2, H1a, H1b, H2a, H2b]; End: [Cm, Cc, Hm1-3, Hc1, Hc2].
def _mid_group():
    out = []
    for p1 in itertools.permutations([2, 3]):
        for p2 in itertools.permutations([4, 5]):
            out.append((0, [0, 1, *p1, *p2]))  # fwd frame
            out.append((1, [1, 0, *p2, *p1]))  # rev frame: C2 plays C1
    return out


def _end_group():
    out = []
    for pm in itertools.permutations([2, 3, 4]):
        for pc in itertools.permutations([5, 6]):
            out.append((0, [0, 1, *pm, *pc]))
    return out


GROUPS = {MID: _mid_group(), END: _end_group()}


# ----------------------------------------------------------------------------- Stage 1


def _frame(e1_from, e1_to, carbon, hydrogens):
    """e1 along carbon-carbon, e2 = normalized symmetric sum of the hydrogens' unit
    bond vectors with e1 projected out, e3 = e1 x e2. Orthonormal by construction."""
    e1 = _normalize(e1_to - e1_from)
    s = torch.zeros_like(e1)
    for h in hydrogens:
        u = _normalize(h - carbon)
        s = s + (u - (u * e1).sum(-1, keepdim=True) * e1)
    e2 = _normalize(s)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)  # columns are basis vectors


def block_frames(X: torch.Tensor, btype: int):
    """X: (B, n, 3) contiguous block positions. Returns origin (B,3), F (2, B, 3, 3)."""
    o = X.mean(1)
    if btype == MID:
        f = _frame(X[:, 0], X[:, 1], X[:, 0], [X[:, 2], X[:, 3]])
        r = _frame(X[:, 1], X[:, 0], X[:, 1], [X[:, 4], X[:, 5]])
    else:
        # Cm -> Cc; the CH2 hydrogens define e2 (a CH3's three H sum to ~0, degenerate).
        f = _frame(X[:, 0], X[:, 1], X[:, 1], [X[:, 5], X[:, 6]])
        r = f
    return o, torch.stack([f, r], 0)


# ----------------------------------------------------------------------------- Stage 2


class BlockEncoder(nn.Module):
    """Dense MLP over a whole block, averaged over the block's symmetry group."""

    def __init__(self, btype: int, d_atom: int, d_block: int, hidden: int, n_species: int = 2):
        super().__init__()
        self.btype = btype
        self.n = BLOCK_SIZE[btype]
        self.d_atom = d_atom
        self.d_block = d_block
        self.species = nn.Embedding(n_species, 8)
        self.n_rbf = 8
        n_pairs = self.n * (self.n - 1) // 2
        self.net = mlp(
            [self.n * (3 + 8) + n_pairs * self.n_rbf, hidden, hidden, hidden,
             self.n * d_atom + d_block]
        )
        self.group = GROUPS[btype]
        self.register_buffer(
            "perms", torch.tensor([pm for _, pm in self.group]), persistent=False
        )  # (G, n)
        self.register_buffer(
            "variants", torch.tensor([v for v, _ in self.group]), persistent=False
        )  # (G,)
        self.register_buffer("triu", torch.triu_indices(self.n, self.n, 1), persistent=False)

    def forward(self, Xt: torch.Tensor, species: torch.Tensor, F: torch.Tensor, o: torch.Tensor):
        # Xt (B,n,3); species (B,n) in {0:H, 1:C}; F (2,B,3,3); o (B,3)
        B = Xt.shape[0]
        rel = Xt - o[:, None, :]
        coords = [torch.einsum("bni,bij->bnj", rel, F[v]) for v in range(2)]  # F^T (r - o)
        emb = self.species(species)
        # All intra-block distances, Gaussian-expanded: the stiff bond / angle terms are
        # simple functions of these, hard functions of raw Cartesian frame coordinates.
        # (cdist has no double backward, which force training needs.)
        dv = Xt[:, :, None, :] - Xt[:, None, :, :]
        D = gaussians(torch.sqrt((dv * dv).sum(-1) + 1e-12), 0.8, 3.2, self.n_rbf)
        # All group elements in one batched MLP call: (G, B, ...) -> (G*B, ...).
        G = self.perms.shape[0]
        P = self.perms
        cs = torch.stack(coords, 0)[self.variants]  # (G, B, n, 3) frame variant per element
        cs = torch.gather(cs, 2, P[:, None, :, None].expand(G, B, self.n, 3))
        es = emb[:, P].permute(1, 0, 2, 3)  # (G, B, n, 8)
        Dp = D[:, P[:, :, None], P[:, None, :]]  # (B, G, n, n, K)
        Dp = Dp[:, :, self.triu[0], self.triu[1]].permute(1, 0, 2, 3)  # (G, B, pairs, K)
        x = torch.cat([cs.reshape(G, B, -1), es.reshape(G, B, -1), Dp.reshape(G, B, -1)], -1)
        y = self.net(x)  # (G, B, out)
        ha = y[..., : self.n * self.d_atom].reshape(G, B, self.n, self.d_atom)
        # un-permute: output slot s of element g belongs to atom P[g, s]
        inv = torch.argsort(P, dim=1)  # (G, n)
        ha = torch.gather(ha, 2, inv[:, None, :, None].expand(G, B, self.n, self.d_atom))
        return ha.mean(0), y[..., self.n * self.d_atom :].mean(0)


# ----------------------------------------------------------------------------- Stage 3


class BondedLayer(nn.Module):
    """One round of message passing on the block chain graph.

    Edge input: [H_I, H_J, R_IJ (9 raw entries), d_IJ (3), |d_IJ|, pair-pooled distance
    expansion (invariant, see PolyFrameNet.forward), edge type (first/second neighbour)].
    Averaged over the two chain-direction frame variants. Sum aggregation, residual."""

    def __init__(self, d: int, hidden: int, d_edge: int):
        super().__init__()
        self.msg = mlp([2 * d + 13 + d_edge, hidden, hidden, d])
        self.upd = mlp([2 * d, hidden, d])

    def forward(self, H, src, dst, R, dvec, efeat):
        # R: (2, E, 9), dvec: (2, E, 3) per frame variant; message flows src -> dst
        m = 0.0
        for v in range(2):
            z = torch.cat(
                [H[dst], H[src], R[v], dvec[v], dvec[v].norm(dim=-1, keepdim=True), efeat], -1
            )
            m = m + self.msg(z)
        m = 0.5 * m
        agg = torch.zeros_like(H).index_add_(0, dst, m)
        return H + self.upd(torch.cat([H, agg], -1))


# ----------------------------------------------------------------------------- valence prior


class Valence(nn.Module):
    """Learnable valence term on topological pairs (Urey-Bradley style): bonds (1-2),
    angles via 1-3 distances, torsion-sensitive 1-4 distances. One shared MLP of a fine
    Gaussian expansion of r, conditioned on the pair class (bond distance x species pair).
    Stiff covalent terms are then one-dimensional functions the optimiser finds at once,
    instead of features buried in the block MLP. Pair energy is split equally onto its two
    atoms. Smooth (the pair set is fixed by topology, never enters or leaves)."""

    def __init__(self, n_rbf: int = 48, hidden: int = 64, n_class: int = 9):
        super().__init__()
        self.n_rbf = n_rbf
        self.n_class = n_class
        self.net = mlp([n_rbf + n_class, hidden, hidden, 1])

    def forward(self, batch: Batch, pos: torch.Tensor) -> torch.Tensor:
        i, j = batch.val_pairs[:, 0], batch.val_pairs[:, 1]
        v = pos[j] - pos[i]
        if batch.cell is not None:
            v = _mic_shift(v, batch.cell[batch.struct[i]])
        r = torch.sqrt((v * v).sum(-1) + 1e-12)
        x = torch.cat(
            [gaussians(r, 0.8, 4.4, self.n_rbf),
             nn.functional.one_hot(batch.val_class, self.n_class).to(pos.dtype)], -1
        )
        e = self.net(x).squeeze(-1)
        out = pos.new_zeros(pos.shape[0])
        return out.index_add(0, i, 0.5 * e).index_add(0, j, 0.5 * e)


# ----------------------------------------------------------------------------- Stage 4


def bessel_envelope(r, cutoff: float, n: int, p: int = 6):
    """sin(k pi r / rc) / r times a polynomial envelope that is C^2-smooth to zero at rc."""
    x = r / cutoff
    k = torch.arange(1, n + 1, device=r.device, dtype=r.dtype)
    b = math.sqrt(2.0 / cutoff) * torch.sin(k * math.pi * x[:, None]) / r[:, None]
    env = (
        1.0
        - (p + 1) * (p + 2) / 2 * x**p
        + p * (p + 2) * x ** (p + 1)
        - p * (p + 1) / 2 * x ** (p + 2)
    )
    env = torch.where(x < 1.0, env, torch.zeros_like(env))
    return b * env[:, None]


_TRIU = (torch.tensor([0, 0, 0, 1, 1, 2]), torch.tensor([0, 1, 2, 1, 2, 2]))


class NonBonded(nn.Module):
    """One pass over the (masked) radius graph.

    Per edge: phi_ij,c = R_c(r_ij; pair species) * g_c(j), with g a linear read of the
    neighbour's features. Per atom: Cartesian moments M0 = sum phi, M1 = sum phi r_hat,
    M2 = sum phi r_hat r_hat^T, accumulated ONCE in the global frame; projecting them into
    a block frame is per-atom (F^T M1, F^T M2 F), which is why the chain-direction average
    costs nothing per edge. Body order from products of linear mixes (A^2, A^3: nu = 3)."""

    def __init__(self, d_in: int, n_rbf: int, n_ch: int, d_out: int, cutoff: float, avg_nbr: float):
        super().__init__()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.n_ch = n_ch
        self.avg_nbr = avg_nbr
        self.radial = nn.Parameter(torch.randn(3, n_rbf, n_ch) / math.sqrt(n_rbf))
        self.gate = nn.Linear(d_in, n_ch)
        na = n_ch * 10
        self.lin1 = nn.Linear(na, d_out)
        self.lin2a = nn.Linear(na, d_out)
        self.lin2b = nn.Linear(na, d_out)
        self.lin3a = nn.Linear(na, d_out)
        self.lin3b = nn.Linear(na, d_out)
        self.lin3c = nn.Linear(na, d_out)
        self.d_out = 3 * d_out

    def moments(self, x_atom, numbers, i, j, vec):
        r = vec.norm(dim=-1)
        rhat = vec / r[:, None]
        pair = (numbers[i] == 6).long() + (numbers[j] == 6).long()  # 0 HH, 1 CH, 2 CC
        rb = bessel_envelope(r, self.cutoff, self.n_rbf)  # (E, n_rbf)
        Rc = torch.einsum("em,emc->ec", rb, self.radial[pair])  # (E, n_ch)
        phi = Rc * self.gate(x_atom)[j]
        N = x_atom.shape[0]
        M0 = x_atom.new_zeros(N, self.n_ch).index_add_(0, i, phi)
        M1 = x_atom.new_zeros(N, self.n_ch, 3).index_add_(0, i, phi[:, :, None] * rhat[:, None, :])
        outer = rhat[:, :, None] * rhat[:, None, :]
        M2 = x_atom.new_zeros(N, self.n_ch, 3, 3).index_add_(
            0, i, phi[:, :, None, None] * outer[:, None]
        )
        s = 1.0 / self.avg_nbr
        return M0 * s, M1 * s, M2 * s

    def project(self, M0, M1, M2, F):
        # F: (N, 3, 3) frame of each atom's block
        P1 = torch.einsum("nci,nij->ncj", M1, F)
        P2 = torch.einsum("nik,ncij,njl->nckl", F, M2, F)  # F^T M2 F
        P2 = P2[:, :, _TRIU[0], _TRIU[1]]
        A = torch.cat([M0[:, :, None], P1, P2], -1).reshape(M0.shape[0], -1)
        b1 = self.lin1(A)
        b2 = self.lin2a(A) * self.lin2b(A)
        b3 = self.lin3a(A) * self.lin3b(A) * self.lin3c(A)
        return torch.cat([b1, b2, b3], -1)


# ----------------------------------------------------------------------------- full model


class PolyFrameNet(nn.Module):
    def __init__(
        self,
        cutoff: float = 4.5,
        d_atom: int = 64,
        d_block: int = 128,
        hidden: int = 128,
        n_bonded: int = 2,
        n_rbf: int = 8,
        n_ch: int = 16,
        d_nb: int = 64,
        avg_nbr: float = 20.0,
        exclude_hops: int | None = None,
        long_range: bool = False,
    ):
        super().__init__()
        self.config = dict(
            cutoff=cutoff, d_atom=d_atom, d_block=d_block, hidden=hidden, n_bonded=n_bonded,
            n_rbf=n_rbf, n_ch=n_ch, d_nb=d_nb, avg_nbr=avg_nbr, exclude_hops=exclude_hops,
            long_range=long_range,
        )
        self.cutoff = cutoff
        self.exclude_hops = n_bonded if exclude_hops is None else exclude_hops
        self.encoders = nn.ModuleDict(
            {str(t): BlockEncoder(t, d_atom, d_block, hidden) for t in (MID, END)}
        )
        self.block_type_emb = nn.Embedding(2, d_block)
        self.n_pair_rbf = 24
        d_edge = 3 * self.n_pair_rbf + 2
        self.bonded = nn.ModuleList(
            [BondedLayer(d_block, hidden, d_edge) for _ in range(n_bonded)]
        )
        d_ctx = d_atom + d_block
        self.nonbonded = NonBonded(d_ctx, n_rbf, n_ch, d_nb, cutoff, avg_nbr)
        self.species_emb = nn.Embedding(2, 16)
        self.readout = mlp([d_ctx + self.nonbonded.d_out + 16, hidden, hidden, 1])
        self.long_range = long_range
        self.valence = Valence()
        # Per-element reference energies and scale; set from training data (fit_reference).
        self.register_buffer("e0", torch.zeros(2))  # [H, C] eV
        self.register_buffer("scale", torch.ones(()))

    # -- Stage 5 hook -------------------------------------------------------------------
    def long_range_energy(self, batch: Batch, x_atom: torch.Tensor) -> torch.Tensor:
        return x_atom.new_zeros(batch.n_struct)

    def forward(self, batch: Batch, edges=None) -> torch.Tensor:
        """Energy per structure (S,). Pure function of batch.positions / batch.cell."""
        pos = batch.positions
        N = batch.n_atoms
        B = batch.block_type.shape[0]
        species = (batch.numbers == 6).long()
        dt = pos.dtype

        # Stages 1+2, per block type ---------------------------------------------------
        h_atom = pos.new_zeros(N, self.encoders[str(MID)].d_atom)
        H = pos.new_zeros(B, self.block_type_emb.embedding_dim)
        origins = pos.new_zeros(B, 3)
        Fb = pos.new_zeros(2, B, 3, 3)
        XB = pos.new_zeros(B, 7, 3)  # contiguous block positions, padded to 7 slots
        SB = torch.full((B, 7), -1, dtype=torch.long, device=pos.device)  # species, -1 = pad
        for t in (MID, END):
            idx = batch.block_atoms[t]
            if idx.shape[0] == 0:
                continue
            ids = batch.block_ids[t]
            X = pos[idx]
            cell_b = None if batch.cell is None else batch.cell[batch.block_struct[ids]]
            if cell_b is not None:  # make each block contiguous across the periodic boundary
                X = X[:, :1] + _mic_shift(X - X[:, :1], cell_b[:, None])
            o, F = block_frames(X, t)
            h, Hb = self.encoders[str(t)](X, species[idx], F, o)
            h_atom[idx.reshape(-1)] = h.reshape(-1, h.shape[-1])
            H[ids] = Hb + self.block_type_emb.weight[t]
            origins[ids] = o
            Fb[:, ids] = F
            XB[ids, : idx.shape[1]] = X
            SB[ids, : idx.shape[1]] = species[idx]

        # Stage 3: bonded channel on the block chain -----------------------------------
        if batch.block_edges.shape[0]:
            e1, e2 = batch.block_edges, batch.block_edges2
            a = torch.cat([e1[:, 0], e2[:, 0]])
            b = torch.cat([e1[:, 1], e2[:, 1]])
            etype = torch.cat([a.new_zeros(len(e1)), a.new_ones(len(e2))])
            src = torch.cat([a, b])
            dst = torch.cat([b, a])
            etype = torch.cat([etype, etype])
            dO = origins[src] - origins[dst]
            if batch.cell is not None:
                dO = _mic_shift(dO, batch.cell[batch.block_struct[dst]])
            R = torch.einsum("veij,veik->vejk", Fb[:, dst], Fb[:, src]).reshape(2, -1, 9)
            dvec = torch.einsum("veij,ei->vej", Fb[:, dst], dO)
            # All atom pairs across the two blocks, Gaussian-expanded and summed within
            # each species-pair class (CC, CH, HH): the sum makes it invariant to hydrogen
            # labels and chain direction, and it carries the junction bond, angles and the
            # 1-4 / 1-5 contacts explicitly.
            rel_s = XB[src] - origins[src][:, None]
            rel_d = XB[dst] - origins[dst][:, None]
            pv = rel_s[:, None, :, :] - rel_d[:, :, None, :] + dO[:, None, None, :]
            r = torch.sqrt((pv * pv).sum(-1) + 1e-12)  # (E, 7, 7)
            ss, sd = SB[src][:, None, :], SB[dst][:, :, None]
            valid = (ss >= 0) & (sd >= 0)
            cls = torch.where(valid, ss + sd, torch.zeros_like(ss + sd))
            onehot = nn.functional.one_hot(cls, 3).to(pos.dtype) * valid[..., None]
            G = gaussians(r, 1.0, 7.0, self.n_pair_rbf)  # (E, 7, 7, K)
            pooled = torch.einsum("eabc,eabk->eck", onehot, G).reshape(len(src), -1) / 4.0
            efeat = torch.cat([pooled, nn.functional.one_hot(etype, 2).to(pos.dtype)], -1)
            for layer in self.bonded:
                H = layer(H, src, dst, R, dvec, efeat)

        # Stage 4: non-bonded channel --------------------------------------------------
        x_atom = torch.cat([h_atom, H[batch.block_of_atom]], -1)
        if edges is None:
            i, j, S = radius_graph(batch, self.cutoff)
        else:
            i, j, S = edges
        keep = exclusion_keep_mask(batch, i, j, self.exclude_hops)
        i, j, S = i[keep], j[keep], S[keep]
        vec = edge_vectors(batch, i, j, S)
        M0, M1, M2 = self.nonbonded.moments(x_atom, batch.numbers, i, j, vec)

        # Stage 6: readout, averaged over the two frame variants -----------------------
        sp = self.species_emb(species)
        eps = 0.0
        for v in range(2):
            Fa = Fb[v][batch.block_of_atom]
            nb = self.nonbonded.project(M0, M1, M2, Fa)
            eps = eps + self.readout(torch.cat([x_atom, nb, sp], -1)).squeeze(-1)
        eps = (0.5 * eps + self.valence(batch, pos)) * self.scale + self.e0[species]
        E = pos.new_zeros(batch.n_struct, dtype=dt).index_add_(0, batch.struct, eps)
        if self.long_range:
            E = E + self.long_range_energy(batch, x_atom)
        return E


def energy_and_forces(model: PolyFrameNet, batch: Batch, create_graph: bool = False, edges=None):
    pos = batch.positions
    if not pos.requires_grad:
        pos.requires_grad_(True)
    E = model(batch, edges=edges)
    (g,) = torch.autograd.grad(E.sum(), pos, create_graph=create_graph)
    return E, -g
