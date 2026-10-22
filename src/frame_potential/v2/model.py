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

Every average is over a block's full variant set, which colour classes make closed under
rotations, reflections and relabellings (graph.py); hence exact O(3) and permutation
invariance without selecting any frame.
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


def _g(x, index):
    """Row gather. index_select's backward is index_add (atomic); x[index]'s backward is a
    sort-based index_put that dominated the step time on CPU (profiled: ~25%)."""
    return x.index_select(0, index)


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
        # atom type: element, degree, ring flag, bonds leaving the block; plus (Q, S)
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
        # valence prior
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
        # Per-element offsets as integer micro-eV: model.to(float32) cannot round them.
        self.register_buffer("e0_micro", torch.zeros(N_ELEM, dtype=torch.long))
        self.register_buffer("scale", torch.ones(()))

    def set_reference(self, e0):
        """e0: per-element reference energies in eV, indexed by atomic number."""
        self.e0_micro.copy_(torch.round(torch.as_tensor(e0, dtype=torch.float64) * 1e6).long())

    def _frames(self, batch, pos):
        cab = batch.var_cab
        free = cab[:, 0] < 0
        c, a, b = (cab[:, k].clamp(min=0) for k in range(3))
        va, vb = _g(pos, a) - _g(pos, c), _g(pos, b) - _g(pos, c)
        if batch.cell is not None:
            cv = batch.cell[batch.block_struct[batch.var_block]]
            va, vb = _mic_shift(va, cv), _mic_shift(vb, cv)
        e1 = _normalize(va)
        e2 = _normalize(vb - (vb * e1).sum(-1, keepdim=True) * e1)
        e3 = batch.var_m.to(pos.dtype)[:, None] * torch.cross(e1, e2, dim=-1)
        F = torch.stack([e1, e2, e3], -1)  # columns are basis vectors
        return torch.where(free[:, None, None], torch.zeros_like(F), F)

    def forward(self, batch, edges=None, lr_edges=None, capture=None):
        """Energy per structure (S,), float64. Pure function of positions and cell."""
        pos = batch.positions
        dt = pos.dtype
        N, B = batch.n_atoms, batch.n_blocks
        qs = torch.stack([batch.charge, batch.spin - 1.0], -1).to(dt)
        u = self.u_lin(torch.cat([
            self.z_emb(batch.numbers),
            _F.one_hot(batch.degree.clamp(max=6), 7).to(dt),
            batch.ring.to(dt)[:, None],
            _F.one_hot(batch.n_ext.clamp(max=4), 5).to(dt)], -1))
        u = u + _g(self.qs_lin(qs), batch.struct)

        # Stage 1: contiguous block coordinates, origins, frames ------------------------
        ba = batch.block_atoms
        mask = ba >= 0
        X = pos[ba.clamp(min=0)]
        if batch.cell is not None:
            X = X[:, :1] + _mic_shift(X - X[:, :1], batch.cell[batch.block_struct][:, None])
        mf = mask.to(dt)[..., None]
        o = (X * mf).sum(1) / mf.sum(1)
        F = self._frames(batch, pos)

        # Ragged bookkeeping (integers only): every listed atom with its block and slot,
        # so no feature tensor is padded to the largest block in the batch.
        with torch.no_grad():
            kb, ks = torch.nonzero(mask, as_tuple=True)
        relA = pos.new_zeros(N, 3).index_put((ba[kb, ks],), X[kb, ks] - _g(o, kb))  # r - o_block

        # Stage 2: shared block encoder -------------------------------------------------
        n = ba.shape[1]
        with torch.no_grad():
            eye = torch.eye(n, dtype=torch.bool, device=pos.device)
            pb, pk, pl = torch.nonzero(mask[:, :, None] & mask[:, None, :] & ~eye,
                                       as_tuple=True)
        ak, al = ba[pb, pk], ba[pb, pl]
        dv = _g(relA, ak) - _g(relA, al)  # same block origin
        r = torch.sqrt((dv * dv).sum(-1) + 1e-12)
        pin = torch.cat([gaussians(r, 0.8, 7.0, self.n_pair_rbf), _g(u, ak), _g(u, al)], -1)
        P = pos.new_zeros(N, self.pair_net[-1].out_features).index_add(0, ak, self.pair_net(pin))
        aa, av = batch.av_atom, batch.av_var  # (atom, variant of its block) rows
        Fa = _g(F, av)
        coords = torch.einsum("ai,aij->aj", _g(relA, aa), Fa)  # F^T (r - o)
        x = self.atom_net(torch.cat([coords, _g(u, aa), _g(P, aa)], -1))
        Hv = self.block_net(x.new_zeros(len(batch.var_block), x.shape[-1]).index_add(0, av, x))
        H = _scatter_mean(Hv, batch.var_block, B)
        h_atom = _scatter_mean(x, aa, N)
        if capture is not None:
            capture["H2"] = H.detach()

        # Stage 3: bonded channel on the block graph ------------------------------------
        # The update layers run for every block, edges or not: a block with no
        # neighbours (benzene alone) must get the same H whether or not other structures
        # in the batch have edges (caught by test_batch_equals_single).
        src, dst = batch.edge_src, batch.edge_dst
        has_edges = bool(src.numel())
        if has_edges:
            dO = _g(o, src) - _g(o, dst)
            if batch.cell is not None:
                dO = _mic_shift(dO, batch.cell[batch.block_struct[dst]])
            ee, ev = batch.ev_edge, batch.ev_var
            with torch.no_grad():  # (edge-variant row, atom of J) pairs
                qi, qslot = torch.nonzero(mask[src[ee]], as_tuple=True)
            aJ = ba[src[ee[qi]], qslot]
            cJ = torch.einsum("qi,qij->qj", _g(relA, aJ) + _g(dO, ee[qi]), _g(F, ev[qi]))
            g = cJ.new_zeros(len(ee), self.geo_net[-1].out_features).index_add(
                0, qi, self.geo_net(torch.cat([cJ, _g(u, aJ)], -1)))
            with torch.no_grad():  # all atom pairs across the two blocks of each edge
                pe_, pa, pc = torch.nonzero(mask[dst][:, :, None] & mask[src][:, None, :],
                                            as_tuple=True)
            aI, aJ2 = ba[dst[pe_], pa], ba[src[pe_], pc]
            pv = _g(relA, aJ2) + _g(dO, pe_) - _g(relA, aI)
            rx = torch.sqrt((pv * pv).sum(-1) + 1e-12)
            pz = self.pz(batch.numbers[aI]) * self.pz(batch.numbers[aJ2])  # symmetric
            G = gaussians(rx, 1.0, 8.0, self.n_x_rbf)
            pooled = pz.new_zeros(len(src), pz.shape[1], self.n_x_rbf).index_add(
                0, pe_, pz[:, :, None] * G[:, None, :]).flatten(1) / 4.0
            efeat = torch.cat([pooled, _F.one_hot(batch.edge_hop, 2).to(dt)], -1)
            dst_e, src_e = dst[ee], src[ee]
        for msg, upd in zip(self.msg, self.upd, strict=True):
            agg = torch.zeros_like(H)
            if has_edges:
                z = torch.cat([_g(H, dst_e), _g(H, src_e), g, _g(efeat, ee)], -1)
                m = _scatter_mean(msg(z), ee, len(src))  # mean over I's variants
                agg = agg.index_add(0, dst, m)
            H = H + upd(torch.cat([H, agg], -1))

        # Valence prior -----------------------------------------------------------------
        vi, vj = batch.val_pairs[:, 0], batch.val_pairs[:, 1]
        vv = _g(pos, vj) - _g(pos, vi)
        if batch.cell is not None:
            vv = _mic_shift(vv, batch.cell[batch.struct[vi]])
        rv = torch.sqrt((vv * vv).sum(-1) + 1e-12)
        ui, uj = self.val_u(_g(u, vi)), self.val_u(_g(u, vj))
        ev_ = self.val_net(torch.cat([gaussians(rv, 0.8, 4.4, 48),
                                      _F.one_hot(batch.val_d - 1, 3).to(dt),
                                      ui + uj, ui * uj], -1)).squeeze(-1)
        val_atom = pos.new_zeros(N).index_add(0, vi, 0.5 * ev_).index_add(0, vj, 0.5 * ev_)

        # Stage 4: non-bonded channel ---------------------------------------------------
        x_atom = torch.cat([h_atom, _g(H, batch.block_of_atom)], -1)
        i4, j4, S4 = radius_graph(batch, self.cutoff) if edges is None else edges
        vec = edge_vectors(batch, i4, j4, S4)
        r4 = torch.sqrt((vec * vec).sum(-1))
        rhat = vec / r4[:, None]
        rb = bessel_envelope(r4, self.cutoff, self.n_rbf)
        pe4 = self.nb_pair(batch.numbers[i4]) * self.nb_pair(batch.numbers[j4])
        phi = self.radial((rb[:, :, None] * pe4[:, None, :]).flatten(1))
        phi = phi * _g(self.gate(x_atom), j4)
        s = 1.0 / self.avg_nbr
        M0 = x_atom.new_zeros(N, self.n_ch).index_add(0, i4, phi) * s
        M1 = x_atom.new_zeros(N, self.n_ch, 3).index_add(
            0, i4, phi[:, :, None] * rhat[:, None, :]) * s
        M2 = x_atom.new_zeros(N, self.n_ch, 3, 3).index_add(
            0, i4, phi[:, :, None, None] * (rhat[:, :, None] * rhat[:, None, :])[:, None]) * s
        P1 = torch.einsum("aci,aij->acj", _g(M1, aa), Fa)
        P2 = torch.einsum("aik,acij,ajl->ackl", Fa, _g(M2, aa), Fa)[:, :, _TRIU[0], _TRIU[1]]
        A = torch.cat([_g(M0, aa)[:, :, None], P1, P2], -1).flatten(1)
        Bv = torch.cat([self.lin1(A), self.lin2a(A) * self.lin2b(A),
                        self.lin3a(A) * self.lin3b(A) * self.lin3c(A)], -1)
        Bn = _scatter_mean(Bv, aa, N)  # products before the mean keep orientation info

        # Stage 6 + 5: readout and long range -------------------------------------------
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
