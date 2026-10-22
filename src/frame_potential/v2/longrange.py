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
    """C2-smooth step: 1 below r_on, 0 above r_off."""
    x = ((r - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 1.0 - x**3 * (10.0 - 15.0 * x + 6.0 * x**2)


def neutralize(q, batch):
    """Shift charges uniformly per structure so they sum to the structure's total charge."""
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
    i, j, S = lr_edges  # full neighbour list within r_c: each pair twice
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
        recip = 2 * math.pi * torch.linalg.inv(cell).T  # rows: reciprocal lattice vectors
        with torch.no_grad():
            nk = [int(math.ceil(kmax * float(cell[d].norm()) / (2 * math.pi))) for d in range(3)]
            g = torch.stack(torch.meshgrid(*[torch.arange(-m, m + 1, device=pos.device)
                                             for m in nk], indexing="ij"), -1).reshape(-1, 3)
            # half space (k and -k contribute equally): first nonzero component positive
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
    i, j, S = lr_edges  # full neighbour list: each pair twice, hence the 0.5
    v = edge_vectors(batch, i, j, S)
    r = torch.sqrt((v * v).sum(-1))
    c6ij = torch.sqrt(c6[i] * c6[j] + 1e-12)
    R0 = r0[i] + r0[j]
    e = -0.5 * c6ij / (r**6 + R0**6) * switch(r, r_off - 2.0, r_off)
    return batch.positions.new_zeros(batch.n_struct).index_add(0, batch.struct[i], e)
