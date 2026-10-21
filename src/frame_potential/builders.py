"""Test-system and melt builders: linear alkanes / polyethylene chains.

Geometry from internal coordinates (NeRF placement): C-C 1.53 A, C-H 1.09 A,
C-C-C 112 deg, tetrahedral hydrogens. Torsions are trans (180) or gauche (+-65) with
optional Gaussian noise, so test systems are not artificially symmetric.
"""

from __future__ import annotations

import math

import numpy as np

CC, CH = 1.53, 1.09
CCC = math.radians(112.0)
TET = math.radians(109.47)


def _place(a, b, c, bond, angle, torsion):
    """NeRF: position d with |cd| = bond, angle bcd = angle, dihedral abcd = torsion."""
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    d2 = np.array(
        [-bond * math.cos(angle), bond * math.sin(angle) * math.cos(torsion),
         bond * math.sin(angle) * math.sin(torsion)]
    )
    return c + d2[0] * bc + d2[1] * m + d2[2] * n


def pe_chain(n_carbons: int, rng: np.random.Generator | None = None, gauche_frac: float = 0.0,
             noise: float = 0.0):
    """One CH3-(CH2)_{n-2}-CH3 chain. Returns (numbers, positions)."""
    rng = rng or np.random.default_rng(0)
    C = [np.zeros(3), np.array([CC, 0.0, 0.0])]
    C.append(C[1] + CC * np.array([-math.cos(CCC), math.sin(CCC), 0.0]))
    for _k in range(3, n_carbons):
        t = math.pi
        if rng.random() < gauche_frac:
            t = math.radians(65.0) * (1 if rng.random() < 0.5 else -1)
        C.append(_place(C[-3], C[-2], C[-1], CC, CCC, t))
    C = np.asarray(C[:n_carbons])
    H = []
    for k in range(n_carbons):
        if 0 < k < n_carbons - 1:
            u1 = C[k - 1] - C[k]
            u2 = C[k + 1] - C[k]
            u1 /= np.linalg.norm(u1)
            u2 /= np.linalg.norm(u2)
            bis = -(u1 + u2)
            bis /= np.linalg.norm(bis)
            nrm = np.cross(u1, u2)
            nrm /= np.linalg.norm(nrm)
            half = TET / 2
            H.append(C[k] + CH * (bis * math.cos(half) + nrm * math.sin(half)))
            H.append(C[k] + CH * (bis * math.cos(half) - nrm * math.sin(half)))
        else:
            nb = C[1] if k == 0 else C[-2]
            ref = C[2] if k == 0 else C[-3]
            for tor in (math.pi, math.pi / 3, -math.pi / 3):
                H.append(_place(ref, nb, C[k], CH, TET, tor))
    pos = np.concatenate([C, np.asarray(H)], 0)
    if noise:
        pos = pos + rng.normal(scale=noise, size=pos.shape)
    numbers = np.array([6] * n_carbons + [1] * len(H))
    return numbers, pos


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def pe_cluster(n_chains: int, n_carbons: int, rng: np.random.Generator, spread: float = 6.0,
               gauche_frac: float = 0.2, noise: float = 0.05, min_dist: float = 2.3):
    """Several randomly oriented chains packed near each other (non-periodic test cluster),
    with inter-chain contacts inside the non-bonded cutoff."""
    zs, ps = [], []
    allp = np.zeros((0, 3))
    for _c in range(n_chains):
        for _t in range(5000):
            z, p = pe_chain(n_carbons, rng, gauche_frac, noise)
            p = p - p.mean(0)
            p = p @ random_rotation(rng).T + rng.uniform(-spread, spread, 3) * (len(zs) > 0)
            if _ok_geometry(z, p) and (
                not len(allp) or np.linalg.norm(p[:, None] - allp[None], axis=-1).min() > min_dist
            ):
                break
        else:
            raise RuntimeError("could not place chain")
        zs.append(z)
        ps.append(p)
        allp = np.concatenate([allp, p])
    return np.concatenate(zs), np.concatenate(ps)


def _ok_geometry(z, p) -> bool:
    """Reject self-clashing chains (e.g. g+g- pentane clash): no non-bonded heavy-H or
    H-H contact closer than 1.6 A, judged by counting close pairs against the bond count."""
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    n_close = (d[np.triu_indices(len(z), 1)] < 1.6).sum()
    n_bonds = (z == 6).sum() - 1 + (z == 1).sum()
    return n_close == n_bonds


def pe_melt(n_chains: int, n_carbons: int, density: float, rng: np.random.Generator,
            gauche_frac: float = 0.35, min_dist: float = 2.0, max_tries: int = 2000):
    """Cubic periodic box of PE chains at `density` g/cm^3 (start low; equilibrate by MD).

    Chains are grown with random torsions and placed with random rotation/translation,
    rejecting placements whose heavy atoms come closer than min_dist to existing atoms
    (minimum image). Returns (numbers, positions wrapped into the box, cell 3x3)."""
    mass_chain = n_carbons * 12.011 + (2 * n_carbons + 2) * 1.008
    total_g = n_chains * mass_chain / 6.02214076e23
    L = (total_g / density) ** (1 / 3) * 1e8
    cell = np.eye(3) * L
    zs, ps = [], []
    allp = np.zeros((0, 3))
    for _c in range(n_chains):
        for _t in range(max_tries):
            z, p = pe_chain(n_carbons, rng, gauche_frac)
            if not _ok_geometry(z, p):
                continue
            p = p - p.mean(0)
            p = p @ random_rotation(rng).T + rng.random(3) * L
            if len(allp):
                d = p[:, None, :] - allp[None, :, :]
                d -= np.round(d / L) * L
                if np.min(np.linalg.norm(d, axis=-1)) < min_dist:
                    continue
            # self-overlap check across the boundary
            d = p[:, None, :] - p[None, :, :]
            d -= np.round(d / L) * L
            r = np.linalg.norm(d, axis=-1)
            raw = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            if np.any((r < min_dist) & (raw > r + 1e-6)):
                continue
            break
        else:
            raise RuntimeError("could not place chain; lower the density")
        zs.append(z)
        ps.append(p)
        allp = np.concatenate([allp, p], 0)
    pos = np.concatenate(ps) % L
    return np.concatenate(zs), pos, cell
