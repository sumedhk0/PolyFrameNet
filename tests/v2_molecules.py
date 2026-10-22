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
