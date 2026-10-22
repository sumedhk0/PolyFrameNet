import math

import numpy as np
import torch
from v2_molecules import g2, ion_pair

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


def _ewald_vs_cluster(L, alpha=None):
    z, p = ion_pair()
    p = p + L / 2
    q = torch.tensor([0.5, -0.5])
    cell = np.eye(3) * L
    b = _batch(z, p, cell)
    e_ew = float(coulomb_energy(b, q, radius_graph(b, 10.0), sigma=1.0, r_c=10.0,
                                alpha=alpha)[0])
    e_cl = float(coulomb_energy(_batch(z, p), q, None, sigma=1.0, r_c=10.0)[0])
    M = (q.numpy()[:, None] * p).sum(0)
    return e_ew, e_cl, KE * 2 * math.pi * float(M @ M) / (3 * L**3)


def test_ewald_alpha_independence():
    e_a, _, _ = _ewald_vs_cluster(40.0, alpha=0.35)
    e_b, _, _ = _ewald_vs_cluster(40.0, alpha=0.45)
    assert abs(e_a - e_b) < 1e-6


def test_ewald_converges_to_cluster_limit():
    """Tin-foil Ewald = vacuum (cluster) energy - 2 pi |M|^2 / (3 V), up to O(1/L^5) image
    terms (the dipole-lattice sum vanishes for a cubic lattice). Measured: residual * L^5
    is constant (about -1.45e4) from L = 30 to 80 A, i.e. the right convergence order."""
    res = {}
    for L in (40.0, 80.0):
        e_ew, e_cl, corr = _ewald_vs_cluster(L)
        res[L] = (e_ew - (e_cl - corr)) * L**5
    assert abs(res[80.0] / res[40.0] - 1.0) < 0.1
    e_ew, e_cl, corr = _ewald_vs_cluster(80.0)
    assert abs(e_ew - (e_cl - corr)) < 1e-5


def test_charge_conservation():
    z, p = g2("C5H5N")
    for Q in (-1.0, 0.0, 1.0):
        b = _batch(z, p, charge=Q)
        q = neutralize(torch.randn(b.n_atoms), b)
        assert abs(float(q.sum()) - Q) < 1e-10


def test_dispersion_negative_and_switched():
    z, _ = ion_pair()
    for d, expect_zero in ((5.0, False), (10.5, True)):
        pp = np.array([[0.0, 0, 0], [d, 0, 0]])
        b = _batch(z, pp)
        e = dispersion_energy(b, torch.tensor([10.0, 10.0]), torch.tensor([1.5, 1.5]),
                              radius_graph(b, 10.0), r_off=10.0)
        assert (float(e[0]) == 0.0) == expect_zero and float(e[0]) <= 0.0
