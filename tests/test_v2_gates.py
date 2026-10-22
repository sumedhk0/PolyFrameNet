"""v2 correctness gates on an untrained model (float64), across hydrocarbons, heteroatom
polymer fragments the model is never trained on, frame-free blocks and periodic boxes.

An untrained model with random weights is the right subject: training cannot hide a
symmetry violation that is structurally absent, and a violation shows up at O(1)."""

import copy

import numpy as np
import pytest
import torch

from frame_potential import gates
from frame_potential.builders import pe_chain, pe_cluster, pe_melt, random_rotation
from frame_potential.v2.batch import collate_v2
from frame_potential.v2.gates import (
    drop_first_variant,
    finite_difference_error_v2,
    make_energy_fn_v2,
    signature_mismatch,
    variant_signatures,
)
from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.model import N_ELEM, PolyFrameNetV2
from v2_molecules import HETERO, HYDROCARBONS, cluster_of, ether_chain, g2, ion_pair

MIRROR = np.diag([-1.0, 1.0, 1.0])


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(1)
    m = PolyFrameNetV2().double()
    e0 = np.zeros(N_ELEM)
    e0[1], e0[6], e0[8] = -13.6, -1029.0, -2040.0
    m.set_reference(e0)
    return m.eval()


@pytest.fixture(scope="module")
def model_noref(model):
    """Same weights, zero offsets: FD differences of a large total lose precision to
    float64 rounding; offsets have zero gradient, so this tests the same forces."""
    m = copy.deepcopy(model)
    m.set_reference(np.zeros(N_ELEM))
    return m


@pytest.fixture(scope="module")
def systems():
    rng = np.random.default_rng(5)
    return {
        "hydrocarbons": cluster_of([g2(n) for n in HYDROCARBONS], rng),
        "hetero": cluster_of([g2(n) for n in HETERO], rng),
        "peo_ions": cluster_of([ether_chain(12), ion_pair()], rng, spacing=12.0),
        "pe_cluster": pe_cluster(3, 10, rng),
    }


@pytest.fixture(scope="module")
def melt():
    return pe_melt(4, 16, 0.35, np.random.default_rng(4))


@pytest.mark.parametrize("name", ["hydrocarbons", "hetero", "peo_ions", "pe_cluster"])
def test_rotation_permutation_reflection_cluster(model, systems, name, rng):
    z, p = systems[name]
    f = make_energy_fn_v2(model)
    assert gates.rotation_translation_error(f, z, p, None, rng) < 1e-9
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9
    assert abs(f(z, p @ MIRROR) - f(z, p)) < 1e-9


def test_periodic_rotation_permutation_reflection(model, melt, rng):
    z, p, cell = melt
    f = make_energy_fn_v2(model)
    assert gates.rotation_translation_error(f, z, p, cell, rng) < 1e-9
    assert gates.permutation_error(f, z, p, cell, rng) < 1e-9
    assert abs(f(z, p @ MIRROR, cell @ MIRROR) - f(z, p, cell)) < 1e-9


def test_wrap_periodic(model, melt, rng):
    z, p, cell = melt
    assert gates.wrap_error(make_energy_fn_v2(model), z, p, cell, rng) < 1e-9


@pytest.mark.parametrize("name", ["hetero", "peo_ions"])
def test_finite_difference_cluster(model_noref, systems, name):
    z, p = systems[name]
    assert finite_difference_error_v2(model_noref, z, p, h=1e-5) < 1e-8


def test_finite_difference_periodic(model_noref, melt):
    z, p, cell = melt
    assert finite_difference_error_v2(model_noref, z, p, cell, h=1e-5) < 1e-8


def test_frame_free_and_linear(model, rng):
    z, p = cluster_of([g2("CO2"), g2("CH3CN"), ion_pair(), g2("2-butyne")], rng)
    f = make_energy_fn_v2(model)
    assert np.isfinite(f(z, p))
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9
    assert abs(f(z, p @ MIRROR) - f(z, p)) < 1e-9


def test_charged_input_invariance(model, systems, rng):
    z, p = systems["hetero"]
    f = make_energy_fn_v2(model, charge=1.0, spin=2.0)
    f0 = make_energy_fn_v2(model)
    assert abs(f(z, p) - f0(z, p)) > 1e-6  # conditioning is live
    assert gates.rotation_translation_error(f, z, p, None, rng) < 1e-9
    assert gates.permutation_error(f, z, p, None, rng) < 1e-9


def test_variant_set_oracle(systems, rng):
    """True oracle: the per-block multiset of variant-frame coordinates is unchanged by
    relabelling + rotation + mirror, i.e. the variant set is closed under the symmetry."""
    for name in ("hydrocarbons", "hetero"):
        z, p = systems[name]
        ref = variant_signatures(z, p)
        perm = rng.permutation(len(z))
        Q = random_rotation(rng) @ MIRROR
        got = variant_signatures(z[perm], (p @ Q.T)[perm])
        got = {frozenset(int(perm[i]) for i in k): v for k, v in got.items()}
        assert ref.keys() == got.keys()
        for k in ref:
            assert signature_mismatch(ref[k], got[k]) < 1e-9


def test_negative_control_dropped_variant(model, systems, rng):
    z, p = systems["hydrocarbons"]
    f = make_energy_fn_v2(model, topo_hook=drop_first_variant)
    worst = max(gates.permutation_error(f, z, p, None, rng),
                abs(f(z, p @ MIRROR) - f(z, p)))
    assert worst > 1e-6


def test_cutoff_and_switch_smooth():
    torch.manual_seed(2)
    z1, p1 = pe_chain(8)
    for kw, d_far in ((dict(use_coulomb=False, use_dispersion=False), 9.0),
                      (dict(use_coulomb=False, use_dispersion=True), 15.0)):
        m = PolyFrameNetV2(**kw).double().eval()
        f = make_energy_fn_v2(m)
        ds = np.linspace(4.0, d_far + 3.0, 500)
        Es = np.array([f(np.concatenate([z1, z1]), np.concatenate([p1, p1 + [0, 0, d]]))
                       for d in ds])
        d1, d2 = np.abs(np.diff(Es)), np.abs(np.diff(Es, 2))
        assert np.ptp(Es[:50]) > 1e-6  # the chains interact at short range
        assert d2.max() < 0.2 * d1.max() + 1e-12
        assert np.ptp(Es[ds > d_far]) < 1e-12


def test_batch_equals_single(model, systems, melt):
    z1, p1 = systems["hetero"]
    z2, p2 = g2("C6H6")
    t1, t2 = topology_from_geometry_v2(z1, p1), topology_from_geometry_v2(z2, p2)
    s1 = float(model(collate_v2([t1], [p1]))[0])
    s2 = float(model(collate_v2([t2], [p2]))[0])
    both = model(collate_v2([t1, t2], [p1, p2 + 100.0]))
    assert abs(float(both[0]) - s1) < 1e-9 and abs(float(both[1]) - s2) < 1e-9
    z, p, cell = melt
    t = topology_from_geometry_v2(z, p, cell)
    single = float(model(collate_v2([t], [p], [cell]))[0])
    both = model(collate_v2([t, t], [p, p + 0.3], [cell, cell]))
    assert abs(float(both[0]) - single) < 1e-9 and abs(float(both[1]) - single) < 1e-9


def test_float32_matches_float64(model, systems):
    z, p = systems["hetero"]
    t = topology_from_geometry_v2(z, p)
    b = collate_v2([t], [p])
    m32 = copy.deepcopy(model).float()
    e64 = float(model(b)[0])
    out32 = m32(b.to("cpu", torch.float32))
    assert out32.dtype == torch.float64
    assert abs(float(out32[0]) - e64) < 1e-4 * len(z) ** 0.5


def test_oracle_can_fail(systems):
    """The oracle comparison must detect a genuine geometric difference."""
    z, p = systems["hydrocarbons"]
    q = p.copy()
    q[0] += [0.01, 0.0, 0.0]
    a, b = variant_signatures(z, p), variant_signatures(z, q)
    assert max(signature_mismatch(a[k], b[k]) for k in a) > 1e-3
