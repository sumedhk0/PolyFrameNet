import numpy as np
import torch

from frame_potential.builders import pe_chain, pe_cluster
from frame_potential.v2.graph import topology_from_geometry_v2
from frame_potential.v2.model import PolyFrameNetV2
from frame_potential.v2.registry import BlockRegistry, block_hashes, block_label
from v2_molecules import g2


class _F:  # minimal frame for registry building
    def __init__(self, z, p):
        self.numbers, self.positions, self.topo = z, p, topology_from_geometry_v2(z, p)
        self.forces, self.energy, self.charge, self.spin = np.zeros_like(p), 0.0, 0.0, 1.0


def test_hashes_permutation_invariant(rng):
    z, p = g2("isobutane")
    t = topology_from_geometry_v2(z, p)
    perm = rng.permutation(len(z))
    t2 = topology_from_geometry_v2(z[perm], p[perm])
    assert sorted(map(tuple, block_hashes(t).tolist())) == sorted(
        map(tuple, block_hashes(t2).tolist()))


def test_labels():
    z, p = g2("C6H6")
    assert block_label(topology_from_geometry_v2(z, p), 0) == "C6H6"
    z, p = g2("isobutane")
    t = topology_from_geometry_v2(z, p)
    assert sorted(block_label(t, k) for k in range(t.n_blocks)) == [
        "CH3[C]", "CH3[C]", "CH3[C]", "CH[C3]"]


def test_registry_seen_and_novel(tmp_path):
    torch.manual_seed(0)
    m = PolyFrameNetV2().double().eval()
    rng = np.random.default_rng(0)
    frames = [_F(*pe_cluster(2, 8, rng, noise=0.0)) for _ in range(3)]
    reg = BlockRegistry.build(m, frames, "cpu")
    reg.save(tmp_path / "r.json")
    reg = BlockRegistry.load(tmp_path / "r.json")
    z, p = pe_chain(6)
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    assert {r["verdict"] for r in rep} == {"seen"}
    z, p = g2("isobutane")  # 3 CH3 (seen in PE: C + 3H, one external C) and 1 CH (3 ext C)
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    verdicts = sorted(r["verdict"] for r in rep)
    assert verdicts.count("seen") == 3 and any(v != "seen" for v in verdicts)
    z, p = g2("C6H6")
    rep = reg.report(m, topology_from_geometry_v2(z, p), p)
    assert rep[0]["level"] is None and rep[0]["verdict"] != "seen"


def test_calculator_warns_on_novel_blocks(tmp_path):
    import warnings

    from ase import Atoms

    from frame_potential.v2.calculator import PolyFrameNetV2Calculator

    torch.manual_seed(0)
    m = PolyFrameNetV2().double().eval()
    frames = [_F(*pe_cluster(2, 8, np.random.default_rng(1), noise=0.0))]
    reg = BlockRegistry.build(m, frames, "cpu")
    z, p = g2("CF4")
    at = Atoms(numbers=z, positions=p)
    at.calc = PolyFrameNetV2Calculator(m, "cpu", torch.float64, registry=reg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        at.get_potential_energy()
    assert any("unseen in training" in str(x.message) for x in w)


def test_registry_report_leaves_model_dtype_alone():
    from ase import Atoms

    from frame_potential.v2.calculator import PolyFrameNetV2Calculator

    torch.manual_seed(0)
    m = PolyFrameNetV2().double().eval()
    reg = BlockRegistry.build(m, [_F(*pe_chain(6))], "cpu")
    z, p = pe_chain(6)
    at = Atoms(numbers=z, positions=p)
    calc = PolyFrameNetV2Calculator(PolyFrameNetV2().eval(), "cpu", torch.float32, registry=reg)
    at.calc = calc
    at.get_potential_energy()
    assert next(calc.model.parameters()).dtype == torch.float32
