"""v2 data: config-selected OPoly26 frames, system-level split. The hackathon config is
H/C only; the cluster config is everything. Same code, different config.

Split by system (frames carved from one MD cell share a key and a split), and v1's PE
systems keep their v1 split, so v1's PE test systems stay held out from v2 training."""

from __future__ import annotations

import pathlib
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..data import load_pe_frames, split_by_system, system_key
from ..topology import TopologyError
from .batch import collate_v2
from .graph import GraphTopology, topology_from_geometry_v2
from .model import N_ELEM

# n_extra: frames drawn (by whole system) from systems outside v1's PE systems. Every
# frame of v1's PE systems is always included (they keep their v1 split, so no leak).
DATA_CONFIGS = {
    "hydrocarbon": {"elements": {1, 6}, "n_extra": 1500},
    "all": {"elements": None, "n_extra": None},
}


@dataclass
class FrameV2:
    numbers: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    energy: float
    topo: GraphTopology
    key: str
    charge: float = 0.0
    spin: float = 1.0


def _index_shard(args):
    f, elements = args
    d = np.load(f)
    off, z, names = d["offsets"], d["numbers"], d["names"]
    out = []
    for k in range(len(off) - 1):
        name = str(names[k])
        if "reactivity/" in name:  # AFIR reaction paths: out of scope (non-reactive)
            continue
        if elements is not None and not set(np.unique(z[off[k]: off[k + 1]]).tolist()) <= elements:
            continue
        out.append((str(f), k, system_key(name)))
    return out


def _load_selected(args):
    f, ks = args
    d = np.load(f)
    # NpzFile re-decompresses on every key access: read each array once.
    off, z_all, r_all, f_all = d["offsets"], d["numbers"], d["positions"], d["forces"]
    e_all, mult, names = d["energy"], d["multiplicity"], d["names"]
    out = []
    for k in ks:
        sl = slice(off[k], off[k + 1])
        try:
            topo = topology_from_geometry_v2(z_all[sl], r_all[sl])
        except TopologyError:
            continue
        out.append(FrameV2(z_all[sl], r_all[sl], f_all[sl], float(e_all[k]), topo,
                           system_key(str(names[k])), 0.0, float(mult[k])))
    return out


def pe_split_keys(raw: pathlib.Path):
    """(train, val, test) system keys of v1's PE split."""
    pe = load_pe_frames(raw, raw / "pe_frames.pkl")
    return tuple({f.key for f in part} for part in split_by_system(pe, seed=0))


def load_v2_frames(raw: pathlib.Path, config: str, seed: int = 0, cache=None):
    if cache is not None and pathlib.Path(cache).exists():
        return pickle.loads(pathlib.Path(cache).read_bytes())
    cfg = DATA_CONFIGS[config]
    files = sorted(raw.glob("OPoly26-*.npz"))
    with ProcessPoolExecutor(max_workers=6) as ex:
        index = [r for part in ex.map(_index_shard, [(f, cfg["elements"]) for f in files])
                 for r in part]
    by_key: dict[str, list] = {}
    for f, k, key in index:
        by_key.setdefault(key, []).append((f, k))
    pe_keys = set().union(*pe_split_keys(raw))
    chosen = [k for k in sorted(by_key) if k in pe_keys]
    rest = sorted(set(by_key) - pe_keys)
    np.random.default_rng(seed).shuffle(rest)
    budget = cfg["n_extra"]
    total = 0
    for k in rest:
        if budget is not None and total >= budget:
            break
        chosen.append(k)
        total += len(by_key[k])
    per_file: dict[str, list[int]] = {}
    for key in chosen:
        for f, k in by_key[key]:
            per_file.setdefault(f, []).append(k)
    with ProcessPoolExecutor(max_workers=6) as ex:
        frames = [fr for part in ex.map(_load_selected, sorted(per_file.items())) for fr in part]
    if cache is not None:
        pathlib.Path(cache).write_bytes(pickle.dumps(frames))
    return frames


def split_v2(frames, pe_split, seed: int = 0, fractions=(0.8, 0.1, 0.1)):
    """v1's PE systems keep their v1 split; every other system is split 80/10/10."""
    pe_tr, pe_va, pe_te = pe_split
    pe_all = pe_tr | pe_va | pe_te
    other = sorted({f.key for f in frames} - pe_all)
    np.random.default_rng(seed).shuffle(other)
    a = int(round(fractions[0] * len(other)))
    b = int(round((fractions[0] + fractions[1]) * len(other)))
    parts = [pe_tr | set(other[:a]), pe_va | set(other[a:b]), pe_te | set(other[b:])]
    return [[f for f in frames if f.key in p] for p in parts]


def fit_reference_general(frames):
    """Least-squares per-element reference energies, indexed by atomic number."""
    els = sorted({int(z) for f in frames for z in np.unique(f.numbers)})
    A = np.array([[np.sum(f.numbers == z) for z in els] for f in frames], float)
    y = np.array([f.energy for f in frames])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    e0 = np.zeros(N_ELEM)
    e0[els] = coef
    return e0


def collate_frames_v2(frames, ids):
    fs = [frames[k] for k in ids]
    return collate_v2([f.topo for f in fs], [f.positions for f in fs], None,
                      [f.energy for f in fs], [f.forces for f in fs],
                      [f.charge for f in fs], [f.spin for f in fs])
