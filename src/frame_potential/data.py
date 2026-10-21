"""OPoly26 polyethylene subset: filter, group, split, batch.

Filter: C/H-only frames (scripts/fetch_opoly26_ch.py) whose geometry reads as a set of
linear saturated alkane chains that cut into C2 blocks (topology.build_topology). In
OPoly26 these are hydrogen-capped fragments carved from polymer MD cells, so they are
n-alkane chains: CH3-(CH2)k-CH3.

Split by system, never by frame: frames carved from the same MD cell share a "system
key" (the source path before `_Hterm_frame`) and all go to the same split. Consecutive
MD frames are near-duplicates, so a random frame split would leak.
"""

from __future__ import annotations

import pathlib
import pickle
import re
from dataclasses import dataclass

import numpy as np

from .batch import collate
from .topology import Topology, TopologyError, topology_from_geometry


@dataclass
class Frame:
    numbers: np.ndarray
    positions: np.ndarray
    forces: np.ndarray
    energy: float
    topo: Topology
    key: str  # system key for grouping
    source: str  # "train" or "val" (OPoly26 split it came from)


def system_key(name: str) -> str:
    part = name.split("/")[-2] if "/" in name else name
    return re.sub(r"_Hterm.*", "", part)


def _quick_reject(z: np.ndarray) -> bool:
    """Saturated acyclic hydrocarbons have H = 2C + 2m (m molecules, m >= 1)."""
    nc = int((z == 6).sum())
    nh = int((z == 1).sum())
    return nh < 2 * nc + 2 or (nh - 2 * nc) % 2 == 1


def _load_shard(f: pathlib.Path) -> list[Frame]:
    d = np.load(f)
    source = "val" if "val" in f.name else "train"
    off = d["offsets"]
    # NpzFile re-decompresses on every key access: read each array once.
    z_all, r_all, f_all = d["numbers"], d["positions"], d["forces"]
    e_all, names = d["energy"], d["names"]
    out = []
    for k in range(len(off) - 1):
        z = z_all[off[k] : off[k + 1]]
        if _quick_reject(z):
            continue
        r = r_all[off[k] : off[k + 1]]
        try:
            topo = topology_from_geometry(z, r)
        except TopologyError:
            continue
        out.append(
            Frame(z, r, f_all[off[k] : off[k + 1]], float(e_all[k]), topo,
                  system_key(str(names[k])), source)
        )
    return out


def load_pe_frames(raw_dir: pathlib.Path, cache: pathlib.Path | None = None) -> list[Frame]:
    if cache is not None and cache.exists():
        return pickle.loads(cache.read_bytes())
    from concurrent.futures import ProcessPoolExecutor

    files = sorted(raw_dir.glob("OPoly26-*.npz"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        frames = [fr for part in ex.map(_load_shard, files) for fr in part]
    if cache is not None:
        cache.write_bytes(pickle.dumps(frames))
    return frames


def split_by_system(frames: list[Frame], fractions=(0.8, 0.1, 0.1), seed: int = 0):
    keys = sorted({f.key for f in frames})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    a = int(round(fractions[0] * n))
    b = int(round((fractions[0] + fractions[1]) * n))
    parts = [set(keys[:a]), set(keys[a:b]), set(keys[b:])]
    return [[f for f in frames if f.key in p] for p in parts]


def fit_reference(frames: list[Frame]):
    """Least-squares per-element reference energies E0[H], E0[C] and the per-atom std of
    the residual (used to scale the readout)."""
    A = np.array([[np.sum(f.numbers == 1), np.sum(f.numbers == 6)] for f in frames], float)
    y = np.array([f.energy for f in frames])
    e0, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = (y - A @ e0) / A.sum(1)
    return e0, float(np.std(res)) if len(frames) > 1 else 1.0


def make_batches(frames: list[Frame], max_atoms: int, rng: np.random.Generator | None = None):
    idx = np.arange(len(frames))
    if rng is not None:
        rng.shuffle(idx)
    out, cur, n = [], [], 0
    for k in idx:
        na = len(frames[k].numbers)
        if cur and n + na > max_atoms:
            out.append(cur)
            cur, n = [], 0
        cur.append(k)
        n += na
    if cur:
        out.append(cur)
    return out


def collate_frames(frames: list[Frame], ids):
    fs = [frames[k] for k in ids]
    return collate([f.topo for f in fs], [f.positions for f in fs], None,
                   [f.energy for f in fs], [f.forces for f in fs])


def to_extxyz(frames: list[Frame], path: pathlib.Path):
    """Write frames as extxyz (for MACE baseline training on identical data)."""
    sym = {1: "H", 6: "C"}
    with open(path, "w") as fh:
        for f in frames:
            fh.write(f"{len(f.numbers)}\n")
            fh.write(
                f'Properties=species:S:1:pos:R:3:REF_forces:R:3 REF_energy={f.energy!r} '
                f'pbc="F F F" key={f.key}\n'
            )
            for z, r, fo in zip(f.numbers, f.positions, f.forces, strict=True):
                fh.write(f"{sym[int(z)]} {r[0]:.8f} {r[1]:.8f} {r[2]:.8f} "
                         f"{fo[0]:.8f} {fo[1]:.8f} {fo[2]:.8f}\n")
