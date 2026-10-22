"""Block registry: which blocks the training data contained, for flagging unfamiliar
chemistry at inference. Diagnostic only: the model never reads it (shared weights, never
shared values). Block identity is computed inside the block, independent of context:
  L0 composition  multiset of (Z, sorted Z of external neighbours)
  L1 local        one WL round over the block's internal bonds
  L2 exact        WL over internal bonds refined to stability
"""

from __future__ import annotations

import copy
import json
import pathlib
from collections import Counter

import numpy as np
import torch
from ase.data import chemical_symbols

from ..data import make_batches
from .batch import collate_v2
from .graph import _h

LEVELS = ("L0", "L1", "L2")
_KEEP = 200  # embeddings kept per exact class for spread statistics


def _adjacency(topo):
    nb = [[] for _ in range(topo.n_atoms)]
    for a, b in topo.bonds.tolist():
        nb[a].append(b)
        nb[b].append(a)
    return nb


def block_hashes(topo) -> np.ndarray:
    nb = _adjacency(topo)
    out = np.zeros((topo.n_blocks, 3), dtype=np.int64)
    for k in range(topo.n_blocks):
        atoms = [int(a) for a in topo.block_atoms[k] if a >= 0]
        inside = set(atoms)
        col = {i: _h(int(topo.numbers[i]),
                     tuple(sorted(int(topo.numbers[j]) for j in nb[i] if j not in inside)))
               for i in atoms}
        out[k, 0] = _h(tuple(sorted(col.values())))
        n_cls = len(set(col.values()))
        for rnd in range(20):
            col = {i: _h(col[i], tuple(sorted(col[j] for j in nb[i] if j in inside)))
                   for i in atoms}
            if rnd == 0:
                out[k, 1] = _h(tuple(sorted(col.values())))
            k_new = len(set(col.values()))
            if rnd > 0 and k_new == n_cls:
                break
            n_cls = k_new
        out[k, 2] = _h(tuple(sorted(col.values())))
    return out


def block_label(topo, k) -> str:
    nb = _adjacency(topo)
    atoms = [int(a) for a in topo.block_atoms[k] if a >= 0]
    inside = set(atoms)
    comp = Counter(chemical_symbols[int(topo.numbers[i])] for i in atoms)
    ext = Counter(chemical_symbols[int(topo.numbers[j])] for i in atoms for j in nb[i]
                  if j not in inside)
    f = "".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(comp.items()))
    e = ",".join(f"{s}{n if n > 1 else ''}" for s, n in sorted(ext.items()))
    return f"{f}[{e}]" if e else f


def _embeddings(model, topos, positions, device, cells=None):
    b = collate_v2(topos, positions, cells).to(device, torch.float64)
    cap = {}
    m = copy.deepcopy(model).to(device, torch.float64).eval()  # never retype the caller's
    with torch.no_grad():
        m(b, capture=cap)
    return cap["H2"].cpu().numpy()


class BlockRegistry:
    def __init__(self, counts, labels, means, spread, threshold):
        self.counts, self.labels, self.means = counts, labels, means
        self.spread, self.threshold = spread, threshold

    @classmethod
    def build(cls, model, frames, device, max_atoms=4000):
        counts = {lv: Counter() for lv in LEVELS}
        labels, samples = {}, {}
        for ids in make_batches(frames, max_atoms):
            fs = [frames[i] for i in ids]
            H = _embeddings(model, [f.topo for f in fs], [f.positions for f in fs], device)
            row = 0
            for f in fs:
                hs = block_hashes(f.topo)
                for k in range(f.topo.n_blocks):
                    for li, lv in enumerate(LEVELS):
                        counts[lv][str(hs[k, li])] += 1
                    key = str(hs[k, 2])
                    labels.setdefault(key, block_label(f.topo, k))
                    s = samples.setdefault(key, [])
                    if len(s) < _KEEP:
                        s.append(H[row + k])
                row += f.topo.n_blocks
        means = {k: np.mean(v, 0) for k, v in samples.items()}
        rms = [float(np.sqrt(((np.asarray(v) - means[k]) ** 2).sum(1).mean()))
               for k, v in samples.items() if len(v) > 1]
        spread = float(np.median(rms)) if rms and np.median(rms) > 0 else 1.0
        within = [np.linalg.norm(x - means[k]) / spread for k, v in samples.items() for x in v]
        threshold = float(np.percentile(within, 95)) if within else 1.0
        return cls({lv: dict(c) for lv, c in counts.items()}, labels, means, spread, threshold)

    def report(self, model, topo, positions, cell=None, device="cpu"):
        H = _embeddings(model, [topo], [positions], device, None if cell is None else [cell])
        hs = block_hashes(topo)
        keys = list(self.means)
        M = np.stack([self.means[k] for k in keys]) if keys else np.zeros((0, H.shape[1]))
        out = []
        for k in range(topo.n_blocks):
            level, count = None, 0
            for li in (2, 1, 0):
                c = self.counts[LEVELS[li]].get(str(hs[k, li]), 0)
                if c:
                    level, count = LEVELS[li], c
                    break
            d = np.linalg.norm(M - H[k], axis=1) / self.spread if len(M) else np.array([np.inf])
            j = int(np.argmin(d))
            if level == "L2":
                verdict = "seen"
            elif level == "L1" or float(d[j]) < self.threshold:
                verdict = "similar"
            else:
                verdict = "novel"
            out.append({"block": k, "label": block_label(topo, k), "level": level,
                        "count": count, "verdict": verdict,
                        "nearest": self.labels[keys[j]] if keys else None,
                        "distance": float(d[j])})
        return out

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps({
            "counts": self.counts, "labels": self.labels,
            "means": {k: v.tolist() for k, v in self.means.items()},
            "spread": self.spread, "threshold": self.threshold}))

    @classmethod
    def load(cls, path):
        d = json.loads(pathlib.Path(path).read_text())
        return cls(d["counts"], d["labels"], {k: np.asarray(v) for k, v in d["means"].items()},
                   d["spread"], d["threshold"])
