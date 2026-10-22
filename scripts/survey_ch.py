"""Survey the raw C/H shards for the v2 (carbon-centred block) design: how many frames have
a clean hydrocarbon bond graph, which carbon environments occur, and how they are mixed.

A frame is "clean" when every H has exactly one neighbour (a C) and every C has 2-4
neighbours. Carbon environment = (number of C neighbours, number of H neighbours).
"""

import collections
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

CC_MAX, CH_MAX, HH_MAX = 1.85, 1.35, 1.0


def survey_shard(f: pathlib.Path):
    d = np.load(f)
    off, z_all, r_all, names = d["offsets"], d["numbers"], d["positions"], d["names"]
    out = []
    for k in range(len(off) - 1):
        z = z_all[off[k] : off[k + 1]]
        r = r_all[off[k] : off[k + 1]]
        pairs = cKDTree(r).query_pairs(CC_MAX, output_type="ndarray")
        dist = np.linalg.norm(r[pairs[:, 0]] - r[pairs[:, 1]], axis=1)
        zi, zj = z[pairs[:, 0]], z[pairs[:, 1]]
        hh = np.where((zi == 1) & (zj == 1), HH_MAX, CH_MAX)
        cut = np.where((zi == 6) & (zj == 6), CC_MAX, hh)
        b = pairs[dist < cut]
        nbr_c = np.zeros(len(z), int)
        nbr_h = np.zeros(len(z), int)
        for a, c in ((0, 1), (1, 0)):
            np.add.at(nbr_c, b[:, a], z[b[:, c]] == 6)
            np.add.at(nbr_h, b[:, a], z[b[:, c]] == 1)
        h, c = z == 1, z == 6
        ok = bool((nbr_c[h] + nbr_h[h] == 1).all() and (nbr_h[h] == 0).all())
        deg = nbr_c[c] + nbr_h[c]
        ok = ok and bool(((deg >= 2) & (deg <= 4)).all())
        envs = collections.Counter(zip(nbr_c[c].tolist(), nbr_h[c].tolist(), strict=True))
        out.append((ok, len(z), str(names[k]), envs))
    return out


if __name__ == "__main__":
    files = sorted(pathlib.Path(sys.argv[1]).glob("OPoly26-*.npz"))
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = [r for part in ex.map(survey_shard, files) for r in part]
    n_ok = sum(r[0] for r in rows)
    print(f"frames {len(rows)}, clean {n_ok}")
    env = collections.Counter()
    sizes = [r[1] for r in rows if r[0]]
    for r in rows:
        if r[0]:
            env.update(r[3])
    print("atoms per clean frame: median", int(np.median(sizes)), "max", max(sizes))
    print("carbon environments (nC, nH): count")
    for k, v in env.most_common():
        print(" ", k, v)
    sp = sum(1 for r in rows if r[0] and any(nc + nh == 2 for nc, nh in r[3]))
    print("clean frames containing a 2-coordinate (sp) carbon:", sp)
    systems = {r[2].split("/")[-2] if "/" in r[2] else r[2] for r in rows if r[0]}
    print("distinct source paths:", len(systems))
    print("sample names:", [r[2] for r in rows[:: max(1, len(rows) // 8)]][:8])
