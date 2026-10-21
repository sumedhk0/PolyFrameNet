"""Filter the raw C/H shards to polyethylene frames, cache them, print a summary."""

import collections
import pathlib
import sys
import time

import numpy as np

from frame_potential.data import load_pe_frames, split_by_system

if __name__ == "__main__":
    raw = pathlib.Path(sys.argv[1])
    t = time.time()
    fr = load_pe_frames(raw, raw / "pe_frames.pkl")
    print(f"{len(fr)} PE frames in {time.time() - t:.0f} s", flush=True)
    print("source split:", dict(collections.Counter(f.source for f in fr)))
    print("systems:", len({f.key for f in fr}))
    n = np.array([len(f.numbers) for f in fr])
    print(f"atoms/frame mean {n.mean():.0f} min {n.min()} max {n.max()} total {n.sum()}")
    print("chains/frame mean", np.mean([f.topo.block_chain.max() + 1 for f in fr]))
    tr, va, te = split_by_system(fr)
    print("split train/val/test:", len(tr), len(va), len(te))
