"""Build and cache the v2 dataset for a data config; print its composition.
Usage: python scripts/build_v2_dataset.py <raw dir> <config>"""

import collections
import pathlib
import sys
import time

import numpy as np

from frame_potential.v2.data import load_v2_frames, pe_split_keys, split_v2

if __name__ == "__main__":
    raw, config = pathlib.Path(sys.argv[1]), sys.argv[2]
    t0 = time.time()
    frames = load_v2_frames(raw, config, cache=raw / f"v2_{config}.pkl")
    pe_split = pe_split_keys(raw)
    tr, va, te = split_v2(frames, pe_split)
    n_at = sum(len(f.numbers) for f in frames)
    sizes = collections.Counter(int((row >= 0).sum()) for f in frames
                                for row in f.topo.block_atoms)
    print(f"{len(frames)} frames, {n_at} atoms, {len({f.key for f in frames})} systems, "
          f"{time.time() - t0:.0f} s")
    print(f"split frames: train {len(tr)} val {len(va)} test {len(te)}")
    pe_te_frames = [f for f in te if f.key in pe_split[2]]
    print(f"frames from v1 PE systems: {sum(f.key in set().union(*pe_split) for f in frames)}, "
          f"PE test frames in v2 test: {len(pe_te_frames)}")
    print("block sizes:", sorted(sizes.items()))
    nv = [len(f.topo.var_block) / f.topo.n_blocks for f in frames]
    print(f"variants per block: mean {np.mean(nv):.1f}")
