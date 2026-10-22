"""Build registry.json for a finished v2 run from its training split.
Usage: python scripts/build_registry.py <raw dir> <run dir>"""

import pathlib
import sys

import torch

from frame_potential.v2.data import load_v2_frames, pe_split_keys, split_v2
from frame_potential.v2.registry import BlockRegistry
from frame_potential.v2.train import load_model_v2

if __name__ == "__main__":
    raw, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = load_v2_frames(raw, "hydrocarbon", cache=raw / "v2_hydrocarbon.pkl")
    train, _, _ = split_v2(frames, pe_split_keys(raw))
    reg = BlockRegistry.build(load_model_v2(run / "best.pt", device), train, device)
    reg.save(run / "registry.json")
    print({lv: len(c) for lv, c in reg.counts.items()}, "exact classes;",
          f"spread {reg.spread:.3f}, similar threshold {reg.threshold:.2f}")
