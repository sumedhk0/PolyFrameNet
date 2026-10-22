"""Exact v2 split as extxyz for all baselines, plus the shared E0 (same reference for every
model). pe_test.xyz holds exactly v1's 77 PE test frames.
Usage: python scripts/export_v2_extxyz.py <raw dir> <out dir>"""

import json
import pathlib
import sys

from frame_potential.data import to_extxyz
from frame_potential.v2.data import fit_reference_general, load_v2_frames, pe_split_keys, split_v2
from frame_potential.v2.train import v1_pe_frames

if __name__ == "__main__":
    raw, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    frames = load_v2_frames(raw, "hydrocarbon", cache=raw / "v2_hydrocarbon.pkl")
    pe_split = pe_split_keys(raw)
    tr, va, te = split_v2(frames, pe_split)
    parts = {"train": tr, "val": va, "test": te, "pe_test": v1_pe_frames(te, pe_split[2])}
    for name, part in parts.items():
        to_extxyz(part, out / f"{name}.xyz")
    e0 = fit_reference_general(tr)
    (out / "e0.json").write_text(json.dumps({"1": e0[1], "6": e0[6]}))
    print({k: len(v) for k, v in parts.items()}, "E0 H, C:", e0[1], e0[6])
