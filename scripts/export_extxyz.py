"""Write the exact train/val/test split used by PolyFrameNet as extxyz for the MACE baseline,
plus the fitted per-element reference energies (so both models subtract the same E0)."""

import json
import pathlib
import sys

from frame_potential.data import fit_reference, load_pe_frames, split_by_system, to_extxyz

if __name__ == "__main__":
    raw = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    frames = load_pe_frames(raw, raw / "pe_frames.pkl")
    train, val, test = split_by_system(frames, seed=0)
    for name, part in (("train", train), ("val", val), ("test", test)):
        to_extxyz(part, out / f"{name}.xyz")
        print(name, len(part))
    e0, _ = fit_reference(train)
    (out / "e0.json").write_text(json.dumps({"1": float(e0[0]), "6": float(e0[1])}))
    print("E0s", e0)
