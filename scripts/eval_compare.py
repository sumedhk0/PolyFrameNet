"""Accuracy on the identical held-out test split: PolyFrameNet vs MACE, plus the
frame-conditioning histogram the spec asks for (smallest H-to-axis angle per block).

Both models are evaluated in float64 on the same frames with the same error definitions:
per-atom energy MAE (meV/atom), force-component MAE/RMSE (meV/A).
"""

import json
import pathlib
import sys

import numpy as np
import torch
from ase import Atoms

from frame_potential.data import load_pe_frames, split_by_system
from frame_potential.topology import END, MID
from frame_potential.train import evaluate, load_model


def mace_metrics(model_path, frames):
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=str(model_path), device="cuda", default_dtype="float64")
    e_err, f_err = [], []
    for f in frames:
        at = Atoms(numbers=f.numbers, positions=f.positions)
        at.calc = calc
        e_err.append((at.get_potential_energy() - f.energy) / len(f.numbers))
        f_err.append((at.get_forces() - f.forces).ravel())
    e = np.asarray(e_err) * 1000
    fe = np.concatenate(f_err) * 1000
    return {
        "e_mae_meV_atom": float(np.abs(e).mean()),
        "e_rmse_meV_atom": float(np.sqrt((e**2).mean())),
        "f_mae_meV_A": float(np.abs(fe).mean()),
        "f_rmse_meV_A": float(np.sqrt((fe**2).mean())),
        "n_frames": len(frames),
    }


def frame_conditioning(frames):
    """For every frame-defining hydrogen: sin of angle between its C-H bond and e1; and per
    block |sum of perpendicular parts|, the norm normalised into e2 (our recipe degenerates
    only if this goes to zero)."""
    out = {MID: [], END: [], "e2norm": []}
    for f in frames:
        for t, (ca, cb, c_h, hs) in ((MID, (0, 1, 0, (2, 3))), (END, (0, 1, 1, (5, 6)))):
            idx = f.topo.block_atoms[t]
            if not len(idx):
                continue
            X = f.positions[idx]
            e1 = X[:, cb] - X[:, ca]
            e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            acc = np.zeros_like(e1)
            for h in hs:
                u = X[:, h] - X[:, c_h]
                u /= np.linalg.norm(u, axis=1, keepdims=True)
                out[t].append(np.linalg.norm(np.cross(u, e1), axis=1))
                acc += u - (u * e1).sum(1, keepdims=True) * e1
            out["e2norm"].append(np.linalg.norm(acc, axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}


if __name__ == "__main__":
    raw, ours, mace, out = map(pathlib.Path, sys.argv[1:5])
    frames = load_pe_frames(raw, raw / "pe_frames.pkl")
    train, val, test = split_by_system(frames, seed=0)
    res = {"ours": evaluate(load_model(ours, "cuda"), test, "cuda"),
           "mace": mace_metrics(mace, test)}
    fc = frame_conditioning(frames)
    res["frame_conditioning"] = {
        str(k): {"min_sin": float(v.min()), "p01_sin": float(np.percentile(v, 1)),
                 "median_sin": float(np.median(v)), "n": int(len(v))} for k, v in fc.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "accuracy.json").write_text(json.dumps(res, indent=1))
    np.savez(out / "frame_conditioning.npz", mid=fc[MID], end=fc[END], e2norm=fc["e2norm"])
    print(json.dumps(res, indent=1))
    torch.cuda.empty_cache()
