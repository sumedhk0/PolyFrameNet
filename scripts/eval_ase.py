"""Accuracy of any ASE calculator on an extxyz split with REF_energy / REF_forces.
Usage: python scripts/eval_ase.py <kind> <model> <split.xyz> <out.json>
kind: v1 | v2 | mace | allegro | deepmd | uma. Runs inside that model's own environment."""

import json
import sys

import numpy as np
from ase.io import read


def make_calc(kind, model, dtype="float64"):
    """dtype: float64 for accuracy; bench_ase passes float32 for speed / MD (as v1 did)."""
    if kind == "v2":
        import torch

        from frame_potential.v2.calculator import PolyFrameNetV2Calculator
        from frame_potential.v2.train import load_model_v2
        return PolyFrameNetV2Calculator(load_model_v2(model, "cuda"), "cuda",
                                        getattr(torch, dtype))
    if kind == "v1":
        import torch

        from frame_potential.calculator import PolyFrameNetCalculator
        from frame_potential.train import load_model
        return PolyFrameNetCalculator(load_model(model, "cuda"), "cuda", getattr(torch, dtype))
    if kind == "mace":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=model, device="cuda", default_dtype=dtype)
    if kind == "allegro":
        # Eager checkpoint, not compiled: nequip 0.19 refuses TorchScript on torch >= 2.10
        # and AOTInductor needs MSVC (not installed). Slower than compiled Allegro.
        from nequip.integrations.ase import NequIPCalculator
        return NequIPCalculator._from_saved_model(model, device="cuda")
    if kind == "deepmd":
        from deepmd.calculator import DP
        return DP(model=model)
    if kind == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        return FAIRChemCalculator(pretrained_mlip.get_predict_unit(model, device="cuda"),
                                  task_name="omol")
    raise ValueError(kind)


if __name__ == "__main__":
    kind, model, split, out = sys.argv[1:5]
    calc = make_calc(kind, model)
    e_err, f_err = [], []
    for at in read(split, ":"):
        e_ref, f_ref = at.info["REF_energy"], at.arrays["REF_forces"]
        at.info["charge"], at.info["spin"] = 0, 1  # UMA omol inputs; ignored by others
        if hasattr(calc, "topo"):
            # Our calculators read topology once per system (right for MD). Distinct
            # frames can share an atom list but not bonding: rebuild it every frame.
            calc.topo = None
        at.calc = calc
        e_err.append((at.get_potential_energy() - e_ref) / len(at))
        f_err.append((at.get_forces() - f_ref).ravel())
    e = np.asarray(e_err) * 1000
    f = np.concatenate(f_err) * 1000
    res = {"kind": kind, "model": model, "split": split, "n_frames": len(e_err),
           "e_mae_meV_atom": float(np.abs(e).mean()),
           "e_rmse_meV_atom": float(np.sqrt((e**2).mean())),
           "f_mae_meV_A": float(np.abs(f).mean()), "f_rmse_meV_A": float(np.sqrt((f**2).mean()))}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(res)
