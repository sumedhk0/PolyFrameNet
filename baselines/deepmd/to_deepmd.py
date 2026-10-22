"""Convert the v2 extxyz split to DeepMD npy systems (run inside the bl_deepmd venv).
Usage: python baselines/deepmd/to_deepmd.py <extxyz_v2 dir> <out dir>
Writes <out>/{train,val}/<formula>/... and <out>/{train,val}.txt listing the systems."""

import pathlib
import sys

import dpdata
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read

if __name__ == "__main__":
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    for split in ("train", "val"):
        ms = dpdata.MultiSystems(type_map=["H", "C"])
        for at in read(src / f"{split}.xyz", ":"):
            at.calc = SinglePointCalculator(at, energy=at.info["REF_energy"],
                                            forces=at.arrays["REF_forces"])
            ms.append(dpdata.LabeledSystem(at, fmt="ase/structure", type_map=["H", "C"]))
        ms.to_deepmd_npy(str(out / split))
        systems = sorted(str(p.parent) for p in (out / split).rglob("type.raw"))
        (out / f"{split}.txt").write_text("\n".join(systems))
        print(split, len(systems), "systems,", sum(len(s) for s in ms.systems.values()), "frames")
