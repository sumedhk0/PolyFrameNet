"""Write the benchmark PE melts (md_bench.melt_atoms, tiled 1x/2x/3x) as extxyz."""

import pathlib
import sys

import numpy as np
from ase import Atoms
from ase.io import write

sys.path.insert(0, "scripts")
from md_bench import melt_atoms  # noqa: E402

from frame_potential.builders import tile  # noqa: E402

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    base = melt_atoms(8, 40, 0.50, seed=1)
    for r in (1, 2, 3):
        z, p, c = tile(base.numbers, base.positions, np.asarray(base.cell), r)
        write(out / f"melt_r{r}.xyz", Atoms(numbers=z, positions=p, cell=c, pbc=True))
    write(out / "melt_nve.xyz", melt_atoms(8, 40, 0.50, seed=2))
