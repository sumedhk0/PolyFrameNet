"""ASE calculator. Topology is read once from the first geometry and then frozen
(non-reactive). Energy and forces are recomputed from scratch at every call: no state
carried between timesteps except the integer topology."""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .batch import collate
from .model import energy_and_forces
from .topology import topology_from_geometry


class PolyFrameNetCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, device="cuda", dtype=torch.float32, **kw):
        super().__init__(**kw)
        self.model = model.to(device, dtype).eval()
        self.device = device
        self.dtype = dtype
        self.topo = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        cell = np.asarray(atoms.cell) if atoms.pbc.any() else None
        if self.topo is None or not np.array_equal(self.topo.numbers, atoms.numbers):
            self.topo = topology_from_geometry(atoms.numbers, pos, cell)
        b = collate([self.topo], [pos], None if cell is None else [cell])
        b = b.to(self.device, self.dtype)
        E, F = energy_and_forces(self.model, b)
        self.results["energy"] = float(E.detach().sum())  # float64 (see model.e0_micro)
        self.results["forces"] = F.detach().double().cpu().numpy()
