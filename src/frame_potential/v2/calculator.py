"""ASE calculator for v2. Topology read once from the first geometry (non-reactive) and
rebuilt only when the atomic numbers change; energy recomputed from scratch every call."""

from __future__ import annotations

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from ..model import energy_and_forces
from .batch import collate_v2
from .graph import topology_from_geometry_v2


class PolyFrameNetV2Calculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, device="cuda", dtype=torch.float32, charge=0.0, spin=1.0, **kw):
        super().__init__(**kw)
        self.model = model.to(device, dtype).eval()
        self.device, self.dtype = device, dtype
        self.charge, self.spin = charge, spin
        self.topo = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        cell = np.asarray(atoms.cell) if atoms.pbc.any() else None
        if self.topo is None or not np.array_equal(self.topo.numbers, atoms.numbers):
            self.topo = topology_from_geometry_v2(atoms.numbers, pos, cell)
        b = collate_v2([self.topo], [pos], None if cell is None else [cell],
                       charges=[self.charge], spins=[self.spin]).to(self.device, self.dtype)
        E, F = energy_and_forces(self.model, b)
        self.results["energy"] = float(E.detach().sum())  # float64 sum (see e0_micro)
        self.results["forces"] = F.detach().double().cpu().numpy()
