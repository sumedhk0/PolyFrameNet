# PolyFrameNet model card

| File | Model | Trained on |
|---|---|---|
| `polyframenet_v2_hydrocarbon.pt` | v2: general polymer architecture, 673k parameters | 3,429 OPoly26 hydrocarbon frames |
| `polyframenet_v2_hydrocarbon.registry.json` | block registry for the v2 model | same training frames |
| `polyframenet_v1_pe.pt` | v1: polyethylene-only architecture, 656k parameters | 494 OPoly26 polyethylene frames |

## Intended use

Energies and forces for non-reactive hydrocarbon polymers: polyethylene, polypropylene,
polystyrene-like aromatics, and branched or unsaturated chains, for molecular dynamics
through ASE.

The v2 *architecture* accepts any element, charge and spin, but this checkpoint has only
seen neutral, closed-shell C/H structures. Run the block registry on any new system. It
reports, for each building block, whether the model saw it in training (`seen`), saw a
close relative (`similar`), or never saw anything like it (`novel`).

```python
import torch
from ase.io import read
from frame_potential.v2.calculator import PolyFrameNetV2Calculator
from frame_potential.v2.registry import BlockRegistry
from frame_potential.v2.train import load_model_v2

model = load_model_v2("models/polyframenet_v2_hydrocarbon.pt", "cuda", torch.float32)
registry = BlockRegistry.load("models/polyframenet_v2_hydrocarbon.registry.json")
atoms = read("my_polymer.xyz")
atoms.calc = PolyFrameNetV2Calculator(model, "cuda", torch.float32, registry=registry)
print(atoms.get_potential_energy(), atoms.get_forces().shape)  # warns on novel blocks
```

To print the registry report on its own:

```bash
python scripts/block_coverage.py <dir with best.pt + registry.json> my_polymer.xyz
```

## Training data

OPoly26 (ColabFit mirror). These are ωB97M-V/def2-TZVPD energies and forces on H-capped
clusters of up to 357 atoms, carved from polymer MD.

The data is split by MD system, not by frame. v1's polyethylene test systems stay held
out of v2. One training frame was removed: a CH3 with an H at exactly 180° to its C–C
bond, which makes a frame singular (see `docs/writeup.md`).

## Evaluation

MAE, float64, identical splits for all models (`scripts/eval_ase.py`):

| | Energy (meV/atom) | Force (meV/Å) |
|---|---|---|
| v2, hydrocarbon test (782 frames) | 1.19 | 43.4 |
| v2, v1's polyethylene test (77 frames) | 0.43 | 18.5 |
| v1, polyethylene test (77 frames) | 0.33 | 13.7 |

The baselines on the same splits are in `docs/writeup.md`.

## Limitations

- **Clusters only.** Training data is non-periodic. Bulk melts are an extrapolation, and
  there is no stress output, so no NPT.
- **Accuracy gap.** v2 trained 90 min (312 epochs) on one NVIDIA L40S. MACE trained the same way has about half of v2's errors (see `docs/writeup.md` section 8).
- **Near-collinear frames.** A frame built from two bonds near 180° is singular. Frames
  are filtered in the data, but not yet guarded at inference (planned: sin²-weighted
  frame averaging).
- **Untrained components.** Charge, spin and electrostatics are present but untrained
  for anything except neutral hydrocarbons.

## License

MIT, see `LICENSE`.
