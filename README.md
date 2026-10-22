# PolyFrameNet: a polymer potential that knows what a polymer is

PolyFrameNet is a machine-learned interatomic potential for polymers. It exploits three
facts that general-purpose potentials ignore:
- polymers are built from a few repeating building blocks;
- their covalent bonds never change during a simulation;
- every block carries its own local coordinate frame.

Expressing geometry in those frames makes every network a plain MLP, with no tensor
products. Averaging over all symmetry-equivalent frames keeps the energy *exactly*
invariant to rotation, reflection and relabelling. Forces are exact gradients, so energy
is conserved in MD.

- **One-page summary:** [docs/submission/one_pager.md](docs/submission/one_pager.md)
- **Full design, results and every bug found:** [docs/writeup.md](docs/writeup.md)
- **Teaching walkthrough of the whole architecture:**
  [docs/explainer/architecture_explained.md](docs/explainer/architecture_explained.md)
- **Pretrained weights and model card:** [models/](models/)

## Two versions

| | v1 | v2 |
|---|---|---|
| Chemistry | Polyethylene only | Any element, any non-reactive polymer (trained here on hydrocarbons) |
| Blocks | Hand-defined C2 units | Rigid fragments read from the bond graph: ring systems, single heavy atoms, terminal atoms join their neighbour |
| Frames | 4 variants (chain direction × mirror) | Every frame drawn from Weisfeiler–Lehman colour classes, × mirror |
| Long range | none | Latent-charge electrostatics (Ewald) + learned dispersion; charge/spin inputs |
| Code | `src/frame_potential/*.py` | `src/frame_potential/v2/` |

The hackathon model and a full-scale model share one architecture. Only the data config
differs (`--config hydrocarbon` vs `--config all`).

## Quick start

```bash
pip install -e ".[dev,bench]"                     # Python 3.12+, PyTorch with CUDA
pytest                                            # 81 correctness tests, about 30 s on Linux
```

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
atoms.get_forces()  # warns if the structure contains block types never seen in training
```

## Layout

| Path | What |
|---|---|
| `src/frame_potential/v2/graph.py` | Bonds by covalent radii, block rule, WL colours, frame variants, block graph |
| `src/frame_potential/v2/model.py` | `PolyFrameNetV2`: encoder, bonded channel, valence prior, many-body layer, readout |
| `src/frame_potential/v2/longrange.py` | Erf-screened Coulomb, Ewald summation, dispersion tail |
| `src/frame_potential/v2/registry.py` | Block registry: seen / similar / novel per block |
| `src/frame_potential/v2/{data,train,calculator}.py` | Config-driven OPoly26 loader, training loop, ASE calculator |
| `tests/test_v2_*.py` | Correctness gates on hydrocarbons and on heteroatom molecules never trained on |
| `baselines/` | MACE, Allegro and DeepMD configs on the identical split |
| `scripts/` | Data build, exports, model-agnostic accuracy and speed benchmarks, figures |

## Reproduce

Data is OPoly26 (ColabFit mirror) at ωB97M-V/def2-TZVPD. `$D` is a data directory.

```bash
python scripts/fetch_opoly26_ch.py --out $D/opoly26_ch          # C/H frames from OPoly26
python scripts/build_v2_dataset.py $D/opoly26_ch hydrocarbon    # v2 dataset + composition
python -m frame_potential.v2.train --raw $D/opoly26_ch --out $D/runs/v2 --minutes 90
python scripts/export_v2_extxyz.py $D/opoly26_ch $D/extxyz_v2  # same split for baselines
python scripts/eval_ase.py v2 $D/runs/v2/best.pt $D/extxyz_v2/test.xyz out.json
```

Baseline training commands are in [baselines/README.md](baselines/README.md).

## License

MIT
