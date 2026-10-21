# Repository for PolyFrameNet, a polymer-specific MLIP for efficient, accurate MD simulation.

PolyFrameNet (block-factored polymer potential) cuts each polymer chain into repeating
two-carbon blocks. It gives every block an orthonormal frame built from its own atoms, so
every network after that is a plain MLP: no tensor products. Forces are exact `-dE/dr` by
autodiff. v1 covers polyethylene and is trained on the polyethylene subset of OPoly26.

The full design, results and benchmark against MACE are in [docs/writeup.md](docs/writeup.md).

## Layout

| Path | What |
|---|---|
| `src/frame_potential/topology.py` | Bonds from geometry (once), chain/block decomposition, valence pairs |
| `src/frame_potential/batch.py` | Batching, radius graph (vesin / brute force), exclusion mask, edge vectors |
| `src/frame_potential/model.py` | Stages 1-6: frames, block encoder, bonded channel, valence prior, non-bonded channel, readout |
| `src/frame_potential/data.py` | OPoly26 polyethylene filter, system-level split, extxyz export |
| `src/frame_potential/train.py` | Training loop (GPU, EMA, float64 evaluation) |
| `src/frame_potential/calculator.py` | ASE calculator for MD |
| `src/frame_potential/gates.py`, `tests/` | Correctness gates: rotation, permutation, wrap, finite-difference forces, cutoff smoothness |
| `scripts/` | Data download, dataset build, MACE export, accuracy comparison, MD benchmark, figures |

## Development setup

Keep the virtual environment and data outside OneDrive: syncing a venv's thousands of files is
slow and can lock files mid-install.

```powershell
uv venv --python 3.13 $HOME\venvs\PolyFrameNet
& $HOME\venvs\PolyFrameNet\Scripts\Activate.ps1
uv pip install -e ".[dev,bench]"
uv pip install --reinstall torch --torch-backend=cu128    # GPU build
```

pyarrow is not used. Windows Application Control blocks its DLL on this machine, so the
parquet download goes through DuckDB instead. For the same reason MACE is run as
`python -m mace.cli.run_train`, not through `mace_run_train.exe`.

## Checks

```powershell
pytest        # every test runs in float64, see tests/conftest.py
ruff check .
```

## Reproduce

See the Reproduce section of [docs/writeup.md](docs/writeup.md).
