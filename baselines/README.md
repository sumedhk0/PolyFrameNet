# Baselines on the v2 hydrocarbon split

Each baseline lives in its own virtual environment, so its pinned dependencies cannot
touch the main environment's CUDA torch. Every model trains on the same files:
`extxyz_v2/{train,val,test,pe_test}.xyz` from `scripts/export_v2_extxyz.py`.
`$D = C:/Users/sumedh/polyframenet_data`.

```bash
uv venv --python 3.12 $HOME/venvs/bl_allegro
uv pip install --python $HOME/venvs/bl_allegro/Scripts/python.exe --torch-backend=cu128 nequip-allegro ase
uv venv --python 3.12 $HOME/venvs/bl_deepmd
uv pip install --python $HOME/venvs/bl_deepmd/Scripts/python.exe --torch-backend=cu128 "deepmd-kit[torch]" dpdata ase
uv pip uninstall --python $HOME/venvs/bl_deepmd/Scripts/python.exe vesin vesin-torch
```

These installs gave torch 2.11.0+cu128, nequip 0.19.1, nequip-allegro 0.8.3 and
deepmd-kit 3.2.0.

## Allegro

The config is `allegro/allegro_v2.yaml`. Its main settings:
- r_max 5.0, l_max 2, 2 layers, 64 scalar and 32 tensor features;
- per-type energy shifts fixed to the shared E0 (`extxyz_v2/e0.json`);
- loss: 20 × per-atom energy MSE + force MSE;
- Adam at lr 1e-3, EMA 0.99;
- float32, with a 90 min wall-clock cap (`trainer.max_time`).

```bash
python -m nequip.scripts.train -cp <repo>/baselines/allegro -cn allegro_v2 hydra.run.dir=$D/runs/allegro_v2
```

**Inference is eager, not compiled.**
- nequip 0.19 refuses TorchScript compilation on torch ≥ 2.10.
- AOTInductor compilation needs MSVC, which this machine does not have.
- So `scripts/eval_ase.py` loads `best.ckpt` through
  `NequIPCalculator._from_saved_model`, uncompiled. Allegro's speed numbers are therefore
  pessimistic for Allegro.

## DeepMD

The config is `deepmd/input.json`. Its main settings:
- descriptor se_e2_a, rcut 6.0, PyTorch backend;
- `sel` = [96, 72], from `dp neighbor-stat` on the training split (at most 90 H and
  67 C neighbours).

```bash
python baselines/deepmd/to_deepmd.py $D/extxyz_v2 $D/deepmd_v2        # dpdata npy systems
python baselines/deepmd/launch.py $D/deepmd_v2 $D/runs/deepmd_v2 <numb_steps>
```

Deviations, all disclosed:
- **Loss schedule.** DeepMD uses its own prefactor schedule (energy 0.02 → 1, force
  1000 → 1), not a fixed E 20 / F 1.
- **Energy reference.** DeepMD fits its own per-type energy bias rather than using the
  shared E0.
- **vesin removed.** The `vesin` DLL in this venv is blocked by Windows Application
  Control (`WinError 4551`). deepmd imports it optionally but does not catch `OSError`,
  so the package is removed and deepmd falls back to its built-in neighbour list.
- **First smoke run.** It crashed with a CUDA illegal memory access. That run used
  `sel` = [60, 30], below the real neighbour counts; it trains cleanly at [96, 72].

## Evaluation and benchmarks

Run in each model's own environment:

```bash
python scripts/eval_ase.py <kind> <model> $D/extxyz_v2/test.xyz <out.json>
python scripts/bench_ase.py <kind> <model> $D/melts/melt_r1.xyz speed <out.json>
```

Kinds and model files:

| kind | model file |
|---|---|
| allegro | `best.ckpt` |
| deepmd | `model.pth` |
| mace | `mace_v2.model` |
| v2 | `best.pt` |

UMA was dropped from this run at the user's request.
