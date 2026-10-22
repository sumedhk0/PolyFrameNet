"""Speed / NVE for any ASE calculator on exported melts, same protocol as md_bench.py:
3 warm-up calls with 1e-4 A rattles, then the median of 20 timed energy + force calls;
NVE from 450 K Maxwell-Boltzmann velocities, velocity Verlet, dt 0.5 fs, 6000 steps.
Runs inside each model's own environment; models run in float32 (energy summed in float64
where the model does so).
Usage: python scripts/bench_ase.py <kind> <model> <structure.xyz> <speed|nve> <out.json>"""

import json
import sys
import time

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

sys.path.insert(0, "scripts")
from eval_ase import make_calc  # noqa: E402


def speed(calc, atoms):
    atoms.calc = calc
    for _ in range(3):
        atoms.rattle(1e-4)
        atoms.get_forces()
    ts = []
    for _ in range(20):
        atoms.rattle(1e-4)
        t = time.perf_counter()
        atoms.get_forces()
        ts.append(time.perf_counter() - t)
    return {"n_atoms": len(atoms), "s_per_step": float(np.median(ts))}


def nve(calc, atoms, steps=6000, dt_fs=0.5):
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=450, rng=np.random.default_rng(0))
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    t, e = [], []
    t0 = time.time()
    for k in range(steps // 10):
        dyn.run(10)
        t.append((k + 1) * 10 * dt_fs / 1000)
        e.append(atoms.get_total_energy() / len(atoms))
    t, e = np.asarray(t), np.asarray(e)
    return {"steps": steps, "dt_fs": dt_fs, "ok": bool(np.isfinite(e).all()),
            "drift_meV_atom_ps": float(np.polyfit(t, e, 1)[0] * 1000),
            "e_std_meV_atom": float(np.std(e) * 1000), "wall_s": time.time() - t0,
            "trace_t_ps": t.tolist(), "trace_e_eV_atom": e.tolist()}


if __name__ == "__main__":
    kind, model, struct, task, out = sys.argv[1:6]
    import torch

    atoms = read(struct)
    calc = make_calc(kind, model, "float32")
    torch.cuda.reset_peak_memory_stats()
    try:
        res = speed(calc, atoms) if task == "speed" else nve(calc, atoms)
    except Exception as exc:  # OOM at large N: record, do not crash the sweep
        res = {"n_atoms": len(atoms), "s_per_step": None, "error": str(exc)[:300]}
    res.update(kind=kind, model=model, peak_gpu_GB=torch.cuda.max_memory_allocated() / 1e9)
    with open(out, "w") as fh:
        json.dump(res, fh)
    print({k: v for k, v in res.items() if not k.startswith("trace")})
