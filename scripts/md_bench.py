"""Speed and stability benchmark on periodic PE melts, PolyFrameNet vs MACE.

Same structures, same GPU, same MD engine (ASE velocity Verlet), float32 for both.
Speed: wall time per energy+force call (median over repeats, after warm-up), reported
also as ns/day at a 1 fs timestep. Neighbour lists differ (vesin for ours, matscipy
inside MACE's calculator); our neighbour-list time is measured and reported separately.

Stability: NVE from 450 K Maxwell-Boltzmann velocities, total-energy drift (linear fit).
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet

from frame_potential.batch import collate, radius_graph
from frame_potential.builders import pe_melt_grown, tile
from frame_potential.calculator import PolyFrameNetCalculator
from frame_potential.topology import topology_from_geometry
from frame_potential.train import load_model


def melt_atoms(n_chains, n_carbons, density, seed):
    z, p, cell = pe_melt_grown(n_chains, n_carbons, density, np.random.default_rng(seed))
    return Atoms(numbers=z, positions=p, cell=cell, pbc=True)


def time_calc(calc, atoms, repeats=20):
    atoms = atoms.copy()
    atoms.calc = calc
    for _ in range(3):
        atoms.rattle(1e-4)
        atoms.get_forces()
    ts = []
    for _ in range(repeats):
        atoms.rattle(1e-4)
        torch.cuda.synchronize()
        t = time.perf_counter()
        atoms.get_forces()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def time_neighbor_list(atoms, cutoff, repeats=20):
    topo = topology_from_geometry(atoms.numbers, atoms.positions, np.asarray(atoms.cell))
    b = collate([topo], [atoms.positions], [np.asarray(atoms.cell)]).to("cuda", torch.float32)
    radius_graph(b, cutoff)
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        radius_graph(b, cutoff)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def nve(calc, atoms, steps, dt_fs, T=450.0, seed=0, log_every=10):
    atoms = atoms.copy()
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(seed))
    Stationary(atoms)
    ZeroRotation(atoms)
    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    t, e, temp = [], [], []

    def log():
        t.append(dyn.nsteps * dt_fs / 1000.0)
        e.append(atoms.get_total_energy() / len(atoms))
        temp.append(atoms.get_temperature())

    dyn.attach(log, interval=log_every)
    t0 = time.perf_counter()
    ok = True
    try:
        dyn.run(steps)
    except Exception as exc:  # blow-up
        print("MD failed:", exc)
        ok = False
    wall = time.perf_counter() - t0
    t, e = np.asarray(t), np.asarray(e)
    slope = float(np.polyfit(t, e, 1)[0]) * 1000 if len(t) > 2 else float("nan")  # meV/atom/ps
    return {
        "steps": int(dyn.nsteps), "dt_fs": dt_fs, "ok": ok and bool(np.isfinite(e).all()),
        "drift_meV_atom_ps": slope, "e_std_meV_atom": float(np.std(e) * 1000),
        "T_mean": float(np.mean(temp)), "T_final": float(temp[-1]), "wall_s": wall,
        "trace_t_ps": t.tolist(), "trace_e_eV_atom": e.tolist(), "final_positions":
        atoms.get_positions().tolist(),
    }


def chain_observables(atoms, topo):
    """Backbone torsion trans fraction, mean end-to-end distance, intermolecular C-C RDF.
    Chains are unwrapped along their bonds (minimum image per C-C step)."""
    pos = atoms.get_positions()
    L = np.diag(np.asarray(atoms.cell))
    chains = {}
    for c in np.unique(topo.chain_of_atom):
        carb = [a for a in np.nonzero((topo.chain_of_atom == c) & (topo.numbers == 6))[0]]
        chains[c] = carb
    tors, ree, unwrapped = [], [], {}
    nbr = {i: [] for i in range(len(pos))}
    for a, b in topo.bonds:
        if topo.numbers[a] == 6 and topo.numbers[b] == 6:
            nbr[a].append(b)
            nbr[b].append(a)
    for c, carb in chains.items():
        start = [a for a in carb if len(nbr[a]) == 1][0]
        order, prev = [start], -1
        while True:
            nxt = [x for x in nbr[order[-1]] if x != prev]
            if not nxt:
                break
            prev = order[-1]
            order.append(nxt[0])
        X = [pos[order[0]]]
        for a in order[1:]:
            d = pos[a] - X[-1]
            d -= np.round(d / L) * L
            X.append(X[-1] + d)
        X = np.asarray(X)
        unwrapped[c] = X
        ree.append(np.linalg.norm(X[-1] - X[0]))
        for k in range(len(X) - 3):
            b1, b2, b3 = X[k + 1] - X[k], X[k + 2] - X[k + 1], X[k + 3] - X[k + 2]
            n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
            cosphi = np.dot(n1, n2) / np.linalg.norm(n1) / np.linalg.norm(n2)
            tors.append(np.degrees(np.arccos(np.clip(cosphi, -1, 1))))
    tors = np.asarray(tors)
    C = np.nonzero(topo.numbers == 6)[0]
    ch = topo.chain_of_atom[C]
    d = pos[C][:, None] - pos[C][None]
    d -= np.round(d / L) * L
    r = np.linalg.norm(d, axis=-1)[ch[:, None] != ch[None]]
    edges = np.linspace(0, min(L) / 2, 61)
    hist, _ = np.histogram(r, bins=edges)
    return {"trans_fraction": float(np.mean(tors > 120)), "mean_ree_A": float(np.mean(ree)),
            "rdf_counts": hist.tolist(), "rdf_edges": edges.tolist()}


def nvt_observables(calc, atoms, steps, dt_fs=0.5, T=450.0, seed=0, every=20):
    from ase.md.langevin import Langevin

    atoms = atoms.copy()
    atoms.calc = calc
    topo = topology_from_geometry(atoms.numbers, atoms.positions, np.asarray(atoms.cell))
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, dt_fs * units.fs, temperature_K=T, friction=0.01 / units.fs,
                   rng=np.random.default_rng(seed))
    samples, temps = [], []

    def sample():
        if dyn.nsteps >= steps // 2:  # second half only
            samples.append(chain_observables(atoms, topo))
        temps.append(atoms.get_temperature())

    dyn.attach(sample, interval=every)
    ok = True
    try:
        dyn.run(steps)
    except Exception as exc:
        print("NVT failed:", exc)
        ok = False
    if not samples:
        return {"ok": False}
    return {
        "ok": ok, "steps": int(dyn.nsteps), "T_mean": float(np.mean(temps)),
        "trans_fraction": float(np.mean([s["trans_fraction"] for s in samples])),
        "mean_ree_A": float(np.mean([s["mean_ree_A"] for s in samples])),
        "rdf_counts": np.mean([s["rdf_counts"] for s in samples], 0).tolist(),
        "rdf_edges": samples[0]["rdf_edges"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", type=pathlib.Path, required=True)
    p.add_argument("--mace", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--nve-steps", type=int, default=4000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--nvt-steps", type=int, default=4000)
    a = p.parse_args()
    from mace.calculators import MACECalculator

    ours = PolyFrameNetCalculator(load_model(a.ours, "cuda", torch.float32), "cuda")
    mace = MACECalculator(model_paths=str(a.mace), device="cuda", default_dtype="float32")
    res = {"speed": [], "nve": {}}
    base = melt_atoms(8, 40, 0.50, seed=1)
    for reps in (1, 2, 3):
        z, p, c = tile(base.numbers, base.positions, np.asarray(base.cell), reps)
        atoms = Atoms(numbers=z, positions=p, cell=c, pbc=True)
        ours.topo = None
        row = {"n_atoms": len(atoms), "ours_s": time_calc(ours, atoms),
               "ours_nl_s": time_neighbor_list(atoms, ours.model.cutoff)}
        try:
            row["mace_s"] = time_calc(mace, atoms)
        except RuntimeError as exc:  # OOM at large N
            row["mace_s"] = None
            print("mace failed", exc)
            torch.cuda.empty_cache()
        for k in ("ours", "mace"):
            s = row[f"{k}_s"]
            row[f"{k}_ns_day_1fs"] = None if s is None else 86400 / s * 1e-6
        print(row, flush=True)
        res["speed"].append(row)
    atoms = melt_atoms(8, 40, 0.50, seed=2)
    for name, calc in (("ours", ours), ("mace", mace)):
        if name == "ours":
            ours.topo = None
        r = nve(calc, atoms, a.nve_steps, a.dt)
        summary = {k: v for k, v in r.items() if not k.startswith(("trace", "final"))}
        print(name, summary, flush=True)
        res["nve"][name] = r
    res["nvt"] = {}
    for name, calc in (("ours", ours), ("mace", mace)):
        if name == "ours":
            ours.topo = None
        r = nvt_observables(calc, atoms, a.nvt_steps)
        print(name, {k: v for k, v in r.items() if not k.startswith("rdf")}, flush=True)
        res["nvt"][name] = r
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "md_bench.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
