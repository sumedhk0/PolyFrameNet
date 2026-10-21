"""Training loop. GPU, float32 training, float64 validation, EMA weights, system-level split.

Loss (per batch):
    L = w_E * mean_s ((E_pred - E_ref)_s / N_s)^2 + w_F * mean_atoms,xyz (F_pred - F_ref)^2
Per-atom energy error squared is the spec's MSE(E)/N^2. Forces are -dE/dr by autodiff
with create_graph=True so the force loss trains the energy model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import time

import numpy as np
import torch

from .batch import exclusion_keep_mask, radius_graph
from .data import collate_frames, fit_reference, load_pe_frames, make_batches, split_by_system
from .model import PolyFrameNet, energy_and_forces


def measure_avg_neighbors(frames, cutoff, exclude_hops, n=50):
    tot_e, tot_a = 0, 0
    for f in frames[:n]:
        b = collate_frames([f], [0])
        i, j, _ = radius_graph(b, cutoff)
        tot_e += int(exclusion_keep_mask(b, i, j, exclude_hops).sum())
        tot_a += b.n_atoms
    return max(tot_e / max(tot_a, 1), 1.0)


@torch.no_grad()
def _ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
    for be, bm in zip(ema.buffers(), model.buffers(), strict=True):
        be.copy_(bm)


def evaluate(model, frames, device, max_atoms=4000, dtype=torch.float64):
    """MAE/RMSE of per-atom energy (meV/atom), total energy (meV) and forces (meV/A)."""
    m = copy.deepcopy(model).to(dtype)
    m.eval()
    e_err, e_tot, f_err = [], [], []
    for ids in make_batches(frames, max_atoms):
        b = collate_frames(frames, ids).to(device, dtype)
        E, F = energy_and_forces(m, b)
        n = torch.bincount(b.struct, minlength=b.n_struct).to(dtype)
        de = (E.detach() - b.energy)
        e_err.append((de / n).cpu().numpy())
        e_tot.append(de.cpu().numpy())
        f_err.append((F.detach() - b.forces).cpu().numpy().ravel())
    e_err = np.concatenate(e_err) * 1000
    e_tot = np.concatenate(e_tot) * 1000
    f_err = np.concatenate(f_err) * 1000
    return {
        "e_mae_meV_atom": float(np.abs(e_err).mean()),
        "e_rmse_meV_atom": float(np.sqrt((e_err**2).mean())),
        "e_mae_meV": float(np.abs(e_tot).mean()),
        "f_mae_meV_A": float(np.abs(f_err).mean()),
        "f_rmse_meV_A": float(np.sqrt((f_err**2).mean())),
        "n_frames": len(frames),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--max-epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-atoms", type=int, default=2500)
    p.add_argument("--w-energy", type=float, default=20.0)
    p.add_argument("--w-forces", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--cutoff", type=float, default=4.5)
    p.add_argument("--n-bonded", type=int, default=2)
    p.add_argument("--exclude-hops", type=int, default=None)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        # A second launch of the same command once overwrote best.pt with epoch-0 weights
        # 20 s after the real run finished. Never train into a non-empty directory.
        raise SystemExit(f"{a.out} is not empty; refusing to overwrite a finished run")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = load_pe_frames(a.raw, a.raw / "pe_frames.pkl")
    train, val, test = split_by_system(frames, seed=a.seed)
    print(f"frames: {len(frames)}  train {len(train)}  val {len(val)}  test {len(test)}")

    model = PolyFrameNet(cutoff=a.cutoff, hidden=a.hidden, n_bonded=a.n_bonded,
                         exclude_hops=a.exclude_hops)
    e0, _ = fit_reference(train)
    f_rms = float(np.sqrt(np.mean(np.concatenate([f.forces.ravel() for f in train]) ** 2)))
    model.e0.copy_(torch.tensor(e0))
    model.scale.fill_(f_rms)  # readout in units of the typical force scale
    avg = measure_avg_neighbors(train, a.cutoff, model.exclude_hops)
    model.nonbonded.avg_nbr = avg
    model.config["avg_nbr"] = avg
    print(f"E0 [H, C] = {e0}, force rms = {f_rms:.3f} eV/A, kept neighbours/atom = {avg:.1f}")
    model.to(device, torch.float32)
    ema = copy.deepcopy(model)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"parameters: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6, amsgrad=True)
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    budget = a.minutes * 60
    best = math.inf
    history = []
    step = 0
    for epoch in range(a.max_epochs):
        model.train()
        batches = make_batches(train, a.max_atoms, rng)
        frac = min((time.time() - t_start) / budget, 1.0)
        for g in opt.param_groups:  # cosine decay on wall-clock fraction
            g["lr"] = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        tl = 0.0
        for ids in batches:
            b = collate_frames(train, ids).to(device, torch.float32)
            E, F = energy_and_forces(model, b, create_graph=True)
            n = torch.bincount(b.struct, minlength=b.n_struct).float()
            le = (((E - b.energy) / n) ** 2).mean()
            lf = ((F - b.forces) ** 2).mean()
            loss = a.w_energy * le + a.w_forces * lf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            _ema_update(ema, model, a.ema)
            tl += float(loss)
            step += 1
        elapsed = time.time() - t_start
        if epoch % 5 == 0 or elapsed > budget:
            metrics = evaluate(ema, val, device)
            metrics.update(epoch=epoch, step=step, minutes=elapsed / 60,
                           train_loss=tl / len(batches), lr=opt.param_groups[0]["lr"])
            history.append(metrics)
            score = metrics["e_rmse_meV_atom"] + metrics["f_rmse_meV_A"] / 20
            tag = ""
            if score < best:
                best = score
                torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "best.pt")
                tag = " *"
            print(
                f"ep {epoch:4d} {elapsed / 60:6.1f} min  loss {tl / len(batches):.4e}  "
                f"val E {metrics['e_mae_meV_atom']:.2f} meV/at  "
                f"F {metrics['f_mae_meV_A']:.1f} meV/A{tag}",
                flush=True,
            )
            (a.out / "history.json").write_text(json.dumps(history, indent=1))
        if elapsed > budget:
            break
    torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "last.pt")
    ck = torch.load(a.out / "best.pt", map_location=device)
    best_model = PolyFrameNet(**ck["config"])
    best_model.load_state_dict(ck["state"])
    best_model.to(device)
    res = {"val": evaluate(best_model, val, device), "test": evaluate(best_model, test, device),
           "n_params": n_params, "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def load_model(path, device="cpu", dtype=torch.float64):
    ck = torch.load(path, map_location=device)
    m = PolyFrameNet(**ck["config"])
    m.load_state_dict(ck["state"])
    return m.to(device, dtype).eval()


if __name__ == "__main__":
    main()
