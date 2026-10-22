"""v2 training loop: same objective, optimiser and schedule as v1 (frame_potential.train),
v2 model and data. --config selects the data; nothing else changes between the hackathon
run and a full-OPoly26 run.

Loss: 20 * mean((E_pred - E_ref) / N)^2 + mean (F_pred - F_ref)^2, forces by autodiff with
create_graph. AdamW (amsgrad), cosine decay over the wall-clock budget, EMA 0.99, float32
training, float64 evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import time

import numpy as np
import torch

from ..batch import radius_graph
from ..data import make_batches
from ..model import energy_and_forces
from ..topology import TopologyError, topology_from_geometry
from ..train import _ema_update
from .data import (
    collate_frames_v2,
    fit_reference_general,
    load_v2_frames,
    pe_split_keys,
    split_v2,
)
from .model import PolyFrameNetV2
from .registry import BlockRegistry


def evaluate_v2(model, frames, device, max_atoms=4000, dtype=torch.float64):
    """MAE/RMSE of per-atom energy (meV/atom) and force components (meV/A)."""
    m = copy.deepcopy(model).to(dtype).eval()
    e_err, f_err = [], []
    for ids in make_batches(frames, max_atoms):
        b = collate_frames_v2(frames, ids).to(device, dtype)
        E, F = energy_and_forces(m, b)
        n = torch.bincount(b.struct, minlength=b.n_struct).double()
        e_err.append(((E.detach() - b.energy) / n).cpu().numpy())
        f_err.append((F.detach() - b.forces).cpu().numpy().ravel())
    e = np.concatenate(e_err) * 1000
    f = np.concatenate(f_err) * 1000
    return {"e_mae_meV_atom": float(np.abs(e).mean()),
            "e_rmse_meV_atom": float(np.sqrt((e**2).mean())),
            "f_mae_meV_A": float(np.abs(f).mean()),
            "f_rmse_meV_A": float(np.sqrt((f**2).mean())), "n_frames": len(frames)}


def load_model_v2(path, device="cpu", dtype=torch.float64):
    ck = torch.load(path, map_location=device)
    m = PolyFrameNetV2(**ck["config"])
    m.load_state_dict(ck["state"])
    return m.to(device, dtype).eval()


def v1_pe_frames(frames, keys):
    """Frames from the given v1 PE systems that pass v1's PE topology: exactly the frames
    v1 used (other frames of those MD cells are copolymer fragments v1 rejected)."""
    out = []
    for f in frames:
        if f.key in keys:
            try:
                topology_from_geometry(f.numbers, f.positions)
            except TopologyError:
                continue
            out.append(f)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=pathlib.Path, required=True)
    p.add_argument("--config", default="hydrocarbon")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--max-epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-atoms", type=int, default=2000)
    p.add_argument("--w-energy", type=float, default=20.0)
    p.add_argument("--w-forces", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--max-train-frames", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty; refusing to overwrite a finished run")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = load_v2_frames(a.raw, a.config, cache=a.raw / f"v2_{a.config}.pkl")
    pe_split = pe_split_keys(a.raw)
    train, val, test = split_v2(frames, pe_split, seed=a.seed)
    if a.max_train_frames:
        train = train[: a.max_train_frames]
    pe_test = v1_pe_frames(test, pe_split[2])
    print(f"frames {len(frames)}: train {len(train)} val {len(val)} test {len(test)} "
          f"(v1 PE test frames {len(pe_test)})", flush=True)

    model = PolyFrameNetV2()
    model.set_reference(fit_reference_general(train))
    f_rms = float(np.sqrt(np.mean(np.concatenate([f.forces.ravel() for f in train]) ** 2)))
    model.scale.fill_(f_rms)
    tot_e = tot_a = 0
    for f in train[:50]:
        b = collate_frames_v2([f], [0])
        tot_e += len(radius_graph(b, model.cutoff)[0])
        tot_a += b.n_atoms
    model.avg_nbr = model.config["avg_nbr"] = max(tot_e / tot_a, 1.0)
    model.to(device, torch.float32)
    ema = copy.deepcopy(model)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"params {n_params}, force rms {f_rms:.3f} eV/A, avg nbr {model.avg_nbr:.1f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6, amsgrad=True)
    rng = np.random.default_rng(a.seed)
    t0, budget, best, history, step = time.time(), a.minutes * 60, math.inf, [], 0
    for epoch in range(a.max_epochs):
        model.train()
        frac = min((time.time() - t0) / budget, 1.0)
        for g in opt.param_groups:  # cosine decay on wall-clock fraction
            g["lr"] = a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        batches = make_batches(train, a.max_atoms, rng)
        tl, nb, n_skip = 0.0, 0, 0
        for ids in batches:
            b = collate_frames_v2(train, ids).to(device, torch.float32)
            E, F = energy_and_forces(model, b, create_graph=True)
            n = torch.bincount(b.struct, minlength=b.n_struct).double()
            loss = (a.w_energy * (((E - b.energy) / n) ** 2).mean().float()
                    + a.w_forces * ((F - b.forces) ** 2).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
                # One non-finite step would poison every weight (and the EMA). Skip it and
                # count it: a run with many skips is diverging, not converging.
                n_skip += 1
                continue
            opt.step()
            _ema_update(ema, model, a.ema)
            tl += float(loss.detach())
            nb += 1
            step += 1
            if time.time() - t0 > budget:
                break
        elapsed = time.time() - t0
        if epoch % a.eval_every == 0 or elapsed > budget:
            mtr = evaluate_v2(ema, val, device)
            mtr.update(epoch=epoch, step=step, minutes=elapsed / 60, train_loss=tl / max(nb, 1),
                       skipped=n_skip)
            history.append(mtr)
            score = mtr["e_rmse_meV_atom"] + mtr["f_rmse_meV_A"] / 20
            tag = ""
            if score < best:
                best = score
                torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "best.pt")
                tag = " *"
            print(f"ep {epoch:4d} {elapsed / 60:6.1f} min loss {tl / max(nb, 1):.3e} skip {n_skip} "
                  f"val E {mtr['e_mae_meV_atom']:.2f} meV/at F {mtr['f_mae_meV_A']:.1f} meV/A{tag}",
                  flush=True)
            (a.out / "history.json").write_text(json.dumps(history, indent=1))
        if elapsed > budget:
            break
    torch.save({"state": ema.state_dict(), "config": ema.config}, a.out / "last.pt")
    bm = load_model_v2(a.out / "best.pt", device)
    BlockRegistry.build(bm, train, device).save(a.out / "registry.json")
    res = {"val": evaluate_v2(bm, val, device), "test": evaluate_v2(bm, test, device),
           "pe_test": evaluate_v2(bm, pe_test, device), "n_params": n_params,
           "args": {k: str(v) for k, v in vars(a).items()}}
    (a.out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
