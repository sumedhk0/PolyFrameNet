"""Figures for docs/writeup.md. Two series everywhere: PolyFrameNet (slot 1 blue) and MACE
(slot 2 orange), the first two slots of the validated reference categorical palette.
Static PNGs, light surface, recessive grid, direct labels instead of legend boxes."""

import json
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS, MACE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})


def mace_curve(log: pathlib.Path):
    pts = []
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"Epoch (\d+):.*MAE_E_per_atom=\s*([\d.]+) meV, MAE_F=\s*([\d.]+)", line)
        if m:
            pts.append(tuple(float(x) for x in m.groups()))
    return pts


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=INK2,
                va="center", fontsize=9)
    ax.plot([x], [y], "o", color=color, ms=4)


def main(runs: pathlib.Path, out: pathlib.Path):
    out.mkdir(parents=True, exist_ok=True)
    hist = json.loads((runs / "pfn_main" / "history.json").read_text())
    mc = mace_curve(runs / "mace_main.log")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, mi, title in ((axes[0], "e_mae_meV_atom", 1, "Validation energy MAE (meV/atom)"),
                               (axes[1], "f_mae_meV_A", 2, "Validation force MAE (meV/Å)")):
        x = [h["epoch"] for h in hist]
        y = [h[key] for h in hist]
        ax.plot(x, y, color=OURS)
        label_end(ax, x[-1], y[-1], "PolyFrameNet", OURS)
        if mc:
            ax.plot([p[0] for p in mc], [p[mi] for p in mc], color=MACE)
            label_end(ax, mc[-1][0], mc[-1][mi], "MACE", MACE)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=160)

    md = runs / "md" / "md_bench.json"
    if md.exists():
        r = json.loads(md.read_text())
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        n = [s["n_atoms"] for s in r["speed"]]
        for k, c, name in (("ours_s", OURS, "PolyFrameNet"), ("mace_s", MACE, "MACE")):
            pts = [(a, s[k] * 1000) for a, s in zip(n, r["speed"], strict=True) if s[k]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=c, ms=5)
            label_end(ax, pts[-1][0], pts[-1][1], name, c)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("atoms (periodic PE melt)")
        ax.set_title("Energy + forces per MD step (ms, RTX 4080 Laptop)", loc="left",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "speed.png", dpi=160)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        for k, c, name in (("ours", OURS, "PolyFrameNet"), ("mace", MACE, "MACE")):
            tr = r["nve"][k]
            e0 = tr["trace_e_eV_atom"][0]
            y = [(e - e0) * 1000 for e in tr["trace_e_eV_atom"]]
            ax.plot(tr["trace_t_ps"], y, color=c, lw=1.5)
            label_end(ax, tr["trace_t_ps"][-1], y[-1], name, c)
        ax.set_xlabel("time (ps)")
        ax.set_title("NVE total energy drift, 450 K start (meV/atom)", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "nve.png", dpi=160)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
