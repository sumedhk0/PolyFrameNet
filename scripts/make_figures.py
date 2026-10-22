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
    hist = json.loads((runs / "pfn_o3" / "history.json").read_text())
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

    md = runs / "md2"
    if md.exists():
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        for k, c, name in (("ours", OURS, "PolyFrameNet"), ("mace", MACE, "MACE")):
            rows = [json.loads(f.read_text()) for f in sorted(md.glob(f"speed_{k}_r*.json"))]
            pts = sorted((r["n_atoms"], r["s_per_step"] * 1000) for r in rows if r["s_per_step"])
            if not pts:
                continue
            ax.plot([q[0] for q in pts], [q[1] for q in pts], "-o", color=c, ms=5)
            label_end(ax, pts[-1][0], pts[-1][1], name, c)
            # The laptop GPU has 12 GB; above that the driver pages into system RAM.
            for r in rows:
                if r["s_per_step"] and r.get("peak_gpu_GB", 0) > 12:
                    ax.annotate(f"peak {r['peak_gpu_GB']:.1f} GB (12 GB card)",
                                (r["n_atoms"], r["s_per_step"] * 1000), xytext=(-8, 8),
                                textcoords="offset points", ha="right", fontsize=8,
                                color=INK2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("atoms (periodic PE melt)")
        ax.set_title("Energy + forces per MD step (ms)", loc="left", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "speed.png", dpi=160)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        # MACE float32 sums ~-3.4e5 eV in float32, which quantises the total energy; the
        # float64 run is the like-for-like comparison, float32 is kept faint for reference.
        for k, c, name, alpha in (("mace", MACE, "MACE float32", 0.3),
                                  ("mace_f64", MACE, "MACE float64", 1.0),
                                  ("ours", OURS, "PolyFrameNet float32", 1.0)):
            f = md / f"nve_{k}.json"
            if not f.exists():
                continue
            tr = json.loads(f.read_text())
            e0 = tr["trace_e_eV_atom"][0]
            y = [(e - e0) * 1000 for e in tr["trace_e_eV_atom"]]
            ax.plot(tr["trace_t_ps"], y, color=c, lw=1.2, alpha=alpha, label=name)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.set_xlabel("time (ps)")
        ax.set_title("NVE total energy minus initial (meV/atom), 976 atoms", loc="left",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "nve.png", dpi=160)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        for k, c, name in (("ours", OURS, "PolyFrameNet"), ("mace", MACE, "MACE")):
            f = md / f"nvt_{k}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            e = r["rdf_edges"]
            mid = [(e[i] + e[i + 1]) / 2 for i in range(len(e) - 1)]
            norm = [cnt / (mid[i] ** 2) for i, cnt in enumerate(r["rdf_counts"])]
            tail = sum(norm[-10:]) / 10
            g = [x / tail for x in norm]
            ax.plot(mid, g, color=c, lw=1.5, label=name)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax.set_xlabel("r (Å)")
        ax.set_title("Intermolecular C-C g(r), NVT 450 K (normalised to tail)", loc="left",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "rdf.png", dpi=160)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
