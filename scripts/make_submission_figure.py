"""One-page submission figure: (a) force MAE per model on the hydrocarbon and PE test
sets, (b) energy+force time per call vs atom count (log-log).
Usage: python scripts/make_submission_figure.py <eval dir> <bench dir> <out.png>
<eval dir> holds <kind>_{test,pe}.json (eval_ase.py); <bench dir> holds
speed_<kind>_r<reps>.json (bench_ase.py / md_bench.py)."""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (reference palette); PolyFrameNet first.
COLORS = {"v2": "#2a78d6", "ours": "#2a78d6", "v1": "#2a78d6", "mace": "#eb6834",
          "allegro": "#1f9e89", "deepmd": "#8b5cc4"}
NAMES = {"v2": "PolyFrameNet v2", "ours": "PolyFrameNet v1", "v1": "PolyFrameNet v1",
         "mace": "MACE", "allegro": "Allegro", "deepmd": "DeepMD"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})


def main(eval_dir: pathlib.Path, bench_dir: pathlib.Path, out: pathlib.Path):
    rows = {}
    for f in eval_dir.glob("*_*.json"):
        if f.name == "all.json":
            continue
        kind, split = f.stem.rsplit("_", 1)
        rows.setdefault(kind, {})[split] = json.loads(f.read_text())["f_mae_meV_A"]
    order = [k for k in ("v2", "mace", "allegro", "deepmd") if k in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.3))
    w = 0.38
    for i, k in enumerate(order):
        for j, (split, hatch) in enumerate((("test", None), ("pe", "///"))):
            v = rows[k].get(split)
            if v is None:
                continue
            a.bar(i + (j - 0.5) * w, v, w * 0.92, color=COLORS[k], hatch=hatch,
                  edgecolor=SURF, linewidth=0)
            a.annotate(f"{v:.0f}", (i + (j - 0.5) * w, v), xytext=(0, 2),
                       textcoords="offset points", ha="center", fontsize=7, color=INK2)
    a.set_xticks(range(len(order)), [NAMES[k].replace("PolyFrameNet ", "PFN ") for k in order])
    a.set_yscale("log")
    a.set_ylabel("force MAE (meV/Å)")
    a.set_title("(a) force MAE: solid = test, hatched = PE test", loc="left", fontsize=8)
    for k in ("v1", "v2", "mace", "allegro", "deepmd"):
        pts = []
        for f in bench_dir.glob(f"speed_{k}_r*.json"):
            r = json.loads(f.read_text())
            if r.get("s_per_step"):
                pts.append((r["n_atoms"], r["s_per_step"] * 1000))
        if not pts:
            continue
        pts.sort()
        b.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, color=COLORS[k],
               label=NAMES[k], linestyle="--" if k == "v1" else "-")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("atoms (periodic PE melt)")
    b.set_ylabel("ms per energy + force call")
    b.set_title("(b) time per energy+force call", loc="left", fontsize=8)
    b.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main(*map(pathlib.Path, sys.argv[1:4]))
