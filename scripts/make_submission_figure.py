"""Submission figure, three panels, all from the committed JSON results:
(a) accuracy vs speed trade-off at 976 atoms (hydrocarbon test force MAE vs ms per call),
(b) time per energy+force call vs system size, out-of-memory runs marked,
(c) polyethylene: PE-only v1 vs MACE trained on the same PE data (v1 round).
Usage: python scripts/make_submission_figure.py <eval dir> <bench dir> <v1 acc.json> <out.png>"""

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"v2": "#2a78d6", "v1": "#2a78d6", "mace": "#eb6834", "allegro": "#1f9e89"}
NAMES = {"v2": "PolyFrameNet", "v1": "PFN (PE-only)", "mace": "MACE",
         "allegro": "Allegro*"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 8.5, "lines.linewidth": 2,
})


def load(p):
    return json.loads(pathlib.Path(p).read_text())


def main(eval_dir, bench_dir, v1_acc, out):
    eval_dir, bench_dir = pathlib.Path(eval_dir), pathlib.Path(bench_dir)
    fmae = {k: load(eval_dir / f"{k}_test.json")["f_mae_meV_A"] for k in ("v2", "mace", "allegro")}
    speed = {}
    for k in ("v2", "v1", "mace", "allegro"):
        for r in (1, 2, 3):
            f = bench_dir / f"speed_{k}_r{r}.json"
            if f.exists():
                d = load(f)
                speed.setdefault(k, []).append((d["n_atoms"], d.get("s_per_step")))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.45),
                                  gridspec_kw={"width_ratios": [1.05, 1.15, 0.8]})

    # (a) trade-off at 976 atoms: lower-left is better
    for k in ("v2", "mace", "allegro"):
        t = dict(speed[k])[976] * 1000
        a.scatter(t, fmae[k], s=70, color=COLORS[k], zorder=3, edgecolor=SURF, linewidth=1.5)
        a.annotate(NAMES[k], (t, fmae[k]), xytext=(6, 4), textcoords="offset points",
                   fontsize=8, color=INK2)
    a.set_xlim(15, 75)
    a.set_ylim(0, 55)
    a.set_xlabel("ms per MD step, 1k atoms")
    a.set_ylabel("force MAE (meV/Å)")
    a.set_title("(a) accuracy vs speed", loc="left", fontsize=8.5)
    a.text(0.04, 0.04, "↙ better", transform=a.transAxes, ha="left", fontsize=7.5, color=INK2)

    # (b) scaling, with out-of-memory runs marked at the top edge
    top = 2000
    for k in ("v1", "v2", "mace", "allegro"):
        pts = sorted(p for p in speed.get(k, []) if p[1])
        oom = sorted(p[0] for p in speed.get(k, []) if not p[1])
        style = "--" if k == "v1" else "-"
        b.plot([p[0] for p in pts], [p[1] * 1000 for p in pts], style, marker="o", ms=4,
               color=COLORS[k], label=NAMES[k].replace("PolyFrameNet", "PFN"))
        shift = {"mace": 0.92, "allegro": 1.09}.get(k, 1.0)  # keep coincident markers visible
        for n in oom:
            b.scatter(n * shift, top, marker="x", s=45, color=COLORS[k], zorder=3, linewidths=2)
    b.annotate("× = out of memory\n(46 GB L40S)", (2.6e4, top), xytext=(-4, -22),
               textcoords="offset points", ha="right", fontsize=7, color=INK2)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_ylim(15, top * 1.6)
    b.set_xlabel("atoms (periodic polyethylene melt)")
    b.set_ylabel("ms per MD step")
    b.set_title("(b) cost vs system size", loc="left", fontsize=8.5)
    b.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.6)

    # (c) polyethylene, both trained on the same PE data
    v1 = load(v1_acc)
    vals = [v1["ours"]["f_mae_meV_A"], v1["mace"]["f_mae_meV_A"]]
    bars = c.bar([0, 1], vals, 0.6, color=[COLORS["v1"], COLORS["mace"]])
    for x, v in zip([0, 1], vals, strict=True):
        c.annotate(f"{v:.1f}", (x, v), xytext=(0, 2), textcoords="offset points", ha="center",
                   fontsize=8, color=INK2)
    c.set_xticks([0, 1], ["PolyFrameNet\n(PE-only)", "MACE"])
    c.set_ylim(0, 18)
    c.set_ylabel("force MAE (meV/Å)")
    c.set_title("(c) polyethylene: a tie", loc="left", fontsize=8.5)
    del bars
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out, dpi=220)


if __name__ == "__main__":
    main(*sys.argv[1:5])
