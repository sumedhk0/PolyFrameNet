"""PolyFrameNet architecture diagram (cover figure), in the spirit of the Transformer figure:
inputs at the bottom, energy and forces at the top, one box per stage.
Usage: python scripts/make_architecture_figure.py docs/figures/architecture.png"""

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

INK, INK2, BG, ATOM, AX = "#14181f", "#52514e", "#ffffff", "#3b3f46", "#d0342c"
C = {  # stage fill (light) and edge colour
    "graph": ("#e8f0fb", "#2a78d6"), "enc": ("#e3f4ee", "#1f9e89"), "bond": ("#fdeee6", "#eb6834"),
    "nb": ("#efe9f8", "#7a52c7"), "lr": ("#fbf3dd", "#c98a00"), "out": ("#f1f2f4", "#52514e"),
}

fig = plt.figure(figsize=(8.6, 9.6), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 112)
ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, kind, title, sub="", fs=10.5):
    fill, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=fill, ec=edge, lw=1.6))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.2,
                color=INK2, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs,
                weight="bold", color=INK)


def arrow(x0, y0, x1, y1, color=INK2, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- title and output
ax.text(50, 108.6, "PolyFrameNet", ha="center", fontsize=19, weight="bold", color=INK)
ax.text(50, 104.9, "a block-factored, frame-based interatomic potential for polymers",
        ha="center", fontsize=9.5, color=INK2)

X, W = 33, 44
cx = X + W / 2
ax.text(cx, 100.4, r"$E=\sum_i(\varepsilon_i+v_i)+E_{\rm Coul}+E_{\rm disp}+\sum_i E_0(Z_i)$"
        r"      $\mathbf{F}=-\partial E/\partial \mathbf{r}$",
        ha="center", va="center", fontsize=10.2, color=INK)

# ---------------------------------------------------------------- main column
box(X, 4, W, 6.5, "out", r"Atom positions  $\mathbf{r}$,  elements  $Z$", "(charge Q, spin S)",
    fs=10)
box(X, 15, W, 8, "graph", "Bond graph → rigid blocks",
    "rings · single heavy atoms · H joins its atom")
box(X, 27.5, W, 8, "graph", "Frame variants per block",
    r"WL colour classes → $(c,a,b,\pm)$,   $F_v=[e_1,e_2,e_3]$")
box(X, 40, W, 8, "enc", "Block encoder  (shared MLP)",
    r"atoms in each frame $F_v^\top(r-o)$ + pair distances")
ax.add_patch(FancyBboxPatch((X - 2.2, 51.3), W + 4.4, 13.4,
                            boxstyle="round,pad=0.3,rounding_size=2",
                            fc="none", ec="#c9ccd2", lw=1.2, ls=(0, (4, 3))))
box(X, 53, W, 9.5, "bond", "Bonded message passing",
    "block graph · message J→I sees J's atoms in I's frame")
ax.text(X - 5.6, 57.7, "×3", fontsize=13, weight="bold", color=C["bond"][1], va="center")
box(X, 69, W, 9, "nb", "Many-body layer  (4.5 Å)",
    r"moments $M_0, M_1, M_2$ → $F^\top M F$ → 2–4-body")
box(X, 82.5, W, 6.5, "out", r"Readout MLP  →  per-atom energy  $\varepsilon_i$", fs=10)
ax.add_patch(Circle((cx, 94.2), 2.6, fc="#ffffff", ec=INK, lw=1.6))
ax.text(cx, 94.2, "Σ", ha="center", va="center", fontsize=14, weight="bold")

for y0, y1 in ((10.9, 14.6), (23.4, 27.1), (35.9, 39.6), (48.4, 52.6), (62.9, 68.6),
               (78.4, 82.1), (89.4, 91.4), (96.9, 98.6)):
    arrow(cx, y0, cx, y1)
for y in (44, 57.7, 73.5):  # frame-dependent stages average over frame variants
    ax.text(X + W + 1.3, y, r"$\langle\cdot\rangle_v$", fontsize=10.5, color=INK2, va="center")

# ---------------------------------------------------------------- right column: physics terms
RX, RW = 83.2, 15.6
box(RX, 31, RW, 9, "lr", "Valence prior", "1-2 / 1-3 / 1-4\ndistances", fs=9.5)
box(RX, 69.5, RW, 13, "lr", "Long range",
    "latent charges $q_i$\nCoulomb / Ewald\ndispersion $C_6$ (10 Å)", fs=9.5)
arrow(RX + RW / 2, 82.9, cx + 2.3, 95.6, color=C["lr"][1], rad=0.25)
arrow(RX + RW / 2, 40.4, cx + 2.6, 93.3, color=C["lr"][1], rad=0.32)
arrow(X + W + 0.4, 19.0, RX - 0.4, 33.0, color=INK2, rad=-0.15, lw=1.0)

# ---------------------------------------------------------------- left inset: polymer -> blocks
ax.text(15, 89.5, "How a polymer is cut", ha="center", fontsize=9.5, weight="bold", color=INK)
ax.text(15, 86.6, "(polystyrene, one repeat unit)", ha="center", fontsize=7.6, color=INK2)


def atom(x, y, z):
    ax.add_patch(Circle((x, y), 1.15 if z == "C" else 0.7,
                        fc=ATOM if z == "C" else "#ffffff", ec=ATOM, lw=0.9, zorder=4))


bb = [(5, 70), (11, 73), (17, 70), (23, 73)]  # backbone C atoms
for (x0, y0), (x1, y1) in zip(bb[:-1], bb[1:], strict=True):
    ax.plot([x0, x1], [y0, y1], color=ATOM, lw=1.6, zorder=3)
ax.plot([2, 5], [68.5, 70], color=ATOM, lw=1.6, ls=":")
ax.plot([23, 26], [73, 71.5], color=ATOM, lw=1.6, ls=":")
ring_c = (17, 58.6)
ring = [(ring_c[0] + 4.2 * math.cos(math.radians(90 + 60 * k)),
         ring_c[1] + 4.2 * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
ax.plot([17, ring[0][0]], [70, ring[0][1]], color=ATOM, lw=1.6, zorder=3)
ax.add_patch(Polygon(ring, closed=True, fc="none", ec=ATOM, lw=1.6, zorder=3))
ax.add_patch(FancyBboxPatch((8.3, 70.2), 5.4, 7.0, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc=C["enc"][0], ec=C["enc"][1], lw=1.3, zorder=1))
ax.add_patch(FancyBboxPatch((12.7, 66.4), 6.8, 6.6, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc=C["bond"][0], ec=C["bond"][1], lw=1.3, zorder=1))
ax.add_patch(Circle(ring_c, 6.4, fc=C["nb"][0], ec=C["nb"][1], lw=1.3, zorder=1))
for x, y in bb:
    atom(x, y, "C")
for x, y in ring:
    atom(x, y, "C")
for x, y in ((9.6, 76.2), (12.4, 76.2), (14.2, 67.6)):
    atom(x, y, "H")
ax.text(11, 79.4, "CH₂", ha="center", fontsize=8, color=C["enc"][1], weight="bold")
ax.text(20.6, 68.2, "CH", ha="left", fontsize=8, color=C["bond"][1], weight="bold")
ax.text(17, 50.6, "phenyl ring = one block", ha="center", fontsize=8, color=C["nb"][1],
        weight="bold")
fx, fy = 17, 70  # a frame on the CH block: e1 to the next backbone C, e2 toward the ring
for (dx, dy), lab, (lx, ly) in (((5.5, 2.8), "$e_1$", (3.2, 4.6)),
                               ((-0.2, -3.6), "$e_2$", (0.9, -3.9))):
    ax.add_patch(FancyArrowPatch((fx, fy), (fx + dx, fy + dy), arrowstyle="-|>",
                                 mutation_scale=8, color=AX, lw=1.5, zorder=6))
    ax.text(fx + lx, fy + ly, lab, fontsize=8.5, color=AX, zorder=6)
ax.text(15, 46.2, "Each block gets local frames from\nits own bonded atoms. Vectors in a\n"
        "frame are rotation-free, so every\nnetwork is a plain MLP: no tensor\nproducts.",
        ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.35)
ax.text(15, 33.0, "one shared encoder:\nsame weights for every block", ha="center",
        va="center", fontsize=7.6, color=INK2)
arrow(26.5, 33.0, X - 0.6, 43.0, color=INK2, rad=-0.15, lw=1.0)
ax.text(15, 9.5, r"$\langle\cdot\rangle_v$ = mean over all frame variants" "\n"
        "→ exact rotation, reflection and\npermutation invariance, smooth energy",
        fontsize=7.8, color=INK2, va="center", ha="center", linespacing=1.4)

fig.savefig(sys.argv[1], dpi=220, facecolor=BG)
