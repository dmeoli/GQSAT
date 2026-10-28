#!/usr/bin/env python3
"""The backgrounds of the slides, 16:9 and vector, in the colors of the random
families of the figures (figstyle) and of the violet of the theme; the slides
fade them (opacity in the background template).

bg-cover.pdf, behind the title page: a wall of 3-SAT clauses, with the AND and
the OR of the CNF drawn large.  bg-content.pdf, behind the other frames:
neighborhoods of a bipartite graph whose messages flow into a node, each arrow
as wide and as opaque as its softmax coefficient, i.e., message passing
weighted by attention.  Run from this directory: python3 make_backgrounds.py;
another deck writes its own cover with --cover-only <pdf> --free <bands>.
"""
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import colors as mcolors
from matplotlib import patheffects as pe

W, H = 16, 9                                     # cm, the page of a 16:9 beamer
VIOLET, CYAN, NAVY, MAGENTA = "#665094", "#17becf", "#1a3a6b", "#8e1b5c"
GOLD = "#d4a52a"                                 # the complement of the violet
PAGE = "white"                                   # the color of the page


def lighter(c, f=0.45):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


PAL = [CYAN, NAVY, MAGENTA, VIOLET, lighter(CYAN), lighter(NAVY), lighter(MAGENTA)]


def page():
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return fig, ax


def sub(n):
    return "".join("₀₁₂₃₄₅₆₇₈₉"[int(c)] for c in str(n))


# the heights (cm from the bottom) of the lines of the title page that the
# clauses must leave free, i.e., from the rule under the subtitle down to the
# name and the institution, measured on the PDF of the deck with pdftotext -bbox
FREE = [(2.16, 3.80)]
ROW = 0.42                                       # the pitch of the rows of clauses


def cover(seed=1, free=FREE):
    rnd = random.Random(seed)
    fig, ax = page()
    y, row = H - 0.25, 0
    while y > 0:
        # a row whose glyphs (about 0.2 cm below its top) would touch a line
        # of text is left out, so that the line falls on an empty band
        if any(y > lo - 0.06 and y - 0.22 < hi + 0.06 for lo, hi in free):
            y -= ROW; row += 1
            continue
        cls = []
        for _ in range(9):
            lits = [("¬" if rnd.random() < .5 else "") + "x" + sub(v)
                    for v in rnd.sample(range(1, 60), 3)]
            cls.append("(" + " ∨ ".join(lits) + ")")
        ax.text(-0.3 - 0.6 * (row % 2), y, " ∧ ".join(cls), fontsize=5.2,
                family="DejaVu Sans Mono", color=[VIOLET, NAVY][row % 2], va="top", ha="left")
        y -= ROW; row += 1
    # the AND and the OR of the CNF, side by side and crossing each other, on
    # the bottom of the page in its right corner, in the violet of the theme and its complement;
    # both are outlined in the color of the page,
    # so that a thin gap keeps the OR, in front, apart from the AND where they
    # cross, and both apart from the clauses
    ax.text(W - 2.75, -0.32, "⋀", fontsize=90, color=VIOLET, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2.2, foreground=PAGE)])
    ax.text(W - 1.3, -0.32, "⋁", fontsize=90, color=GOLD, ha="center", va="bottom",
            path_effects=[pe.withStroke(linewidth=2.2, foreground=PAGE)])
    return fig


def content(seed=4):
    rnd = random.Random(seed)
    fig, ax = page()
    centers = []
    while len(centers) < 16:                     # neighborhoods not too close
        c = (rnd.uniform(.4, W - .4), rnd.uniform(.4, H - .4))
        if all((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 > 3.2 for d in centers):
            centers.append(c)
    # a few long, faint edges between the neighborhoods: the rest of the graph
    for i, c in enumerate(centers):
        for d in sorted(centers, key=lambda d: (d[0] - c[0]) ** 2 + (d[1] - c[1]) ** 2)[1:3]:
            ax.plot([c[0], d[0]], [c[1], d[1]], color=VIOLET, lw=.4, alpha=.35, zorder=0)
    for k, (cx, cy) in enumerate(centers):
        col = PAL[k % 3]                         # the node in a full color
        n = rnd.randint(4, 7)
        r = rnd.uniform(.9, 1.4)
        s = [rnd.gauss(0, 1.1) for _ in range(n)]
        z = sum(math.exp(x) for x in s)
        for t in range(n):
            a = 2 * math.pi * t / n + rnd.random() * .4
            x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            alpha = math.exp(s[t]) / z           # the attention coefficient
            ax.annotate("", (cx, cy), (x, y), zorder=1,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=.3 + 3.0 * alpha,
                                        alpha=.3 + .7 * alpha, shrinkA=3, shrinkB=6,
                                        mutation_scale=4 + 6 * alpha))
            # filled shapes with no outline: a dashed outline on so small a
            # square gives jagged corners
            ax.add_patch(Rectangle((x - .08, y - .08), .16, .16, zorder=2, lw=0,
                                   facecolor=lighter(col, .35)))
        ax.add_patch(Circle((cx, cy), .17, facecolor=col, lw=0, zorder=3))
    return fig


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--free", default=None,
                    help="the bands to leave free on the cover, as lo:hi,lo:hi (cm)")
    ap.add_argument("--cover-only", default=None, metavar="PDF",
                    help="write only the cover, to this file")
    a = ap.parse_args()
    free = [tuple(float(x) for x in b.split(":")) for b in a.free.split(",")] if a.free else FREE
    if a.cover_only:
        fig = cover(free=free); fig.savefig(a.cover_only); plt.close(fig)
        print("wrote", a.cover_only)
    else:
        for name, f in [("bg-cover", lambda: cover(free=free)), ("bg-content", content)]:
            fig = f(); fig.savefig(name + ".pdf"); plt.close(fig)
            print("wrote", name + ".pdf")
