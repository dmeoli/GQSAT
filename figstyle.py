"""The common style of the figures of the note (report/gatqsat.tex).

Every figure is drawn at the size at which it is printed, the text block of
the note being 15.8 cm wide, and is included with the same fraction of
\\textwidth (FULL at \\textwidth, SINGLE at 0.75\\textwidth, HALF in a
subfigure of 0.49\\textwidth); hence the fonts come out at the same size in
all of them, and no figure is scaled up or down by LaTeX.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT = 15.8 / 2.54                # width of the text block, in inches
FULL = (TEXT, 2.5)                # a figure across the text block
SINGLE = (0.75 * TEXT, 2.5)       # a single plot
HALF = (0.49 * TEXT, 2.7)         # one of two plots side by side, legend below
DPI = 300

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.dpi": DPI,
})

# The colors.  The hue says the kind of the test instances, and is the same in
# every figure: graph coloring (the flat and the small-world families) blue,
# satisfiable uniform random 3-SAT green, unsatisfiable red, the structured
# SATLIB families violet.  Within a hue the shade orders the objects: the
# families by size (light the smallest) in the curves of one model, the models
# (light Graph-Q-SAT, dark GAT-Q-SAT) where the models are compared, the two
# controls in between and hatched.
from matplotlib import colormaps

KIND_CMAP = {"coloring": "Blues", "sat": "Greens", "unsat": "Reds", "structured": "Purples"}
MODEL_SHADE = {"Graph-Q-SAT": 0.0, "Graph-Q-SAT wide": 0.35,
               "Attention aggregation": 0.65, "GAT-Q-SAT": 1.0}
MODEL_HATCH = {"Graph-Q-SAT wide": "////", "Attention aggregation": "...."}
MODEL_LS = {"Graph-Q-SAT wide": "--", "Attention aggregation": ":"}


def kind(dataset):
    """The kind of a family, i.e., its hue."""
    if dataset.startswith(("flat", "sw")):
        return "coloring"
    if dataset.startswith("uuf"):
        return "unsat"
    if dataset.startswith("uf"):
        return "sat"
    return "structured"


def shade(k, t):
    """The color of kind k at shade t in [0, 1] (0 light, 1 dark); the ends of
    the colormaps are left out, too pale on white and too close to black."""
    return colormaps[KIND_CMAP[k]](0.38 + 0.57 * t)


def gradient(datasets):
    """One color per family, the families of each kind from light to dark in
    the order given (i.e., by size)."""
    out = {}
    for k in KIND_CMAP:
        ds = [d for d in datasets if kind(d) == k]
        for i, d in enumerate(ds):
            out[d] = shade(k, i / (len(ds) - 1) if len(ds) > 1 else 1.0)
    return out


def model_color(model, k):
    return shade(k, MODEL_SHADE[model])


def model_legend(ax, models, **kw):
    """A legend of the models in gray, since their hue changes with the data."""
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=colormaps["Greys"](0.3 + 0.55 * MODEL_SHADE[m]),
                hatch=MODEL_HATCH.get(m, ""), edgecolor="white", label=m) for m in models]
    ax.legend(handles=hs, **kw)
