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

# The colors.  A color always means a family of instances, and the same one
# in every figure: the eight graph-coloring families have eight distinct
# colors, the uniform random ones three more, and an unsatisfiable family has
# the color of the satisfiable one with as many variables, lighter (and its
# curve dashed).  The models carry no color: light gray Graph-Q-SAT, dark gray
# GAT-Q-SAT, the two controls in between and hatched or dashed.
from matplotlib import colors as mcolors

DATASET_COLOR = {
    # graph coloring: deep, muted tones in the order of the wheel, from teal to
    # ochre as the graphs grow, so that the order of the sizes can be read
    "flat30-60": "#0f7c78", "flat50-115": "#3b7dbf", "flat75-180": "#4c46a8",
    "flat100-239": "#7d4fa3", "flat125-301": "#b8457a", "flat150-360": "#d0624a",
    "flat175-417": "#c88a1c", "flat200-479": "#6f8a2a",
    # uniform random 3-SAT
    "uf50-218": "#17becf", "uf100-430": "#1a3a6b", "uf250-1065": "#8e1b5c",
}


def _lighter(c, f=0.45):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


for _n in ("50-218", "100-430", "250-1065"):
    DATASET_COLOR["uuf" + _n] = _lighter(DATASET_COLOR["uf" + _n])

MODEL_GRAY = {"Graph-Q-SAT": "#b4b4b4", "Graph-Q-SAT wide": "#8a8a8a",
              "Attention aggregation": "#5e5e5e", "GAT-Q-SAT": "#262626"}
MODEL_HATCH = {"Graph-Q-SAT wide": "////", "Attention aggregation": "...."}
MODEL_LS = {"Graph-Q-SAT wide": "--", "Attention aggregation": ":"}


def dataset_color(d):
    return DATASET_COLOR[d]


def model_color(model):
    return MODEL_GRAY[model]


def model_legend(ax, models, **kw):
    """A legend of the models, by their gray."""
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=MODEL_GRAY[m], hatch=MODEL_HATCH.get(m, ""),
                edgecolor="white", label=m) for m in models]
    ax.legend(handles=hs, **kw)
