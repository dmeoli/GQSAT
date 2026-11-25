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
# curve dashed).  The two main models are in the violet of the theme, light
# lilac Graph-Q-SAT and deep violet GAT-Q-SAT, which are told apart by their
# lightness; the other three variants, which only appear next to them, take
# hues far from violet and from one another (orange, green and blue, checked
# for protanopia and deuteranopia).  The marks stay plain (no hatches, solid
# lines): the colors alone tell the models apart.
from matplotlib import colors as mcolors

DATASET_COLOR = {
    # graph coloring: the "muted" scheme of Paul Tol, safe for protanopia and
    # deuteranopia between neighbors, from cool to warm as the graphs grow, so
    # that the order of the sizes can be read
    "flat30-60": "#332288", "flat50-115": "#88ccee", "flat75-180": "#44aa99",
    "flat100-239": "#117733", "flat125-301": "#999933", "flat150-360": "#ddcc77",
    "flat175-417": "#cc6677", "flat200-479": "#882255",
    # uniform random 3-SAT
    "uf50-218": "#33bbee", "uf100-430": "#1a3a6b", "uf250-1065": "#ee7733",
}


def _lighter(c, f=0.45):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


for _n in ("50-218", "100-430", "250-1065"):
    DATASET_COLOR["uuf" + _n] = _lighter(DATASET_COLOR["uf" + _n])

MODEL_COLOR = {"Graph-Q-SAT": "#b9a7d6", "GAT-Q-SAT": "#4b2e83",
               "Graph-Q-SAT wide": "#e69f00", "Attention aggregation": "#009e73",
               "GAT-Q-SAT pool": "#0072b2"}
MODEL_HATCH = {}
MODEL_LS = {}


def dataset_color(d):
    return DATASET_COLOR[d]


def model_color(model):
    return MODEL_COLOR[model]


def model_legend(ax, models, **kw):
    """A legend of the models, by their violet."""
    from matplotlib.patches import Patch
    hs = [Patch(facecolor=MODEL_COLOR[m], hatch=MODEL_HATCH.get(m, ""),
                edgecolor="white", label=m) for m in models]
    ax.legend(handles=hs, **kw)
