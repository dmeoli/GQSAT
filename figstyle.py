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
