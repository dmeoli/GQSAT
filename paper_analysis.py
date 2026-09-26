#!/usr/bin/env python3
"""Train-set-aware analysis of the Graph-Q-SAT / GAT-Q-SAT runs for the report.

The runs, their grouping and the baseline are those of paper_numbers.py, so the
bars and the curves drawn here are the numbers of the tables: the colouring
group is the pair trained on flat50-115 (both seeds), the random group the runs
trained on satisfiable uniform random 3-SAT (the four trained on unsatisfiable
formulas are a control and are not pooled with them), and the MRIR is read
against MiniSat with restarts on colouring and without on random 3-SAT.

Pure matplotlib/stdlib. Run from the GQSAT root:
    python3 paper_analysis.py [--reeval-root DIR]      (or OUT_ROOT=DIR)
"""
import argparse
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_numbers as pn

CAPS = [10, 50, 100, 300, 500, 1000]
FLAT = pn.FLAT_ALL
RANDOM = pn.RAND
SIZE = {"flat30-60": 90, "flat50-115": 150, "flat75-180": 225, "flat100-239": 300,
        "flat125-301": 375, "flat150-360": 450, "flat175-417": 525, "flat200-479": 600,
        "uf50-218": 50, "uf100-430": 100, "uf250-1065": 250,
        "uuf50-218": 50, "uuf100-430": 100, "uuf250-1065": 250}
COL = {"GAT-Q-SAT": "#4b2e83", "Graph-Q-SAT": "#b9a7d6"}  # deep violet vs light lilac
TRAIN = {"coloring": pn.COL, "random": pn.SAT}
BASELINE = "family"


def ratio_axis(ax, lo=None, hi=None):
    """The MRIR is a ratio: a log2 axis puts 1/2 and 2 at the same distance
    from parity, which is drawn as a dashed line."""
    ax.set_yscale("log", base=2)
    ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8]
    if lo is not None and hi is not None:
        ax.set_ylim(lo, hi)
        ticks = [t for t in ticks if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.axhline(1.0, color="gray", lw=.8, ls="--")


def ratio_bars(ax, xs, vals, width, **kw):
    """Bars that start at parity, going up for MRIR > 1 and down for MRIR < 1."""
    heights = [(v - 1) if v is not None else 0 for v in vals]
    ax.bar(xs, heights, width, bottom=1, **kw)
    for x, v in zip(xs, vals):
        if v is not None:
            ax.text(x, v * (1.03 if v >= 1 else 0.97), f"{v:.2f}", ha="center",
                    va="bottom" if v >= 1 else "top", fontsize=8)


def mean_mrir(data, key, dataset, cap):
    train, model = key
    return pn.group(model, TRAIN[train], dataset, cap, BASELINE)


def mean_sec(data, key, dataset, cap):
    train, model = key
    return pn.group(model, TRAIN[train], dataset, cap, BASELINE, idx=1)


def fig_thesis(data, out):
    """Mean MRIR (cap 500) per regime, GAT vs Graph, on a log2 axis."""
    SAT_R = [d for d in RANDOM if d.startswith("uf")]
    UNSAT_R = [d for d in RANDOM if d.startswith("uuf")]
    regimes = [("coloring", FLAT, "colouring-trained\n→ colouring"),
               ("random", FLAT, "random-trained\n→ colouring"),
               ("random", SAT_R, "random-trained\n→ satisfiable random"),
               ("random", UNSAT_R, "random-trained\n→ unsatisfiable random")]
    cap = 500
    gat, graph = [], []
    for train, dss, _ in regimes:
        g = [mean_mrir(data, (train, "GAT-Q-SAT"), d, cap) for d in dss]
        h = [mean_mrir(data, (train, "Graph-Q-SAT"), d, cap) for d in dss]
        gat.append(statistics.fmean([x for x in g if x is not None]))
        graph.append(statistics.fmean([x for x in h if x is not None]))
    x = range(len(regimes)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ratio_bars(ax, [i - w/2 for i in x], graph, w, label="Graph-Q-SAT", color=COL["Graph-Q-SAT"])
    ratio_bars(ax, [i + w/2 for i in x], gat, w, label="GAT-Q-SAT", color=COL["GAT-Q-SAT"])
    ratio_axis(ax, 0.5, 4)
    ax.set_xticks(list(x)); ax.set_xticklabels([r[2] for r in regimes], fontsize=9)
    ax.set_ylabel("mean MRIR (cap 500, log scale)")
    ax.legend(); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def fig_time(data, out):
    """Wall-clock companion to fig_thesis: median seconds-to-solve (cap 500) per
    regime, GAT vs Graph. Fewer iterations (MRIR) does NOT mean less wall-clock ---
    each model decision is a GNN forward, and attention makes it costlier still."""
    regimes = [("coloring", FLAT, "colouring\n→ colouring"),
               ("random", FLAT, "random\n→ colouring"),
               ("random", RANDOM, "random\n→ random")]
    cap = 500
    gat, graph = [], []
    for train, dss, _ in regimes:
        g = [mean_sec(data, (train, "GAT-Q-SAT"), d, cap) for d in dss]
        h = [mean_sec(data, (train, "Graph-Q-SAT"), d, cap) for d in dss]
        gat.append(statistics.fmean([x for x in g if x is not None]))
        graph.append(statistics.fmean([x for x in h if x is not None]))
    x = range(len(regimes)); w = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - w/2 for i in x], graph, w, label="Graph-Q-SAT", color=COL["Graph-Q-SAT"])
    plt.bar([i + w/2 for i in x], gat,  w, label="GAT-Q-SAT",   color=COL["GAT-Q-SAT"])
    for i, (a, b) in enumerate(zip(graph, gat)):
        plt.text(i - w/2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
        plt.text(i + w/2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(list(x), [r[2] for r in regimes], fontsize=9)
    plt.ylabel("mean wall-clock sec to solve (cap 500)")
    plt.title("Mean wall-clock seconds to solve (cap 500)")
    plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


def fig_mrir_time(data, out):
    """Side-by-side: iteration reduction (MRIR) and wall-clock cost (sec), Graph vs
    GAT, per regime --- the two together show that fewer iterations is not less time."""
    regimes = [("coloring", FLAT, "col\n→col"),
               ("random", FLAT, "rand\n→col"),
               ("random", RANDOM, "rand\n→rand")]
    cap = 500
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, ylabel, title in [
            (axes[0], mean_mrir, "mean MRIR vs MiniSat (cap 500)",
             "Iterations (MRIR, cap 500)"),
            (axes[1], mean_sec, "mean wall-clock sec to solve (cap 500)",
             "Wall-clock seconds to solve (cap 500)")]:
        gat, graph = [], []
        for train, dss, _ in regimes:
            g = [metric(data, (train, "GAT-Q-SAT"), d, cap) for d in dss]
            h = [metric(data, (train, "Graph-Q-SAT"), d, cap) for d in dss]
            gat.append(statistics.fmean([x for x in g if x is not None]))
            graph.append(statistics.fmean([x for x in h if x is not None]))
        x = range(len(regimes)); w = 0.38
        ax.bar([i - w/2 for i in x], graph, w, label="Graph-Q-SAT", color=COL["Graph-Q-SAT"])
        ax.bar([i + w/2 for i in x], gat,  w, label="GAT-Q-SAT",   color=COL["GAT-Q-SAT"])
        for i, (a, b) in enumerate(zip(graph, gat)):
            ax.text(i - w/2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w/2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
        if metric is mean_mrir:
            ax.axhline(1.0, color="gray", lw=.8, ls="--")
        ax.set_xticks(list(x)); ax.set_xticklabels([r[2] for r in regimes], fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    axes[0].legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(); print("wrote", out)


def fig_curves(data, train, datasets, title, out):
    """MRIR vs decision-cap, GAT vs Graph, averaged over the datasets; cap 0 is
    MiniSat without restarts against the baseline, read from the METADATA."""
    # only the 2021 colouring pair is evaluated at every cap
    runs = pn.CURVE_RUNS if train == "coloring" else None
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for model in ("Graph-Q-SAT", "GAT-Q-SAT"):
        xs, ys = [0], [statistics.fmean(pn.cap0(d, BASELINE) for d in datasets)]
        for c in CAPS:
            vals = [pn.group(model, TRAIN[train], d, c, BASELINE, runs=runs) for d in datasets]
            xs.append(c)
            ys.append(statistics.fmean(vals) if all(v is not None for v in vals) else float("nan"))
        ax.plot(xs, ys, marker="o", lw=1.8, label=model, color=COL[model])
    ax.set_xscale("symlog", linthresh=10); ax.set_xticks([0]+CAPS)
    ax.set_xticklabels(["0"]+[str(c) for c in CAPS])
    ratio_axis(ax)
    ax.set_xlabel("cap on the decisions of the model"); ax.set_ylabel("mean MRIR (log scale)")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def fig_generalization(data, out):
    """Random-trained: MRIR (cap 500) against the size of the test instances, on
    graph colouring and on random 3-SAT (satisfiable solid, unsatisfiable dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    panels = [(axes[0], [("", FLAT, "-")], "graph colouring", "# variables (3 per node)"),
              (axes[1], [(" (sat)", [d for d in RANDOM if d.startswith("uf")], "-"),
                         (" (unsat)", [d for d in RANDOM if d.startswith("uuf")], "--")],
               "uniform random 3-SAT", "# variables")]
    for ax, series, name, xlabel in panels:
        for model in ("Graph-Q-SAT", "GAT-Q-SAT"):
            for suffix, dss, ls in series:
                pts = [(SIZE[d], mean_mrir(data, ("random", model), d, 500)) for d in dss]
                pts = sorted((s_, v) for s_, v in pts if v is not None)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", lw=1.8,
                            ls=ls, label=model + suffix, color=COL[model])
        ratio_axis(ax, 0.5, 6)
        ax.set_xlabel(xlabel); ax.set_title(name); ax.grid(alpha=.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("MRIR (cap 500, log scale)")
    fig.suptitle("Models trained on satisfiable random 3-SAT, against the size of the test instances")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig); print("wrote", out)


def write_summary(data, out):
    lines = ["# Experiment summary (mean MRIR over runs, cap 500)\n"]
    for train, dss, name in [("coloring", FLAT, "Colouring-trained on colouring"),
                             ("random", FLAT, "Random-trained on colouring (transfer)"),
                             ("random", RANDOM, "Random-trained on random")]:
        lines.append(f"\n## {name}\n")
        lines.append("| dataset | Graph-Q-SAT | GAT-Q-SAT | Δ (GAT−Graph) |")
        lines.append("|---|---|---|---|")
        for d in dss:
            g = mean_mrir(data, (train, "Graph-Q-SAT"), d, 500)
            a = mean_mrir(data, (train, "GAT-Q-SAT"), d, 500)
            if g is None and a is None:
                continue
            dlt = f"{a-g:+.2f}" if (g is not None and a is not None) else "—"
            lines.append(f"| {d} | {g:.2f} | {a:.2f} | {dlt} |" if g and a
                         else f"| {d} | {g} | {a} | {dlt} |")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reeval-root", default=os.environ.get("OUT_ROOT"))
    pn.REEVAL_ROOT = ap.parse_args().reeval_root
    data = None
    od = os.environ.get("PAPER_IMG_DIR", "../img/paper")
    os.makedirs(od, exist_ok=True)
    fig_thesis(data, f"{od}/thesis.png")
    fig_time(data, f"{od}/time.png")
    fig_mrir_time(data, f"{od}/mrir_time.png")
    fig_curves(data, "coloring", FLAT, "Trained & tested on graph colouring (in-distribution)",
               f"{od}/coloring_indist.png")
    fig_curves(data, "random", RANDOM, "Trained & tested on uniform-random 3-SAT",
               f"{od}/random_indist.png")
    fig_generalization(data, f"{od}/generalization.png")
    write_summary(data, f"{od}/summary.md")


if __name__ == "__main__":
    main()
