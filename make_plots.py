#!/usr/bin/env python3
"""The per-family figures: MRIR ("iterations improvement") and wall-clock
seconds against the number of model decisions (the cap), one line per dataset,
for each model, on the graph-colouring and on the random families.

The logs, the runs and the baseline are those of paper_numbers.py. Outputs PNGs
under ../img/ (pure matplotlib/stdlib). Run from the GQSAT root:

    python3 make_plots.py [--reeval-root DIR] [--out-dir DIR]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_numbers as pn

CAPS = [10, 50, 100, 300, 500, 1000]

FLAT = ["flat30-60", "flat50-115", "flat75-180", "flat100-239",
        "flat125-301", "flat150-360", "flat175-417", "flat200-479"]
RANDOM = ["uf50-218", "uf100-430", "uf250-1065", "uuf50-218", "uuf100-430", "uuf250-1065"]
TITLE = {"gatqsat": "GAT-Q-SAT", "graphqsat": "Graph-Q-SAT"}


def mean_over_runs(agg, model, dataset, cap, idx):
    """Mean over the runs of the per-run median (idx 0 the MRIR, 1 the seconds),
    with the runs and the baseline of paper_numbers.py: the 2021 colouring pair on
    the flat families (the only one evaluated at every cap), the runs trained on
    satisfiable random 3-SAT on the random families."""
    variant = TITLE[model]
    if dataset.startswith("flat"):
        return pn.group(variant, pn.COL, dataset, cap, "family", idx, runs=pn.CURVE_RUNS)
    return pn.group(variant, pn.SAT, dataset, cap, "family", idx)


def shared_ylim(agg, models, datasets, idx, start_one=False, pad=0.06):
    """Common y range over the models, so two panels can be read side by side."""
    vals = [1.0] if start_one else []
    for m in models:
        for d in datasets:
            for c in CAPS:
                v = mean_over_runs(agg, m, d, c, idx)
                if v is not None:
                    vals.append(v)
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span


def plot_curves(agg, model, datasets, idx, ylabel, title, out_path, start_one=False,
                ylim=None):
    plt.figure(figsize=(7, 4.3))
    # one line per dataset: use a categorical palette (a violet gradient makes the
    # per-dataset legend unreadable) + varied markers, so each curve is identifiable.
    palette = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for i, d in enumerate(datasets):
        xs, ys = ([0], [1.0]) if start_one else ([], [])
        for c in CAPS:
            v = mean_over_runs(agg, model, d, c, idx)
            if v is not None:
                xs.append(c)
                ys.append(v)
        if len(ys) > (1 if start_one else 0):
            plt.plot(xs, ys, marker=markers[i % len(markers)], markersize=4,
                     linewidth=1.6, label=d, color=palette[i % len(palette)])
    plt.xscale("symlog")
    plt.xticks([0] + CAPS, ["0"] + [str(c) for c in CAPS])
    plt.xlabel("model decisions")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if not start_one:
        plt.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    plt.legend(fontsize=7, ncol=2, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reeval-root", default=os.environ.get("OUT_ROOT"))
    ap.add_argument("--out-dir", default="../img")
    args = ap.parse_args()
    pn.REEVAL_ROOT = args.reeval_root
    agg = None
    os.makedirs(args.out_dir, exist_ok=True)

    MODELS = ("graphqsat", "gatqsat")
    flat_mrir = shared_ylim(agg, MODELS, FLAT, 0, start_one=True)
    flat_time = shared_ylim(agg, MODELS, FLAT, 1)
    rand_mrir = shared_ylim(agg, MODELS, RANDOM, 0, start_one=True)

    for model in MODELS:
        # MRIR ("iterations improvement") vs model decisions on graph colouring
        plot_curves(agg, model, FLAT, idx=0,
                    ylabel="iterations improvement (MRIR)",
                    title=f"{TITLE[model]} on graph colouring (flat)",
                    out_path=os.path.join(args.out_dir, f"{model}.png"),
                    start_one=True, ylim=flat_mrir)
        # wall-clock time vs model decisions
        plot_curves(agg, model, FLAT, idx=1,
                    ylabel="median sec to solve",
                    title=f"{TITLE[model]} solving time (flat)",
                    out_path=os.path.join(args.out_dir, f"{model}_time.png"),
                    ylim=flat_time)
        # MRIR on random 3-SAT
        plot_curves(agg, model, RANDOM, idx=0,
                    ylabel="iterations improvement (MRIR)",
                    title=f"{TITLE[model]} on uniform-random 3-SAT",
                    out_path=os.path.join(args.out_dir, f"{model}_random.png"),
                    start_one=True, ylim=rand_mrir)


if __name__ == "__main__":
    main()
