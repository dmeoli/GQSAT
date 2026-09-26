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


def point(agg, model, dataset, cap, idx, ratio):
    """One point of a curve; cap 0 is MiniSat without restarts, left alone,
    whose MRIR against the baseline is read from the METADATA of the family."""
    if cap == 0:
        return pn.cap0(dataset, "family") if ratio else None
    return mean_over_runs(agg, model, dataset, cap, idx)


def shared_ylim(agg, models, datasets, idx, ratio=False, pad=0.06):
    """Common y range over the models, so two panels can be read side by side
    (in log space for the MRIR)."""
    vals = []
    for m in models:
        for d in datasets:
            for c in [0] + CAPS:
                v = point(agg, m, d, c, idx, ratio)
                if v is not None:
                    vals.append(v)
    if not vals:
        return None
    if ratio:
        lo, hi = min(vals + [1.0]), max(vals + [1.0])
        return lo / (1 + pad), hi * (1 + pad)
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return max(0.0, lo - pad * span), hi + pad * span


def plot_curves(agg, model, datasets, idx, ylabel, title, out_path, ratio=False,
                ylim=None, dashed=()):
    fig, ax = plt.subplots(figsize=(7, 4.3))
    # one line per dataset: use a categorical palette (a violet gradient makes the
    # per-dataset legend unreadable) + varied markers, so each curve is identifiable;
    # a cap with no log is a gap in the line, not a straight segment across it.
    palette = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    for i, d in enumerate(datasets):
        xs = [0] + CAPS if ratio else CAPS
        ys = [point(agg, model, d, c, idx, ratio) for c in xs]
        if all(y is None for y in (ys[1:] if ratio else ys)):
            continue
        ys = [float("nan") if y is None else y for y in ys]
        ax.plot(xs, ys, marker=markers[i % len(markers)], markersize=4, linewidth=1.6,
                ls="--" if d in dashed else "-", label=d, color=palette[i % len(palette)])
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(([0] if ratio else []) + CAPS)
    ax.set_xticklabels((["0"] if ratio else []) + [str(c) for c in CAPS])
    ax.set_xlabel("cap on the decisions of the model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ratio:
        ax.set_yscale("log", base=2)
        lo, hi = ylim if ylim else ax.get_ylim()
        ticks = [t for t in (0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8) if lo <= t <= hi]
        ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
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
    UNSAT = [d for d in RANDOM if d.startswith("uuf")]
    flat_mrir = shared_ylim(agg, MODELS, FLAT, 0, ratio=True)
    flat_time = shared_ylim(agg, MODELS, FLAT, 1)
    rand_mrir = shared_ylim(agg, MODELS, RANDOM, 0, ratio=True)

    for model in MODELS:
        # MRIR against the cap on graph colouring
        plot_curves(agg, model, FLAT, idx=0, ylabel="MRIR (log scale)",
                    title=f"{TITLE[model]} on graph colouring",
                    out_path=os.path.join(args.out_dir, f"{model}.png"),
                    ratio=True, ylim=flat_mrir)
        # wall-clock time against the cap
        plot_curves(agg, model, FLAT, idx=1, ylabel="median seconds per instance",
                    title=f"{TITLE[model]} solving time on graph colouring",
                    out_path=os.path.join(args.out_dir, f"{model}_time.png"),
                    ylim=flat_time)
        # MRIR on random 3-SAT, unsatisfiable families dashed
        plot_curves(agg, model, RANDOM, idx=0, ylabel="MRIR (log scale)",
                    title=f"{TITLE[model]} on uniform random 3-SAT",
                    out_path=os.path.join(args.out_dir, f"{model}_random.png"),
                    ratio=True, ylim=rand_mrir, dashed=UNSAT)

if __name__ == "__main__":
    main()
