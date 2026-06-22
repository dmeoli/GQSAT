#!/usr/bin/env python3
"""Generate the result figures locally from runs/*/*.tsv (replacing the old,
never-updating Google Sheets). Reproduces the README plots --- MRIR ("iterations
improvement") vs the number of model decisions (the cap), one line per dataset ---
and adds wall-clock-time figures from the "sec to solve" column.

Outputs PNGs under ../img/ (and is pure matplotlib/stdlib). Run from GQSAT root:

    python3 make_plots.py
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FNAME_RE = re.compile(r"^(?P<dataset>.+)-(?P<model>gatqsat|graphqsat)-max(?P<cap>\d+)\.tsv$")
CAPS = [10, 50, 100, 300, 500, 1000]

FLAT = ["flat30-60", "flat50-115", "flat75-180", "flat100-239",
        "flat125-301", "flat150-360", "flat175-417", "flat200-479"]
RANDOM = ["uf50-218", "uf100-430", "uf250-1065", "uuf50-218", "uuf100-430", "uuf250-1065"]
TITLE = {"gatqsat": "GAT-Q-SAT", "graphqsat": "Graph-Q-SAT"}


def collect(runs_dir):
    """(model, dataset, cap) -> list of per-run (median_score, median_sec)."""
    agg = defaultdict(list)
    for path in glob.glob(os.path.join(runs_dir, "*", "*.tsv")):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        scores, secs = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                try:
                    s = float(row["score"])
                    t = float(row["sec to solve"])
                except (ValueError, KeyError, TypeError):
                    continue
                if s == s:
                    scores.append(s)
                    secs.append(t)
        if scores:
            agg[(m["model"], m["dataset"], int(m["cap"]))].append(
                (statistics.median(scores), statistics.median(secs)))
    return agg


def mean_over_runs(agg, model, dataset, cap, idx):
    vals = [v[idx] for v in agg.get((model, dataset, cap), [])]
    return statistics.fmean(vals) if vals else None


def plot_curves(agg, model, datasets, idx, ylabel, title, out_path, start_one=False):
    plt.figure(figsize=(7, 4.3))
    from matplotlib.colors import LinearSegmentedColormap
    vmap = LinearSegmentedColormap.from_list("violet", ["#cdbfe6", "#4b2e83"])
    n = len(datasets)
    for i, d in enumerate(datasets):
        xs, ys = ([0], [1.0]) if start_one else ([], [])
        for c in CAPS:
            v = mean_over_runs(agg, model, d, c, idx)
            if v is not None:
                xs.append(c)
                ys.append(v)
        if len(ys) > (1 if start_one else 0):
            plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=d,
                     color=vmap(i / max(1, n - 1)))
    plt.xscale("symlog")
    plt.xticks([0] + CAPS, ["0"] + [str(c) for c in CAPS])
    plt.xlabel("model decisions")
    plt.ylabel(ylabel)
    plt.title(title)
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
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="../img")
    args = ap.parse_args()
    agg = collect(args.runs_dir)
    if not agg:
        raise SystemExit("no result .tsv found")
    os.makedirs(args.out_dir, exist_ok=True)

    for model in ("graphqsat", "gatqsat"):
        # MRIR ("iterations improvement") vs model decisions on graph colouring
        plot_curves(agg, model, FLAT, idx=0,
                    ylabel="iterations improvement (MRIR)",
                    title=f"{TITLE[model]} on graph colouring (flat)",
                    out_path=os.path.join(args.out_dir, f"{model}.png"),
                    start_one=True)
        # wall-clock time vs model decisions
        plot_curves(agg, model, FLAT, idx=1,
                    ylabel="median sec to solve",
                    title=f"{TITLE[model]} solving time (flat)",
                    out_path=os.path.join(args.out_dir, f"{model}_time.png"))
        # MRIR on random 3-SAT
        plot_curves(agg, model, RANDOM, idx=0,
                    ylabel="iterations improvement (MRIR)",
                    title=f"{TITLE[model]} on uniform-random 3-SAT",
                    out_path=os.path.join(args.out_dir, f"{model}_random.png"),
                    start_one=True)


if __name__ == "__main__":
    main()
