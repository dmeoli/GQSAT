#!/usr/bin/env python3
"""Train-set-aware analysis of the Graph-Q-SAT / GAT-Q-SAT runs for the report.

Each run is labelled by (model, train-family) from its model.yaml/status.yaml:
  * model        = GAT-Q-SAT (use_attention) | Graph-Q-SAT
  * train-family = coloring (flat*) | random (uniform 3-SAT)
Runs are then grouped so every comparison is a clean attention on/off ablation at a
FIXED training set. Produces the paper figures under ../img/paper/ and a summary.

Pure matplotlib/stdlib + yaml. Run from the GQSAT root:  python3 paper_analysis.py
"""
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

FNAME = re.compile(r"^(?P<dataset>.+)-(?P<model>gatqsat|graphqsat)-max(?P<cap>\d+)\.tsv$")
CAPS = [10, 50, 100, 300, 500, 1000]
FLAT = ["flat30-60", "flat50-115", "flat75-180", "flat100-239",
        "flat125-301", "flat150-360", "flat175-417", "flat200-479"]
RANDOM = ["uf50-218", "uf100-430", "uf250-1065", "uuf50-218", "uuf100-430", "uuf250-1065"]
SIZE = {"flat30-60": 90, "flat50-115": 150, "flat75-180": 225, "flat100-239": 300,
        "flat125-301": 375, "flat150-360": 450, "flat175-417": 525, "flat200-479": 600,
        "uf50-218": 50, "uf100-430": 100, "uf250-1065": 250,
        "uuf50-218": 50, "uuf100-430": 100, "uuf250-1065": 250}
COL = {"GAT-Q-SAT": "#4b2e83", "Graph-Q-SAT": "#b9a7d6"}  # deep violet vs light lilac


def run_label(run_dir):
    """Return (model, train_family) or None."""
    try:
        with open(os.path.join(run_dir, "model.yaml")) as f:
            attn = bool(yaml.load(f, Loader=yaml.Loader)["call_args"].get("use_attention"))
        model = "GAT-Q-SAT" if attn else "Graph-Q-SAT"
    except Exception:
        return None
    train = "random"
    try:
        txt = open(os.path.join(run_dir, "status.yaml"), errors="ignore").read()
        m = re.search(r"train_problems_paths[^\n]*", txt)
        if m and ("flat" in m.group(0) or "graph-coloring" in m.group(0)):
            train = "coloring"
    except Exception:
        pass
    return model, train


def parse_tsv(path):
    scores, secs = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                s, t = float(row["score"]), float(row["sec to solve"])
            except (ValueError, KeyError, TypeError):
                continue
            if s == s:
                scores.append(s); secs.append(t)
    return (statistics.median(scores), statistics.median(secs)) if scores else None


def collect():
    """data[(train, model)][dataset][cap] = list over runs of (median_score, median_sec)."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in glob.glob("runs/*/"):
        lab = run_label(run)
        if lab is None:
            continue
        model, train = lab
        for path in glob.glob(os.path.join(run, "*.tsv")):
            m = FNAME.match(os.path.basename(path))
            if not m:
                continue
            r = parse_tsv(path)
            if r:
                data[(train, model)][m["dataset"]][int(m["cap"])].append(r)
    return data


def mean_mrir(data, key, dataset, cap):
    vals = [v[0] for v in data.get(key, {}).get(dataset, {}).get(cap, [])]
    return statistics.fmean(vals) if vals else None


def mean_sec(data, key, dataset, cap):
    vals = [v[1] for v in data.get(key, {}).get(dataset, {}).get(cap, [])]
    return statistics.fmean(vals) if vals else None


def fig_thesis(data, out):
    """THE money figure: mean MRIR (cap 500) per regime, GAT vs Graph."""
    regimes = [("coloring", FLAT, "Trained on colouring\n→ colouring"),
               ("random", FLAT, "Trained on random\n→ colouring (transfer)"),
               ("random", RANDOM, "Trained on random\n→ random")]
    cap = 500
    gat, graph = [], []
    for train, dss, _ in regimes:
        g = [mean_mrir(data, (train, "GAT-Q-SAT"), d, cap) for d in dss]
        h = [mean_mrir(data, (train, "Graph-Q-SAT"), d, cap) for d in dss]
        gat.append(statistics.fmean([x for x in g if x is not None]))
        graph.append(statistics.fmean([x for x in h if x is not None]))
    x = range(len(regimes)); w = 0.38
    plt.figure(figsize=(8, 4.5))
    plt.bar([i - w/2 for i in x], graph, w, label="Graph-Q-SAT", color=COL["Graph-Q-SAT"])
    plt.bar([i + w/2 for i in x], gat,  w, label="GAT-Q-SAT",   color=COL["GAT-Q-SAT"])
    for i, (a, b) in enumerate(zip(graph, gat)):
        plt.text(i - w/2, a + .02, f"{a:.2f}", ha="center", fontsize=8)
        plt.text(i + w/2, b + .02, f"{b:.2f}", ha="center", fontsize=8)
    plt.axhline(1.0, color="gray", lw=.8, ls="--")
    plt.xticks(list(x), [r[2] for r in regimes], fontsize=9)
    plt.ylabel("mean MRIR vs MiniSat (cap 500)")
    plt.title("Graph attention helps on structured problems, not on random SAT")
    plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


def fig_curves(data, train, datasets, title, out):
    """MRIR vs decision-cap, GAT vs Graph, averaged over the datasets."""
    plt.figure(figsize=(7, 4.3))
    for model in ("Graph-Q-SAT", "GAT-Q-SAT"):
        xs, ys = [0], [1.0]
        for c in CAPS:
            vals = [mean_mrir(data, (train, model), d, c) for d in datasets]
            vals = [v for v in vals if v is not None]
            if vals:
                xs.append(c); ys.append(statistics.fmean(vals))
        plt.plot(xs, ys, marker="o", lw=1.8, label=model, color=COL[model])
    plt.xscale("symlog"); plt.xticks([0]+CAPS, ["0"]+[str(c) for c in CAPS])
    plt.xlabel("model decisions"); plt.ylabel("mean MRIR vs MiniSat")
    plt.title(title); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(out, dpi=130); plt.close(); print("wrote", out)


def fig_generalization(data, out):
    """Random-trained: MRIR vs problem size (cap 500), GAT vs Graph, colouring + random."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, dss, name in [(axes[0], FLAT, "graph colouring (transfer)"),
                          (axes[1], RANDOM, "uniform-random 3-SAT")]:
        for model in ("Graph-Q-SAT", "GAT-Q-SAT"):
            pts = [(SIZE[d], mean_mrir(data, ("random", model), d, 500)) for d in dss]
            pts = [(s, v) for s, v in pts if v is not None]
            pts.sort()
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker="o", lw=1.8, label=model, color=COL[model])
        ax.axhline(1.0, color="gray", lw=.8, ls="--")
        ax.set_xlabel("# variables"); ax.set_title(name); ax.grid(alpha=.3)
    axes[0].set_ylabel("mean MRIR vs MiniSat (cap 500)"); axes[0].legend()
    fig.suptitle("Generalisation across problem size (random-trained models)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(); print("wrote", out)


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
    data = collect()
    print("groups found:", {k: sum(len(c) for c in v.values()) for k, v in data.items()})
    od = "../img/paper"; os.makedirs(od, exist_ok=True)
    fig_thesis(data, f"{od}/thesis.png")
    fig_curves(data, "coloring", FLAT, "Trained & tested on graph colouring (in-distribution)",
               f"{od}/coloring_indist.png")
    fig_curves(data, "random", FLAT, "Random-trained, transfer to graph colouring",
               f"{od}/random_transfer.png")
    fig_curves(data, "random", RANDOM, "Trained & tested on uniform-random 3-SAT",
               f"{od}/random_indist.png")
    fig_generalization(data, f"{od}/generalization.png")
    write_summary(data, f"{od}/summary.md")


if __name__ == "__main__":
    main()
