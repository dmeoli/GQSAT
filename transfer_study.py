#!/usr/bin/env python3
"""Cross-domain transfer of the colouring-trained heuristics.

Evaluates the models trained on graph colouring (flat50-115), with no
retraining, on the structured SATLIB families of ../data (AIM, planning and
those under ../data/satlib), and on the small-world colouring instances at
the nine rewiring levels of SATLIB, from a random graph (p = 1) to a ring
lattice (p = 0).

Every evaluation writes one line per problem, with the iterations of the model
and of both MiniSat baselines, as reeval.sh does; a complete log is not
evaluated again, so the study can be stopped and resumed. The runs are those of
paper_numbers.py, and the pair Graph-Q-SAT / GAT-Q-SAT (all its seeds) is
evaluated before the two controls, so that an interrupted session leaves the
main comparison complete. The MRIR is read from the logs with paper_numbers.py,
against the stronger of the two MiniSat runs on each family, and averaged over
the seeds.

Produces transfer.png, transfer_sw.png and transfer_summary.md under
PAPER_IMG_DIR (../img/paper by default). Run from the GQSAT root:
    python3 transfer_study.py [--plot-only]
On Colab:
    OUT_ROOT=<Drive>/transfer_logs CKPT_ROOT=<Drive> PAPER_IMG_DIR=<Drive>/transfer \
        DEVICE_FLAG= python3 transfer_study.py
"""
import argparse
import os
import re
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_numbers as pn
from paper_analysis import COL as _COL, ratio_axis, ratio_bars

COL = dict(_COL, **{"Graph-Q-SAT wide": "#8c8c8c", "Attention aggregation": "#e39b3a"})
# (label, dataset) of the structured families, i.e. the directory under ../data
# (aim, planning) or ../data/satlib; the order is the one of the figure
FAMILIES = [
    ("AIM", "aim"), ("jnh", "jnh"), ("dubois", "dubois"), ("pret", "pret"),
    ("pigeon hole", "pigeon-hole"), ("parity", "parity"), ("ais", "ais"),
    ("inductive\ninference", "inductive-inference"), ("circuit\nfaults", "circuit"),
    ("Beijing", "beijing"), ("hanoi", "hanoi"), ("planning", "planning"),
    ("quasigroup", "quasigroup"),
]
# the small-world levels: lpk has rewiring probability 2^-k, p0 is the lattice
SW = [(f"sw-lp{k}", 2.0 ** -k) for k in range(9)] + [("sw-p0", 0.0)]
CAP = 200
MAIN = ["Graph-Q-SAT", "GAT-Q-SAT"]
CONTROLS = ["Graph-Q-SAT wide", "Attention aggregation"]
DEVICE_FLAG = os.environ.get("DEVICE_FLAG", "--no-cuda")
OUT_ROOT = os.environ.get("OUT_ROOT", "runs")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "")
ROW = re.compile(r"^(sec to solve|[0-9])")


def runs_of(variant):
    return [r for r, (v, t) in pn.RUNS.items() if v == variant and t in pn.COL]


def run_dir(run):
    """runs/<run> in the repository, else the Drive folder of the checkpoints."""
    return os.path.join("runs", run) if os.path.isdir(os.path.join("runs", run)) \
        else os.path.join(CKPT_ROOT, run)


def last_checkpoint(d):
    steps = [int(m.group(1)) for f in os.listdir(d)
             if (m := re.fullmatch(r"model_(\d+)\.chkp", f))]
    return f"model_{max(steps)}.chkp" if steps else None


def n_problems(path):
    return sum(f.endswith(".cnf") for f in os.listdir(path))


def n_rows(log):
    if not os.path.isfile(log):
        return 0
    with open(log) as f:
        return sum(1 for _ in f) - 1


def log_dir(run):
    return os.path.join(OUT_ROOT, run) if OUT_ROOT != "runs" \
        else os.path.join("runs", run, "transfer")


def run_eval(run, variant, name):
    """Write the per-problem log of one checkpoint on one family, unless done."""
    path = pn.data_dir(name)
    if n_rows(os.path.join(path, "METADATA")) + 1 < n_problems(path):
        print(f"[skip] METADATA of {path} incomplete (run make_metadata.py first)")
        return
    out = os.path.join(log_dir(run), f"{name}-{pn.NAME[variant]}-max{CAP}.tsv")
    if n_rows(out) >= n_problems(path):
        return
    d = run_dir(run)
    ck = last_checkpoint(d) if os.path.isdir(d) else None
    if ck is None:
        print(f"[skip] {run}: no checkpoint")
        return
    os.makedirs(log_dir(run), exist_ok=True)
    cmd = [
        sys.executable, "evaluate.py", "--env-name", "sat-v0", "--core-steps", "-1",
        "--eps-final", "0.0", "--no_restarts", *DEVICE_FLAG.split(),
        "--test_time_max_decisions_allowed", str(CAP),
        "--eval-problems-paths", path, "--model-dir", d, "--model-checkpoint", ck,
    ]
    print(f"{run} {name} cap {CAP}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if ROW.match(l)]
    if len(lines) - 1 >= n_problems(path):
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print(f"[incomplete, discarded] {run} {name}", file=sys.stderr)


def mrir(variant, name):
    """(mean over the seeds, per-seed values) of the MRIR on one family."""
    vals = []
    for run in runs_of(variant):
        if OUT_ROOT == "runs":
            p = os.path.join(log_dir(run), f"{name}-{pn.NAME[variant]}-max{CAP}.tsv")
            sc = pn.scores(p, "stronger")[0] if os.path.isfile(p) else None
            v = statistics.median(sc) if sc else None
        else:
            v = pn.cell(run, name, CAP, "family", pn.NAME[variant])[0]
        if v is not None:
            vals.append(v)
    return (statistics.fmean(vals) if vals else None), vals


def trivial_share(name):
    """Share of the instances that MiniSat solves in fewer than 10 decisions."""
    rows = []
    with open(os.path.join(pn.data_dir(name), "METADATA")) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 3:
                rows.append(max(float(p[1]), float(p[2])))
    return sum(r < 10 for r in rows) / len(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="read the logs, evaluate nothing")
    a = ap.parse_args()
    pn.REEVAL_ROOT = None if OUT_ROOT == "runs" else OUT_ROOT
    names = [n for _, n in FAMILIES] + [n for n, _ in SW]
    if not a.plot_only:
        for group in (MAIN, CONTROLS):
            for variant in group:
                for run in runs_of(variant):
                    for name in names:
                        run_eval(run, variant, name)

    variants = [v for v in MAIN + CONTROLS if any(mrir(v, n)[0] for n in names)]
    od = os.environ.get("PAPER_IMG_DIR", "../img/paper")
    os.makedirs(od, exist_ok=True)

    # the structured families: bars from parity on a log2 axis
    fams = [(l, n) for l, n in FAMILIES if any(mrir(v, n)[0] for v in variants)]
    w = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, v in enumerate(variants):
        xs = [k + (i - (len(variants) - 1) / 2) * w for k in range(len(fams))]
        ratio_bars(ax, xs, [mrir(v, n)[0] for _, n in fams], w, label=v, color=COL[v])
    ratio_axis(ax, 0.25, 8)
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([f"{l}\n({n_problems(pn.data_dir(n))})" for l, n in fams], fontsize=8)
    ax.set_ylabel(f"MRIR (cap {CAP}, log scale)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(f"{od}/transfer.png", dpi=130); plt.close(fig); print("wrote", f"{od}/transfer.png")

    # the small-world levels: MRIR against the rewiring probability
    fig, ax = plt.subplots(figsize=(7, 4.3))
    xs = [k for k in range(len(SW))]
    for v in variants:
        ys = [mrir(v, n)[0] for n, _ in SW]
        ax.plot(xs, [y if y is not None else float("nan") for y in ys], marker="o",
                lw=1.8, label=v, color=COL[v])
    ratio_axis(ax)
    ax.set_xticks(xs); ax.set_xticklabels(["1"] + [f"$2^{{-{k}}}$" for k in range(1, 9)] + ["0"])
    ax.set_xlabel("rewiring probability (1: random graph, 0: ring lattice)")
    ax.set_ylabel(f"MRIR (cap {CAP}, log scale)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(f"{od}/transfer_sw.png", dpi=130); plt.close(fig); print("wrote", f"{od}/transfer_sw.png")

    with open(f"{od}/transfer_summary.md", "w") as f:
        f.write(f"# Transfer of the colouring-trained models (MRIR against the stronger "
                f"MiniSat, cap {CAP}, mean over the seeds [per seed])\n\n")
        f.write("| family | n | <10 decisions | " + " | ".join(variants) + " |\n")
        f.write("|---|---|---|" + "---|" * len(variants) + "\n")
        for label, n in fams + [(s, s) for s, _ in SW]:
            if not os.path.isfile(os.path.join(pn.data_dir(n), "METADATA")):
                continue
            cells = []
            for v in variants:
                m, vals = mrir(v, n)
                cells.append(f"{m:.2f} [{', '.join(f'{x:.2f}' for x in vals)}]" if m else "-")
            ts = trivial_share(n)
            f.write(f"| {label.replace(chr(10), ' ')} | {n_problems(pn.data_dir(n))} | "
                    f"{'' if ts is None else f'{ts:.0%}'} | " + " | ".join(cells) + " |\n")
    print(f"wrote {od}/transfer_summary.md")


if __name__ == "__main__":
    main()
