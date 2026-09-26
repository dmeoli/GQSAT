#!/usr/bin/env python3
"""Cross-domain knowledge-transfer study for Graph-Q-SAT vs GAT-Q-SAT.

Evaluates the *colouring-trained* checkpoints (no retraining) on a panel of
structured domains they never saw, and reports the median MRIR per domain. The
question: does the attention advantage transfer to unseen structured domains?

Every evaluation writes one line per problem, with the iterations of the model
and of both MiniSat baselines, as reeval.sh does; a log that is complete is not
evaluated again, so the study can be stopped and resumed. The MRIR is then read
from the logs with paper_numbers.py, against the stronger of the two MiniSat
runs on each domain, and averaged over the seeds.

Produces transfer.png and a summary under PAPER_IMG_DIR (../img/paper by
default). Run from the GQSAT root:  python3 transfer_study.py
On Colab:  OUT_ROOT=<Drive>/transfer_logs CKPT_ROOT=<Drive> \
           PAPER_IMG_DIR=<Drive>/transfer DEVICE_FLAG= python3 transfer_study.py
"""
import os
import re
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_numbers as pn

# colouring-trained runs, both seeds (the 2021 pair in runs/, the 2026 pair under
# CKPT_ROOT), each evaluated at its last checkpoint, as in the 2021 logs
MODELS = [
    ("Graph-Q-SAT", "graphqsat", ["Dec08_08-39-57_e63e47f25457", "gqsat_graphqsat"], "#b9a7d6"),
    ("GAT-Q-SAT",   "gatqsat",   ["Dec09_12-16-16_d4e65e7af705", "gqsat_gatqsat"],   "#4b2e83"),
]
# (label, dataset name of the logs, path) transfer domains; the models were
# trained on flat graph colouring
DOMAINS = [
    ("small-world\ncolouring", "small-world", "../data/small-world-coloring/transfer_eval"),
    ("AIM",                    "aim",         "../data/aim"),
    ("quasigroup",             "quasigroup",  "../data/quasigroup"),
    ("planning",               "planning",    "../data/planning"),
]
CAP = 200
# DEVICE_FLAG: --no-cuda on a machine without a GPU, which is the case locally,
# empty to use the GPU, as on Colab; the same knobs as reeval.sh.
DEVICE_FLAG = os.environ.get("DEVICE_FLAG", "--no-cuda")
OUT_ROOT = os.environ.get("OUT_ROOT", "runs")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "")
ROW = re.compile(r"^(sec to solve|[0-9])")


def run_dir(run):
    """runs/<run> in the repository, else the Drive folder of the checkpoints."""
    return os.path.join("runs", run) if os.path.isdir(os.path.join("runs", run)) \
        else os.path.join(CKPT_ROOT, run)


def last_checkpoint(d):
    """The checkpoint with the largest step in a run directory."""
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


def run_eval(run, model, name, path):
    """Write the per-problem log of one checkpoint on one domain, unless done."""
    out = os.path.join(log_dir(run), f"{name}-{model}-max{CAP}.tsv")
    if n_rows(out) >= n_problems(path):
        return
    d = run_dir(run)
    ck = last_checkpoint(d) if os.path.isdir(d) else None
    if ck is None:
        print(f"[skip] no checkpoint in {d}")
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


def main():
    pn.REEVAL_ROOT = None if OUT_ROOT == "runs" else OUT_ROOT
    for _, model, runs, _ in MODELS:
        for run in runs:
            for _, name, path in DOMAINS:
                if not os.path.isfile(os.path.join(path, "METADATA")):
                    print(f"[skip] no METADATA in {path}")
                    continue
                run_eval(run, model, name, path)

    # the logs are read as paper_numbers.py reads the re-evaluation ones; the
    # directory of the 2021 pair is runs/<run>/transfer when run locally
    def per_run(run, model, name):
        if OUT_ROOT == "runs":
            p = os.path.join(log_dir(run), f"{name}-{model}-max{CAP}.tsv")
            if not os.path.isfile(p):
                return None
            sc, _ = pn.scores(p, "stronger")
            return statistics.median(sc) if sc else None
        return pn.cell(run, name, CAP, "family", model)[0]

    results = {}  # variant -> {domain label: (mean over seeds, per-seed list)}
    for variant, model, runs, _ in MODELS:
        results[variant] = {}
        for label, name, _ in DOMAINS:
            vals = [v for v in (per_run(r, model, name) for r in runs) if v is not None]
            results[variant][label] = (statistics.fmean(vals) if vals else None, vals)
            print(f"{variant:12s} {name:12s} MRIR={results[variant][label]}", flush=True)

    # grouped bar chart: domains on x, one bar per model
    labels = [d[0] for d in DOMAINS]
    x = range(len(labels)); w = 0.38
    plt.figure(figsize=(8, 4.5))
    for i, (variant, _, _, col) in enumerate(MODELS):
        ys = [results[variant][d][0] or 0 for d in labels]
        xs = [k + (i - 0.5) * w for k in x]
        plt.bar(xs, ys, w, label=variant, color=col)
        for xi, yv in zip(xs, ys):
            plt.text(xi, yv + 0.02, f"{yv:.2f}", ha="center", fontsize=8)
    plt.axhline(1.0, color="gray", lw=.8, ls="--")
    plt.xticks(list(x), labels)
    plt.ylabel(f"median MRIR vs MiniSat (cap {CAP})")
    plt.title("Cross-domain transfer of the colouring-trained heuristic")
    plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    od = os.environ.get("PAPER_IMG_DIR", "../img/paper")
    os.makedirs(od, exist_ok=True)
    out = f"{od}/transfer.png"
    plt.savefig(out, dpi=130); plt.close(); print("wrote", out)

    with open(f"{od}/transfer_summary.md", "w") as f:
        f.write("# Cross-domain transfer (median MRIR against the stronger MiniSat, "
                "colouring-trained, cap %d, mean over the seeds [per seed])\n\n" % CAP)
        f.write("| domain | Graph-Q-SAT | GAT-Q-SAT |\n|---|---|---|\n")
        for d in labels:
            cells = []
            for variant in ("Graph-Q-SAT", "GAT-Q-SAT"):
                m, vals = results[variant][d]
                cells.append(f"{m:.2f} [{', '.join(f'{v:.2f}' for v in vals)}]" if m else "-")
            f.write(f"| {d.replace(chr(10), ' ')} | {cells[0]} | {cells[1]} |\n")
    print(f"wrote {od}/transfer_summary.md")


if __name__ == "__main__":
    main()
