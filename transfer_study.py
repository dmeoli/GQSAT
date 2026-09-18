#!/usr/bin/env python3
"""Cross-domain knowledge-transfer study for Graph-Q-SAT vs GAT-Q-SAT.

Evaluates the *colouring-trained* checkpoints (no retraining) on a panel of
structured domains they never saw, and reports the median MRIR per domain. The
question: does the attention advantage transfer to unseen structured domains?

Eval-only, runs locally on CPU. Produces ../img/paper/transfer.png and a summary.
Run from the GQSAT root:  python3 transfer_study.py
"""
import os
import re
import sys
import subprocess
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# colouring-trained runs (Dec08 = Graph-Q-SAT, Dec09 = GAT-Q-SAT on flat50-115),
# each evaluated at its last checkpoint, as in the 2021 logs
MODELS = [
    ("Graph-Q-SAT", "runs/Dec08_08-39-57_e63e47f25457", "#b9a7d6"),
    ("GAT-Q-SAT",   "runs/Dec09_12-16-16_d4e65e7af705", "#4b2e83"),
]
# (label, path) transfer domains; the model was trained on flat graph colouring
DOMAINS = [
    ("small-world\ncolouring", "../data/small-world-coloring/transfer_eval"),
    ("AIM",                    "../data/aim"),
    ("quasigroup",             "../data/quasigroup"),
    ("planning",               "../data/planning"),
]
CAP = 200
MEDIAN_RE = re.compile(r"median_relative_score:\s*([0-9.]+)")


def last_checkpoint(run_dir):
    """The checkpoint with the largest step in a run directory."""
    steps = [int(m.group(1)) for f in os.listdir(run_dir)
             if (m := re.fullmatch(r"model_(\d+)\.chkp", f))]
    return f"model_{max(steps)}.chkp"


def run_eval(run_dir, checkpoint, problems_path):
    """Return the median MRIR of one checkpoint on one domain (None on failure)."""
    cmd = [
        sys.executable, "evaluate.py", "--env-name", "sat-v0", "--core-steps", "-1",
        "--eps-final", "0.0", "--no_restarts", "--no-cuda",
        "--test_time_max_decisions_allowed", str(CAP),
        "--eval-problems-paths", problems_path,
        "--model-dir", run_dir, "--model-checkpoint", checkpoint,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600).stdout
    except subprocess.TimeoutExpired:
        return None
    vals = [float(m) for m in MEDIAN_RE.findall(out)]
    return statistics.fmean(vals) if vals else None


def main():
    results = {}  # model -> {domain_label: mrir}
    for name, run_dir, _ in MODELS:
        ck = last_checkpoint(run_dir)
        results[name] = {}
        for dlabel, dpath in DOMAINS:
            if not os.path.isfile(os.path.join(dpath, "METADATA")):
                print(f"[skip] no METADATA in {dpath}")
                results[name][dlabel] = None
                continue
            mrir = run_eval(run_dir, ck, dpath)
            results[name][dlabel] = mrir
            print(f"{name:12s} {dlabel:20s} MRIR={mrir}", flush=True)

    # grouped bar chart: domains on x, one bar per model
    labels = [d[0] for d in DOMAINS]
    x = range(len(labels)); w = 0.38
    plt.figure(figsize=(8, 4.5))
    for i, (name, _, col) in enumerate(MODELS):
        ys = [results[name].get(d[0]) or 0 for d in DOMAINS]
        xs = [k + (i - 0.5) * w for k in x]
        plt.bar(xs, ys, w, label=name, color=col)
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
        f.write("# Cross-domain transfer (median MRIR, colouring-trained, cap %d)\n\n" % CAP)
        f.write("| domain | Graph-Q-SAT | GAT-Q-SAT |\n|---|---|---|\n")
        for d in labels:
            g = results["Graph-Q-SAT"].get(d)
            a = results["GAT-Q-SAT"].get(d)
            f.write(f"| {d.replace(chr(10), ' ')} | {g} | {a} |\n")
    print(f"wrote {od}/transfer_summary.md")


if __name__ == "__main__":
    main()
