#!/usr/bin/env python3
"""
The numbers of the note, recomputed from the re-evaluation logs.

A log written by the current evaluate.py carries, for every problem, the
iterations of the model and both MiniSat baselines (with and without restarts),
so the same run can be read against either of them, or against the stronger of
the two on each instance.  The note uses "family": MiniSat with restarts on
graph colouring, where it is the stronger one and the one Kurin et al. compare
with, and MiniSat without restarts on uniform random 3-SAT, where the restarts
cost iterations (and whose 2021 logs only carry that baseline).

Usage:
  python paper_numbers.py [--baseline family|no_restarts|with_restarts|best] [--cap 500]
"""
import argparse
import csv
import glob
import os
import statistics

# run -> (variant, training set); the four runs trained on unsatisfiable
# formulas are a control, and are kept apart from the others.  The colouring
# pair has two seeds: the 2021 runs and the pair retrained in 2026 on the
# current stack with the same configuration (checkpoints on Drive, logs under
# --reeval-root only).
RUNS = {
    "Dec08_08-39-57_e63e47f25457": ("Graph-Q-SAT", "flat50-115"),
    "Dec09_12-16-16_d4e65e7af705": ("GAT-Q-SAT", "flat50-115"),
    "gqsat_graphqsat": ("Graph-Q-SAT", "flat50-115"),
    "gqsat_gatqsat": ("GAT-Q-SAT", "flat50-115"),
    "Dec21_01-59-59_6bed2aa9b612": ("Graph-Q-SAT", "uf50-218"),
    "Dec23_01-48-54_90582559eea7": ("Graph-Q-SAT", "uf100-430"),
    "Nov12_14-06-54_c42e8ad320d8": ("GAT-Q-SAT", "uf50-218"),
    "Nov13_03-55-51_c42e8ad320d8": ("GAT-Q-SAT", "uf100-430"),
    "Dec21_14-55-50_5eccdc34d583": ("Graph-Q-SAT", "uuf50-218"),
    "Dec23_14-42-44_90582559eea7": ("Graph-Q-SAT", "uuf100-430"),
    "Nov12_20-35-32_c42e8ad320d8": ("GAT-Q-SAT", "uuf50-218"),
    "Nov14_03-26-36_54337a27a809": ("GAT-Q-SAT", "uuf100-430"),
}
FLAT = ["flat30-60", "flat75-180", "flat125-301", "flat150-360", "flat200-479"]
FLAT_ALL = ["flat30-60", "flat50-115", "flat75-180", "flat100-239",
            "flat125-301", "flat150-360", "flat175-417", "flat200-479"]
# the 2021 colouring pair, the only one evaluated at every cap
CURVE_RUNS = {"Dec08_08-39-57_e63e47f25457", "Dec09_12-16-16_d4e65e7af705"}
COL, SAT, UNSAT = {"flat50-115"}, {"uf50-218", "uf100-430"}, {"uuf50-218", "uuf100-430"}
RAND = ["uf50-218", "uf100-430", "uf250-1065",
        "uuf50-218", "uuf100-430", "uuf250-1065"]


def scores(path, baseline):
    """Per-problem MRIR of one evaluation, read against the chosen baseline;
    "stronger" takes, over the problems of the log, the MiniSat run with fewer
    iterations in total."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if baseline == "stronger":
        if not rows or not rows[0].get("model_iters"):
            return None, None
        nr = sum(float(r["minisat_no_restarts"]) for r in rows)
        wr = sum(float(r["minisat_with_restarts"]) for r in rows)
        baseline = "with_restarts" if wr < nr else "no_restarts"
    out, secs = [], []
    for row in rows:
        try:
            secs.append(float(row["sec to solve"]))
        except (TypeError, ValueError):
            pass
        if row.get("model_iters"):          # re-evaluation log
            it = float(row["model_iters"])
            nr, wr = float(row["minisat_no_restarts"]), float(row["minisat_with_restarts"])
            b = {"no_restarts": nr, "with_restarts": wr, "best": min(nr, wr)}[baseline]
            out.append(b / it)
        elif row.get("score"):              # 2021 log: only the score is there
            if baseline != "no_restarts":
                return None, None
            out.append(float(row["score"]))
    return out, secs


REEVAL_ROOT = None   # set by --reeval-root when the logs live outside the repo


def resolve(baseline, dataset):
    """"family": with restarts on graph colouring, without on uniform random
    3-SAT, the stronger of the two on any other family (the transfer panel)."""
    if baseline == "family":
        if dataset.startswith("flat"):
            return "with_restarts"
        if dataset.startswith("u"):
            return "no_restarts"
        return "stronger"
    return baseline


def cell(run, dataset, cap, baseline, model):
    baseline = resolve(baseline, dataset)
    dirs = [os.path.join("runs", run, "reeval"), os.path.join("runs", run)]
    if REEVAL_ROOT:
        dirs.insert(0, os.path.join(REEVAL_ROOT, run))
    for d in dirs:
        p = os.path.join(d, f"{dataset}-{model}-max{cap}.tsv")
        if os.path.exists(p):
            sc, secs = scores(p, baseline)
            if sc:
                return statistics.median(sc), statistics.median(secs) if secs else None
    return None, None


def group_runs(variant, trained, dataset, cap, baseline, idx=0, runs=None):
    """Per-run medians (idx 0 the MRIR, 1 the seconds) of the runs of a group
    that have a log for this cell; runs, if given, restricts the group."""
    name = "gatqsat" if variant == "GAT-Q-SAT" else "graphqsat"
    vals = [cell(r, dataset, cap, baseline, name)[idx]
            for r, (v, t) in RUNS.items()
            if v == variant and t in trained and (runs is None or r in runs)]
    return [v for v in vals if v is not None]


def group(variant, trained, dataset, cap, baseline, idx=0, runs=None):
    """Mean over the runs of the group of the per-run median."""
    vals = group_runs(variant, trained, dataset, cap, baseline, idx, runs)
    return statistics.fmean(vals) if vals else None


def fmt(v, w=6, d=2):
    return f"{v:{w}.{d}f}" if v is not None else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="family",
                    choices=["family", "no_restarts", "with_restarts", "best", "stronger"])
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--reeval-root", default=os.environ.get("OUT_ROOT"),
                    help="directory holding <run>/ subdirectories of re-evaluation logs")
    a = ap.parse_args()
    global REEVAL_ROOT
    REEVAL_ROOT = a.reeval_root
    b, cap = a.baseline, a.cap

    print(f"baseline = {b}, cap = {cap}\n")
    print("Table 1 - colouring-trained, on graph colouring (per seed, then mean)")
    print(f"  {'family':14s} {'Graph':>17} {'GAT':>17} {'delta':>7}")
    for d in FLAT_ALL:
        gs, as_ = group_runs("Graph-Q-SAT", COL, d, cap, b), group_runs("GAT-Q-SAT", COL, d, cap, b)
        g, a_ = group("Graph-Q-SAT", COL, d, cap, b), group("GAT-Q-SAT", COL, d, cap, b)
        delta = f"{a_ - g:+7.2f}" if (g and a_) else "      -"
        seeds = lambda v: "/".join(f"{x:.2f}" for x in v)
        print(f"  {d:14s} {seeds(gs):>10} {fmt(g,6)} {seeds(as_):>10} {fmt(a_,6)} {delta}")

    print("\n§6.2 - trained on satisfiable random, tested on colouring")
    print(f"  {'family':14s} {'Graph':>7} {'GAT':>7}")
    for d in FLAT_ALL:
        print(f"  {d:14s} {fmt(group('Graph-Q-SAT', SAT, d, cap, b),7)} "
              f"{fmt(group('GAT-Q-SAT', SAT, d, cap, b),7)}")

    print("\nTable 2 - trained on satisfiable random (control: trained on unsatisfiable)")
    print(f"  {'family':14s} {'Graph':>7} {'GAT':>7} {'Graph*':>8} {'GAT*':>8}")
    for d in RAND:
        print(f"  {d:14s} {fmt(group('Graph-Q-SAT', SAT, d, cap, b),7)} "
              f"{fmt(group('GAT-Q-SAT', SAT, d, cap, b),7)} "
              f"{fmt(group('Graph-Q-SAT', UNSAT, d, cap, b),8)} "
              f"{fmt(group('GAT-Q-SAT', UNSAT, d, cap, b),8)}")

    print("\nTable 3 - flat200-479, colouring-trained, against the budget (mean over the seeds)")
    print(f"  {'model':12s} " + " ".join(f"{'cap '+str(c):>20}" for c in (50, 500, 1000)))
    for v in ("Graph-Q-SAT", "GAT-Q-SAT"):
        cells = []
        for c in (50, 500, 1000):
            m = group(v, COL, "flat200-479", c, b)
            s = group(v, COL, "flat200-479", c, b, idx=1)
            n = len(group_runs(v, COL, "flat200-479", c, b))
            cells.append(f"{fmt(s,6)}s {fmt(m,6)} n={n}")
        print(f"  {v:12s} " + " ".join(f"{c:>20}" for c in cells))

if __name__ == "__main__":
    main()
