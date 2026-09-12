#!/usr/bin/env python3
"""
The numbers of the note, recomputed from the re-evaluation logs.

A log written by the current evaluate.py carries, for every problem, the
iterations of the model and both MiniSat baselines (with and without restarts),
so the same run can be read against either of them, or against the stronger of
the two, which is the comparison of Kurin et al.

Usage:
  python paper_numbers.py [--baseline no_restarts|with_restarts|best] [--cap 500]
"""
import argparse
import csv
import glob
import os
import statistics

# run -> (variant, training set); the four runs trained on unsatisfiable
# formulas are a control, and are kept apart from the others.
RUNS = {
    "Dec08_08-39-57_e63e47f25457": ("Graph-Q-SAT", "flat50-115"),
    "Dec09_12-16-16_d4e65e7af705": ("GAT-Q-SAT", "flat50-115"),
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
RAND = ["uf50-218", "uf100-430", "uf250-1065",
        "uuf50-218", "uuf100-430", "uuf250-1065"]


def scores(path, baseline):
    """Per-problem MRIR of one evaluation, read against the chosen baseline."""
    out, secs = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
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


def cell(run, dataset, cap, baseline, model):
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


def group(variant, trained, dataset, cap, baseline):
    """Mean over the runs of the group of the per-run median."""
    name = "gatqsat" if variant == "GAT-Q-SAT" else "graphqsat"
    vals = [cell(r, dataset, cap, baseline, name)[0]
            for r, (v, t) in RUNS.items() if v == variant and t in trained]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def fmt(v, w=6, d=2):
    return f"{v:{w}.{d}f}" if v is not None else " " * (w - 1) + "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="with_restarts",
                    choices=["no_restarts", "with_restarts", "best"])
    ap.add_argument("--cap", type=int, default=500)
    ap.add_argument("--reeval-root", default=os.environ.get("OUT_ROOT"),
                    help="directory holding <run>/ subdirectories of re-evaluation logs")
    a = ap.parse_args()
    global REEVAL_ROOT
    REEVAL_ROOT = a.reeval_root
    b, cap = a.baseline, a.cap
    COL, SAT, UNSAT = {"flat50-115"}, {"uf50-218", "uf100-430"}, {"uuf50-218", "uuf100-430"}

    print(f"baseline = {b}, cap = {cap}\n")
    print("Table 1 - colouring-trained, on graph colouring")
    print(f"  {'family':14s} {'Graph':>7} {'GAT':>7} {'delta':>7}")
    for d in FLAT:
        g, a_ = group("Graph-Q-SAT", COL, d, cap, b), group("GAT-Q-SAT", COL, d, cap, b)
        delta = f"{a_ - g:+7.2f}" if (g and a_) else "      -"
        print(f"  {d:14s} {fmt(g,7)} {fmt(a_,7)} {delta}")

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

    print("\nTable 3 - flat200-479, colouring-trained, against the budget")
    print(f"  {'model':12s} " + " ".join(f"{'cap '+str(c):>16}" for c in (50, 500, 1000)))
    for v in ("Graph-Q-SAT", "GAT-Q-SAT"):
        cells = []
        for c in (50, 500, 1000):
            run = [r for r, (vv, t) in RUNS.items() if vv == v and t in COL][0]
            name = "gatqsat" if v == "GAT-Q-SAT" else "graphqsat"
            m, s = cell(run, "flat200-479", c, b, name)
            cells.append(f"{fmt(s,6)}s {fmt(m,6)}")
        print(f"  {v:12s} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
