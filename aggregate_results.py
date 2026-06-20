#!/usr/bin/env python3
"""Aggregate Graph-Q-SAT / GAT-Q-SAT evaluation results from runs/*/*.tsv into
CSV + Markdown tables, replacing the manual Google Sheets workflow.

Each evaluation .tsv is named ``{dataset}-{model}-max{cap}.tsv`` and has columns
``sec to solve``, ``episode``, ``score``. ``score`` is the per-problem relative
iteration reduction w.r.t. MiniSat, so the per-file median is the MRIR (Median
Relative Iteration Reduction) used as the headline metric in the paper.

Outputs (under ``results/``):
  * ``summary.csv``      -- one row per (run, model, dataset, cap) with
                            n / median / mean / min / max of the score.
  * ``mrir_<model>.md``  -- pivot table (dataset x cap) of the MRIR, averaged
                            over the runs of that model (paper-style mean MRIR),
                            with the run count noted.

Pure standard library: no pandas/torch needed. Run from the GQSAT root:

    python3 aggregate_results.py [--runs-dir runs] [--out-dir results]
"""
import argparse
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

FNAME_RE = re.compile(r"^(?P<dataset>.+)-(?P<model>gatqsat|graphqsat)-max(?P<cap>\d+)\.tsv$")

# stable ordering for nicer tables
CAP_ORDER = [10, 50, 100, 300, 500, 1000]


def dataset_sort_key(name):
    """Sort flat/uf/uuf datasets by family then by size (first number)."""
    family = {"flat": 0, "uf": 1, "uuf": 2}
    m = re.match(r"([a-z]+)(\d+)", name)
    fam = family.get(m.group(1), 9) if m else 9
    size = int(m.group(2)) if m else 0
    return (fam, size, name)


def parse_tsv(path):
    """Return the list of finite per-problem scores in a result file."""
    scores = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw = row.get("score")
            if raw is None or raw == "":
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val == val and val not in (float("inf"), float("-inf")):  # drop NaN/inf
                scores.append(val)
    return scores


def collect(runs_dir):
    """records[(run, model, dataset, cap)] = stats dict."""
    records = []
    for path in glob.glob(os.path.join(runs_dir, "*", "*.tsv")):
        run = os.path.basename(os.path.dirname(path))
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        scores = parse_tsv(path)
        if not scores:
            continue
        records.append({
            "run": run,
            "model": m.group("model"),
            "dataset": m.group("dataset"),
            "cap": int(m.group("cap")),
            "n": len(scores),
            "median": statistics.median(scores),
            "mean": statistics.fmean(scores),
            "min": min(scores),
            "max": max(scores),
        })
    return records


def write_summary_csv(records, out_path):
    fields = ["run", "model", "dataset", "cap", "n", "median", "mean", "min", "max"]
    records = sorted(records, key=lambda r: (r["model"], dataset_sort_key(r["dataset"]), r["cap"], r["run"]))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = dict(r)
            for k in ("median", "mean", "min", "max"):
                row[k] = f"{r[k]:.4f}"
            w.writerow(row)


def write_mrir_markdown(records, model, out_path):
    """Pivot table dataset x cap of mean-over-runs of the per-file MRIR (median score)."""
    # group per-run MRIR (the per-file median) by (dataset, cap)
    by_cell = defaultdict(list)       # (dataset, cap) -> [per-run MRIR]
    runs_per_cell = defaultdict(set)
    for r in records:
        if r["model"] != model:
            continue
        by_cell[(r["dataset"], r["cap"])].append(r["median"])
        runs_per_cell[(r["dataset"], r["cap"])].add(r["run"])

    if not by_cell:
        return False

    datasets = sorted({d for (d, _c) in by_cell}, key=dataset_sort_key)
    caps = [c for c in CAP_ORDER if any((d, c) in by_cell for d in datasets)]
    caps += sorted({c for (_d, c) in by_cell if c not in caps})

    n_runs = max((len(s) for s in runs_per_cell.values()), default=0)
    lines = []
    title = "GAT-Q-SAT" if model == "gatqsat" else "Graph-Q-SAT"
    lines.append(f"## {title} — MRIR (median relative iteration reduction vs MiniSat)")
    lines.append("")
    lines.append(f"Mean over {n_runs} run(s) of the per-problem-set median score. "
                 "Columns are the cap on the number of model decisions before handing control to MiniSat. "
                 "Values > 1 mean fewer iterations than MiniSat.")
    lines.append("")
    header = "| dataset | " + " | ".join(f"max{c}" for c in caps) + " |"
    sep = "|" + "---|" * (len(caps) + 1)
    lines.append(header)
    lines.append(sep)
    for d in datasets:
        cells = []
        for c in caps:
            vals = by_cell.get((d, c))
            cells.append(f"{statistics.fmean(vals):.2f}" if vals else "—")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    records = collect(args.runs_dir)
    if not records:
        raise SystemExit(f"No parseable result .tsv files found under {args.runs_dir!r}")

    os.makedirs(args.out_dir, exist_ok=True)
    write_summary_csv(records, os.path.join(args.out_dir, "summary.csv"))

    models = sorted({r["model"] for r in records})
    for model in models:
        write_mrir_markdown(records, model, os.path.join(args.out_dir, f"mrir_{model}.md"))

    n_runs = len({r["run"] for r in records})
    print(f"Parsed {len(records)} (run, model, dataset, cap) cells from {n_runs} runs.")
    print(f"Wrote {args.out_dir}/summary.csv and mrir_<model>.md for: {', '.join(models)}")


if __name__ == "__main__":
    main()
