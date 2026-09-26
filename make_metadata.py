#!/usr/bin/env python3
"""Compute the MiniSat baselines (iterations without and with restarts) of the
instances of one or more families, one instance at a time, with a time limit.

add_metadata.py writes the METADATA of a whole directory in one run, and one
instance that MiniSat cannot solve blocks it; here each instance is run on its
own, under `timeout`, and its line is appended to <dir>/METADATA as soon as it
is known. An instance that is not solved, with and without restarts, within the
limit is moved to <dir>/excluded/ (the environment reads only the .cnf files of
the directory itself) and listed in <dir>/EXCLUDED, so that the rule "an
instance enters the panel if MiniSat solves it within the limit" is applied and
recorded. Lines already in METADATA and names already in EXCLUDED are skipped,
so the script can be stopped and resumed.

With --mirror DIR, METADATA and EXCLUDED of every family are also copied to
DIR/<family>/ after each instance, and restored from there at the start: on
Colab this keeps them on Drive across sessions.

Usage (from the GQSAT root):
  python make_metadata.py ../data/satlib/* [--timeout 600] [--mirror <Drive>/metadata]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def read_lines(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [l.rstrip("\n") for l in f if l.strip()]


def mirror_out(d, mirror):
    if not mirror:
        return
    m = os.path.join(mirror, os.path.basename(os.path.normpath(d)))
    os.makedirs(m, exist_ok=True)
    for name in ("METADATA", "EXCLUDED"):
        if os.path.isfile(os.path.join(d, name)):
            shutil.copy(os.path.join(d, name), os.path.join(m, name))


def mirror_in(d, mirror):
    """Take back what a previous session computed (the longer list wins)."""
    if not mirror:
        return
    m = os.path.join(mirror, os.path.basename(os.path.normpath(d)))
    for name in ("METADATA", "EXCLUDED"):
        src, dst = os.path.join(m, name), os.path.join(d, name)
        if os.path.isfile(src) and len(read_lines(src)) > len(read_lines(dst)):
            shutil.copy(src, dst)


def exclude(d, name):
    os.makedirs(os.path.join(d, "excluded"), exist_ok=True)
    p = os.path.join(d, name)
    if os.path.isfile(p):
        shutil.move(p, os.path.join(d, "excluded", name))


def one(d, name, timeout):
    """The METADATA line of one instance, or None if MiniSat does not finish."""
    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(os.path.abspath(os.path.join(d, name)), os.path.join(tmp, name))
        cmd = [sys.executable, "add_metadata.py", "--eval-problems-paths", tmp]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout, check=True)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return None
        lines = read_lines(os.path.join(tmp, "METADATA"))
        return lines[0] if lines else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--timeout", type=int, default=600,
                    help="seconds for the two MiniSat runs of one instance")
    ap.add_argument("--mirror", default=None)
    a = ap.parse_args()
    for d in a.dirs:
        mirror_in(d, a.mirror)
        done = {l.split(",")[0] for l in read_lines(os.path.join(d, "METADATA"))}
        skipped = set(read_lines(os.path.join(d, "EXCLUDED")))
        for name in skipped:
            exclude(d, name)
        todo = sorted(f for f in os.listdir(d)
                      if f.endswith(".cnf") and f not in done and f not in skipped)
        print(f"{d}: {len(done)} done, {len(skipped)} excluded, {len(todo)} to do", flush=True)
        for name in todo:
            line = one(d, name, a.timeout)
            if line is None:
                with open(os.path.join(d, "EXCLUDED"), "a") as f:
                    f.write(name + "\n")
                exclude(d, name)
                print(f"  {name}: not solved within {a.timeout} s, excluded", flush=True)
            else:
                with open(os.path.join(d, "METADATA"), "a") as f:
                    f.write(line + "\n")
                print(f"  {line}", flush=True)
            mirror_out(d, a.mirror)
        # keep METADATA sorted, as add_metadata.py writes it
        lines = sorted(set(read_lines(os.path.join(d, "METADATA"))))
        if lines:
            with open(os.path.join(d, "METADATA"), "w") as f:
                f.write("\n".join(lines) + "\n")
        mirror_out(d, a.mirror)


if __name__ == "__main__":
    main()
