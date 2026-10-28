#!/usr/bin/env python3
"""The arXiv package of the note, in arxiv/ and arxiv.tar.gz.

The source is copied without comments, with every \\upd{...} replaced by its
content and \\pdfoutput=1 on its first line; the package has the .bbl produced
by bibtex (arXiv does not run it) and only the figures that the note includes.
It is then compiled alone, in a directory of its own, to check that nothing is
missing. The script stops if a placeholder of the campaign is still in the
text ([To be written ...], $X$, -- in a table, the TODO on the logs), unless
--allow-pending is given, which is only for trying the package out.

    python3 make_arxiv.py [--allow-pending]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = "gatqsat.tex"
OUT = os.path.join(HERE, "arxiv")


def strip_comments(text):
    """Drop the comment lines and the comments at the end of a line; a % that
    is escaped (\\%) is text, and a % right after text is kept alone, since it
    may be there to suppress the space at the end of the line."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        m = re.search(r"(?<!\\)%", line)
        if m:
            line = line[:m.start() + 1] if line[:m.start()].strip() else ""
            if not line:
                continue
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n"


def resolve_upd(text):
    """Replace \\upd{...} (balanced braces) by its content, and drop the
    definition of the macro."""
    text = re.sub(r"\\newcommand\{\\upd\}\[1\]\{#1\}\n", "", text)
    out, i = [], 0
    while True:
        j = text.find("\\upd{", i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j])
        k, depth = j + len("\\upd{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        out.append(text[j + len("\\upd{"):k - 1])
        i = k
    return "".join(out)


def pending(text):
    pats = [r"\[To be written", r"\$X\$", r"\{--\}", r"& -- &", r"\[One sentence",
            r"\[Second, the conclusion", r"TODO", r"\\upd\{"]
    return [p for p in pats if re.search(p, text)]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-pending", action="store_true")
    a = ap.parse_args()

    src = open(os.path.join(HERE, TEX)).read()
    left = pending(src)
    if left and not a.allow_pending:
        sys.exit("placeholders still in %s: %s" % (TEX, ", ".join(left)))

    # the .bbl, from a build of the note in a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        for f in (TEX, "gatqsat.bib"):
            shutil.copy(os.path.join(HERE, f), tmp)
        shutil.copytree(os.path.join(HERE, "img"), os.path.join(tmp, "img"))
        for cmd in (["pdflatex", "-interaction=nonstopmode", TEX], ["bibtex", "gatqsat"]):
            run(cmd, tmp)
        bbl = open(os.path.join(tmp, "gatqsat.bbl")).read()

    text = "\\pdfoutput=1\n" + strip_comments(resolve_upd(src))
    figs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", text)

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "img"))
    open(os.path.join(OUT, TEX), "w").write(text)
    open(os.path.join(OUT, "gatqsat.bbl"), "w").write(bbl)
    for f in figs:
        shutil.copy(os.path.join(HERE, "img", f), os.path.join(OUT, "img", f))

    # the package compiled alone, twice for the references, with no bibtex
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(OUT, tmp, dirs_exist_ok=True)
        for _ in range(2):
            code, log = run(["pdflatex", "-interaction=nonstopmode", TEX], tmp)
        errors = [l for l in log.splitlines() if l.startswith("!")]
        undefined = re.findall(r"(?:Citation|Reference) `[^']+' .*undefined", log)
        if code or errors or undefined:
            sys.exit("the package does not compile alone:\n" + "\n".join(errors + undefined))
        pages = re.search(r"\((\d+) pages", log)
        shutil.copy(os.path.join(tmp, "gatqsat.pdf"), os.path.join(OUT, "gatqsat-arxiv.pdf"))

    with tarfile.open(os.path.join(HERE, "arxiv.tar.gz"), "w:gz") as t:
        for f in [TEX, "gatqsat.bbl"] + ["img/" + f for f in figs]:
            t.add(os.path.join(OUT, f), arcname=f)
    print("arxiv/ and arxiv.tar.gz: %s, the .bbl and %d figures (%s); %s pages%s"
          % (TEX, len(figs), ", ".join(figs), pages.group(1) if pages else "?",
             "; PENDING: " + ", ".join(left) if left else ""))


if __name__ == "__main__":
    main()
