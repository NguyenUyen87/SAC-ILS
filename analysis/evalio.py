"""Reading the result files of experiments/one_thread.py.

Columns were appended to the one-thread script over time (mean_edges, max_edges,
cut_frac); a file created earlier may keep the old header while newer rows carry the
extra columns at the end. This reader takes the current column list and pads shorter
rows, so old and new rows are read consistently.
"""
import csv

import pandas as pd

FIELDS = ["source", "name", "n", "method", "budget", "singles", "stable", "wall", "iters", "repairs",
          "repair_gain", "repair_time", "status", "bound", "s10", "s25", "s50", "s100",
          "mean_edges", "max_edges", "cut_frac"]
NUMERIC = ["n", "budget", "singles", "stable", "wall", "iters", "repairs", "repair_gain", "repair_time", "bound",
           "s10", "s25", "s50", "s100", "mean_edges", "max_edges", "cut_frac"]


def read_eval(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    assert header == FIELDS[:len(header)], "unexpected header"
    out = [dict(zip(FIELDS, r + [""] * (len(FIELDS) - len(r)))) for r in body if r]
    d = pd.DataFrame(out, columns=FIELDS)
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

