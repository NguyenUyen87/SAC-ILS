"""Single-thread CP-SAT against its starting matching (the Gale-Shapley hint) on the
scalability set: how many singles does CP-SAT remove within the budget? Recomputes the
hint exactly as experiments/one_thread.py does (same strategy seed per run). No timing;
needs one_thread_scal.csv with the CPSAT1 runs. Output: cpsat1_vs_gs.csv.

    python experiments/cpsat1_vs_gs.py
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import instances, load, out  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402


def main():
    import pandas as pd
    d = pd.read_csv(out("one_thread_scal.csv"))
    cp = d[d.method == "CPSAT1"].set_index("name")
    with open(out("cpsat1_vs_gs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "n", "gs_singles", "cpsat_singles"])
        for j, (name, n, budget, k, spec) in enumerate(instances("scal")):
            if name not in cp.index:
                continue
            men, women = load(spec)
            gs = SparseSearch(men, women, 0.05, k, k, 0.9, None, MatlabRNG(1000 + j)).pm
            w.writerow([name, n, int((gs < 0).sum()), int(cp.loc[name, "singles"])])


if __name__ == "__main__":
    main()
