"""Statistics of the chains applied by SAC (exact counts, no rounding).

From Gale-Shapley, chains are applied until none is found (L_max = 16 men, B = 5000
nodes). For every applied chain we record: its number of edges; whether some agent
is worse off (equivalently, whether it contains an edge strictly blocked in M);
and the number of edges of a SHORTEST stable augmenting chain from the same single
man in the same matching, found by iterative deepening on the maximum number of men
(1, 2, ..., 16) with a large node budget. Deterministic. Output: chain_stats.csv and a summary.

    python experiments/chain_stats.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import bench_path, out, scal_path  # noqa: E402
from sacsmti import gen  # noqa: E402
from sacsmti.io import read_json  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.sac import apply_path, find_stable_augmenting_path  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402


def worse_off(men, women, pm, pw, P):
    for i, (m, w) in enumerate(P):
        if i > 0 and men[m][w] > men[m][int(pm[m])]:
            return True
        q = int(pw[w])
        if q >= 0 and women[w][m] > women[w][q]:
            return True
    return False


def strictly_blocked(men, women, pm, pw, P):
    for m, w in P:
        q = int(pw[w])
        if (q >= 0 and women[w][m] > women[w][q]) or (pm[m] >= 0 and men[m][w] > men[m][int(pm[m])]):
            return True
    return False


def shortest(men, women, pm, pw, m0, rng, cap):
    for L in range(1, cap + 1):
        P, _ = find_stable_augmenting_path(men, women, pm, pw, m0, rng, L, 200000, "chain")
        if P:
            return 2 * len(P) - 1
    return None


def main(first=None):
    cases = []
    for p2 in ("0.2", "0.5", "0.8"):
        cases.append((f"large p1=0.99 p2={p2}", ("file", bench_path(1000, "0.99", p2, "flip_geometric", 0))))
    cases.append(("scal n=2000 i=0", ("file", scal_path(2000, 0))))
    cases.append(("cfg n=5000 L=20 p2=0.2", ("gen", 5000, 20, 0.2, 1)))
    rows = []
    for name, spec in cases[:first]:
        if spec[0] == "file":
            men, women = read_json(spec[1])
        else:
            men, women = gen.sparse_c1(spec[1], spec[2], spec[3], rng=np.random.default_rng(spec[4]))
        pm = SparseSearch(men, women, 0.05, 5, 5, 0.9, None, MatlabRNG(1)).pm.copy()
        pw = np.full(len(women), -1, dtype=np.int64)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        rng = np.random.default_rng(1)
        progress = True
        while progress:
            progress = False
            for m0 in rng.permutation(np.flatnonzero(pm < 0)):
                if not men[m0]:
                    continue
                P, _ = find_stable_augmenting_path(men, women, pm, pw, int(m0), rng, 16, 5000)
                if P:
                    short = shortest(men, women, pm, pw, int(m0), np.random.default_rng(0), len(P))
                    rows.append({"case": name, "edges": 2 * len(P) - 1, "shortest_edges": short,
                                 "worse_off": int(worse_off(men, women, pm, pw, P)),
                                 "strictly_blocked": int(strictly_blocked(men, women, pm, pw, P))})
                    apply_path(pm, pw, P)
                    progress = True
        print(name, "done", flush=True)
    with open(out("chain_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby("case").agg(chains=("edges", "size"), worse=("worse_off", "sum"), strict=("strictly_blocked", "sum"),
                                edges_mean=("edges", "mean"), shortest_mean=("shortest_edges", "mean"),
                                edges_max=("edges", "max"), shortest_max=("shortest_edges", "max")).round(2))
    print("all:", len(d), "chains;", int(d.worse_off.sum()), "with an agent worse off;", int(d.strictly_blocked.sum()),
          "with a strictly blocked edge; identical:", bool((d.worse_off == d.strictly_blocked).all()))
    print("edges: mean %.1f median %d; shortest: mean %.1f median %d" % (d.edges.mean(), d.edges.median(),
                                                                         d.shortest_edges.mean(), d.shortest_edges.median()))
    print("shortest length histogram:", d.shortest_edges.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    main(ap.parse_args().first)
