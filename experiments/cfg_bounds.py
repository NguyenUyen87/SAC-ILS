"""Upper bounds for the generated configurations (Table 'cfg').

For each instance: the maximum matching of the acceptability graph (stability ignored)
and the LP relaxation of the integer programming model of Delorme et al. (2019),
solved by HiGHS (scipy.optimize.linprog). Agents with empty lists are counted.
Only the configurations with lists of about five entries (with lists of about 20 the
matchings are nearly perfect). Deterministic. Output: cfg_bounds.csv.

    python experiments/cfg_bounds.py
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import instances, load, out  # noqa: E402


def bounds(men, women):
    n, nw = len(men), len(women)
    pairs = [(m, w) for m, d in enumerate(men) for w in d]
    idx = {p: j for j, p in enumerate(pairs)}
    A = csr_matrix((np.ones(len(pairs)), ([m for m, _ in pairs], [w for _, w in pairs])), shape=(n, nw))
    mcm = int((maximum_bipartite_matching(A, perm_type="column") >= 0).sum())
    rows, cols, r = [], [], 0
    for m in range(n):                                   # sum_w x_mw <= 1
        for w in men[m]:
            rows.append(r); cols.append(idx[(m, w)])
        r += 1
    for w in range(nw):                                  # sum_m x_mw <= 1
        for m in women[w]:
            rows.append(r); cols.append(idx[(m, w)])
        r += 1
    n_cap = r
    for (m, w), j in idx.items():                        # stability: -(...) <= -1
        rm, rw = men[m][w], women[w][m]
        for w2, r2 in men[m].items():
            if r2 <= rm:
                rows.append(r); cols.append(idx[(m, w2)])
        for m2, r2 in women[w].items():
            if r2 <= rw and m2 != m:
                rows.append(r); cols.append(idx[(m2, w)])
        r += 1
    data = np.ones(len(rows))
    sign = np.where(np.array(rows) >= n_cap, -1.0, 1.0)
    A_ub = coo_matrix((data * sign, (rows, cols)), shape=(r, len(pairs))).tocsr()
    b_ub = np.concatenate([np.ones(n_cap), -np.ones(r - n_cap)])
    res = linprog(-np.ones(len(pairs)), A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method="highs")
    return mcm, (-res.fun if res.status == 0 else float("nan")), res.status


def main(first=None):
    path = out("cfg_bounds.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {r["name"] for r in csv.DictReader(fh)}
    new = not done
    with open(path, "a", newline="") as fh:
        wr = csv.writer(fh)
        if new:
            wr.writerow(["name", "n", "empty_m", "empty_w", "mcm", "lp", "lp_status"])
        k = 0
        for name, n, _, _, spec in instances("cfg", 5):
            if ",L=5," not in name:                    # lists of about 20: near-perfect matchings, no bound needed
                continue
            k += 1
            if first is not None and k > first:
                break
            if name in done:
                continue
            men, women = load(spec)
            mcm, lp, st = bounds(men, women)
            row = [name, n, sum(1 for d in men if not d), sum(1 for d in women if not d), mcm, round(lp, 2), st]
            wr.writerow(row)
            fh.flush()
            print(row, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    main(ap.parse_args().first)
