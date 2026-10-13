"""Chains from single women, and stable paths that are not chains, on large instances.

For each instance: apply stable augmenting chains from single men until a full pass
finds none (SAC, final configuration: random order, at most 16 men, 5000 nodes).
At this local optimum, count
  - single women from which a chain exists (the same search with the sides swapped),
  - single men from which the path search (mode "path", pending requirements) finds a
    stable augmenting path although no chain from them was found,
both with the budget of SAC and with a 20-times larger one (100,000 nodes, 24 men).
Then continue with chains from both sides until a pass finds none, and report the
extra pairs. Every matching is checked for weak stability.

    python experiments/chain_sides.py
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import instances, load, out  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.sac import apply_path, find_stable_augmenting_path as F  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402

SMALL_B, BIG_B = (16, 5000), (24, 100000)


def chains(men, women, pm, pw, rng, sides):
    """Apply chains (random order, 16 men, 5000 nodes) until a pass finds none."""
    progress = True
    while progress:
        progress = False
        starts = [("m", int(m)) for m in np.flatnonzero(pm < 0) if men[m]]
        if sides == "both":
            starts += [("w", int(w)) for w in np.flatnonzero(pw < 0) if women[w]]
        for i in rng.permutation(len(starts)):
            side, v = starts[i]
            if side == "m" and pm[v] < 0:
                p, _ = F(men, women, pm, pw, v, rng, *SMALL_B, "chain", "random")
                if p:
                    apply_path(pm, pw, p)
                    progress = True
            elif side == "w" and pw[v] < 0:
                p, _ = F(women, men, pw, pm, v, rng, *SMALL_B, "chain", "random")
                if p:
                    apply_path(pw, pm, p)
                    progress = True
    return pm, pw


def main(first=None):
    cases = [c for c in instances("large") if "p1=0.99" in c[0]][:12]
    cases += [c for c in instances("scal") if c[1] == 2000][:3] + [c for c in instances("scal") if c[1] == 10000][:2]
    cases = cases[:first]
    fields = ["name", "n", "sac_singles", "single_w", "wchain_small", "wchain_big", "single_m", "path_not_chain_small",
              "path_not_chain_big", "mchain_big", "both_singles", "stable", "time"]
    with open(out("chain_sides.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for name, n, _, _, spec in cases:
            t0 = time.perf_counter()
            men, women = load(spec)
            rng = np.random.default_rng(7)
            pm = SparseSearch(men, women, 0.05, 5, 5, 0.9, None, MatlabRNG(1)).pm.copy()
            pw = np.full(len(women), -1, dtype=np.int64)
            pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
            pm, pw = chains(men, women, pm, pw, rng, "m")
            row = {"name": name, "n": n, "sac_singles": int((pm < 0).sum())}
            sw = [int(x) for x in np.flatnonzero(pw < 0) if women[x]]
            sm = [int(x) for x in np.flatnonzero(pm < 0) if men[x]]
            row["single_w"], row["single_m"] = len(sw), len(sm)
            for tag, b in (("small", SMALL_B), ("big", BIG_B)):
                row[f"wchain_{tag}"] = sum(F(women, men, pw, pm, x, rng, *b, "chain", "random")[0] is not None for x in sw)
                cnt = mc = 0
                for x in sm:
                    c = F(men, women, pm, pw, x, rng, *b, "chain", "random")[0] is not None
                    mc += c
                    if not c and F(men, women, pm, pw, x, rng, *b, "path", "random")[0] is not None:
                        cnt += 1
                row[f"path_not_chain_{tag}"] = cnt
                if tag == "big":
                    row["mchain_big"] = mc
            pm2, pw2 = chains(men, women, pm.copy(), pw.copy(), rng, "both")
            row["both_singles"] = int((pm2 < 0).sum())
            row["stable"] = int(is_stable(men, women, pm) and is_stable(men, women, pm2))
            row["time"] = round(time.perf_counter() - t0, 1)
            w.writerow(row)
            fh.flush()
            print(row, flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    main(ap.parse_args().first)
