"""Small random instances: how often does a stable augmenting path exist without a chain?

6000 random instances (gen.c1, three to eight agents per side), Gale-Shapley matching of
a random strategy. For every single man m0, all M-augmenting paths are enumerated; for
each end wk of a stable augmenting path from m0 we record whether a chain exists from m0
(man side) and from wk (woman side; the search with the sides swapped). Output:
chain_enum.csv (one row per pair (m0, wk)). Seeded, no timing.

    python experiments/chain_enum.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import out  # noqa: E402
from sacsmti import gen  # noqa: E402
from sacsmti.io import from_matrices  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.sac import apply_path, find_stable_augmenting_path as F  # noqa: E402
from sacsmti.stability import is_weakly_stable  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402


def _all_aug_paths(men, pm, pw, m0, path=None, used=None):
    """Every M-augmenting path from the single man m0 (brute force), as [(m0, w1), (m1, w2), ...]."""
    path = path or []
    used = used or set()
    for w in men[m0]:
        if w in used or w == pm[m0]:
            continue
        P = path + [(m0, w)]
        if pw[w] < 0:
            yield P
        else:
            yield from _all_aug_paths(men, pm, pw, int(pw[w]), P, used | {w})


def main():
    rng = np.random.default_rng(11)
    with open(out("chain_enum.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "n", "m0", "wk", "chain_from_m0", "chain_from_wk"])
        for t in range(6000):
            n = int(rng.integers(3, 9))
            inst = gen.c1(n, float(rng.choice([0.3, 0.5, 0.7])), float(rng.choice([0.5, 0.8])), rng=rng, geom="p2")
            men, women = from_matrices(inst.r_rank, inst.h_rank)
            pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
            pw = np.full(n, -1)
            pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
            for m0 in np.flatnonzero(pm < 0):
                ends = sorted({P[-1][1] for P in _all_aug_paths(men, pm, pw, int(m0))
                               if is_weakly_stable(inst, apply_path(pm.copy(), pw.copy(), P)[0])})
                if not ends:
                    continue
                cm = F(men, women, pm, pw, int(m0), np.random.default_rng(0), 2 * n, 10 ** 6)[0] is not None
                for wk in ends:
                    cw = F(women, men, pw, pm, int(wk), np.random.default_rng(0), 2 * n, 10 ** 6)[0] is not None
                    w.writerow([t, n, int(m0), int(wk), int(cm), int(cw)])


if __name__ == "__main__":
    main()
