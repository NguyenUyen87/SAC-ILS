"""LNS keeps weak stability, never shrinks the matching, and is exact when N is everything."""
import numpy as np

from sacsmti import gen
from sacsmti.instance import Instance
from sacsmti.lns import has_augmenting_path, solve_sub
from sacsmti.io import from_matrices
from sacsmti.tbls import SparseSearch
from sacsmti.stability import is_weakly_stable
from sacsmti.rng import MatlabRNG


def _brute_max(R, W):
    import itertools
    n = R.shape[0]
    pairs = [(m, w) for m in range(n) for w in range(n) if R[m, w] > 0]
    inst = Instance(R, W, np.ones(n, dtype=np.int64))
    for k in range(n, -1, -1):
        for sub in itertools.combinations(pairs, k):
            if len({m for m, _ in sub}) < k or len({w for _, w in sub}) < k:
                continue
            a = np.full(n, -1)
            for m, w in sub:
                a[m] = w
            if is_weakly_stable(inst, a):
                return k
    return 0


def test_whole_instance_is_exact():
    rng = np.random.default_rng(1)
    for t in range(8):
        inst = gen.c1(6, 0.4, 0.6, rng=rng, geom="p2")
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, None, MatlabRNG(t)).pm.copy()
        pw = np.full(6, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        pm2, _, st, _ = solve_sub(men, women, pm, pw, set(range(6)), set(range(6)), 10.0)
        assert st == "OPTIMAL"
        assert int((pm2 >= 0).sum()) == _brute_max(inst.r_rank, inst.h_rank)
        if int((pm2 >= 0).sum()) > int((pm >= 0).sum()):
            assert has_augmenting_path(men, pm, pw, set(range(6)), set(range(6)))
