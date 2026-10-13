"""Reference values computed with the code that produced the results of the paper.

A failure of test_generators means that this NumPy version draws other random numbers:
the generated instances (cfg, development, very large) would then differ from those of
the paper. A failure of the algorithm tests means that the code behaves differently.
The algorithm runs are made deterministic by a fixed repair threshold (T) and an
unlimited time budget.
"""
import hashlib
import json

import numpy as np

from sacsmti import gen
from sacsmti.io import from_matrices
from sacsmti.rng import MatlabRNG
from sacsmti.sac import sac_search
from sacsmti.tbls import SparseSearch, run


def _h(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def _lists(men, women):
    return _h([[sorted(d.items()) for d in men], [sorted(d.items()) for d in women]])


def _dev_instance():
    inst = gen.c1(1000, 0.99, 0.8, rng=np.random.default_rng(7_000_000 + 990 * 100 + 80 + 1), geom="p2")
    return from_matrices(inst.r_rank, inst.h_rank)


def test_generators():
    men, women = gen.sparse_c1(5000, 5, 0.2, rng=np.random.default_rng(1_000_003 * 5000 + 10_007 * 5 + 20 * 101))
    assert _lists(men, women) == "6ee81674e30a8132"
    assert _lists(*_dev_instance()) == "330bacdc11d2b258"


def test_tbls():
    men, women = _dev_instance()
    for variant, singles, digest in (("TBLS", 22, "0dc3faf5e0c950a1"), ("TBLS+ILS", 21, "35ae6ba9eae66f91")):
        r = run(men, women, variant, max_iters=400, ku=5, kw=5, T=1e9, seed=1001)
        assert (r["singles"], _h(r["assign"].tolist())) == (singles, digest), variant


def test_gale_shapley_and_sac():
    men, women = _dev_instance()
    gs = SparseSearch(men, women, 0.05, 5, 5, 0.9, None, MatlabRNG(1001)).pm.copy()
    assert _h(gs.tolist()) == "e57c5db8d5183ae4"
    r = sac_search(men, women, gs, 1e9, max_len=16, node_budget=5000, seed=1001, order="random")
    assert (r["singles"], r["paths"], _h(r["assign"].tolist())) == (6, 38, "552b860596777c36")
