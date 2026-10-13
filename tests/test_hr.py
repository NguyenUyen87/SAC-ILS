"""HR reproduces the worked example of Section 3.2 of Viet et al. (2022)."""
import numpy as np

from sacsmti.baselines.hr import hr

MEN = {1: [[3], [8], [5], [2], [1, 7]], 2: [[5]], 3: [[8], [2, 3, 7], [5], [4]], 4: [[8], [5], [3]],
       5: [[1, 2, 7]], 6: [[6, 8]], 7: [[1], [3], [8], [7]], 8: [[1, 4], [7, 8], [2, 3, 5, 6]]}
WOMEN = {1: [[8], [1], [5], [7]], 2: [[5], [1, 8], [3]], 3: [[1], [4, 7, 8], [3]], 4: [[3, 8]],
         5: [[1, 3], [8], [4], [2]], 6: [[8], [6]], 7: [[5], [3, 7], [1], [8]], 8: [[8], [7], [6], [1], [3, 4]]}


def ranks(lists, n=8):
    A = np.zeros((n, n), dtype=np.int64)
    for v, groups in lists.items():
        for r, g in enumerate(groups, 1):
            for x in g:
                A[v - 1, x - 1] = r
    return A


def pairs(pm):
    return {m + 1: (int(w) + 1 if w >= 0 else None) for m, w in enumerate(pm)}


def test_example_of_the_paper():
    R, W = ranks(MEN), ranks(WOMEN)
    assert ((R > 0) == (W > 0).T).all()
    trace = []
    out = hr(R, W, trace=trace)
    # (1) first Gale-Shapley phase, (2) after the first Improve, (3) perfect matching
    assert pairs(trace[0]) == {1: 3, 2: None, 3: 8, 4: 5, 5: 2, 6: 6, 7: 1, 8: 4}
    assert pairs(trace[1]) == {1: 3, 2: 5, 3: 8, 4: None, 5: 2, 6: 6, 7: 1, 8: 4}
    assert pairs(trace[2]) == {1: 3, 2: 5, 3: 7, 4: 8, 5: 2, 6: 6, 7: 1, 8: 4}
    assert out["singles"] == 0 and out["iters"] == 2


def test_tie_reading_keeps_every_phase_stable():
    from sacsmti import gen
    from sacsmti.baselines.hr import _weakly_stable
    rng = np.random.default_rng(3)
    for _ in range(20):
        inst = gen.c1(60, 0.9, 0.5, rng=rng, geom="p2")
        R, W = inst.r_rank, inst.h_rank
        trace = []
        out = hr(R, W, trace=trace, candidate="tie")
        for pm in trace:
            pw = np.full(W.shape[0], -1)
            pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
            assert _weakly_stable(R, W, pm, pw)
        assert out["singles"] == int((trace[-1] < 0).sum())
