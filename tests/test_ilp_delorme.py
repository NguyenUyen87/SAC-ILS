"""Preprocessing and models of Delorme et al. (2019): on small random instances, the
preprocessing never removes a pair that belongs to some weakly stable matching, and all
models (with and without preprocessing) reach the brute-force maximum, with a matching
that is weakly stable in the original instance."""
import numpy as np

from sacsmti import gen
from sacsmti.ilp_delorme import preprocess, solve
from sacsmti.io import from_matrices
from sacsmti.stability import is_weakly_stable


def _stable_matchings(inst, men):
    n = len(men)
    out = []

    def rec(m, used, a):
        if m == n:
            if is_weakly_stable(inst, np.array(a)):
                out.append(list(a))
            return
        a.append(-1)
        rec(m + 1, used, a)
        a.pop()
        for w in men[m]:
            if w not in used:
                a.append(w)
                rec(m + 1, used | {w}, a)
                a.pop()
    rec(0, frozenset(), [])
    return out


def test_preprocessing_and_models_match_brute_force():
    rng = np.random.default_rng(5)
    removed_total = checked = 0
    for t in range(60):
        n = int(rng.integers(3, 7))
        inst = gen.c1(n, float(rng.choice([0.2, 0.4, 0.6])), float(rng.choice([0.3, 0.6, 0.9])), rng=rng, geom="p2")
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        stable = _stable_matchings(inst, men)
        best = max(sum(1 for w in a if w >= 0) for a in stable)
        men2, women2, removed = preprocess(men, women)
        removed_total += removed
        kept = {(m, w) for m, d in enumerate(men2) for w in d}
        for a in stable:                                      # no stable pair is removed
            assert all((m, w) in kept for m, w in enumerate(a) if w >= 0), t
        for model in ("M1", "M2", "M3", "M4"):
            for pre in (False, True):
                r = solve(men, women, model, pre=pre, time_limit=10, workers=1)
                assert r["status"] == "OPTIMAL", (t, model, pre)
                a = r["assign"]
                assert is_weakly_stable(inst, a), (t, model, pre)
                assert int((a >= 0).sum()) == best, (t, model, pre)
                checked += 1
    assert checked == 480 and removed_total > 0
