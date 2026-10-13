"""Stable augmenting paths and chains: results are weakly stable and one larger, and
on small instances each search mode finds an object exactly when brute force over
all augmenting paths finds one (mode "path": stable paths; mode "chain": stable chains)."""
import numpy as np

from sacsmti import gen
from sacsmti.sac import apply_path, find_stable_augmenting_path
from sacsmti.io import from_matrices
from sacsmti.rng import MatlabRNG
from sacsmti.tbls import SparseSearch
from sacsmti.stability import is_weakly_stable


def _all_aug_paths(men, pm, pw, m0, path=None, used=None):
    path, used = path or [], used or set()
    for w in men[m0]:
        if w in used or w == pm[m0]:
            continue
        q = int(pw[w])
        if q < 0:
            yield path + [(m0, w)]
        else:
            yield from _all_aug_paths(men, pm, pw, q, path + [(m0, w)], used | {w})


def test_paths_found_are_stable_and_complete():
    rng = np.random.default_rng(7)
    found = exists = 0
    for t in range(60):
        n = int(rng.integers(6, 12))
        inst = gen.c1(n, 0.6, 0.5, rng=rng, geom="p2")
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
        pw = np.full(n, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        for m0 in np.flatnonzero(pm < 0):
            path, _ = find_stable_augmenting_path(men, women, pm, pw, int(m0), np.random.default_rng(t),
                                                  max_len=2 * n, node_budget=10 ** 6, mode="path")
            brute = []
            for P in _all_aug_paths(men, pm, pw, int(m0)):
                a, b = apply_path(pm.copy(), pw.copy(), P)
                if is_weakly_stable(inst, a):
                    brute.append(P)
            exists += bool(brute)
            if path is not None:
                found += 1
                a, b = apply_path(pm.copy(), pw.copy(), path)
                assert is_weakly_stable(inst, a)
                assert (a >= 0).sum() == (pm >= 0).sum() + 1
            assert (path is not None) == bool(brute)
    assert exists > 0 and found == exists


def test_differential_many_instance_types():
    """Search == brute force (existence of a stable augmenting path from each free man)
    on 400 random states: both tie distributions, several densities, starting matchings
    from Gale-Shapley and from earlier path applications."""
    rng = np.random.default_rng(11)
    checked = 0
    for t in range(400):
        n = int(rng.integers(5, 13))
        p1 = float(rng.choice([0.4, 0.55, 0.7]))
        geom = ["p2", "1-p2", None][t % 3]
        inst = gen.c1(n, p1, float(rng.choice([0.3, 0.6, 0.9])), rng=rng, geom=geom)
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
        pw = np.full(n, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        for _ in range(3):                         # a few improvement steps per instance
            progressed = False
            for m0 in np.flatnonzero(pm < 0):
                path, _ = find_stable_augmenting_path(men, women, pm, pw, int(m0), np.random.default_rng(t),
                                                      max_len=2 * n, node_budget=10 ** 6, mode="path")
                brute = any(is_weakly_stable(inst, apply_path(pm.copy(), pw.copy(), P)[0])
                            for P in _all_aug_paths(men, pm, pw, int(m0)))
                assert (path is not None) == brute, (t, int(m0))
                checked += 1
                if path is not None:
                    pm, pw = apply_path(pm, pw, path)
                    assert is_weakly_stable(inst, pm)
                    progressed = True
                    break
            if not progressed:
                break
    assert checked > 300


def _is_chain(inst, pm, pw, P):
    """Brute force: every intermediate matching has no blocking pair avoiding the displaced man."""
    n = len(pm)
    for i in range(1, len(P) + 1):
        a = pm.copy()
        for m, w in P[:i]:
            old = pw[w]
            if old >= 0 and a[old] == w:
                a[old] = -1
        for m, w in P[:i]:
            a[m] = w
        displaced = P[i][0] if i < len(P) else None
        R, W = inst.r_rank, inst.h_rank
        pw_a = np.full(n, -1)
        pw_a[a[a >= 0]] = np.flatnonzero(a >= 0)
        for m in range(n):
            if m == displaced:
                continue
            for w in np.flatnonzero(R[m] > 0):
                if a[m] == w or (a[m] >= 0 and R[m, w] >= R[m, a[m]]):
                    continue
                h = pw_a[w]
                if h == displaced:
                    continue
                if h < 0 or W[w, m] < W[w, h]:
                    return False
    return True


def test_chain_mode_matches_brute_force():
    rng = np.random.default_rng(13)
    checked = found = 0
    for t in range(300):
        n = int(rng.integers(5, 12))
        inst = gen.c1(n, float(rng.choice([0.4, 0.55, 0.7])), float(rng.choice([0.3, 0.6, 0.9])), rng=rng,
                      geom=["p2", "1-p2", None][t % 3])
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
        pw = np.full(n, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        for m0 in np.flatnonzero(pm < 0):
            chain, _ = find_stable_augmenting_path(men, women, pm, pw, int(m0), np.random.default_rng(t),
                                                   max_len=2 * n, node_budget=10 ** 6, mode="chain")
            brute = any(_is_chain(inst, pm, pw, P) for P in _all_aug_paths(men, pm, pw, int(m0)))
            assert (chain is not None) == brute, (t, int(m0))
            checked += 1
            if chain is not None:
                found += 1
                assert _is_chain(inst, pm, pw, chain)
                assert is_weakly_stable(inst, apply_path(pm.copy(), pw.copy(), chain)[0])
    assert checked > 200 and found > 20


def test_improving_paths_are_stable_chains():
    """Proposition: an augmenting path on which no agent except m0 is worse off
    (the improvement chains of Erdil and Ergin) is a stable augmenting chain."""
    rng = np.random.default_rng(19)
    improving = 0
    for t in range(200):
        n = int(rng.integers(5, 12))
        inst = gen.c1(n, float(rng.choice([0.4, 0.55])), float(rng.choice([0.3, 0.6, 0.9])), rng=rng, geom="p2")
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
        pw = np.full(n, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        for m0 in np.flatnonzero(pm < 0):
            for P in _all_aug_paths(men, pm, pw, int(m0)):
                ok = True
                for i, (m, w) in enumerate(P):
                    if i > 0 and men[m][w] > men[m][int(pm[m])]:
                        ok = False
                    old = int(pw[w])
                    if old >= 0 and women[w][m] > women[w][old]:
                        ok = False
                if ok:
                    improving += 1
                    assert _is_chain(inst, pm, pw, P)
    assert improving > 20


def test_paper_examples():
    """The two instances of the manuscript (Section 4): the worked example, where a chain
    leads to a perfect matching and TBLS has no adjustment, and the counterexample of
    Proposition 'local', where no stable augmenting path exists although a perfect
    weakly stable matching does."""
    from sacsmti.instance import Instance
    # worked example (Table 'example'); agents 0-based
    R = np.array([[0, 1, 0], [2, 2, 1], [0, 1, 1]])
    W = np.array([[0, 1, 0], [2, 1, 1], [0, 1, 1]])
    inst = Instance(R, W, np.ones(3, dtype=np.int64))
    men, women = from_matrices(R, W)
    M = np.array([-1, 2, 1])
    pw = np.full(3, -1)
    pw[M[M >= 0]] = np.flatnonzero(M >= 0)
    assert is_weakly_stable(inst, M)
    chain, _ = find_stable_augmenting_path(men, women, M, pw, 0, np.random.default_rng(0), 10, 10 ** 6, "chain")
    assert chain == [(0, 1), (2, 2), (1, 0)]
    assert apply_path(M.copy(), pw.copy(), chain)[0].tolist() == [1, 0, 2]
    # no TBLS adjustment: every single agent's candidates strictly prefer their partners
    for f, x in ((0, 1),):                                   # m1 -> w2 (partner m3)
        assert W[x, f] > W[x, pw[x]]
    assert R[1, 0] > R[1, M[1]]                              # w1 -> m2 (partner w3)
    # counterexample of Proposition 'local'
    R = np.array([[0, 0, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 0, 2, 1]])
    W = np.array([[0, 2, 1, 1], [0, 0, 1, 0], [1, 0, 2, 2], [1, 0, 0, 1]])
    inst = Instance(R, W, np.ones(4, dtype=np.int64))
    men, women = from_matrices(R, W)
    M = np.array([3, -1, 0, 2])
    pw = np.full(4, -1)
    pw[M[M >= 0]] = np.flatnonzero(M >= 0)
    assert is_weakly_stable(inst, M)
    paths = list(_all_aug_paths(men, M, pw, 1))
    assert paths == [[(1, 0), (2, 1)]]
    assert not is_weakly_stable(inst, apply_path(M.copy(), pw.copy(), paths[0])[0])
    assert is_weakly_stable(inst, np.array([2, 0, 1, 3]))


def test_chains_from_single_women_match_brute_force():
    """The search with the sides swapped finds chains from single women (the displaced
    agent is a woman) exactly when brute force on the transposed instance does."""
    from sacsmti.instance import Instance
    rng = np.random.default_rng(17)
    checked = found = 0
    for t in range(200):
        n = int(rng.integers(5, 11))
        inst = gen.c1(n, float(rng.choice([0.4, 0.55, 0.7])), float(rng.choice([0.3, 0.6, 0.9])), rng=rng,
                      geom=["p2", "1-p2", None][t % 3])
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        T = Instance(inst.h_rank, inst.r_rank, np.ones(n, dtype=np.int64))
        pm = SparseSearch(men, women, 0.05, 1, 1, 0.9, 1e9, MatlabRNG(t)).pm.copy()
        pw = np.full(n, -1)
        pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
        for w0 in np.flatnonzero(pw < 0):
            chain, _ = find_stable_augmenting_path(women, men, pw, pm, int(w0), np.random.default_rng(t),
                                                   max_len=2 * n, node_budget=10 ** 6, mode="chain")
            brute = any(_is_chain(T, pw, pm, P) for P in _all_aug_paths(women, pw, pm, int(w0)))
            assert (chain is not None) == brute, (t, int(w0))
            checked += 1
            if chain is not None:
                found += 1
                pw2, pm2 = apply_path(pw.copy(), pm.copy(), chain)
                assert is_weakly_stable(inst, pm2) and (pm2 >= 0).sum() == (pm >= 0).sum() + 1
    assert checked > 100 and found > 20


def test_chain_counterexamples():
    """Proposition 'chainpath': the earlier 3x3 example is a chain from the single woman;
    the new 3x3 example is a stable augmenting path that is a chain in neither direction."""
    from sacsmti.instance import Instance
    # earlier example: m1: (w2 w3) w1, m2: w3, m3: (w1 w2 w3); w1: (m1 m3), w2: (m1 m3), w3: (m1 m2 m3)
    R = np.array([[2, 1, 1], [0, 0, 1], [1, 1, 1]])
    W = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]])
    men, women = from_matrices(R, W)
    M = np.array([2, -1, 0])
    pw = np.full(3, -1)
    pw[M[M >= 0]] = np.flatnonzero(M >= 0)
    P = [(1, 2), (0, 0), (2, 1)]
    inst = Instance(R, W, np.ones(3, dtype=np.int64))
    T = Instance(W, R, np.ones(3, dtype=np.int64))
    assert is_weakly_stable(inst, apply_path(M.copy(), pw.copy(), P)[0])
    assert not _is_chain(inst, M, pw, P)                                  # not a chain from m2
    assert _is_chain(T, pw, M, [(w, m) for m, w in reversed(P)])          # a chain from w2
    # new example: m1: (w1 w2) w3, m2: (w1 w3), m3: (w2 w3); w1: (m1 m2), w2: (m1 m3), w3: (m2 m3) m1
    R = np.array([[1, 1, 2], [1, 0, 1], [0, 1, 1]])
    W = np.array([[1, 1, 0], [1, 0, 1], [2, 1, 1]])
    inst = Instance(R, W, np.ones(3, dtype=np.int64))
    men, women = from_matrices(R, W)
    M = np.array([0, -1, 2])
    pw = np.full(3, -1)
    pw[M[M >= 0]] = np.flatnonzero(M >= 0)
    assert is_weakly_stable(inst, M)
    P = [(1, 0), (0, 2), (2, 1)]
    T = Instance(W, R, np.ones(3, dtype=np.int64))
    assert is_weakly_stable(inst, apply_path(M.copy(), pw.copy(), P)[0])
    assert not _is_chain(inst, M, pw, P)
    assert not _is_chain(T, pw, M, [(w, m) for m, w in reversed(P)])


def test_ils_both_sides_weakly_stable():
    from sacsmti.instance import Instance
    from sacsmti.sac import sac_ils
    rng = np.random.default_rng(3)
    for t in range(15):
        n = int(rng.integers(20, 60))
        inst = gen.c1(n, 0.85, 0.6, rng=rng, geom="p2")
        men, women = from_matrices(inst.r_rank, inst.h_rank)
        r = sac_ils(men, women, 0.3, kick=2, max_len=16, node_budget=2000, seed=t, sides="both", order="random")
        assert is_weakly_stable(Instance(inst.r_rank, inst.h_rank, np.ones(n, dtype=np.int64)), r["assign"])
