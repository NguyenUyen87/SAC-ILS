"""Port of MCS.m and MCS1.m: max-conflicts-based heuristic search for SMTI (Hoang Huu Viet)."""
import time

import numpy as np

from .common import (RNG, blocking_matrix, check_blocking_pair, find_first,
                     free_slot, matching_cost, partners, remove_blocking_pair)


def find_UBPs(men_rank_list, women_rank_list, M):
    """Undominated blocking pairs from the men's side.

    X rows are [mi, wi, mj, wj] (one per man, in M's column order) and y[w-1]
    counts how often woman w appears in X. Vectorised version of find_UBPs.
    """
    n = men_rank_list.shape[0]
    B = blocking_matrix(men_rank_list, women_rank_list, M)
    _, pw = partners(M, n)
    # best woman = lowest rank, then lowest index (MATLAB's stable sort)
    key = np.where(B, men_rank_list, np.iinfo(np.int64).max)
    best = np.argmin(key, axis=1)
    has = B.any(axis=1)
    X = []
    y = np.zeros(n, dtype=np.int64)
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0 or not has[mi - 1]:
            continue
        wj = int(best[mi - 1]) + 1
        X.append([mi, int(M[1, i]), int(pw[wj]), wj])
        y[wj - 1] += 1
    return np.array(X, dtype=np.int64).reshape(-1, 4), y


def find_UBPs_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port of find_UBPs in MCS.m (slow, used by tests)."""
    n = men_rank_list.shape[0]
    X = []
    y = np.zeros(n, dtype=np.int64)
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        mr_wi = men_rank_list[mi - 1, wi - 1] if wi > 0 else n + 1
        x = men_rank_list[mi - 1]
        idx = np.argsort(x, kind="stable")
        for j in range(n):
            mr_wj = x[idx[j]]
            if 0 < mr_wj < mr_wi:
                wj = int(idx[j]) + 1
                mj = int(M[0, find_first(M[1] == wj)])
                if check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
                    X.append([mi, wi, mj, wj])
                    y[wj - 1] += 1
                    break
    return np.array(X, dtype=np.int64).reshape(-1, 4), y


def find_UBPs1_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port of find_UBPs1 in MCS1.m (slow, used by tests)."""
    n = men_rank_list.shape[0]
    X = []
    y = np.zeros(n, dtype=np.int64)
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        mr_wi = men_rank_list[mi - 1, wi - 1] if wi > 0 else n + 1
        mi_rank_list = men_rank_list[mi - 1].copy()
        for _ in range(n):
            pos = mi_rank_list[mi_rank_list > 0]
            if pos.size == 0:
                break
            mr_wj = pos.min()
            if mr_wj == mr_wi:
                break
            wj = find_first(mi_rank_list == mr_wj) + 1
            mj = int(M[0, find_first(M[1] == wj)])
            if check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
                X.append([mi, wi, mj, wj])
                y[wj - 1] += 1
                break
            mi_rank_list[wj - 1] = 0
    return np.array(X, dtype=np.int64).reshape(-1, 4), y


def escape_local_minima(men_rank_list, women_rank_list, M, rng=None):
    rng = rng or RNG
    if rng.rand() <= 0.5:
        # pick a random single man and make every woman on his list single
        idxm = np.flatnonzero((M[0] != 0) & (M[1] == 0))
        mi = M[0, idxm[rng.randi(idxm.size) - 1]]
        for wi in np.flatnonzero(men_rank_list[mi - 1] > 0) + 1:
            j = find_first((M[0] != 0) & (M[1] == wi))
            if j is not None:
                k = free_slot(M)
                M[1, k] = M[1, j]
                M[1, j] = 0
    else:
        # pick a random single woman and make every man on her list single
        idxw = np.flatnonzero((M[0] == 0) & (M[1] != 0))
        wi = M[1, idxw[rng.randi(idxw.size) - 1]]
        for mi in np.flatnonzero(women_rank_list[wi - 1] > 0) + 1:
            j = find_first((M[0] == mi) & (M[1] != 0))
            if j is not None:
                k = free_slot(M)
                M[0, k] = M[0, j]
                M[0, j] = 0
    return M


def MCS(men_rank_list, women_rank_list, M, p=0.03, max_iters=10000, ubps=find_UBPs, rng=None):
    """Returns (f_time, f_cost, f_stable, f_iter, f_reset, f_nbps, M_best)."""
    rng = rng or RNG
    n = men_rank_list.shape[0]
    M = M.copy()
    M_best = M.copy()
    f_best = n
    f_stable = 0
    num_reset = 0
    it = 0
    f_nbps = []
    t0 = time.perf_counter()
    while it <= max_iters:
        X, y = ubps(men_rank_list, women_rank_list, M)
        f_nbps.append(X.shape[0])
        # no undominated blocking pair: M is stable
        if X.shape[0] == 0:
            f_stable = 1
            f = matching_cost(M)
            if f_best > f:
                M_best = M.copy()
                f_best = f
            if f > 0:
                num_reset += 1
                M = escape_local_minima(men_rank_list, women_rank_list, M, rng)
                continue
            break
        # heuristic h(mi) = n*y(wj) - rank_wj(mi)
        h = np.zeros(n, dtype=np.int64)
        h[X[:, 0] - 1] = n * y[X[:, 3] - 1] - women_rank_list[X[:, 3] - 1, X[:, 0] - 1]
        if rng.rand() < p:
            mi = X[rng.randi(X.shape[0]) - 1, 0]
        else:
            mi = int(np.argmax(h)) + 1  # max-conflicts, first maximum as in MATLAB
        row = find_first(X[:, 0] == mi)
        if row is None:
            # MATLAB would crash here (h is all zero and man 1 has no UBP);
            # fall back to the first UBP.
            row = 0
        mi, wi, _, wj = (int(v) for v in X[row])
        remove_blocking_pair(M, mi, wi, wj)
        it += 1
    f_time = time.perf_counter() - t0
    return f_time, f_best, f_stable, it, num_reset, f_nbps, M_best


def MCS1(men_rank_list, women_rank_list, M, rng=None):
    """MCS1.m: MCS with p = 0 and find_UBPs1 (equivalent result, different scan)."""
    return MCS(men_rank_list, women_rank_list, M, p=0.0, ubps=find_UBPs1_loop, rng=rng)
