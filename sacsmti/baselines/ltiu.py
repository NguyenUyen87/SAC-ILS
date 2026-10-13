"""Port of LTIU.m: local search for SMTI.

Reference: Gelain, Pini, Rossi, Venable, Walsh. Local search for stable
marriage problems with ties and incomplete lists.
"""
import time

import numpy as np

from .common import (RNG, blocking_matrix, check_blocking_pair, find_first,
                     make_random_matching, number_of_blocking_pairs, partners,
                     remove_blocking_pair)


def find_undominated_blocking_pairs1(men_rank_list, women_rank_list, M):
    """Undominated BPs from the men's point of view (vectorised)."""
    n = men_rank_list.shape[0]
    B = blocking_matrix(men_rank_list, women_rank_list, M)
    _, pw = partners(M, n)
    best = np.argmin(np.where(B, men_rank_list, np.iinfo(np.int64).max), axis=1)
    has = B.any(axis=1)
    X = []
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0 or not has[mi - 1]:
            continue
        wj = int(best[mi - 1]) + 1
        X.append([mi, int(M[1, i]), int(pw[wj]), wj])
    return np.array(X, dtype=np.int64).reshape(-1, 4)


def find_undominated_blocking_pairs2(men_rank_list, women_rank_list, M):
    """Undominated BPs from the women's point of view (vectorised)."""
    n = men_rank_list.shape[0]
    B = blocking_matrix(men_rank_list, women_rank_list, M).T  # [w-1, m-1]
    pm, _ = partners(M, n)
    best = np.argmin(np.where(B, women_rank_list, np.iinfo(np.int64).max), axis=1)
    has = B.any(axis=1)
    X = []
    for j in range(M.shape[1]):
        wj = int(M[1, j])
        if wj == 0 or not has[wj - 1]:
            continue
        mi = int(best[wj - 1]) + 1
        X.append([mi, int(pm[mi]), int(M[0, j]), wj])
    return np.array(X, dtype=np.int64).reshape(-1, 4)


def find_undominated_blocking_pairs1_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port (slow, used by tests)."""
    n = men_rank_list.shape[0]
    X = []
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
                    break
    return np.array(X, dtype=np.int64).reshape(-1, 4)


def find_undominated_blocking_pairs2_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port (slow, used by tests)."""
    n = women_rank_list.shape[0]
    X = []
    for j in range(M.shape[1]):
        wj = int(M[1, j])
        if wj == 0:
            continue
        mj = int(M[0, j])
        wr_mj = women_rank_list[wj - 1, mj - 1] if mj > 0 else n + 1
        x = women_rank_list[wj - 1]
        idx = np.argsort(x, kind="stable")
        for i in range(n):
            wr_mi = x[idx[i]]
            if 0 < wr_mi < wr_mj:
                mi = int(idx[i]) + 1
                wi = int(M[1, find_first(M[0] == mi)])
                if check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
                    X.append([mi, wi, mj, wj])
                    break
    return np.array(X, dtype=np.int64).reshape(-1, 4)


def find_cost_of_matching(men_rank_list, women_rank_list, M):
    """f = #blocking pairs + #singles not in any blocking pair."""
    return number_of_blocking_pairs(men_rank_list, women_rank_list, M)[0]


def LTIU(men_rank_list, women_rank_list, M, p=0.05, max_iters=3000, rng=None):
    """Returns (f_time, f_cost, f_stable, f_iter, f_reset, M_best)."""
    rng = rng or RNG
    n = men_rank_list.shape[0]
    M = M.copy()
    f = find_cost_of_matching(men_rank_list, women_rank_list, M)
    # f_best only tracks stable matchings (n = none found yet, as in MCS).
    # LTIU.m started from the cost of the initial matching, so an unstable
    # initial matching with a low cost was never replaced by a stable one.
    M_best = M.copy()
    f_best = n
    f_stable = 0
    num_reset = 0
    it = 0
    t0 = time.perf_counter()
    while it <= max_iters:
        it += 1
        if f_best == 0 and f_stable == 1:
            break
        # alternate between men's and women's point of view
        if it % 2 == 1:
            X = find_undominated_blocking_pairs1(men_rank_list, women_rank_list, M)
        else:
            X = find_undominated_blocking_pairs2(men_rank_list, women_rank_list, M)
        if X.shape[0] == 0:
            f_stable = 1
            if f_best > f:
                M_best = M.copy()
                f_best = f
            if f_best > 0:
                num_reset += 1
                M = make_random_matching(men_rank_list, women_rank_list, n, rng)
                f = find_cost_of_matching(men_rank_list, women_rank_list, M)
            continue
        # neighbours: remove each undominated blocking pair
        neighbors = []
        costs = []
        for mi, wi, _, wj in X:
            child = M.copy()
            remove_blocking_pair(child, int(mi), int(wi), int(wj))
            neighbors.append(child)
            costs.append(find_cost_of_matching(men_rank_list, women_rank_list, child))
        if rng.rand() < p:
            idx = rng.randi(len(costs)) - 1  # random walk
        else:
            idx = int(np.argmin(costs))
        M = neighbors[idx]
        f = costs[idx]
    f_time = time.perf_counter() - t0
    return f_time, f_best, f_stable, it, num_reset, M_best
