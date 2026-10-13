"""Port of AS.m: Adaptive Search for SMTI.

Reference: Munera, Diaz, Abreu, Rossi. Solving Hard Stable Matching Problems
via Local Search and Cooperative Parallelization.
"""
import time

import numpy as np

from .common import (RNG, blocking_matrix, check_blocking_pair, find_first,
                     free_slot, partners, remove_blocking_pair)


def blocking_pair_error(women_rank_list, mj, wj, mi):
    """> 0 if (mi, wj) is a BP; larger when wj prefers mi much more than mj."""
    if mj == 0:
        return 1
    wr_mj = women_rank_list[wj - 1, mj - 1]
    wr_mi = women_rank_list[wj - 1, mi - 1]
    if wr_mi == 0:
        return 0
    return max(0, int(wr_mj - wr_mi))


def cost_of_marriage(men_rank_list, women_rank_list, M):
    """Returns (f, nbp, nsg, blocking_pairs), f = nbp*n + nsg.

    blocking_pairs rows are [mi, wi, mj, wj, error], one per man (his
    undominated BP), in M's column order. Vectorised version.
    """
    n = men_rank_list.shape[0]
    B = blocking_matrix(men_rank_list, women_rank_list, M)
    _, pw = partners(M, n)
    key = np.where(B, men_rank_list, np.iinfo(np.int64).max)
    best = np.argmin(key, axis=1)
    has = B.any(axis=1)
    rows = []
    nsg = 0
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        check_bp = False
        if has[mi - 1]:
            wj = int(best[mi - 1]) + 1
            mj = int(pw[wj])
            err = blocking_pair_error(women_rank_list, mj, wj, mi)
            if err > 0:
                rows.append([mi, wi, mj, wj, err])
                check_bp = True
        if not check_bp and wi == 0:
            nsg += 1
    nbp = len(rows)
    return nbp * n + nsg, nbp, nsg, np.array(rows, dtype=np.int64).reshape(-1, 5)


def cost_of_marriage_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port of cost_of_marriage in AS.m (slow, used by tests)."""
    n = men_rank_list.shape[0]
    nbp = 0
    nsg = 0
    rows = []
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        check_bp = False
        mr_wi = men_rank_list[mi - 1, wi - 1] if wi > 0 else n + 1
        x = men_rank_list[mi - 1]
        idx = np.argsort(x, kind="stable")
        for j in range(n):
            mr_wj = x[idx[j]]
            if 0 < mr_wj < mr_wi:
                wj = int(idx[j]) + 1
                mj = int(M[0, find_first(M[1] == wj)])
                if check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
                    err = blocking_pair_error(women_rank_list, mj, wj, mi)
                    if err > 0:
                        nbp += 1
                        rows.append([mi, wi, mj, wj, err])
                        check_bp = True
                        break
        if not check_bp and wi == 0:
            nsg += 1
    return nbp * n + nsg, nbp, nsg, np.array(rows, dtype=np.int64).reshape(-1, 5)


def _swap_pair(M, mi, wi, mj, wj):
    """Remove BP (mi, wj) given the exact pairs (mi, wi) and (mj, wj) of M."""
    mi_idx = find_first((M[0] == mi) & (M[1] == wi))
    wj_idx = find_first((M[0] == mj) & (M[1] == wj))
    if mi_idx is None or wj_idx is None:
        return False
    M[1, mi_idx] = wj
    M[1, wj_idx] = 0
    M[1, free_slot(M)] = wi
    return True


def reset(men_rank_list, women_rank_list, M, p, rng=None):
    rng = rng or RNG
    _, nbp, nsg, bps = cost_of_marriage(men_rank_list, women_rank_list, M)
    if nbp >= 1:
        bps = bps.copy()
        idx1 = int(np.argmax(bps[:, 4]))
        mi, wi, mj, wj = (int(v) for v in bps[idx1, :4])
        # second worst variable: ignore BPs that share (mi, mj, wj)
        same = (bps[:, 0] == mi) & (bps[:, 2] == mj) & (bps[:, 3] == wj)
        bps[same, 4] = 0
        idx2 = int(np.argmax(bps[:, 4]))
        # In MATLAB an empty position here would crash; it cannot happen
        # because (mi, wi) and (mj, wj) were read from M itself.
        _swap_pair(M, mi, wi, mj, wj)
        if nbp >= 2 and rng.rand() <= p:
            mi, wi, mj, wj = (int(v) for v in bps[idx2, :4])
            _swap_pair(M, mi, wi, mj, wj)
            return M
    if nsg >= 1:
        # a random single man and a random single woman
        idxm = np.flatnonzero((M[0] != 0) & (M[1] == 0))
        mi_idx = int(idxm[rng.randi(idxm.size) - 1])
        mi = M[0, mi_idx]
        idxw = np.flatnonzero((M[0] == 0) & (M[1] != 0))
        wj_idx = int(idxw[rng.randi(idxw.size) - 1])
        wj = M[1, wj_idx]
        if men_rank_list[mi - 1, wj - 1] > 0:
            M[1, mi_idx] = wj
            M[1, wj_idx] = 0
    else:
        # two random pairs of M: mi takes mj's partner
        idx = np.flatnonzero((M[0] != 0) & (M[1] != 0))
        mi_idx = int(idx[rng.randi(idx.size) - 1])
        mi, wi = M[0, mi_idx], M[1, mi_idx]
        wj_idx = int(idx[rng.randi(idx.size) - 1])
        wj = M[1, wj_idx]
        if men_rank_list[mi - 1, wj - 1] > 0:
            M[1, mi_idx] = wj
            M[1, wj_idx] = 0
            M[1, free_slot(M)] = wi
    return M


def AS(men_rank_list, women_rank_list, M, p=0.98, max_iters=3000, rng=None):
    """Returns (f_time, f_cost, f_stable, f_iter, f_reset, M_best)."""
    rng = rng or RNG
    n = men_rank_list.shape[0]
    M = M.copy()
    f, *_ = cost_of_marriage(men_rank_list, women_rank_list, M)
    M_best = M.copy()
    f_best = f
    f_stable = 0
    num_reset = 0
    it = 0
    t0 = time.perf_counter()
    while it <= max_iters:
        it += 1
        f, _, _, bps = cost_of_marriage(men_rank_list, women_rank_list, M)
        # f < n means no blocking pair: M is stable
        if f < n:
            f_stable = 1
            if f_best > f:
                M_best = M.copy()
                f_best = f
            if f_best == 0:
                break
            num_reset += 1
            M = reset(men_rank_list, women_rank_list, M, p, rng)
            continue
        # fix the worst variable
        idx = int(np.argmax(bps[:, 4]))
        mi, wi, _, wj = (int(v) for v in bps[idx, :4])
        remove_blocking_pair(M, mi, wi, wj)
        f_last = f
        f, *_ = cost_of_marriage(men_rank_list, women_rank_list, M)
        if f >= f_last:
            num_reset += 1
            M = reset(men_rank_list, women_rank_list, M, p, rng)
    f_time = time.perf_counter() - t0
    return f_time, f_best, f_stable, it, num_reset, M_best
