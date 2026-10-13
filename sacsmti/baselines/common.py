"""Shared helpers of the baselines MCS, AS and LTIU, ported from their MATLAB code.

Conventions kept from MATLAB so results stay comparable:
  * men and women are numbered 1..n; 0 means "nobody".
  * men_rank_list[m-1, w-1] is the rank man m gives woman w (0 = unacceptable),
    women_rank_list[w-1, m-1] likewise.
  * a matching M is a 2 x 2n int array: M[0, i] is a man (or 0), M[1, i] his
    partner (or 0). A column (m, 0) is a single man, (0, w) a single woman and
    (0, 0) a free slot. Column order matters for tie-breaking, exactly as in MATLAB.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Random numbers
# ---------------------------------------------------------------------------
from ..rng import MatlabRNG  # noqa: E402


RNG = MatlabRNG()


def seed(s):
    RNG.seed(s)


# ---------------------------------------------------------------------------
# Small MATLAB-like helpers
# ---------------------------------------------------------------------------
def find_first(mask):
    """MATLAB find(mask,1,'first') as a 0-based index, or None."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def free_slot(M):
    """First (0,0) column of M."""
    return find_first((M[0] == 0) & (M[1] == 0))


def remove_blocking_pair(M, mi, wi, wj):
    """Satisfy blocking pair (mi, wj): mi marries wj, mj and wi become single.

    This is the block repeated in MCS.m, AS.m and LTIU.m.
    """
    mi_idx = find_first(M[0] == mi)
    wj_idx = find_first(M[1] == wj)
    M[1, mi_idx] = wj
    M[1, wj_idx] = 0
    wi_idx = free_slot(M)
    M[1, wi_idx] = wi


def partners(M, n):
    """pm[m] = partner of man m, pw[w] = partner of woman w (1-based, 0 = single)."""
    pm = np.zeros(n + 1, dtype=np.int64)
    pw = np.zeros(n + 1, dtype=np.int64)
    both = (M[0] > 0) & (M[1] > 0)
    pm[M[0, both]] = M[1, both]
    pw[M[1, both]] = M[0, both]
    return pm, pw


def blocking_matrix(men_rank_list, women_rank_list, M):
    """B[m-1, w-1] is True iff (m, w) is a blocking pair of M.

    Vectorised equivalent of calling check_blocking_pair for every (m, w).
    """
    n = men_rank_list.shape[0]
    mr = men_rank_list
    wr_t = women_rank_list.T  # wr_t[m-1, w-1] = rank woman w gives man m
    pm, pw = partners(M, n)
    rows = np.arange(n)
    big = np.iinfo(np.int64).max
    man_cur = np.where(pm[1:] > 0, mr[rows, np.maximum(pm[1:] - 1, 0)], big)
    woman_cur = np.where(pw[1:] > 0, women_rank_list[rows, np.maximum(pw[1:] - 1, 0)], big)
    return ((mr > 0) & (wr_t > 0)
            & (mr < man_cur[:, None])
            & (wr_t < woman_cur[None, :]))


# ---------------------------------------------------------------------------
# check_blocking_pair.m
# ---------------------------------------------------------------------------
def check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
    """(mi, wj) is a blocking pair of M, where (mi, wi) and (mj, wj) are in M."""
    mr_wj = men_rank_list[mi - 1, wj - 1]
    wr_mi = women_rank_list[wj - 1, mi - 1]
    if not (mr_wj > 0 and wr_mi > 0):
        return False
    if wi != 0 and not (mr_wj < men_rank_list[mi - 1, wi - 1]):
        return False
    if mj != 0 and not (wr_mi < women_rank_list[wj - 1, mj - 1]):
        return False
    return True


# ---------------------------------------------------------------------------
# number_of_blocking_pairs.m
# ---------------------------------------------------------------------------
def number_of_blocking_pairs(men_rank_list, women_rank_list, M):
    """Returns (f, nbp, nsg, BPs) with f = nbp + nsg.

    BPs rows are [mi, wi, mj, wj, wr_mi, mr_wj], in the same order as MATLAB.
    """
    n = men_rank_list.shape[0]
    B = blocking_matrix(men_rank_list, women_rank_list, M)
    pm, pw = partners(M, n)
    # MATLAB visits women in the column order of M(2,:)
    women_order = M[1][M[1] > 0]
    rows = []
    nsg = 0
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        hit = B[mi - 1, women_order - 1]
        for wj in women_order[hit]:
            wj = int(wj)
            rows.append([mi, wi, int(pw[wj]), wj,
                         int(women_rank_list[wj - 1, mi - 1]),
                         int(men_rank_list[mi - 1, wj - 1])])
        if not hit.any() and wi == 0:
            nsg += 1
    nbp = len(rows)
    BPs = np.array(rows, dtype=np.int64).reshape(-1, 6)
    return nbp + nsg, nbp, nsg, BPs


def number_of_blocking_pairs_loop(men_rank_list, women_rank_list, M):
    """Line-by-line port of number_of_blocking_pairs.m (slow, used by tests)."""
    nbp = 0
    nsg = 0
    X = []
    for i in range(M.shape[1]):
        mi = int(M[0, i])
        if mi == 0:
            continue
        wi = int(M[1, i])
        check_bp = False
        for j in range(M.shape[1]):
            mj = int(M[0, j])
            wj = int(M[1, j])
            if wj == 0:
                continue
            if check_blocking_pair(men_rank_list, women_rank_list, mi, wi, mj, wj):
                nbp += 1
                check_bp = True
                X.append([mi, wi, mj, wj,
                          int(women_rank_list[wj - 1, mi - 1]),
                          int(men_rank_list[mi - 1, wj - 1])])
        if not check_bp and wi == 0:
            nsg += 1
    return nbp + nsg, nbp, nsg, np.array(X, dtype=np.int64).reshape(-1, 6)


# ---------------------------------------------------------------------------
# make_random_matching.m
# ---------------------------------------------------------------------------
def make_random_matching(men_rank_list, women_rank_list, n, rng=None):
    rng = rng or RNG
    a = np.argsort(rng.rand_matrix(1, n)[0], kind="stable") + 1
    b = np.argsort(rng.rand_matrix(1, n)[0], kind="stable") + 1
    M = np.zeros((2, 2 * n), dtype=np.int64)
    M[0, :n] = a
    M[1, n:] = b
    k = rng.randi(n)
    # only k men are matched with k women who accept each other
    for i in range(k):
        mi = M[0, i]
        for j in range(n, 2 * n):
            wj = M[1, j]
            if wj > 0 and men_rank_list[mi - 1, wj - 1] > 0 and women_rank_list[wj - 1, mi - 1] > 0:
                M[1, j] = 0
                M[1, i] = wj
                break
    both = (M[0] > 0) & (M[1] > 0)
    if np.any((men_rank_list[M[0, both] - 1, M[1, both] - 1] == 0)
              | (women_rank_list[M[1, both] - 1, M[0, both] - 1] == 0)):
        print("\nThere exist unaceptable pairs")
        return np.zeros((2, 0), dtype=np.int64)
    return M


# ---------------------------------------------------------------------------
# verify_result_matching.m
# ---------------------------------------------------------------------------
def verify_result_matching(men_rank_list, women_rank_list, M, verbose=True):
    """1 if M is a valid stable matching, else 0."""
    def say(msg):
        if verbose:
            print(msg)

    f = 1
    if blocking_matrix(men_rank_list, women_rank_list, M).any():
        f = 0
    n = men_rank_list.shape[0]
    if set(range(n + 1)) - set(M[0].tolist()):
        f = 0
        say("There exist men are not matched !!")
    if set(range(n + 1)) - set(M[1].tolist()):
        f = 0
        say("There exist women are not matched !!")
    for row, who in ((0, "men"), (1, "women")):
        y = M[row][M[row] > 0]
        if np.unique(y).size != y.size:
            f = 0
            say(f"There exists a duplicate {who}!")
    both = (M[0] > 0) & (M[1] > 0)
    if np.any((men_rank_list[M[0, both] - 1, M[1, both] - 1] == 0)
              | (women_rank_list[M[1, both] - 1, M[0, both] - 1] == 0)):
        f = 0
        say("There exist unaceptable pairs")
    return f


def matching_cost(M):
    """Number of single men (matching_cost in MCS.m)."""
    return int(np.count_nonzero((M[0] != 0) & (M[1] == 0)))
