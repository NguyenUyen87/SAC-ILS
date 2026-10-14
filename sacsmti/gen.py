"""Random MAX-SMTI instances.

c1         the generator of Gent and Prosser (2002) as used by TBLS (Qiu 2025): each pair
           acceptable with probability 1-p1, uniformly random orders, the next agent tied
           with the previous one with probability p2 (or geometric tie lengths, `geom`).
           Returns a dense Instance (rank matrices).
sparse_c1  sparse version of c1 for large n (lists of mean length `length`) with the
           geometric ties of Qiu (2025); used for the generated configurations and the
           very large and development instances. Returns sparse lists.

The functions are written for HRT (capacities); SMTI is the case of n "hospitals"
(women) of capacity 1, the only case used here.
"""
import numpy as np

from .instance import Instance


def _ranks_from_scores(scores, acceptable, rng, p2=None, tau=None, geom=None):
    """Ranks (1 = best, 0 = unacceptable) for one row.

    p2:   Gent-Prosser ties along the order given by scores.
    geom: with p2, the tie generator of Qiu (2025) Section 5.1: a tie starts
          with probability p2 and has i extra members, i ~ Geom(p2) when
          geom="p2" or Geom(1-p2) when geom="1-p2".
    tau:  ties by rounding scores (in [0,1]) to tau levels.
    """
    rank = np.zeros(len(scores), dtype=np.int64)
    idx = np.flatnonzero(acceptable)
    if idx.size == 0:
        return rank
    if tau is not None:
        level = np.minimum((scores[idx] * tau).astype(int), tau - 1)
        distinct = np.unique(-level)  # best (highest level) first
        rank[idx] = np.searchsorted(distinct, -level) + 1
        return rank
    order = idx[np.argsort(-scores[idx], kind="stable")]
    if geom is not None:
        q = p2 if geom == "p2" else 1 - p2
        j, r = 0, 1
        while j < len(order):
            if rng.random() <= p2:
                i = len(order) if q <= 0 else int(rng.geometric(min(q, 1.0)))
                rank[order[j:j + i + 1]] = r
                j += i + 1
            else:
                rank[order[j]] = r
                j += 1
            r += 1
        return rank
    r = 1
    for j, x in enumerate(order):
        if j > 0 and rng.random() >= p2:
            r += 1
        rank[x] = r
    return rank


def _capacities(m, total, rng):
    cap = np.full(m, total // m, dtype=np.int64)
    cap[rng.choice(m, size=total % m, replace=False)] += 1
    return np.maximum(cap, 1)


def _build(acc, r_scores, h_scores, cap, rng, p2=None, tau=None, geom=None):
    n, m = acc.shape
    r_rank = np.array([_ranks_from_scores(r_scores[s], acc[s], rng, p2, tau, geom) for s in range(n)])
    h_rank = np.array([_ranks_from_scores(h_scores[h], acc[:, h], rng, p2, tau, geom) for h in range(m)])
    return Instance(r_rank, h_rank, cap)


def _acceptability(n, m, p1, rng, need_hospitals, max_tries=1000, per_row=False):
    if per_row:
        # redraw every empty resident list until it is non-empty, as in the
        # HRT instances of Qiu (2025) (mean list length (1-p1) m / (1 - p1^m))
        acc = rng.random((n, m)) >= p1
        for s in np.flatnonzero(~acc.any(axis=1)):
            while not acc[s].any():
                acc[s] = rng.random(m) >= p1
        return acc
    for _ in range(max_tries):
        acc = rng.random((n, m)) >= p1
        if acc.any(axis=1).all() and (not need_hospitals or acc.any(axis=0).all()):
            return acc
    raise ValueError(f"no instance without empty lists after {max_tries} tries "
                     f"(n={n}, m={m}, p1={p1}); lists have {m * (1 - p1):.2f} entries on average")


def c1(n, p1, p2, m=None, cap_total=None, rng=None, geom=None, per_row=False):
    """Gent-Prosser instance. m=None gives SMTI (m = n, capacities 1).
    geom="p2" / "1-p2" switches to the geometric tie lengths of Qiu (2025)."""
    rng = rng or np.random.default_rng()
    smti = m is None
    m = n if smti else m
    cap = np.ones(m, dtype=np.int64) if smti else _capacities(m, cap_total or n, rng)
    acc = _acceptability(n, m, p1, rng, need_hospitals=smti, per_row=per_row)
    return _build(acc, rng.random((n, m)), rng.random((m, n)), cap, rng, p2=p2, geom=geom)


def _tie_ranks(order, p2, geom, rng):
    """Ranks 1, 2, ... along `order` with the tie process of _ranks_from_scores (geom)."""
    q = p2 if geom == "p2" else 1 - p2
    out, j, r = {}, 0, 1
    while j < len(order):
        if rng.random() <= p2:
            i = len(order) if q <= 0 else int(rng.geometric(min(q, 1.0)))
            for x in order[j:j + i + 1]:
                out[int(x)] = r
            j += i + 1
        else:
            out[int(order[j])] = r
            j += 1
        r += 1
    return out


def sparse_c1(n, length, p2, geom="p2", rng=None):
    """Sparse SMTI instance for large n: pair (m, w) acceptable with probability
    length / n (as c1 with p1 = 1 - length/n), uniformly random orders and the
    geometric ties of Qiu (2025). Empty lists are allowed (such agents stay
    single in every matching). Returns (men, women) as lists of {partner: rank}
    dicts, the format of sacsmti.io."""
    rng = rng or np.random.default_rng()
    acc_m = []
    lists_w = [[] for _ in range(n)]
    for m in range(n):
        ws = rng.choice(n, size=rng.binomial(n, length / n), replace=False)
        acc_m.append(ws)
        for w in ws:
            lists_w[int(w)].append(m)
    men = [_tie_ranks(rng.permutation(ws), p2, geom, rng) for ws in acc_m]
    women = [_tie_ranks(rng.permutation(np.array(ms, dtype=np.int64)), p2, geom, rng) for ms in lists_w]
    return men, women
