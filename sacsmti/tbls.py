"""TBLS (Qiu 2025, arXiv:2409.10575) for MAX-SMTI, and the TBLS-ILS control, in a sparse implementation.

The search keeps a strict tie-breaking strategy S and a matching that is stable for S
(hence weakly stable). Moves: an adjustment (a single agent is promoted in the tie of an
indifferent partner) or, with probability pd or when no adjustment exists, a disruption
(the ties of ku men and kw women are broken again at random); the matching is then
repaired by removing blocking pairs under S. TBLS has no published code; this is a
re-implementation of its pseudocode. Random numbers come from MatlabRNG. Every per-agent structure is stored only for the agent's own list, so
memory grows with the number of acceptable pairs instead of n^2.

SparseSearch is also the source of the Gale-Shapley start and of the perturbation of
SAC-ILS (sacsmti.sac).

Input: men[m] = {w: rank}, women[w] = {m: rank} (0-based ids, ranks >= 1).
"""
import time
import numpy as np

from .rng import MatlabRNG

MEN, WOMEN = 0, 1


class SparseStrategy:
    """order[v]: strict list of v (ties contiguous); pos[v]: {x: index in order[v]}."""

    def __init__(self, ranks, cands, rng):
        self.ranks = ranks          # ranks[v] = {x: original rank}
        self.order = []
        self.pos = []
        for v in range(len(ranks)):
            cand = cands[v]         # acceptable agents of v, ascending id
            r = np.fromiter((ranks[v][x] for x in cand), dtype=np.int64, count=len(cand))
            self.order.append(cand[np.argsort(r, kind="stable")])
            self.pos.append({})
            self.break_ties(v, rng)

    def copy(self):
        s = SparseStrategy.__new__(SparseStrategy)
        s.ranks = self.ranks
        s.order = [o.copy() for o in self.order]
        s.pos = [dict(p) for p in self.pos]
        return s

    def _reindex(self, v):
        self.pos[v] = {int(x): i for i, x in enumerate(self.order[v])}

    def break_ties(self, v, rng):
        o = self.order[v]
        rk = self.ranks[v]
        r = [rk[int(x)] for x in o]
        start = 0
        while start < len(o):
            end = start
            while end + 1 < len(o) and r[end + 1] == r[start]:
                end += 1
            if end > start:
                o[start:end + 1] = o[start:end + 1][np.argsort(rng.rand_matrix(1, end - start + 1)[0])]
            start = end + 1
        self._reindex(v)

    def promote(self, x, f):
        o = self.order[x]
        p = self.pos[x]
        i = p[f]
        rk = self.ranks[x]
        target = rk[f]
        start = 0
        while rk[int(o[start])] != target:
            start += 1
        if i > start:
            o[start + 1:i + 1] = o[start:i].copy()
            o[start] = f
            for j in range(start, i + 1):
                p[int(o[j])] = j


class SparseSearch:
    def __init__(self, men, women, pd, ku, kw, c, T, rng):
        self.n = len(men)
        self.ranks = (men, women)
        self.cands = ([np.array(sorted(d), dtype=np.int64) for d in men],
                      [np.array(sorted(d), dtype=np.int64) for d in women])
        self.pd, self.ku, self.kw, self.rng = pd, ku, kw, rng
        self.lens = [np.array([len(d) for d in men]), np.array([len(d) for d in women])]
        self.S = [SparseStrategy(men, self.cands[MEN], rng), SparseStrategy(women, self.cands[WOMEN], rng)]
        t0 = time.perf_counter()
        self.pm, self.pw = self.gale_shapley()
        self.T = time.perf_counter() - t0 if T is None else T
        size0 = int((self.pm >= 0).sum())
        self.bigM = (self.lens[MEN].max() + self.lens[WOMEN].max()) * (self.n - c * size0)
        self.disruptions = 0
        self.fallbacks = 0

    def gale_shapley(self):
        n = self.n
        pm = np.full(n, -1, dtype=np.int64)
        pw = np.full(n, -1, dtype=np.int64)
        nxt = [0] * n
        free = list(range(n - 1, -1, -1))
        order_m, pos_w = self.S[MEN].order, self.S[WOMEN].pos
        while free:
            m = free.pop()
            if nxt[m] >= len(order_m[m]):
                continue
            w = int(order_m[m][nxt[m]])
            nxt[m] += 1
            pwm = pos_w[w].get(m, -1)
            if pwm < 0:
                free.append(m)
                continue
            cur = int(pw[w])
            if cur < 0:
                pm[m], pw[w] = w, m
            elif pwm < pos_w[w][cur]:
                pm[m], pw[w] = w, m
                pm[cur] = -1
                free.append(cur)
            else:
                free.append(m)
        return pm, pw

    def partner(self, side):
        return self.pm if side == MEN else self.pw

    def evaluate(self, pm, pw):
        size = int((pm >= 0).sum())
        free = self.lens[MEN][pm < 0].sum() + self.lens[WOMEN][pw < 0].sum()
        return size * self.bigM + free

    def adjustments(self):
        out = []
        for side in (MEN, WOMEN):
            other = 1 - side
            Ro = self.ranks[other]
            p_other = self.partner(other)
            for f in np.flatnonzero(self.partner(side) < 0):
                f = int(f)
                xs = [int(x) for x in self.cands[side][f]
                      if p_other[x] >= 0 and Ro[x][f] == Ro[x][int(p_other[x])]]
                if xs:
                    out.append((side, f, xs[self.rng.randi(len(xs)) - 1]))
        return out

    def disrupt(self):
        self.disruptions += 1
        Qa = []
        for side, k in ((MEN, self.ku), (WOMEN, self.kw)):
            for v in np.argsort(self.rng.rand_matrix(1, self.n)[0])[:k]:
                self.S[side].break_ties(int(v), self.rng)
                Qa.append((side, int(v)))
        return Qa

    def refine(self):
        adj = self.adjustments()
        if not adj or self.rng.rand() < self.pd:
            return self.disrupt()
        side, f, x = adj[self.rng.randi(len(adj)) - 1]
        self.S[1 - side].promote(x, f)
        return [(1 - side, x)]

    def obtain_stable(self, Qa):
        X = list(dict.fromkeys(Qa))
        t0 = time.perf_counter()
        while X and time.perf_counter() - t0 <= self.T:
            side, v = X.pop(self.rng.randi(len(X)) - 1)
            other = 1 - side
            mine, theirs = self.partner(side), self.partner(other)
            Sv, So = self.S[side], self.S[other]
            pv = Sv.pos[v]
            for y in Sv.order[v]:
                y = int(y)
                mv = int(mine[v])
                if mv == y:
                    continue
                if mv >= 0 and pv[y] > pv[mv]:
                    break
                py = So.pos[y]
                yv = py.get(v, -1)
                ty = int(theirs[y])
                if yv >= 0 and (ty < 0 or yv < py[ty]):
                    for agent in ((other, mv), (side, ty)):
                        if agent[1] >= 0 and agent not in X:
                            X.append(agent)
                    m, w = (v, y) if side == MEN else (y, v)
                    if self.pm[m] >= 0:
                        self.pw[self.pm[m]] = -1
                    if self.pw[w] >= 0:
                        self.pm[self.pw[w]] = -1
                    self.pm[m], self.pw[w] = w, m
        if X:
            self.fallbacks += 1
            self.pm, self.pw = self.gale_shapley()

    def op_adj(self):
        """One adjustment step of TBLS (used by the control ILS-ADJ); None if none exists."""
        adj = self.adjustments()
        if not adj:
            return None
        side, f, x = adj[self.rng.randi(len(adj)) - 1]
        self.S[1 - side].promote(x, f)
        return [(1 - side, x)]



def run(men, women, variant="TBLS", max_iters=3000, ku=5, kw=5, pd=0.05, c=0.9, T=None, restart_after=50,
        seed=None, rng=None, stop_when_perfect=True, time_limit=None, checkpoints=()):
    """variant: "TBLS" (Qiu 2025) or "TBLS+ILS" (TBLS that returns to its best matching
    and strategy after `restart_after` iterations below the best size).

    Returns a dict with assign (man -> woman or -1), singles, time, iters and, with
    checkpoints, "anytime" (best singles at each checkpoint).
    """
    rng = rng or MatlabRNG(seed)
    if variant not in ("TBLS", "TBLS+ILS"):
        raise ValueError(variant)
    use_ils = variant == "TBLS+ILS"
    t0 = time.perf_counter()
    s = SparseSearch(men, women, pd, ku, kw, c, T, rng)
    best = (s.pm.copy(), s.pw.copy())
    e_best = s.evaluate(*best)
    best_size = int((best[0] >= 0).sum())
    best_S = [x.copy() for x in s.S] if use_ils else None
    drift = restores = it = 0
    cps = sorted(checkpoints)
    anytime = []
    while it <= max_iters:
        el = time.perf_counter() - t0
        while cps and el >= cps[0]:
            anytime.append(int((best[0] < 0).sum()))
            cps.pop(0)
        if time_limit is not None and el >= time_limit:
            break
        if stop_when_perfect and (best[0] >= 0).all():
            break
        Qa = s.refine()
        s.obtain_stable(Qa)
        e = s.evaluate(s.pm, s.pw)
        if e >= e_best:
            best, e_best = (s.pm.copy(), s.pw.copy()), e
            if use_ils:
                best_S = [x.copy() for x in s.S]
        size = int((s.pm >= 0).sum())
        best_size = max(best_size, size)
        if use_ils:
            drift = drift + 1 if size < best_size else 0
            if drift >= restart_after:
                s.S = [x.copy() for x in best_S]
                s.pm, s.pw = best[0].copy(), best[1].copy()
                drift = 0
                restores += 1
        it += 1
    anytime += [int((best[0] < 0).sum())] * len(cps)
    return {"assign": best[0].copy(), "singles": int((best[0] < 0).sum()), "anytime": anytime,
            "time": time.perf_counter() - t0, "iters": it, "restores": restores, "fallbacks": s.fallbacks}
