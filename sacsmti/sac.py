"""Stable augmenting paths and chains for MAX-SMTI.

Let M be weakly stable. An M-augmenting path P = m0, w1, m1, ..., m_{k-1}, wk
(m0 and wk free) is *stable* if M (+) P is weakly stable. Only the agents of P
change partners, so a blocking pair of M (+) P contains an agent of P:

  for a man m of P with new partner w: every woman w' that m strictly prefers to w
  must strictly prefer her own partner in M (+) P to m (or be unacceptable);
  for a woman w of P with new partner m: every man m' that w strictly prefers to m
  must strictly prefer his own partner in M (+) P to w.

P is a *stable augmenting chain* if, in addition, every intermediate matching
M_i (the first i edges of P applied, the man m_i they displace left single) has
no blocking pair other than those containing m_i: the switch can be carried out
step by step, an ejection chain in which only the ejected man is ever unsettled.
Every stable chain is a stable path (M_k = M (+) P).

Mode "chain" searches stable chains: when an agent receives its new partner it
is checked against every other agent with its current partner, and any
violation not involving the displaced man cuts the branch. This search is sound
and, with unbounded length and budget, complete for chains.

Mode "path" searches stable paths. P is grown from a free man depth first. When an agent receives its new
partner, its conditions are checked against every other agent. A violation with
an agent already on P (whose new partner is known) cuts the branch. A violation
with an agent y not yet on P is not final, because y may still join P (as the
woman taken next, or as the man she leaves) and change partner; it becomes a
*pending requirement*: y must join P and receive a partner it ranks at least as
high as a given bound. A free man outside P can never join P, so a violation
with him is final. The branch is cut when a requirement fails as its agent
joins P, or when more requirements are pending than the rest of the path can
still meet (each further step adds one woman and one man); a path is accepted
only when no requirement is pending. Every blocking pair is checked when the
later of its two agents receives its new partner, so the search is sound, and
it only cuts branches that cannot be completed, so with unbounded length and
budget it is complete. Women are tried free ones first, then in the man's order
of preference. Depth and the number of expanded nodes are bounded.

In the paper: SAC = sac_search (mode "chain", random candidate order) and SAC-ILS =
sac_ils (kick 5, at most 16 men per chain, 5000 nodes per search, random order).

Data: lists of {partner: rank} dicts (sacsmti.io); pm/pw partner arrays.
"""
import numpy as np

MEN, WOMEN = 0, 1


def _partners_first(S, pm, pw):
    """Put every agent's partner first in its tie: M is then stable for S."""
    for m in np.flatnonzero(pm >= 0):
        S[MEN].promote(int(m), int(pm[m]))
    for w in np.flatnonzero(pw >= 0):
        S[WOMEN].promote(int(w), int(pw[w]))


class _Search:
    def __init__(self, men, women, pm, pw, max_len, node_budget, mode="chain"):
        self.men, self.women, self.pm, self.pw = men, women, pm, pw
        self.chain = mode == "chain"
        self.max_len, self.budget = max_len, node_budget
        self.nm, self.nw = {}, {}          # new partners of the agents of P
        self.open = None                   # displaced man without a new partner yet
        self.req_m, self.req_w = {}, {}    # pending requirements: agent -> worst acceptable rank
        self.nodes = 0
        self.order = "pref"                # "pref": single women first, then by preference; "random"

    def _require(self, side, v, bound, log):
        req = self.req_m if side == "m" else self.req_w
        old = req.get(v)
        if old is None or bound < old:
            log.append((side, v, old))
            req[v] = bound

    def _man_ok(self, m, w, log):
        men, women = self.men, self.women
        rw = men[m][w]
        b = self.req_m.get(m)
        if b is not None and rw > b:
            return False
        for x, r in men[m].items():
            if r >= rw:
                continue
            if x in self.nw:
                q = self.nw[x]
                if women[x][m] < women[x][q]:
                    return False
            else:
                q = int(self.pw[x])
                if q < 0 or women[x][m] < women[x][q]:
                    if self.chain:
                        return False
                    self._require("w", x, women[x][m], log)   # x must join P with a partner >= m
        return True

    def _woman_ok(self, w, m, log):
        men, women = self.men, self.women
        rm = women[w][m]
        b = self.req_w.get(w)
        if b is not None and rm > b:
            return False
        for y, r in women[w].items():
            if r >= rm or y == self.open:
                continue
            if y in self.nm:
                if men[y][w] < men[y][self.nm[y]]:
                    return False
            else:
                q = int(self.pm[y])
                if q < 0:
                    return False                              # a free man never joins P
                if men[y][w] < men[y][q]:
                    if self.chain:
                        return False
                    self._require("m", y, men[y][w], log)    # y must join P with a partner >= w
        return True

    def _undo(self, log):
        for side, v, old in reversed(log):
            req = self.req_m if side == "m" else self.req_w
            if old is None:
                del req[v]
            else:
                req[v] = old

    def _pending(self):
        return sum(1 for v in self.req_m if v not in self.nm) + sum(1 for v in self.req_w if v not in self.nw)

    def grow(self, m, depth, rng):
        """Extend P from the open man m; returns the list of (man, new woman) or None."""
        self.nodes += 1
        if self.nodes > self.budget:
            return None
        cands = [w for w in self.men[m] if w not in self.nw and w != self.pm[m]]
        if self.order == "random":
            order = [cands[i] for i in rng.permutation(len(cands))]
        else:
            free = [w for w in cands if self.pw[w] < 0]
            order = free + sorted((w for w in cands if self.pw[w] >= 0), key=lambda w: (self.men[m][w], rng.random()))
        for w in order:
            q = int(self.pw[w])
            if q >= 0 and depth + 1 >= self.max_len:
                continue
            log = []
            self.nm[m], self.nw[w] = w, m
            prev_open, self.open = self.open, q if q >= 0 else None
            if self._man_ok(m, w, log) and self._woman_ok(w, m, log):
                pend = self._pending()
                if q < 0:
                    if pend == 0:
                        return [(m, w)]
                elif pend <= 2 * (self.max_len - depth - 1):
                    tail = self.grow(q, depth + 1, rng)
                    if tail is not None:
                        return [(m, w)] + tail
            self._undo(log)
            del self.nm[m], self.nw[w]
            self.open = prev_open
            if self.nodes > self.budget:
                return None
        return None


def find_stable_augmenting_path(men, women, pm, pw, m0, rng, max_len=12, node_budget=2000, mode="chain", order="pref"):
    """From free man m0, a stable augmenting chain (mode "chain") or a stable augmenting
    path (mode "path") as [(m0, w1), (m1, w2), ...], or None."""
    s = _Search(men, women, pm, pw, max_len, node_budget, mode)
    s.order = order
    return s.grow(m0, 0, rng), s.nodes


def apply_path(pm, pw, path):
    for m, w in path:
        old = pw[w]
        if old >= 0 and pm[old] == w:
            pm[old] = -1
    for m, w in path:
        pm[m], pw[w] = w, m
    return pm, pw


def sac_search(men, women, pm, time_limit, max_len=24, node_budget=50000, seed=0, checkpoints=(), mode="chain",
               order="pref"):
    """Repeatedly apply stable augmenting paths from free men (random order) to the
    weakly stable matching pm until a full sweep finds none or time runs out."""
    import time
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    pm = np.asarray(pm, dtype=np.int64).copy()
    pw = np.full(len(women), -1, dtype=np.int64)
    pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
    cps, anytime, paths = sorted(checkpoints), [], 0

    def tick():
        el = time.perf_counter() - t0
        while cps and el >= cps[0]:
            anytime.append(int((pm < 0).sum()))
            cps.pop(0)
        return el
    done = False
    while not done and tick() < time_limit:
        done = True
        for m0 in rng.permutation(np.flatnonzero(pm < 0)):
            if not men[m0]:
                continue
            path, _ = find_stable_augmenting_path(men, women, pm, pw, int(m0), rng, max_len, node_budget, mode, order)
            if path:
                apply_path(pm, pw, path)
                paths += 1
                done = False
            if tick() >= time_limit:
                break
    tick()
    anytime += [int((pm < 0).sum())] * len(cps)
    return {"assign": pm, "singles": int((pm < 0).sum()), "anytime": anytime, "paths": paths,
            "time": time.perf_counter() - t0}


def sac_ils(men, women, time_limit, kick=5, max_len=24, node_budget=50000, seed=0, checkpoints=(), sweep_budget=None, sides="m", ls="chain",
            lns_every=0, lns_vars=1000, lns_sec_per_kvar=0.25,
            mode="chain", order="pref", init="gs", deepen=False):
    """Iterated local search over weakly stable matchings.

    Local search: stable augmenting chains from free men until none is found.
    Perturbation: the ties of `kick` random men and `kick` random women are broken
    again at random and the matching is repaired (the disruption of TBLS, which keeps
    the matching stable for the tie-breaking strategy and hence weakly stable).
    Acceptance: the perturbed and re-optimised matching replaces the current one if it
    is not smaller; otherwise the search returns to the best matching. The strategy is
    kept consistent with the matching by putting every partner first in its tie."""
    import time
    from .rng import MatlabRNG
    from .tbls import SparseSearch
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    s = SparseSearch(men, women, 0.05, kick, kick, 0.9, None, MatlabRNG(seed))
    if init == "cky":                      # start from the 1.5-approximation of Csaji, Kiraly and Yokoi
        from .baselines.cky import cky
        from .instance import Instance
        n = len(men)
        R = np.zeros((n, len(women)), dtype=np.int64)
        Wr = np.zeros((len(women), n), dtype=np.int64)
        for m, d in enumerate(men):
            for w, r in d.items():
                R[m, w] = r
        for w, d in enumerate(women):
            for m, r in d.items():
                Wr[w, m] = r
        a = np.asarray(cky(Instance(R, Wr, np.ones(len(women), dtype=np.int64)), np.random.default_rng(seed)), dtype=np.int64)
        s.pm = a
        s.pw = np.full(len(women), -1, dtype=np.int64)
        s.pw[a[a >= 0]] = np.flatnonzero(a >= 0)
        _partners_first(s.S, s.pm, s.pw)
    cps, anytime = sorted(checkpoints), []
    st = {"paths": 0, "kicks": 0, "accepted": 0, "searches": 0, "cut": 0, "edges": 0, "max_edges": 0,
          "lns_calls": 0, "lns_gain": 0, "lns_time": 0.0}
    lrng = np.random.default_rng(seed + 7_919)   # only used by the LNS step, so the default stream is unchanged

    def lns_step():
        """SAC-ILS-LNS: CP-SAT re-optimises alternating balls around random single men, about
        `lns_vars` variables, with the rest of the matching fixed and the stability
        constraints of the whole matching; the result replaces M if it is larger."""
        from .lns import path_balls, solve_sub
        t1 = time.perf_counter()
        Nm, Nw = path_balls(men, women, s.pm, s.pw, lrng, lns_vars, rho=1)
        if Nm:
            nv = sum(1 for m in Nm for w in men[m] if w in Nw)
            lim = min(max(0.05, lns_sec_per_kvar * nv / 1000), max(0.0, time_limit - tick()))
            if lim > 0:
                pm2, pw2, status, _ = solve_sub(men, women, s.pm, s.pw, Nm, Nw, time_limit=lim)
                gain = int((pm2 >= 0).sum() - (s.pm >= 0).sum())
                if gain > 0:
                    s.pm, s.pw = pm2, pw2
                    _partners_first(s.S, s.pm, s.pw)
                    st["lns_gain"] += gain
        st["lns_calls"] += 1
        st["lns_time"] += time.perf_counter() - t1
    best = None

    def tick():
        el = time.perf_counter() - t0
        while cps and el >= cps[0]:
            # before the first local search ends, the current matching (always weakly stable)
            anytime.append(int((best[0] < 0).sum()) if best is not None else int((s.pm < 0).sum()))
            cps.pop(0)
        return el

    def local_search_adj():
        """Control: the local search of TBLS inside the same ILS. Adjustment steps of TBLS
        (promote a single agent in the tie of an indifferent partner, then repair) until
        50 consecutive steps do not enlarge the matching or no adjustment exists; the
        best matching of the run is kept (with its strategy)."""
        nonlocal_best = (s.pm.copy(), s.pw.copy(), [x.copy() for x in s.S])
        idle = 0
        while idle < 50 and tick() < time_limit:
            Qa = s.op_adj()
            if Qa is None:
                break
            s.obtain_stable(Qa)
            st["searches"] += 1
            if (s.pm >= 0).sum() > (nonlocal_best[0] >= 0).sum():
                nonlocal_best = (s.pm.copy(), s.pw.copy(), [x.copy() for x in s.S])
                st["paths"] += 1
                idle = 0
            else:
                idle += 1
        s.pm, s.pw, s.S = nonlocal_best[0], nonlocal_best[1], nonlocal_best[2]

    def local_search():
        if ls == "adj":
            return local_search_adj()
        progress = True
        while progress and tick() < time_limit:
            progress = False
            if sides == "both":            # chains from single women too: the same search with the sides swapped
                starts = [("m", int(m)) for m in np.flatnonzero(s.pm < 0)] + \
                         [("w", int(w)) for w in np.flatnonzero(s.pw < 0)]
                seq = [starts[i] for i in rng.permutation(len(starts))]
            else:
                seq = [("m", int(m)) for m in rng.permutation(np.flatnonzero(s.pm < 0))]
            for side, m0 in seq:
                if side == "m" and (not men[m0] or s.pm[m0] >= 0) or side == "w" and (not women[m0] or s.pw[m0] >= 0):
                    continue               # empty list, or matched by an earlier chain of this pass
                if side == "w":
                    path, nodes = find_stable_augmenting_path(women, men, s.pw, s.pm, m0, rng, max_len,
                                                              node_budget, mode, order)
                    st["searches"] += 1
                    st["cut"] += int(path is None and nodes > node_budget)
                    if path:
                        apply_path(s.pw, s.pm, path)
                        st["paths"] += 1
                        st["edges"] += 2 * len(path) - 1
                        st["max_edges"] = max(st["max_edges"], 2 * len(path) - 1)
                        progress = True
                    if tick() >= time_limit:
                        break
                    continue
                if deepen:                 # shortest chains first: iterative deepening on the length
                    path, nodes = None, 0
                    for L in sorted({2, 4, 8, max_len}):
                        if L > max_len:
                            continue
                        path, k = find_stable_augmenting_path(men, women, s.pm, s.pw, int(m0), rng, L, node_budget,
                                                              mode, order)
                        nodes = max(nodes, k)
                        if path:
                            break
                else:
                    path, nodes = find_stable_augmenting_path(men, women, s.pm, s.pw, int(m0), rng, max_len,
                                                              node_budget, mode, order)
                st["searches"] += 1
                st["cut"] += int(path is None and nodes > node_budget)
                if path:
                    apply_path(s.pm, s.pw, path)
                    st["paths"] += 1
                    st["edges"] += 2 * len(path) - 1
                    st["max_edges"] = max(st["max_edges"], 2 * len(path) - 1)
                    progress = True
                if tick() >= time_limit:
                    break
        _partners_first(s.S, s.pm, s.pw)

    local_search()
    best = (s.pm.copy(), s.pw.copy(), [x.copy() for x in s.S])
    while tick() < time_limit and (best[0] < 0).any():
        s.obtain_stable(s.disrupt())
        st["kicks"] += 1
        local_search()
        if lns_every and st["kicks"] % lns_every == 0 and tick() < time_limit:
            lns_step()
            local_search()
        if (s.pm >= 0).sum() >= (best[0] >= 0).sum():
            best = (s.pm.copy(), s.pw.copy(), [x.copy() for x in s.S])
            st["accepted"] += 1
        else:
            s.pm, s.pw, s.S = best[0].copy(), best[1].copy(), [x.copy() for x in best[2]]
    tick()
    anytime += [int((best[0] < 0).sum())] * len(cps)
    return {"assign": best[0].copy(), "singles": int((best[0] < 0).sum()), "anytime": anytime,
            "time": time.perf_counter() - t0, **st}
