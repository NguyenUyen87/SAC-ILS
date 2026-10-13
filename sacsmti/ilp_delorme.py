"""Integer programming for MAX-SMTI following Delorme et al. (2019), Sections 3-6.

Preprocessing (their Section 4, Theorem 1, Algorithms 1 and 2, applied to both sides and
repeated until nothing changes) removes acceptable pairs that cannot belong to any stable
matching. Models (their Table 3 notation; "children" = men, "families" = women):
  M1  x variables, stability constraint (4) per acceptable pair
  M2  M1 with dummy variables y^c_{ik}, y^f_{jk} (constraints (11)-(19))
  M3  M1 with stability constraints merged per child and rank (24)
  M4  M2 with merged stability constraints (24) on the dummy variables
All models are solved with CP-SAT (OR-Tools), with an optional solution hint.

Data: men[m] = {w: rank}, women[w] = {m: rank}; ranks are integers, equal ranks are ties,
smaller is better (sacsmti.io format).
"""
import time
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------- preprocessing
def _remove(prefs_a, prefs_b, a, b):
    if b in prefs_a[a]:
        del prefs_a[a][b]
        del prefs_b[b][a]
        return 1
    return 0


def _pass_full(ch, fa):
    """Algorithm 2 (full-child-preferences) with ch = children side, fa = families side."""
    removed = 0
    for i in range(len(ch)):
        if not ch[i]:
            continue
        F, C = [], set()
        for j in sorted(ch[i], key=lambda j: (ch[i][j], j)):
            F.append(j)
            ri = fa[j][i]
            C.update(p for p, r in fa[j].items() if r <= ri)
            if len(F) >= len(C):
                worst = max(ch[i][q] for q in F)
                for jj in [q for q, r in ch[i].items() if r > worst]:
                    removed += _remove(ch, fa, i, jj)
                break
    return removed


def _pass_first(ch, fa):
    """Algorithm 1 (first-rank-family): families grouped by the set of children they rank first."""
    groups = defaultdict(list)
    for j in range(len(fa)):
        if fa[j]:
            best = min(fa[j].values())
            groups[frozenset(i for i, r in fa[j].items() if r == best)].append(j)
    removed = 0
    for C, F in groups.items():
        if len(F) >= len(C):
            for i in C:
                if not all(j in ch[i] for j in F):
                    continue
                worst = max(ch[i][j] for j in F)
                for jj in [q for q, r in ch[i].items() if r > worst]:
                    removed += _remove(ch, fa, i, jj)
    return removed


def preprocess(men, women):
    """Copies of the lists without the pairs that cannot be in any stable matching, and the
    number of removed pairs. Both algorithms are applied to both sides until no change."""
    men = [dict(d) for d in men]
    women = [dict(d) for d in women]
    total = 0
    while True:
        r = _pass_first(men, women) + _pass_first(women, men) + _pass_full(men, women) + _pass_full(women, men)
        total += r
        if r == 0:
            return men, women, total


# ---------------------------------------------------------------- models
def build(men, women, model="M1", hint=None):
    """CP-SAT model of the given type; returns (model, x dict)."""
    from ortools.sat.python import cp_model
    md = cp_model.CpModel()
    x = {(m, w): md.NewBoolVar("") for m, d in enumerate(men) for w in d}
    by_m = defaultdict(list)
    by_w = defaultdict(list)
    for (m, w), v in x.items():
        by_m[m].append(v)
        by_w[w].append(v)
    for vs in by_m.values():
        md.Add(sum(vs) <= 1)
    for vs in by_w.values():
        md.Add(sum(vs) <= 1)

    def levels(d):
        return sorted(set(d.values()))

    # sums over "at least as good as rank r" for each agent, as expressions or dummy variables
    dummy = model in ("M2", "M4")
    yc, yf = {}, {}                                   # (agent, rank value) -> bool var or expression
    for side, prefs, store, key in (("c", men, yc, lambda a, b: (a, b)), ("f", women, yf, lambda a, b: (b, a))):
        for a, d in enumerate(prefs):
            acc = []
            prev = None
            for r in levels(d):
                acc += [x[key(a, b)] for b, rb in d.items() if rb == r]
                if dummy:
                    y = md.NewBoolVar("")
                    tier = [x[key(a, b)] for b, rb in d.items() if rb == r]
                    md.Add(y == sum(tier) + (prev if prev is not None else 0))
                    prev = y
                    store[(a, r)] = y
                else:
                    store[(a, r)] = list(acc)
    def val(store, a, r):
        s = store[(a, r)]
        return s if dummy else sum(s)

    if model in ("M1", "M2"):
        for (m, w) in x:
            md.Add(val(yc, m, men[m][w]) + val(yf, w, women[w][m]) >= 1)
    elif model in ("M3", "M4"):
        for m, d in enumerate(men):
            for r in levels(d):
                Fk = [w for w, rw in d.items() if rw == r]
                md.Add(len(Fk) * (1 - val(yc, m, r)) <= sum(val(yf, w, women[w][m]) for w in Fk))
    else:
        raise ValueError(model)
    md.Maximize(sum(x.values()))
    if hint is not None:
        for (m, w), v in x.items():
            md.AddHint(v, int(hint[m] == w))
    return md, x


def solve(men, women, model="M3", pre=True, hint=None, time_limit=60.0, workers=8, checkpoints=(), t0=None):
    """Preprocess (optionally), build and solve. Times count from t0 (default: now).
    Returns dict with assign, status, bound, removed pairs, build/preprocess times, and the best
    size at each checkpoint (-1 if no solution yet)."""
    from ortools.sat.python import cp_model
    t0 = time.perf_counter() if t0 is None else t0
    removed = 0
    if pre:
        men2, women2, removed = preprocess(men, women)
    else:
        men2, women2 = men, women
    t_pre = time.perf_counter() - t0
    if hint is not None:                              # a hint pair removed by preprocessing is dropped
        hint = np.array([w if w >= 0 and w in men2[m] else -1 for m, w in enumerate(hint)])
    md, x = build(men2, women2, model, hint)
    t_build = time.perf_counter() - t0

    class Track(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.events = []

        def on_solution_callback(self):
            self.events.append((time.perf_counter() - t0, int(round(self.ObjectiveValue()))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, time_limit - t_build)
    solver.parameters.num_workers = workers
    solver.parameters.linearization_level = 2
    tr = Track()
    status = solver.Solve(md, tr)
    a = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        a = np.full(len(men), -1, dtype=np.int64)
        for (m, w), v in x.items():
            if solver.Value(v):
                a[m] = w
    at = [max([s for t, s in tr.events if t <= c], default=-1) for c in checkpoints]
    return {"assign": a, "status": solver.StatusName(status), "bound": int(round(solver.BestObjectiveBound())),
            "removed": removed, "pairs": sum(len(d) for d in men), "pre_s": round(t_pre, 2),
            "build_s": round(t_build, 2), "at": at,
            "proved_s": round(time.perf_counter() - t0, 2) if status == cp_model.OPTIMAL else None}
