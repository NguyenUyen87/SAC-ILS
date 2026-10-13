"""CP-SAT on the integer programming model of MAX-SMTI (Delorme et al. 2019, model M1
without preprocessing): x_mw for every acceptable pair, at most one partner per agent,
and the weak-stability constraint of every pair. Default presolve, linearization level
2, a solution hint (a weakly stable matching), and the best size over time recorded
through a solution callback (the hint included)."""
import time

import numpy as np


def cpsat(men, women, hint, budget, cps, workers=8, max_mem_mb=None):
    """Returns (info, sizes at the checkpoints cps (-1: no solution yet), assign or None).
    Model building counts against the budget."""
    from ortools.sat.python import cp_model
    t0 = time.perf_counter()
    model = cp_model.CpModel()
    x, mlist, wlist = {}, [[] for _ in men], [[] for _ in women]
    for m, prefs in enumerate(men):
        for w, r in prefs.items():
            v = model.NewBoolVar("")
            x[(m, w)] = v
            mlist[m].append((r, v))
            wlist[w].append((women[w][m], m, v))
    for lst in mlist:
        if lst:
            model.Add(sum(v for _, v in lst) <= 1)
    for lst in wlist:
        if lst:
            model.Add(sum(v for _, _, v in lst) <= 1)
    for (m, w), v in x.items():
        rm, rw = men[m][w], women[w][m]
        model.Add(sum([vv for r, vv in mlist[m] if r <= rm] + [vv for r, mm, vv in wlist[w] if r <= rw and mm != m]) >= 1)
    for (m, w), v in x.items():
        model.AddHint(v, int(hint[m] == w))
    model.Maximize(sum(x.values()))
    build = time.perf_counter() - t0

    class Track(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.events = []

        def on_solution_callback(self):
            self.events.append((time.perf_counter() - t0, int(round(self.ObjectiveValue()))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, budget - build)
    solver.parameters.num_workers = workers
    solver.parameters.linearization_level = 2
    if max_mem_mb:
        solver.parameters.max_memory_in_mb = max_mem_mb
    tr = Track()
    status = solver.Solve(model, tr)
    a = None
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        a = np.full(len(men), -1, dtype=np.int64)
        for (m, w), v in x.items():
            if solver.Value(v):
                a[m] = w
    first = tr.events[0][0] if tr.events else -1
    at = [max([s for t, s in tr.events if t <= c], default=-1) for c in cps]
    return {"status": solver.StatusName(status), "bound": int(round(solver.BestObjectiveBound())),
            "build_s": round(build, 2), "first_s": round(first, 2),
            "proved_s": round(time.perf_counter() - t0, 2) if status == cp_model.OPTIMAL else ""}, at, a
