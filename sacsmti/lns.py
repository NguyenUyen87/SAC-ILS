"""Exact re-optimisation of a neighbourhood of a weakly stable matching (CP-SAT).

path_balls          alternating breadth-first balls around single men, each grown until it
                    contains an M-augmenting path (plus rho layers), up to a budget of variables
has_augmenting_path does the neighbourhood contain an M-augmenting path?
solve_sub           maximum matching inside the neighbourhood, the rest of M fixed, subject to
                    weak stability of the whole matching

Used by SAC-ILS-LNS (sacsmti.sac) and by experiments/repair_paths.py.
Data: lists of {partner: rank} dicts (sacsmti.io); pm: man -> woman or -1.
"""
from collections import deque


def path_balls(men, women, pm, pw, rng, budget, rho=1, ball_cap=None):
    """Union of alternating balls around random free men, each grown until it reaches a
    free woman (then rho more layers), until about `budget` variables; closed under M."""
    free_m = [m for m in range(len(men)) if pm[m] < 0 and men[m]]
    rng.shuffle(free_m)
    Nm, Nw, count = set(), set(), 0
    for s in free_m:
        if count >= budget:
            break
        if s in Nm:
            continue
        Nm.add(s)
        count += len(men[s])
        start_count = count
        cap = budget if ball_cap is None else ball_cap
        layer, extra = [s], None
        while layer and count < budget and count - start_count < cap and (extra is None or extra > 0):
            nxt = []
            for m in layer:
                for w in men[m]:
                    if w in Nw:
                        continue
                    Nw.add(w)
                    p = int(pw[w])
                    if p < 0:
                        if extra is None:
                            extra = rho + 1          # augmenting path found
                    elif p not in Nm:
                        Nm.add(p)
                        count += len(men[p])
                        nxt.append(p)
            layer = nxt
            if extra is not None:
                extra -= 1
    for m in list(Nm):
        if pm[m] >= 0:
            Nw.add(int(pm[m]))
    for w in list(Nw):
        if pw[w] >= 0:
            Nm.add(int(pw[w]))
    return Nm, Nw


def has_augmenting_path(men, pm, pw, Nm, Nw):
    """Is there an M-augmenting path using only agents of N (necessary for a gain)?"""
    q = deque(m for m in Nm if pm[m] < 0)
    seen_m, seen_w = set(q), set()
    while q:
        m = q.popleft()
        for w in men[m]:
            if w not in Nw or w in seen_w or w == pm[m]:
                continue
            seen_w.add(w)
            p = int(pw[w])
            if p < 0:
                return True
            if p in Nm and p not in seen_m:
                seen_m.add(p)
                q.append(p)
    return False


def solve_sub(men, women, pm, pw, Nm, Nw, time_limit=2.0, workers=1):
    """Exact re-optimisation inside (Nm, Nw). Returns (pm, pw, status, n_vars)."""
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    x = {}
    mvars = {m: [] for m in Nm}
    wvars = {w: [] for w in Nw}
    for m in Nm:
        for w, r in men[m].items():
            if w in Nw:
                v = model.NewBoolVar("")
                x[(m, w)] = v
                mvars[m].append((r, v))
                wvars[w].append((women[w][m], m, v))
    for lst in mvars.values():
        if lst:
            model.Add(sum(v for _, v in lst) <= 1)
    for lst in wvars.values():
        if lst:
            model.Add(sum(v for _, _, v in lst) <= 1)
    for (m, w), v in x.items():
        rm, rw = men[m][w], women[w][m]
        model.Add(sum([vv for r, vv in mvars[m] if r <= rm] + [vv for r, mm, vv in wvars[w] if r <= rw and mm != m]) >= 1)
    for m in Nm:  # m inside, w outside (w keeps her partner)
        for w, rm in men[m].items():
            if w in Nw:
                continue
            p = pw[w]
            if p < 0 or women[w][m] < women[w][p]:
                model.Add(sum(vv for r, vv in mvars[m] if r <= rm) >= 1)
    for w in Nw:  # w inside, m outside
        for m, rw in women[w].items():
            if m in Nm:
                continue
            p = pm[m]
            if p < 0 or men[m][w] < men[m][p]:
                model.Add(sum(vv for r, _, vv in wvars[w] if r <= rw) >= 1)
    for (m, w), v in x.items():
        model.AddHint(v, int(pm[m] == w))
    model.Maximize(sum(x.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return pm, pw, solver.StatusName(status), len(x)
    pm, pw = pm.copy(), pw.copy()
    for m in Nm:
        pm[m] = -1
    for w in Nw:
        pw[w] = -1
    for (m, w), v in x.items():
        if solver.Value(v):
            pm[m], pw[w] = w, m
    return pm, pw, solver.StatusName(status), len(x)
