"""Are the improvements found by exact neighbourhood re-optimisation single stable
augmenting paths, and are those paths chains?

Table "why" of the paper. Instances: three of the scalability set (n = 10,000), six of the
benchmark (n = 1000, p1 = 0.99, instance 3) and two generated (n = 5000). TBLS runs for a
warm-up (60, 5 or 30 s), then 40 exact repairs (CP-SAT, 5 s each) of alternating-path
neighbourhoods (sacsmti.lns). For each repair that enlarges M, every M-augmenting component
P of M (+) M' is tested: is M (+) P weakly stable (a stable augmenting path), and is P a
stable augmenting chain (checked step by step)? Output: repair_paths.log (one line per
instance and a TOTAL line). The warm-up and the repairs are limited by time.

    python experiments/repair_paths.py
"""
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from chain_stats import worse_off  # noqa: E402
from common import SCALE, bench_path, out, scal_path  # noqa: E402
from sacsmti import gen  # noqa: E402
from sacsmti.io import read_json  # noqa: E402
from sacsmti.lns import has_augmenting_path, path_balls, solve_sub  # noqa: E402
from sacsmti.sac import _Search, apply_path  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import run  # noqa: E402


def components(pm, pm2):
    """Components of M (+) M' as lists of edges (m, w, which) with which in {'M', "M'"}."""
    adj = defaultdict(list)
    edges = []
    for m in range(len(pm)):
        if pm[m] != pm2[m]:
            if pm[m] >= 0:
                edges.append((m, int(pm[m]), "M"))
            if pm2[m] >= 0:
                edges.append((m, int(pm2[m]), "M'"))
    for e in edges:
        adj[("m", e[0])].append(e)
        adj[("w", e[1])].append(e)
    seen, comps = set(), []
    for e in edges:
        if e in seen:
            continue
        stack, comp = [e], []
        seen.add(e)
        while stack:
            x = stack.pop()
            comp.append(x)
            for node in (("m", x[0]), ("w", x[1])):
                for y in adj[node]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
        comps.append(comp)
    return comps




def as_path(comp, pm, pw):
    """Order the edges of an M-AUGMENTING component (one more M' edge than M edges) as
    [(m0, w1), (m1, w2), ...]; None for every other component (cycles, even paths)."""
    n_new = sum(1 for e in comp if e[2] == "M'")
    if n_new != len(comp) - n_new + 1:
        return None
    new = {m: w for m, w, which in comp if which == "M'"}
    start = [m for m in new if pm[m] < 0]
    if len(start) != 1:
        return None
    path, m = [], start[0]
    while m is not None:
        w = new[m]
        path.append((m, w))
        q = int(pw[w])
        m = q if q >= 0 and q in new else None
    return path if len(path) == len(new) else None


def is_chain(men, women, pm, pw, P):
    s = _Search(men, women, pm, pw, len(P) + 1, 10 ** 9, "chain")
    for m, w in P:
        q = int(pw[w])
        s.nm[m], s.nw[w] = w, m
        s.open = q if q >= 0 else None
        if not (s._man_ok(m, w, []) and s._woman_ok(w, m, [])):
            return False
    return True


def study(name, men, women, warm, repairs, seed=0):
    g = run(men, women, "TBLS", max_iters=10 ** 9, ku=5, kw=5, seed=seed, time_limit=warm)
    pm = g["assign"].copy()
    pw = np.full(len(women), -1, dtype=np.int64)
    pw[pm[pm >= 0]] = np.flatnonzero(pm >= 0)
    rng = np.random.default_rng(seed)
    total = sum(len(d) for d in men)
    c = {"gains": 0, "components": 0, "aug_components": 0, "stable_path": 0, "chain": 0, "cycles": 0,
         "reverse_paths": 0, "even_paths": 0, "unstable_fixed_by_cycles": 0, "worse_off": 0}
    for _ in range(repairs):
        Nm, Nw = path_balls(men, women, pm, pw, rng, min(20000, total // 4), 1, ball_cap=500)
        if not has_augmenting_path(men, pm, pw, Nm, Nw):
            continue
        pm2, pw2, _, _ = solve_sub(men, women, pm, pw, Nm, Nw, max(0.5, 5.0 * SCALE))
        if (pm2 >= 0).sum() > (pm >= 0).sum():
            c["gains"] += int((pm2 >= 0).sum() - (pm >= 0).sum())
            comps = components(pm, pm2)
            others = []
            for comp in comps:
                c["components"] += 1
                nM = sum(1 for e in comp if e[2] == "M")
                nN = len(comp) - nM
                if nN == nM + 1:
                    pass
                elif nN == nM:
                    # cycle if every agent has degree 2, else an even alternating path
                    deg = {}
                    for m_, w_, _ in comp:
                        deg[("m", m_)] = deg.get(("m", m_), 0) + 1
                        deg[("w", w_)] = deg.get(("w", w_), 0) + 1
                    c["cycles" if all(v == 2 for v in deg.values()) else "even_paths"] += 1
                    others.append(comp)
                    continue
                else:
                    c["reverse_paths"] += 1
                    others.append(comp)
                    continue
            for comp in comps:
                P = as_path(comp, pm, pw)
                if P is None:
                    continue
                c["aug_components"] += 1
                a = pm.copy()
                b = pw.copy()
                apply_path(a, b, P)
                c["worse_off"] += int(worse_off(men, women, pm, pw, P))
                if is_stable(men, women, a):
                    c["stable_path"] += 1
                    c["chain"] += int(is_chain(men, women, pm, pw, P))
                else:
                    # does switching P together with all the other components of M (+) M' restore stability?
                    a2, b2 = a.copy(), b.copy()
                    for comp2 in others:
                        for m_, w_, which in comp2:
                            if which == "M" and a2[m_] == w_:
                                a2[m_] = -1
                                b2[w_] = -1
                        for m_, w_, which in comp2:
                            if which == "M'":
                                a2[m_], b2[w_] = w_, m_
                    c["unstable_fixed_by_cycles"] += int(is_stable(men, women, a2))
        pm, pw = pm2, pw2
    return c


def main(first=None):
    tot = {}
    cases = []
    for i in range(3):
        cases.append((f"scal n=10000 i={i}", lambda i=i: read_json(scal_path(10000, i)), 60, 40))
    for p2 in ("0.2", "0.5", "0.8"):
        for g in ("geometric", "flip_geometric"):
            p = bench_path(1000, "0.99", p2, g, 3)
            cases.append((f"large p1=0.99 p2={p2} {g}", lambda p=p: read_json(p), 5, 40))
    for L, p2 in ((5, 0.8), (20, 0.2)):
        cases.append((f"cfg n=5000 L={L} p2={p2}",
                      lambda L=L, p2=p2: gen.sparse_c1(5000, L, p2, rng=np.random.default_rng(7)), 30, 40))
    with open(out("repair_paths.log"), "w") as log:
        for name, make, warm, reps in cases[:first]:
            warm *= SCALE
            men, women = make()
            c = study(name, men, women, warm, reps)
            print(name, c, file=log, flush=True)
            print(name, c, flush=True)
            for k, v in c.items():
                tot[k] = tot.get(k, 0) + v
        print("TOTAL", tot, file=log)
    print("TOTAL", tot)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    main(ap.parse_args().first)
