"""Eight threads, same wall-clock budget, one instance and one method at a time
(Section 5.4 of the paper). Nothing else should run on the machine.

Methods
  CP-SAT   CP-SAT (eight workers) on the whole model, Gale-Shapley hint (sacsmti.exact)
  ILSx8    eight independent SAC-ILS runs (seeds 1000-1007), best of the eight
  TBLSx8   eight independent TBLS runs, best of the eight (not on the very large instances)
  HYB      eight SAC-ILS runs for the first 10% of the budget, then CP-SAT (eight workers)
           started from the best of the eight matchings for the rest of the budget
Instances
  large  benchmark n = 1000, p1 in {0.98, 0.99}, instance 0 of the 40 configurations, 30 s
  scal   scalability set, n in {2000, 5000, 10000}, instances 0-4, 0.03 n s
  huge   generated (sparse_c1, lists of about ten, p2 = 0.5), n in {50000, 100000},
         two instances each (seeds 77,000,000+), 600 s; written to <data>/huge on first use
Model building (and, for CP-SAT, process start and loading) counts against the budget.
Each CP-SAT solve runs in a fresh process, so its peak memory is its own; CP-SAT is
limited to 12 GB. The best size is recorded at 10, 25, 50 and 100% of the budget.
Every matching is checked for weak stability and saved. A rerun resumes.

    python experiments/eight_threads.py large
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import HUGE, P2, SCALE, bench_path, out, scal_path  # noqa: E402
from sacsmti.io import read_json, write_json  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402

WORKERS = 8
MEM_MB = 12000
HUGE_CASES = [(n, i) for n in (50000, 100000) for i in range(2)]
FRACS = (0.1, 0.25, 0.5, 1.0)
TAGS = ("size_10", "size_25", "size_50", "size_100")


def cases(kind):
    """(name, path) of the instances of a set; the very large instances are generated if missing."""
    if kind == "large":
        return [(f"p1={p1},p2={p2},{g}", bench_path(1000, p1, p2, g, 0))
                for p1 in ("0.98", "0.99") for g in ("geometric", "flip_geometric") for p2 in P2]
    if kind == "scal":
        return [(f"n={n},i={i}", scal_path(n, i)) for n in (2000, 5000, 10000) for i in range(5)]
    from sacsmti.gen import sparse_c1
    os.makedirs(HUGE, exist_ok=True)
    res = []
    for n, i in HUGE_CASES:
        p = os.path.join(HUGE, f"n_{n}_L_10_p2_0.5_i_{i}.json")
        if not os.path.exists(p):
            men, women = sparse_c1(n, 10, 0.5, rng=np.random.default_rng(77_000_000 + 1000 * (n // 1000) + i))
            write_json(p, men, women)
            print("wrote", p, flush=True)
        res.append((f"n={n},i={i}", p))
    return res


def budget_of(kind, n):
    return {"large": 30.0, "scal": 0.03 * n, "huge": 600.0}[kind] * SCALE


def heuristic_one(args):
    """One single-thread run (in a pool worker): sizes at the checkpoints, matching, iterations, memory."""
    path, seed, budget, cps, method = args
    from sacsmti.sysinfo import peak_mem_mb
    men, women = read_json(path)
    if method == "ILS":
        from sacsmti.sac import sac_ils
        r = sac_ils(men, women, budget, kick=5, max_len=16, node_budget=5000, seed=seed, checkpoints=cps,
                    order="random")
        r["iters"] = r["kicks"]
    else:
        from sacsmti.tbls import run
        r = run(men, women, "TBLS", max_iters=10 ** 9, ku=5, kw=5, seed=seed, time_limit=budget, checkpoints=cps)
    n = len(men)
    return [n - s for s in r["anytime"]], r["assign"], r["iters"], peak_mem_mb()


def _cpsat_child(path, hint, budget, cps, start, q):
    from sacsmti.exact import cpsat
    from sacsmti.sysinfo import peak_mem_mb
    men, women = read_json(path)
    lag = time.time() - start                  # process start and loading count against the budget
    try:
        info, at, a = cpsat(men, women, hint, budget - lag, [c - lag for c in cps], workers=WORKERS, max_mem_mb=MEM_MB)
        info["load_s"] = round(lag, 2)
    except Exception as e:  # noqa: BLE001  (out of memory, model too large, ...)
        info, at, a = {"status": f"ERROR {type(e).__name__}"}, [-1] * len(cps), None
    info["mem_mb"] = peak_mem_mb()
    q.put((info, at, None if a is None else a.tolist()))


def cpsat_proc(path, hint, budget, cps):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_cpsat_child, args=(path, hint, budget, cps, time.time(), q))
    p.start()
    try:
        info, at, a = q.get(timeout=budget + 1800)
    except Exception:  # noqa: BLE001  (child killed, e.g. by the OS)
        info, at, a = {"status": "ERROR killed"}, [-1] * len(cps), None
    p.join(60)
    if p.is_alive():
        p.kill()
    return info, at, None if a is None else np.asarray(a, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=("large", "scal", "huge"))
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    args = ap.parse_args()
    methods = ("CP-SAT", "ILSx8", "HYB") if args.kind == "huge" else ("CP-SAT", "TBLSx8", "ILSx8", "HYB")
    path_csv = out(f"eight_threads_{args.kind}.csv")
    mfile = out(f"matchings_eight_threads_{args.kind}.jsonl.gz")
    done = set()
    if os.path.exists(path_csv):
        with open(path_csv) as fh:
            done = {(r["name"], r["method"]) for r in csv.DictReader(fh)}
    fields = ["name", "n", "method", "budget", "status", "bound", "build_s", "first_s", "proved_s", "load_s", "mem_mb",
              "heur_mem_mb", "gs_size", "phase1_size", "final", "stable", *TAGS]
    new = not os.path.exists(path_csv)
    with open(path_csv, "a", newline="") as fh, ProcessPoolExecutor(WORKERS) as ex:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        for name, path in cases(args.kind)[:args.first]:
            if all((name, m) in done for m in methods):
                continue
            men, women = read_json(path)
            n = len(men)
            budget = budget_of(args.kind, n)
            cps = [f * budget for f in FRACS]
            gs = SparseSearch(men, women, 0.05, 5, 5, 0.9, None, MatlabRNG(1)).pm.copy()   # random ties + GS
            base = {"name": name, "n": n, "budget": budget, "gs_size": int((gs >= 0).sum())}
            for meth in methods:
                if (name, meth) in done:
                    continue
                row = dict(base, method=meth)
                if meth == "CP-SAT":
                    info, at, a = cpsat_proc(path, gs, budget, cps)
                    row.update(info)
                elif meth in ("ILSx8", "TBLSx8"):
                    res = list(ex.map(heuristic_one, [(path, 1000 + s, budget, cps, meth[:-2]) for s in range(WORKERS)]))
                    at = [max(r[0][j] for r in res) for j in range(len(cps))]
                    a = res[int(np.argmax([r[0][-1] for r in res]))][1]
                    row["status"] = "-"
                    row["heur_mem_mb"] = max(r[3] for r in res)
                else:  # HYB
                    t0 = time.perf_counter()
                    b1 = FRACS[0] * budget
                    res = list(ex.map(heuristic_one, [(path, 1000 + s, b1, [b1], "ILS") for s in range(WORKERS)]))
                    a1 = np.asarray(res[int(np.argmax([r[0][-1] for r in res]))][1], dtype=np.int64)
                    s1 = int((a1 >= 0).sum())
                    t1 = time.perf_counter() - t0
                    info, at2, a2 = cpsat_proc(path, a1, budget - t1, [max(0.0, c - t1) for c in cps])
                    row.update(info)
                    row["phase1_size"] = s1
                    at = [s1 if c <= t1 + 1e-9 else max(s1, v) for c, v in zip(cps, at2)]
                    a = a2 if a2 is not None and int((a2 >= 0).sum()) >= s1 else a1
                row["final"] = int((np.asarray(a) >= 0).sum()) if a is not None else -1
                row["stable"] = int(is_stable(men, women, a)) if a is not None else -1
                if row["stable"] == 0:
                    raise SystemExit(f"not weakly stable: {name} {meth}")
                row.update(dict(zip(TAGS, at)))
                if a is not None:
                    with gzip.open(mfile, "ab") as mf:
                        mf.write((json.dumps({"name": name, "method": meth, "file": os.path.basename(path),
                                              "assign": [int(x) for x in a]}) + "\n").encode())
                w.writerow(row)
                fh.flush()
                print(name, meth, {k: row.get(k) for k in ("status", "build_s", "mem_mb", "phase1_size", "final")},
                      at, flush=True)
    print("wrote", path_csv)


if __name__ == "__main__":
    main()
