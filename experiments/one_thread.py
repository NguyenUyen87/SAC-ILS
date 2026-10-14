"""One-thread runs with equal wall-clock budgets (Sections 5.3-5.5 of the paper).

Instance sets: see experiments/common.py (small, large, opt, scal, cfg).
Methods
  SACILS        SAC-ILS, final configuration (kick k, at most 16 men, 5000 nodes, random order)
  SAC           chains from the Gale-Shapley matching until none is found, no perturbation
  TBLS          TBLS (Qiu 2025), sacsmti.tbls
  TBLS+ILS      control TBLS-ILS: TBLS returning to its best matching after 50 idle iterations
  ILS[adj]      control ILS-ADJ: the ILS of SAC-ILS with the TBLS adjustments as local search
  ILS           SAC-ILS with the initial candidate order (single women first, then by preference)
  SACILS[path]  stable augmenting paths instead of chains
  SACILS[both]  chains from single women too
  SACILS[lns]   SAC-ILS-LNS: CP-SAT re-optimises a neighbourhood every 10th perturbation
  CPSAT1        CP-SAT on the whole model, one worker, Gale-Shapley hint
The best number of singles is recorded at 10, 25, 50 and 100% of the budget. Every final
matching is checked for weak stability and saved (matchings_one_thread_<set>.jsonl.gz).
Rows are appended as runs finish; a rerun skips the runs already in the file.
--rep r > 0 is an independent repetition: every seed changes and the files get suffix _r<r>.

    python experiments/one_thread.py large --methods SACILS TBLS --workers 8
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import FRACS, instances, load, out  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402,F401  (imported once per worker, outside the budgets)
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import SparseSearch, run  # noqa: E402

# parameters of sac_ils beyond kick = k (1 for n = 100, 5 otherwise), max_len 16, node_budget 5000
ILS_VARIANTS = {"SACILS": {"order": "random"}, "ILS": {},
                "SACILS[path]": {"order": "random", "mode": "path"},
                "SACILS[both]": {"order": "random", "sides": "both"},
                "ILS[adj]": {"order": "random", "ls": "adj"},
                "SACILS[lns]": {"order": "random", "lns_every": 10}}
METHODS = ["SACILS", "SAC", "TBLS", "TBLS+ILS", *[m for m in ILS_VARIANTS if m != "SACILS"], "CPSAT1"]
FIELDS = ["source", "name", "n", "method", "budget", "singles", "stable", "wall", "iters", "repairs",
          "repair_gain", "repair_time", "status", "bound", "s10", "s25", "s50", "s100",
          "mean_edges", "max_edges", "cut_frac"]


def gs_start(men, women, k, seed):
    """Gale-Shapley matching of a random tie-breaking (the start of SAC, SAC-ILS and TBLS)."""
    return SparseSearch(men, women, 0.05, k, k, 0.9, None, MatlabRNG(seed)).pm.copy()


def job(a):
    source, name, n, budget, k, spec, method, seed = a
    men, women = load(spec)
    cps = [f * budget for f in FRACS]
    t0 = time.perf_counter()
    extra = {}
    if method in ("TBLS", "TBLS+ILS"):
        r = run(men, women, method, max_iters=10 ** 9, ku=k, kw=k, seed=seed, time_limit=budget, checkpoints=cps)
        assign, anytime = r["assign"], r["anytime"]
        extra["iters"] = r["iters"]
    elif method == "SAC":
        from sacsmti.sac import sac_search
        r = sac_search(men, women, gs_start(men, women, k, seed), budget, max_len=16, node_budget=5000, seed=seed,
                       checkpoints=cps, order="random")
        assign, anytime = r["assign"], r["anytime"]
        extra = {"iters": r["paths"]}
    elif method in ILS_VARIANTS:
        from sacsmti.sac import sac_ils
        kw = {"kick": k, "max_len": 16, "node_budget": 5000, **ILS_VARIANTS[method]}
        r = sac_ils(men, women, budget, seed=seed, checkpoints=cps, **kw)
        assign, anytime = r["assign"], r["anytime"]
        extra = {"iters": r["kicks"], "repairs": r["paths"], "repair_gain": r.get("lns_gain", 0),
                 "repair_time": round(r.get("lns_time", 0.0), 2),
                 "mean_edges": round(r["edges"] / max(r["paths"], 1), 2), "max_edges": r["max_edges"],
                 "cut_frac": round(r["cut"] / max(r["searches"], 1), 4)}
    elif method == "CPSAT1":
        from sacsmti.exact import cpsat
        gs = gs_start(men, women, k, seed)
        info, at, assign = cpsat(men, women, gs, budget, cps, workers=1)
        anytime = [len(men) - s if s >= 0 else -1 for s in at]
        if assign is None:
            assign = gs
        extra = {"status": info["status"], "bound": info["bound"]}
    else:
        raise ValueError(method)
    row = {"source": source, "name": name, "n": n, "method": method, "budget": budget,
           "singles": int((assign < 0).sum()), "stable": int(is_stable(men, women, assign)),
           "wall": round(time.perf_counter() - t0, 2), **extra}
    row.update({f"s{int(100 * f)}": v for f, v in zip(FRACS, anytime)})
    pairs = [[int(m), int(w)] for m, w in enumerate(assign) if w >= 0]
    return row, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=("small", "large", "opt", "scal", "cfg"))
    ap.add_argument("--methods", nargs="+", default=["SACILS"], choices=METHODS)
    ap.add_argument("--limit", type=int, default=None, help="instances per size/configuration (scal, cfg)")
    ap.add_argument("--workers", type=int, default=4, help="runs in parallel (one thread each)")
    ap.add_argument("--rep", type=int, default=0, help="independent repetition: changes every seed and the file names")
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    args = ap.parse_args()
    suffix = f"_r{args.rep}" if args.rep else ""
    path = out(f"one_thread_{args.source}{suffix}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["name"], r["method"]) for r in csv.DictReader(fh)}
    jobs = []
    for j, (name, n, budget, k, spec) in enumerate(instances(args.source, args.limit)):
        if args.first is not None and j >= args.first:
            break
        for m in args.methods:
            if (name, m) not in done:
                jobs.append((args.source, name, n, budget, k, spec, m, 1000 + j + 100003 * args.rep))
    jobs.sort(key=lambda a: -a[3])          # long budgets first
    print(f"{args.source}{suffix}: {len(done)} runs done, {len(jobs)} to run", flush=True)
    if not jobs:
        return
    new = not os.path.exists(path)
    mpath = out(f"matchings_one_thread_{args.source}{suffix}.jsonl.gz")
    with open(path, "a", newline="") as fh, ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=FIELDS, restval="", extrasaction="ignore")
        if new:
            w.writeheader()
        for c, (row, pairs) in enumerate(ex.map(job, jobs, chunksize=1), 1):
            if not row["stable"]:
                raise SystemExit(f"not weakly stable: {row}")
            w.writerow(row)
            fh.flush()
            with gzip.open(mpath, "ab") as mfh:
                mfh.write((json.dumps({"name": row["name"], "method": row["method"], "pairs": pairs},
                                      separators=(",", ":")) + "\n").encode())
            if c % 10 == 0 or c == len(jobs):
                print(f"  [{c}/{len(jobs)}] {row['name']} {row['method']} singles={row['singles']}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
