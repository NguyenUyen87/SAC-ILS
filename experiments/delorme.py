"""Eight threads: CP-SAT with the preprocessing and models of Delorme et al. (2019)
(Section 5.4 of the paper; compared with the methods of experiments/eight_threads.py).

Methods: CPSAT-M1pre (model M1) and CPSAT-M3pre (merged stability constraints), both
after the preprocessing of Delorme et al. (sacsmti.ilp_delorme), eight workers.
Same instances, budgets and Gale-Shapley hint as experiments/eight_threads.py;
preprocessing and model building count against the budget. Each solve runs in a
fresh process (its own peak memory). Every matching is checked for weak stability
against the original lists and saved. A rerun resumes.

    python experiments/delorme.py large
"""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import out  # noqa: E402
from eight_threads import FRACS, TAGS, WORKERS, budget_of, cases  # noqa: E402
from sacsmti.io import read_json  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import SparseSearch  # noqa: E402

METHODS = {"CPSAT-M1pre": "M1", "CPSAT-M3pre": "M3"}


def _child(path, model, hint, budget, cps, start, q):
    from sacsmti.ilp_delorme import solve
    from sacsmti.sysinfo import peak_mem_mb
    men, women = read_json(path)
    lag = time.time() - start                       # process start and loading count against the budget
    t0 = time.perf_counter() - lag
    try:
        r = solve(men, women, model, pre=True, hint=hint, time_limit=budget, workers=WORKERS, checkpoints=cps, t0=t0)
        a = r.pop("assign")
        info = {k: r[k] for k in ("status", "bound", "removed", "pairs", "pre_s", "build_s", "proved_s")}
        at = r["at"]
    except Exception as e:  # noqa: BLE001
        info, at, a = {"status": f"ERROR {type(e).__name__}"}, [-1] * len(cps), None
    info["load_s"] = round(lag, 2)
    info["mem_mb"] = peak_mem_mb()
    q.put((info, at, None if a is None else a.tolist()))


def run_one(path, model, hint, budget, cps):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(path, model, hint, budget, cps, time.time(), q))
    p.start()
    try:
        info, at, a = q.get(timeout=budget + 1800)
    except Exception:  # noqa: BLE001
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
    path_csv = out(f"delorme_{args.kind}.csv")
    mfile = out(f"matchings_delorme_{args.kind}.jsonl.gz")
    done = set()
    if os.path.exists(path_csv):
        with open(path_csv) as fh:
            done = {(r["name"], r["method"]) for r in csv.DictReader(fh)}
    fields = ["name", "n", "method", "budget", "status", "bound", "removed", "pairs", "pre_s", "build_s", "proved_s",
              "load_s", "mem_mb", "gs_size", "final", "stable", *TAGS]
    new = not os.path.exists(path_csv)
    with open(path_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        for name, path in cases(args.kind)[:args.first]:
            if all((name, m) in done for m in METHODS):
                continue
            men, women = read_json(path)
            n = len(men)
            budget = budget_of(args.kind, n)
            cps = [f * budget for f in FRACS]
            gs = SparseSearch(men, women, 0.05, 5, 5, 0.9, None, MatlabRNG(1)).pm.copy()
            for meth, model in METHODS.items():
                if (name, meth) in done:
                    continue
                info, at, a = run_one(path, model, gs, budget, cps)
                row = {"name": name, "n": n, "method": meth, "budget": budget, "gs_size": int((gs >= 0).sum()), **info,
                       "final": int((a >= 0).sum()) if a is not None else -1,
                       "stable": int(is_stable(men, women, a)) if a is not None else -1}
                if row["stable"] == 0:
                    raise SystemExit(f"not weakly stable: {name} {meth}")
                row.update(dict(zip(TAGS, at)))
                if a is not None:
                    with gzip.open(mfile, "ab") as mf:
                        mf.write((json.dumps({"name": name, "method": meth, "file": os.path.basename(path),
                                              "assign": [int(x) for x in a]}) + "\n").encode())
                w.writerow(row)
                fh.flush()
                print(name, meth, {k: row.get(k) for k in ("status", "removed", "build_s", "mem_mb", "final")}, at,
                      flush=True)
    print("wrote", path_csv)


if __name__ == "__main__":
    main()
