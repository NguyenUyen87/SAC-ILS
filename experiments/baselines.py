"""Published methods on the small and large sets (Table "published" of the paper).

  CKY   1.5-approximation of Csaji, Kiraly and Yokoi (sacsmti.baselines.cky)
  HR    heuristic repair of Viet et al. (no published code; Algorithms 1-2 of the paper), 50 iterations
  MCS   max-conflicts search, AS, LTIU: ports of the original MATLAB code with its default
        iteration limits (MCS 10,000, AS 3000, LTIU 3000), started from a random matching
LTIU is run on the small set only, as in the TBLS paper. Instances: those of the small
and large sets (instances 0-1 of each configuration). Every matching is checked for weak
stability. Rows are appended as runs finish and a rerun resumes.

    python experiments/baselines.py small --workers 8
"""
import argparse
import csv
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import P2, bench_path, out  # noqa: E402
from sacsmti.baselines.as_alg import AS  # noqa: E402
from sacsmti.baselines.cky import cky  # noqa: E402
from sacsmti.baselines.common import make_random_matching  # noqa: E402
from sacsmti.baselines.hr import hr  # noqa: E402
from sacsmti.baselines.ltiu import LTIU  # noqa: E402
from sacsmti.baselines.mcs import MCS  # noqa: E402
from sacsmti.instance import Instance  # noqa: E402
from sacsmti.rng import MatlabRNG  # noqa: E402
from sacsmti.stability import is_weakly_stable  # noqa: E402

SETS = {"small": {"n": 100, "p1": ("0.7", "0.8", "0.9"), "algs": ("CKY", "HR", "MCS", "AS", "LTIU")},
        "large": {"n": 1000, "p1": ("0.98", "0.99"), "algs": ("CKY", "HR", "MCS", "AS")}}
SLOW_FIRST = {"LTIU": 0, "AS": 1, "MCS": 2, "HR": 3, "CKY": 4}


def load_dense(path):
    """Rank matrices R[m, w], W[w, m] (1 = best, 0 = unacceptable)."""
    with open(path) as fh:
        d = json.load(fh)
    n = len(d["men_preference"])
    R = np.zeros((n, n), dtype=np.int64)
    W = np.zeros((n, n), dtype=np.int64)
    for m, pref in d["men_preference"].items():
        for w, r in pref.items():
            R[int(m[2:]), int(w[2:])] = r + 1
    for w, pref in d["women_preference"].items():
        for m, r in pref.items():
            W[int(w[2:]), int(m[2:])] = r + 1
    return R, W


def assign_from(M, n):
    a = np.full(n, -1, dtype=np.int64)
    both = (M[0] > 0) & (M[1] > 0)
    a[M[0, both] - 1] = M[1, both] - 1
    return a


def job(a):
    path, cfg, i, alg, seed = a
    R, W = load_dense(path)
    n = R.shape[0]
    inst = Instance(R, W, np.ones(n, dtype=np.int64))
    t0 = time.perf_counter()
    if alg == "CKY":
        a, iters = cky(inst, np.random.default_rng(seed)), 0
    elif alg == "HR":
        r = hr(R, W, max_iters=50, candidate="paper")
        a, iters = r["assign"], r["iters"]
    else:
        rng = MatlabRNG(seed)
        M = make_random_matching(R, W, n, rng)
        r = {"MCS": MCS, "AS": AS, "LTIU": LTIU}[alg](R, W, M, rng=rng)
        a, iters = assign_from(r[-1], n), r[3]
    t = time.perf_counter() - t0
    row = {**cfg, "inst": i, "alg": alg, "singles": int((a < 0).sum()), "time": round(t, 4), "iters": iters,
           "stable": int(is_weakly_stable(inst, a))}
    return row, [[int(m), int(w)] for m, w in enumerate(a) if w >= 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", choices=SETS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=None, help="only the first K configurations (quick test)")
    args = ap.parse_args()
    s = SETS[args.size]
    jobs = []
    for g in ("geometric", "flip_geometric"):
        for p1 in s["p1"]:
            for p2 in P2:
                cfg = {"n": s["n"], "p1": p1, "p2": p2, "g": "Geom(1-p2)" if g == "geometric" else "Geom(p2)"}
                for alg in s["algs"]:
                    for i in range(2):
                        jobs.append((bench_path(s["n"], p1, p2, g, i), cfg, i, alg, 7919 * i + 13))
    if args.first is not None:
        jobs = jobs[:args.first * 2 * len(s["algs"])]
    path = out(f"baselines_{args.size}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["p1"], r["p2"], r["g"], r["inst"], r["alg"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if (j[1]["p1"], j[1]["p2"], j[1]["g"], str(j[2]), j[3]) not in done]
    todo.sort(key=lambda j: SLOW_FIRST[j[3]])
    print(f"baselines {args.size}: {len(done)} runs done, {len(todo)} to run", flush=True)
    if not todo:
        return
    fields = ["n", "p1", "p2", "g", "inst", "alg", "singles", "time", "iters", "stable"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh, gzip.open(out(f"matchings_baselines_{args.size}.jsonl.gz"), "ab") as mfh, \
            ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for j, (row, pairs) in enumerate(ex.map(job, todo, chunksize=1), 1):
            if not row["stable"]:
                raise SystemExit(f"not weakly stable: {row}")
            w.writerow(row)
            fh.flush()
            mfh.write((json.dumps({k: row[k] for k in ("p1", "p2", "g", "inst", "alg")} | {"pairs": pairs},
                                  separators=(",", ":")) + "\n").encode())
            if j % max(1, len(todo) // 20) == 0:
                print(f"  [{j}/{len(todo)}]", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
