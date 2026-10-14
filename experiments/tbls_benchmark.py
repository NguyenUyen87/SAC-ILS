"""TBLS on the whole benchmark of Qiu (2025) with the settings of the TBLS paper:
3000 iterations, k_u = k_w = 1 (n = 100) or 5 (n = 1000). Reproduces the published
results of TBLS (Section 5.1) and shows that the large instances with p1 <= 0.97 leave
little room for improvement.

  small  n = 100,  p1 0.7-0.9,   p2 0.1-1.0, geometric and flip_geometric ties, 100 instances each
  large  n = 1000, p1 0.95-0.99, p2 0.1-1.0, geometric and flip_geometric ties, 100 instances each

The seed of a run is that of the original study, in which TBLS ran next to two other
algorithms (hence the factor 3 in the job index). Rows are appended as runs finish and
a rerun resumes. Matchings: matchings_tbls_benchmark_<size>.jsonl.gz.

    python experiments/tbls_benchmark.py large --workers 8
"""
import argparse
import csv
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import P2, bench_path, out  # noqa: E402
from sacsmti.io import read_json  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402
from sacsmti.tbls import run  # noqa: E402

SIZES = {"small": {"n": 100, "p1": ("0.7", "0.8", "0.9"), "k": 1},
         "large": {"n": 1000, "p1": ("0.95", "0.96", "0.97", "0.98", "0.99"), "k": 5}}
INSTANCES = 100


def configurations(size):
    sz = SIZES[size]
    for g in ("geometric", "flip_geometric"):
        for p1 in sz["p1"]:
            for p2 in P2:
                yield g, p1, p2, {"n": sz["n"], "p1": p1, "p2": p2, "g": "Geom(1-p2)" if g == "geometric" else "Geom(p2)"}


def job(a):
    path, cfg, i, k, seed = a
    men, women = read_json(path)
    r = run(men, women, "TBLS", max_iters=3000, ku=k, kw=k, seed=seed)
    row = {**cfg, "inst": i, "alg": "TBLS", "singles": r["singles"], "time": round(r["time"], 4), "iters": r["iters"],
           "stable": int(is_stable(men, women, r["assign"]))}
    return row, [[int(m), int(w)] for m, w in enumerate(r["assign"]) if w >= 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("size", choices=SIZES)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances of each configuration (quick test)")
    args = ap.parse_args()
    k = SIZES[args.size]["k"]
    jobs = []
    for c, (g, p1, p2, cfg) in enumerate(configurations(args.size)):
        for i in range(INSTANCES if args.first is None else args.first):
            seed = 1000003 * i + ((c * INSTANCES + i) * 3) % 7
            jobs.append((bench_path(cfg["n"], p1, p2, g, i), cfg, i, k, seed))
    path = out(f"tbls_benchmark_{args.size}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            done = {(r["p1"], r["p2"], r["g"], r["inst"]) for r in csv.DictReader(fh)}
    todo = [j for j in jobs if (j[1]["p1"], j[1]["p2"], j[1]["g"], str(j[2])) not in done]
    print(f"tbls_benchmark {args.size}: {len(done)} runs done, {len(todo)} to run", flush=True)
    if not todo:
        return
    fields = ["n", "p1", "p2", "g", "inst", "alg", "singles", "time", "iters", "stable"]
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh, gzip.open(out(f"matchings_tbls_benchmark_{args.size}.jsonl.gz"), "ab") as mfh, \
            ProcessPoolExecutor(args.workers) as ex:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        for j, (row, pairs) in enumerate(ex.map(job, todo, chunksize=4), 1):
            if not row["stable"]:
                raise SystemExit(f"not weakly stable: {row}")
            w.writerow(row)
            fh.flush()
            mfh.write((json.dumps({k2: row[k2] for k2 in ("p1", "p2", "g", "inst")} | {"pairs": pairs},
                                  separators=(",", ":")) + "\n").encode())
            if j % max(1, len(todo) // 20) == 0:
                print(f"  [{j}/{len(todo)}]", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
