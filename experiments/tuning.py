"""Parameter selection for SAC-ILS on development instances only.

The development instances are generated with seeds disjoint from every evaluation
set and are never used for evaluation:
  dev-1000   sacsmti.gen.c1, n = 1000, p1 in {0.98, 0.99}, p2 in {0.2, 0.5, 0.8},
             ties Geom(p2), 4 instances per configuration (24), budget 30 s
  dev-5000   sacsmti.gen.sparse_c1, n = 5000, mean list length 10, p2 in {0.2, 0.5, 0.8},
             4 instances per configuration (12), budget 150 s
Configurations: one parameter changed at a time from the default (kick 5, maximum
chain length 16 men, 5000 search nodes per single man, candidates tried single
women first and then by preference, start from Gale-Shapley): kick 2 or 10, length
8 or 32, 1000 or 20,000 nodes, random candidate order, start from CKY, shortest
chains first (iterative deepening on the length: 2, 4, 8, 16 men). The configuration with
the smallest mean rank over the instances is selected; the default is kept unless
another configuration is better with Wilcoxon p < 0.05 on the pooled instances.

    python experiments/tuning.py run --workers 8
    python experiments/tuning.py report
"""
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SCALE, out  # noqa: E402
from sacsmti import gen  # noqa: E402
from sacsmti.io import from_matrices  # noqa: E402
from sacsmti.sac import sac_ils  # noqa: E402
from sacsmti.stability import is_stable  # noqa: E402

DEFAULT = {"kick": 5, "max_len": 16, "node_budget": 5000}
CONFIGS = {"default": {}, "kick=2": {"kick": 2}, "kick=10": {"kick": 10}, "len=8": {"max_len": 8},
           "len=32": {"max_len": 32}, "nodes=1000": {"node_budget": 1000}, "nodes=20000": {"node_budget": 20000},
           "order=random": {"order": "random"}, "init=cky": {"init": "cky"}, "deepen": {"deepen": True},
           # robustness check after the selection (review): not used to select
           "random+nodes=1000": {"order": "random", "node_budget": 1000}}
OUT = out("tuning.csv")


def instances():
    for p1 in (0.98, 0.99):
        for p2 in (0.2, 0.5, 0.8):
            for i in range(4):
                yield f"dev1000,p1={p1},p2={p2},i={i}", ("c1", 1000, p1, p2, 7_000_000 + int(1000 * p1) * 100 + int(10 * p2) * 10 + i), 30.0
    for p2 in (0.2, 0.5, 0.8):
        for i in range(4):
            yield f"dev5000,p2={p2},i={i}", ("sparse", 5000, 10, p2, 8_000_000 + int(10 * p2) * 10 + i), 150.0


def load(spec):
    if spec[0] == "c1":
        _, n, p1, p2, seed = spec
        inst = gen.c1(n, p1, p2, rng=np.random.default_rng(seed), geom="p2")
        return from_matrices(inst.r_rank, inst.h_rank)
    _, n, L, p2, seed = spec
    return gen.sparse_c1(n, L, p2, rng=np.random.default_rng(seed))


def job(a):
    name, spec, budget, cfg = a
    men, women = load(spec)
    r = sac_ils(men, women, budget * SCALE, seed=12345, **{**DEFAULT, **CONFIGS[cfg]})
    return {"name": name, "config": cfg, "singles": r["singles"], "stable": int(is_stable(men, women, r["assign"])),
            "kicks": r["kicks"], "mean_edges": round(r["edges"] / max(r["paths"], 1), 2),
            "cut_frac": round(r["cut"] / max(r["searches"], 1), 4)}


def run(workers, first=None):
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(r["name"], r["config"]) for r in csv.DictReader(fh)}
    jobs = [(name, spec, b, c) for name, spec, b in list(instances())[:first] for c in CONFIGS if (name, c) not in done]
    jobs.sort(key=lambda a: -a[2])
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh, ProcessPoolExecutor(workers) as ex:
        w = csv.DictWriter(fh, fieldnames=["name", "config", "singles", "stable", "kicks", "mean_edges", "cut_frac"])
        if new:
            w.writeheader()
        for row in ex.map(job, jobs, chunksize=1):
            w.writerow(row)
            fh.flush()
    print("wrote", OUT)


def report():
    import pandas as pd
    from scipy.stats import wilcoxon
    d = pd.read_csv(OUT)
    assert d.stable.min() == 1
    piv = d.pivot_table(index="name", columns="config", values="singles")
    ranks = piv.rank(axis=1).mean().sort_values()
    print("mean singles:\n", piv.mean().round(3).sort_values())
    print("mean rank:\n", ranks.round(2))
    best = ranks.index[0]
    chosen = "default"
    if best != "default":
        p = wilcoxon(piv[best], piv["default"]).pvalue if (piv[best] != piv["default"]).any() else 1.0
        print(f"best {best} vs default: p = {p:.3g}")
        if p < 0.05 and piv[best].mean() < piv["default"].mean():
            chosen = best
    print("selected:", chosen, {**DEFAULT, **CONFIGS[chosen]})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("run", "report"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--first", type=int, default=None, help="only the first K instances (quick test)")
    a = ap.parse_args()
    run(a.workers, a.first) if a.action == "run" else report()
