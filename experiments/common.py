"""Paths, instance sets and output files shared by the experiment scripts.

Locations (environment variables, set by run_experiments.py):
  SACSMTI_DATA     data directory (default <repo>/data): Instances/SMTI (benchmark of Qiu
                   et al. 2025), scal/Instances/SMTI (its scalability set), huge/ (generated),
                   opt_set.csv (instances with a proven optimum)
  SACSMTI_RESULTS  result directory (default <repo>/results/local)
  SACSMTI_BUDGET_SCALE  factor applied to every time budget (default 1; only for a quick
                   test of the whole pipeline, e.g. run_experiments.py --smoke)

Instance sets of the paper (one-thread runs):
  small  benchmark n = 100, instances 0-1 of all 60 configurations, budget 3 s, kick 1
  large  benchmark n = 1000, p1 in {0.98, 0.99}, instances 0-1 of the 40 configurations, 30 s
  opt    68 benchmark instances with n = 1000 (p1 0.95-0.99) whose optimum was proven by
         integer programming (data/opt_set.csv), 30 s
  scal   scalability set, n in {2000, 5000, 10000}, instances 0-19, 0.03 n s
  cfg    generated (sacsmti.gen.sparse_c1), n in {5000, 10000}, mean list length in {5, 20},
         p2 in {0.2, 0.8}, instances 0-4, 0.03 n s
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sacsmti.gen import sparse_c1  # noqa: E402
from sacsmti.io import read_json  # noqa: E402

DATA = os.path.abspath(os.environ.get("SACSMTI_DATA", os.path.join(ROOT, "data")))
RESULTS = os.path.abspath(os.environ.get("SACSMTI_RESULTS", os.path.join(ROOT, "results", "local")))
BENCH = os.path.join(DATA, "Instances", "SMTI")
SCAL = os.path.join(DATA, "scal", "Instances", "SMTI")
HUGE = os.path.join(DATA, "huge")
OPT_SET = os.path.join(ROOT, "data", "opt_set.csv")
P2 = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
FRACS = (0.1, 0.25, 0.5, 1.0)
SCALE = float(os.environ.get("SACSMTI_BUDGET_SCALE", "1"))


def out(name):
    """Path of a result file in the result directory (created if needed)."""
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def bench_path(n, p1, p2, g, i):
    return os.path.join(BENCH, f"n_{n}_p1_{p1}_p2_{p2}_{g}", f"instance_{i}.json")


def scal_path(n, i):
    return os.path.join(SCAL, f"n_{n}_p1_10_p2_0.5_geometric", f"instance_{i}.json")


def cfg_seed(n, L, p2, i):
    return 1_000_003 * n + 10_007 * L + int(100 * p2) * 101 + i


def instances(source, limit=None):
    """Yields (name, n, budget, k, spec); load(spec) returns (men, women). The order is
    fixed: the seed of a run depends on the position of its instance."""
    for name, n, budget, k, spec in _instances(source, limit):
        yield name, n, budget * SCALE, k, spec


def _instances(source, limit):
    if source in ("small", "large"):
        n, p1s, budget, k = (100, ("0.7", "0.8", "0.9"), 3.0, 1) if source == "small" else \
            (1000, ("0.98", "0.99"), 30.0, 5)
        for i in range(2):
            for p1 in p1s:
                for g in ("geometric", "flip_geometric"):
                    for p2 in P2:
                        yield f"p1={p1},p2={p2},{g},i={i}", n, budget, k, ("file", bench_path(n, p1, p2, g, i))
    elif source == "opt":
        import pandas as pd
        o = pd.read_csv(OPT_SET)
        for _, r in o[o.status == "OPTIMAL"].iterrows():
            g = "geometric" if r.g == "Geom(1-p2)" else "flip_geometric"
            yield (f"p1={r.p1},p2={r.p2},{g},i={int(r.inst)}", 1000, 30.0, 5,
                   ("file", bench_path(1000, r.p1, r.p2, g, int(r.inst))))
    elif source == "scal":
        for i in range(limit or 20):
            for n in (10000, 5000, 2000):
                yield f"n={n},i={i}", n, 0.03 * n, 5, ("file", scal_path(n, i))
    elif source == "cfg":
        for i in range(limit or 10):
            for n in (10000, 5000):
                for L in (5, 20):
                    for p2 in (0.2, 0.8):
                        yield f"n={n},L={L},p2={p2},i={i}", n, 0.03 * n, 5, ("gen", n, L, p2, cfg_seed(n, L, p2, i))
    else:
        raise ValueError(source)


def load(spec):
    if spec[0] == "file":
        return read_json(spec[1])
    _, n, L, p2, seed = spec
    return sparse_c1(n, L, p2, rng=np.random.default_rng(seed))
