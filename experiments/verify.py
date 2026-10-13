"""Independent check of the matchings saved by experiments/one_thread.py against the
ORIGINAL data.

The instance is read again from its JSON file (small, large, opt, scal) or generated
again from its seed (cfg), so a change made to the in-memory preference lists during a
run cannot affect this check. The check itself uses only the standard library: every
pair mutually acceptable, nobody matched twice, no blocking pair (m, w) in which both
strictly prefer each other to their partners (single = worst), and the number of singles
equal to the one recorded in the CSV. Without arguments, every one_thread_*.csv of the
result directory is checked.

    python experiments/verify.py              # all sets and repetitions
    python experiments/verify.py large large_r1
"""
import csv
import glob
import gzip
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.abspath(os.environ.get("SACSMTI_DATA", os.path.join(ROOT, "data")))
RESULTS = os.path.abspath(os.environ.get("SACSMTI_RESULTS", os.path.join(ROOT, "results", "local")))
BENCH = os.path.join(DATA, "Instances", "SMTI")
SCAL = os.path.join(DATA, "scal", "Instances", "SMTI")


def load_original(source, name):
    """Preference lists as {man: {woman: rank}}, {woman: {man: rank}} with integer agents."""
    f = dict(kv.split("=") if "=" in kv else ("g", kv) for kv in name.split(","))
    if source in ("small", "large", "opt"):
        n = 100 if source == "small" else 1000
        path = os.path.join(BENCH, f"n_{n}_p1_{f['p1']}_p2_{f['p2']}_{f['g']}", f"instance_{f['i']}.json")
    elif source == "scal":
        path = os.path.join(SCAL, f"n_{f['n']}_p1_10_p2_0.5_geometric", f"instance_{f['i']}.json")
    else:  # generated instances: regenerate (needs numpy and sacsmti.gen, not the algorithms)
        sys.path.insert(0, ROOT)
        import numpy as np
        from sacsmti.gen import sparse_c1
        n, L, p2, i = int(f["n"]), int(f["L"]), float(f["p2"]), int(f["i"])
        seed = 1_000_003 * n + 10_007 * L + int(100 * p2) * 101 + i
        men, women = sparse_c1(n, L, p2, rng=np.random.default_rng(seed))
        return dict(enumerate(men)), dict(enumerate(women))
    with open(path) as fh:
        d = json.load(fh)
    men = {int(m[2:]): {int(w[2:]): r for w, r in p.items()} for m, p in d["men_preference"].items()}
    women = {int(w[2:]): {int(m[2:]): r for m, r in p.items()} for w, p in d["women_preference"].items()}
    return men, women


def check(men, women, pairs):
    partner_m, partner_w = {}, {}
    for m, w in pairs:
        if w not in men.get(m, {}) or m not in women.get(w, {}):
            return f"unacceptable pair ({m}, {w})"
        if m in partner_m or w in partner_w:
            return f"agent matched twice in ({m}, {w})"
        partner_m[m], partner_w[w] = w, m
    for m, prefs in men.items():
        cur = partner_m.get(m)
        for w, r in prefs.items():
            if w == cur:
                continue
            if cur is not None and r >= prefs[cur]:
                continue
            her = partner_w.get(w)
            if her is None or women[w][m] < women[w][her]:
                return f"blocking pair ({m}, {w})"
    return None


def main():
    sources = sys.argv[1:] or sorted(os.path.basename(f)[len("one_thread_"):-4]
                                     for f in glob.glob(os.path.join(RESULTS, "one_thread_*.csv")))
    status = 0
    for tag in sources:                    # e.g. "large" or "large_r2" (repetition 2)
        src = tag.split("_r")[0]
        mpath = os.path.join(RESULTS, f"matchings_one_thread_{tag}.jsonl.gz")
        cpath = os.path.join(RESULTS, f"one_thread_{tag}.csv")
        if not os.path.exists(mpath):
            print(f"{tag}: no matchings file")
            continue
        recorded = {}
        with open(cpath) as fh:
            for r in csv.DictReader(fh):
                recorded[(r["name"], r["method"])] = int(r["singles"])
        total, bad, cache = 0, Counter(), {}
        with gzip.open(mpath, "rt") as fh:
            for line in fh:
                rec = json.loads(line)
                if rec["name"] not in cache:
                    cache = {rec["name"]: load_original(src, rec["name"])}
                men, women = cache[rec["name"]]
                problem = check(men, women, [tuple(p) for p in rec["pairs"]])
                singles = len(men) - len(rec["pairs"])
                key = (rec["name"], rec["method"])
                if problem is None and key in recorded and recorded[key] != singles:
                    problem = f"CSV records {recorded[key]} singles, matching has {singles}"
                total += 1
                if problem:
                    bad[rec["method"]] += 1
                    print(src, rec["name"], rec["method"], problem)
        print(f"{tag}: {total} matchings checked against the original data, "
              f"{total - sum(bad.values())} weakly stable and consistent, problems: {dict(bad)}", flush=True)
        status |= bool(bad)
    sys.exit(status)


if __name__ == "__main__":
    main()
