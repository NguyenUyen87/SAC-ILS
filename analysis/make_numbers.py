"""Every number and table of the SAC-ILS manuscript, computed from the result files.

Writes <out>/numbers.tex (macros \\N{key}), <out>/tab_*.tex (table bodies) and
<out>/family.csv (all paired comparisons). The manuscript contains no hand-copied result.
claims() checks the qualitative statements of the text against the data; with --report
the failed statements are listed in <out>/claims.txt instead of stopping.

    python analysis/make_numbers.py                                  # results/reference
    python analysis/make_numbers.py --results results/local --report

Conventions
- A run is one (instance, method, seed). Main runs and the repetitions r1..r4 of
  experiments/one_thread.py are pooled; for each instance and method the number of singles is
  averaged over the available seeds, and the per-instance means are compared.
- Paired comparisons: mean difference (other - SAC-ILS) with a 95% bootstrap CI over
  instances (10,000 resamples, percentile), numbers of instances on which SAC-ILS is
  better / worse, two-sided Wilcoxon signed-rank p (zero differences discarded),
  Holm-adjusted over the whole family listed in generated/tab_family.tex.
- A checkpoint at which a method had no matching yet is reported as "no solution",
  never averaged as a number of singles.
"""
import ast
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = os.path.join(ROOT, "results", "reference")
OUT = os.path.join(HERE, "out", "reference")
OPT_SET = os.path.join(ROOT, "data", "opt_set.csv")
sys.path.insert(0, HERE)
from evalio import read_eval  # noqa: E402


def configure(results=None, out=None):
    """Set the result directory and the output directory (default: analysis/out/<name of results>)."""
    global RUNS, OUT
    if results:
        RUNS = os.path.abspath(results)
    OUT = os.path.abspath(out) if out else os.path.join(HERE, "out", os.path.basename(os.path.normpath(RUNS)))

NUM = {}
FAMILY = []
RNG = np.random.default_rng(0)
NAMES = {"SACILS": "\\sacils{}", "SAC": "SAC", "ILS[adj]": "ILS-ADJ", "TBLS+ILS": "TBLS-ILS", "TBLS": "TBLS",
         "CPSAT1": "CP-SAT, one worker", "CP-SAT": "CP-SAT", "CPSAT-M1pre": "CP-SAT, preproc.", "SACILS[path]": "stable paths",
         "ILS": "preference order", "SACILS[both]": "both sides", "SACILS[lns]": "\\sacils{}-LNS"}


def put(key, value, fmt="{:.2f}"):
    if not isinstance(value, str) and value != value:      # NaN: no value (e.g. no matching at a checkpoint)
        NUM[key] = "--"
        return
    NUM[key] = value if isinstance(value, str) else fmt.format(value)


def fmt_p(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(p)))
        return f"{p / 10 ** e:.1f}\\cdot 10^{{{e}}}"
    return f"{p:.3f}"


# ---------------------------------------------------------------- one-thread runs
def runs(src):
    """All seeds of a set: columns name, n, method, seed, singles, s10, iters, ..."""
    frames = []
    for r in range(5):
        f = os.path.join(RUNS, f"one_thread_{src}{'_r%d' % r if r else ''}.csv")
        if os.path.exists(f):
            d = read_eval(f)
            d["seed"] = r
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    assert (d.stable == 1).all(), src
    return d


def seed_means(d, col="singles"):
    """Per-instance mean over seeds; also the number of seeds per (instance, method)."""
    g = d.groupby(["name", "method"])
    return g[col].mean().unstack(), g.seed.nunique().unstack()


def compare(piv, other, family, label):
    pair = piv[["SACILS", other]].dropna()
    diff = (pair[other] - pair["SACILS"]).to_numpy(float)
    idx = RNG.integers(0, len(diff), size=(10000, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    p = wilcoxon(diff[diff != 0]).pvalue if (diff != 0).any() else 1.0
    FAMILY.append({"family": family, "label": label, "vs": other, "n": len(diff), "mean_ref": pair.SACILS.mean(),
                   "mean_other": pair[other].mean(), "diff": diff.mean(), "lo": lo, "hi": hi,
                   "better": int((diff > 0).sum()), "worse": int((diff < 0).sum()), "p": p})
    return len(FAMILY) - 1


def one_thread():
    sets = {"small": None, "large": None, "opt": None, "scal": "n", "cfg": "n"}
    data = {s: runs(s) for s in sets}
    # --- effort, seeds
    for s, d in data.items():
        for m, g in d.groupby("method"):
            put(f"{s}.{m}.seeds", str(g.seed.nunique()))
    # --- main means per set (and per n for scal)
    rows = {}
    for s in ("small", "large", "opt"):
        piv, _ = seed_means(data[s])
        rows[s] = piv
        for m in piv:
            put(f"{s}.{m}", piv[m].mean())
    for s in ("scal", "cfg"):
        for n, g in data[s].groupby("n"):
            piv, _ = seed_means(g)
            rows[(s, n)] = piv
            for m in piv:
                put(f"{s}{int(n)}.{m}", piv[m].mean(), "{:.1f}")
            s10, _ = seed_means(g, "s10")
            for m in s10:
                put(f"{s}{int(n)}.{m}.at10", s10[m].mean(), "{:.1f}")
    for key, g in data["cfg"].assign(key=data["cfg"].name.str.replace(r",i=\d+", "", regex=True)).groupby("key"):
        piv, _ = seed_means(g)
        kk = key.replace("n=", "n").replace(",L=", "L").replace(",p2=", "p")
        for m in piv:
            put(f"cfg.{kk}.{m}", piv[m].mean(), "{:.1f}")
    s10, _ = seed_means(data["large"], "s10")
    for m in s10:
        put(f"large.{m}.at10", s10[m].mean())
    # --- effort counts
    for s in ("large",):
        it = data[s].groupby("method").iters.mean()
        put("large.SACILS.perturb", it["SACILS"], "{:,.0f}")
        put("large.TBLS.iters", it["TBLS"], "{:,.0f}")
    it = data["scal"].groupby(["method", "n"]).iters.mean()
    put("scal.SACILS.perturb.min", it["SACILS"].min(), "{:,.0f}")
    put("scal.SACILS.perturb.max", it["SACILS"].max(), "{:,.0f}")
    put("scal.TBLS.iters.max", it["TBLS"].max(), "{:,.0f}")
    put("SAC.large.time", data["large"][data["large"].method == "SAC"].wall.mean(), "{:.1f}")
    put("SAC.opt.time", data["opt"][data["opt"].method == "SAC"].wall.mean(), "{:.1f}")
    # SAC-ILS with the first seed only, for the one-seed design variants (Table "variants")
    for s in ("large", "opt"):
        d0 = data[s][(data[s].method == "SACILS") & (data[s].seed == 0)]
        put(f"{s}.SACILS.seed0", d0.singles.mean())
    for n, g in data["scal"][(data["scal"].method == "SACILS") & (data["scal"].seed == 0)].groupby("n"):
        put(f"scal{int(n)}.SACILS.seed0", g.singles.mean(), "{:.1f}")
    for n, g in data["scal"][data["scal"].method == "SAC"].groupby("n"):
        put(f"SAC.scal{int(n)}.time", g.wall.mean(), "{:.1f}")
    # --- comparisons: effect of chains (RQ2)
    for s in ("large", "opt"):
        for m in ("ILS[adj]", "TBLS+ILS", "TBLS"):
            compare(rows[s], m, "Effect of the chains", s)
    for n in (2000, 5000, 10000):
        for m in ("ILS[adj]", "TBLS+ILS", "TBLS"):
            compare(rows[("scal", n)], m, "Effect of the chains", f"scal $n={n}$")
    # --- exact solver, one thread; published sets (RQ3)
    for s in ("small", "large", "opt"):
        compare(rows[s], "CPSAT1", "Exact solver, one thread", s)
    for n in (2000, 5000, 10000):
        compare(rows[("scal", n)], "CPSAT1", "Exact solver, one thread", f"scal $n={n}$")
    for n in (5000, 10000):
        compare(rows[("cfg", n)], "TBLS", "Generated configurations", f"cfg $n={n}$")
    # --- design choices (RQ4)
    for m in ("SACILS[path]", "ILS", "SACILS[both]", "SACILS[lns]"):
        fam = "Hybrid with an exact solver" if m == "SACILS[lns]" else "Design choices"
        if m in rows["large"]:
            compare(rows["large"], m, fam, "large")
        if m in ("SACILS[both]", "SACILS[lns]"):
            if m in rows["opt"]:
                compare(rows["opt"], m, fam, "opt")
            for n in (2000, 5000, 10000):
                if m in rows[("scal", n)]:
                    compare(rows[("scal", n)], m, fam, f"scal $n={n}$")
    # --- path mode against chains on the large set (first seed)
    g = data["large"][(data["large"].seed == 0) & data["large"].method.isin(["SACILS", "SACILS[path]"])]
    for m, h in g.groupby("method"):
        put(f"large.{m}.cutpct", 100 * h.cut_frac.mean(), "{:.0f}")
        put(f"large.{m}.applied", h.repairs.mean(), "{:.0f}")
    # --- seed variability on large
    for m in ("SACILS", "TBLS", "TBLS+ILS", "ILS[adj]"):
        per = data["large"][data["large"].method == m].groupby("seed").singles.mean()
        if len(per) > 1:
            put(f"large.{m}.seedmin", per.min())
            put(f"large.{m}.seedmax", per.max())
    # --- single-thread CP-SAT against its Gale-Shapley start (scalability set)
    g = pd.read_csv(os.path.join(RUNS, "cpsat1_vs_gs.csv"))
    g["gain"] = g.gs_singles - g.cpsat_singles
    for n, h in g.groupby("n"):
        put(f"cpsat1.gsgain.{int(n)}", h.gain.mean(), "{:.1f}")
        put(f"cpsat1.gsimproved.{int(n)}", str(int((h.gain > 0).sum())))
        put(f"cpsat1.gscount.{int(n)}", str(len(h)))
    # --- optimality gaps
    o = pd.read_csv(OPT_SET)
    o = o[o.status == "OPTIMAL"].copy()
    o["name"] = o.apply(lambda r: f"p1={r.p1},p2={r.p2},{'geometric' if r.g == 'Geom(1-p2)' else 'flip_geometric'},"
                                  f"i={int(r.inst)}", axis=1)
    d = data["opt"].merge(o[["name", "ilp_size"]], on="name")
    d["gap"] = d.ilp_size - (d.n - d.singles)
    assert (d.gap >= 0).all()
    put("opt.count", str(d.name.nunique()))
    for m, g in d.groupby("method"):
        put(f"opt.{m}.gap", g.gap.mean())
        put(f"opt.{m}.gapmax", g.gap.max(), "{:.0f}")
        put(f"opt.{m}.optimal", (g.gap == 0).groupby(g.seed).sum().mean(), "{:.1f}")
    # --- cfg LP bounds (lambda = 5)
    bfile = os.path.join(RUNS, "cfg_bounds.csv")
    if os.path.exists(bfile):
        b = pd.read_csv(bfile)
        b["lb_singles"] = np.ceil(b.n - b.lp - 1e-6)
        sm, _ = seed_means(data["cfg"])
        b = b.join(sm["SACILS"], on="name")
        put("cfg.bounds.count", str(len(b)))
        b["key"] = b.name.str.replace(r",i=\d+", "", regex=True)
        gaps = []
        for k, g in b.groupby("key"):
            kk = k.replace("n=", "n").replace(",L=", "L").replace(",p2=", "p")
            put(f"cfg.{kk}.lb", g.lb_singles.mean(), "{:.1f}")
            put(f"cfg.{kk}.lbcount", str(len(g)))
            gap = (g.SACILS - g.lb_singles)
            assert (gap >= -1e-9).all(), k           # a lower bound can never be beaten
            gaps.append((gap.mean(), 100 * gap.mean() / g.n.iloc[0], 100 * gap.mean() / g.lb_singles.mean()))
        put("cfg.lbgap.min", min(x[0] for x in gaps), "{:.0f}")
        put("cfg.lbgap.max", max(x[0] for x in gaps), "{:.0f}")
        put("cfg.lbgap.maxpctn", max(x[1] for x in gaps), "{:.1f}")
    return data, rows


# ---------------------------------------------------------------- published methods
def published(data):
    out = {}
    for s, n in (("small", 100), ("large", 1000)):
        names = set(data[s].name)
        b = pd.read_csv(os.path.join(RUNS, f"baselines_{s}.csv"))
        b["name"] = b.apply(lambda r: f"p1={r.p1},p2={r.p2},{'geometric' if r.g == 'Geom(1-p2)' else 'flip_geometric'},"
                                      f"i={int(r.inst)}", axis=1)
        b = b[b.name.isin(names)]
        assert (b.stable == 1).all()
        for a, g in b.groupby("alg"):
            out[(s, a)] = (g.singles.mean(), g.time.mean(), g.name.nunique())
            put(f"pub.{s}.{a}", g.singles.mean(), "{:.2f}")
            put(f"pub.{s}.{a}.time", g.time.mean(), "{:.2f}")
    return out


# ---------------------------------------------------------------- eight threads
def eight_threads():
    res = {}
    for kind in ("large", "scal", "huge"):
        f = os.path.join(RUNS, f"eight_threads_{kind}.csv")
        if not os.path.exists(f) or os.path.getsize(f) == 0:
            continue
        x = pd.read_csv(f)
        if x.empty:
            continue
        g2 = os.path.join(RUNS, f"delorme_{kind}.csv")         # CP-SAT with the preprocessing of Delorme et al.
        if os.path.exists(g2) and os.path.getsize(g2) > 0:
            x = pd.concat([x, pd.read_csv(g2)], ignore_index=True)
        assert (x.stable == 1).all(), kind
        for c in ("10", "100"):
            x[f"s{c}"] = np.where(x[f"size_{c}"] >= 0, x.n - x[f"size_{c}"], np.nan)
        res[kind] = x
        for (n, m), g in x.groupby(["n", "method"]):
            key = f"x8.{int(n)}.{m}"
            put(f"{key}.count", str(len(g)))
            put(f"{key}.at100", g.s100.mean(), "{:.2f}" if n <= 1000 else "{:.1f}")
            put(f"{key}.at10", g.s10.mean(), "{:.2f}" if n <= 1000 else "{:.1f}")
            put(f"{key}.nosol10", str(int(g.s10.isna().sum())))
            if m in ("CP-SAT", "HYB", "CPSAT-M1pre", "CPSAT-M3pre"):
                put(f"{key}.optimal", str(int((g.status == "OPTIMAL").sum())))
                put(f"{key}.build", g.build_s.mean(), "{:.0f}")
                put(f"{key}.mem", g.mem_mb.max() / 1024, "{:.1f}")
            if "heur_mem_mb" in g and g.heur_mem_mb.notna().any():
                put(f"{key}.mem", g.heur_mem_mb.max(), "{:.0f}")
        for n, g in x.groupby("n"):
            piv = g.pivot_table(index="name", columns="method", values="s100")
            if "CP-SAT" in piv and "ILSx8" in piv:
                dd = piv["CP-SAT"] - piv["ILSx8"]
                put(f"x8.{int(n)}.better", str(int((dd > 0).sum())))
                put(f"x8.{int(n)}.worse", str(int((dd < 0).sum())))
                if kind != "huge" and len(dd) >= 5:
                    piv2 = piv.rename(columns={"ILSx8": "SACILS"})
                    compare(piv2, "CP-SAT", "Exact solver, eight threads", f"{kind} $n={int(n)}$, 8 threads")
            if "CPSAT-M1pre" in piv and "ILSx8" in piv:
                dd = (piv["CPSAT-M1pre"] - piv["ILSx8"]).dropna()
                put(f"pre.{int(n)}.better", str(int((dd > 0).sum())))
                put(f"pre.{int(n)}.worse", str(int((dd < 0).sum())))
                if kind != "huge" and len(dd) >= 5:
                    piv3 = piv.rename(columns={"ILSx8": "SACILS"})
                    compare(piv3, "CPSAT-M1pre", "Exact solver, eight threads", f"{kind} $n={int(n)}$, 8 threads")
            if "CPSAT-M1pre" in g.method.values:
                h = g[g.method == "CPSAT-M1pre"]
                put(f"pre.{int(n)}.removedpct", 100 * (h.removed / h.pairs).mean(), "{:.0f}")
            if "HYB" in piv and "ILSx8" in piv:
                dd = piv["HYB"] - piv["ILSx8"]
                put(f"hyb.{int(n)}.better", str(int((dd > 0).sum())))
                put(f"hyb.{int(n)}.worse", str(int((dd < 0).sum())))
            if "HYB" in piv:
                h = g[g.method == "HYB"]
                gain = (h.size_100 - h.phase1_size)
                put(f"hyb.{int(n)}.gainmax", gain.max(), "{:.0f}")
                put(f"hyb.{int(n)}.gainmean", gain.mean(), "{:.1f}")
    return res


# ---------------------------------------------------------------- analyses
def analyses():
    txt = open(os.path.join(RUNS, "repair_paths.log")).read()
    tot = ast.literal_eval(re.search(r"TOTAL (\{.*\})", txt).group(1))
    for k, v in tot.items():
        put(f"why.{k}", str(v))
    put("why.stablepct", 100 * tot["stable_path"] / tot["aug_components"], "{:.1f}")
    put("why.worsepct", 100 * tot["worse_off"] / tot["aug_components"], "{:.1f}")
    put("why.sizepreserving", str(tot["cycles"] + tot["even_paths"]))
    assert tot["gains"] == tot["aug_components"] - tot["reverse_paths"]
    c = pd.read_csv(os.path.join(RUNS, "chain_stats.csv"))
    put("chains.count", str(len(c)))
    put("chains.worse", str(int(c.worse_off.sum())))
    put("chains.strict", str(int(c.strictly_blocked.sum())))
    assert (c.worse_off == c.strictly_blocked).all()
    put("chains.worsepct", 100 * c.worse_off.mean(), "{:.1f}")
    put("chains.edges.mean", c.edges.mean(), "{:.1f}")
    put("chains.edges.median", c.edges.median(), "{:.0f}")
    put("chains.short.mean", c.shortest_edges.mean(), "{:.1f}")
    put("chains.short.median", c.shortest_edges.median(), "{:.0f}")
    put("chains.short.le7", str(int((c.shortest_edges <= 7).sum())))
    put("chains.ee", str(int((c.worse_off == 0).sum())))
    s = pd.read_csv(os.path.join(RUNS, "chain_sides.csv"))
    put("sides.instances", str(len(s)))
    for k in ("single_w", "wchain_small", "wchain_big", "path_not_chain_small", "path_not_chain_big", "sac_singles",
              "both_singles"):
        put(f"sides.{k}", str(int(s[k].sum())))
    put("sides.wchain.instances", str(int((s.wchain_big > 0).sum())))
    f = os.path.join(RUNS, "chain_enum.csv")
    if os.path.exists(f):
        e = pd.read_csv(f)
        put("enum.instances", "6000")
        put("enum.pairs", str(len(e)))
        put("enum.nochain.m", str(int((e.chain_from_m0 == 0).sum())))
        put("enum.nochain.w", str(int((e.chain_from_wk == 0).sum())))
        nn = e[(e.chain_from_m0 == 0) & (e.chain_from_wk == 0)]
        put("enum.neither", str(len(nn)))
        put("enum.neither.n", ", ".join(str(v) for v in sorted(nn.n.unique())) if len(nn) else "--")
    # TBLS on the large configurations that are not used (p1 <= 0.97), 3000 iterations
    d = pd.read_csv(os.path.join(RUNS, "tbls_benchmark_large.csv"))
    d = d[(d.alg == "TBLS") & (d.p1 <= 0.97)]
    put("tbls.p197.max", d.groupby("p1").singles.mean().max(), "{:.2f}")
    # TBLS reproduction (3000 iterations, all instances)
    for s in ("small", "large"):
        d = pd.read_csv(os.path.join(RUNS, f"tbls_benchmark_{s}.csv"))
        d = d[d.alg == "TBLS"]
        assert (d.stable == 1).all()
        put(f"repro.{s}", d.singles.mean(), "{:.3f}")
        put(f"repro.{s}.count", f"{len(d):,}".replace(",", "{,}") if len(d) >= 10000 else str(len(d)))


def tuning():
    d = pd.read_csv(os.path.join(RUNS, "tuning.csv"))
    assert (d.stable == 1).all()
    piv = d.pivot_table(index="name", columns="config", values="singles")
    check = "random+nodes=1000"
    sel = piv.drop(columns=[check]) if check in piv else piv
    ranks = sel.rank(axis=1).mean()
    agg = d.groupby("config").agg(kicks=("kicks", "mean"), edges=("mean_edges", "mean"), cut=("cut_frac", "mean"))
    label = {"order=random": "random order (selected)", "nodes=1000": "1000 nodes", "len=8": "at most 8 men",
             "init=cky": "start from CKY", "kick=10": "$\\kappa = 10$", "deepen": "shortest chains first",
             "default": "initial", "len=32": "at most 32 men", "kick=2": "$\\kappa = 2$", "nodes=20000": "20{,}000 nodes"}
    lines = []
    for cfg in ranks.sort_values().index:
        diff = sel[cfg] - sel["default"]
        if cfg == "default":
            bw, p = "--", "--"
        else:
            bw = f"{int((diff < 0).sum())}/{int((diff > 0).sum())}"
            pv = wilcoxon(sel[cfg], sel["default"]).pvalue if (diff != 0).any() else 1.0
            p = "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"
        a = agg.loc[cfg]
        lines.append(f"{label[cfg]} & {sel[cfg].mean():.2f} & {ranks[cfg]:.2f} & {a.kicks:.0f} & {a.edges:.1f} & "
                     f"{a.cut:.2f} & {bw} & {p}\\\\")
    write("tab_tuning", "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrrrrrrr}\n\\toprule\nConfiguration & singles & rank & perturb. & edges & cut & better/worse & $p$\\\\\n"
          "\\midrule\n" + "\n".join(lines) + "\n\\botrule\n\\end{tabular*}")
    put("tuning.instances", str(len(piv)))
    if check in piv:
        dd = piv[check] - piv["order=random"]
        put("tuning.check.mean", piv[check].mean())
        put("tuning.check.better", str(int((dd < 0).sum())))
        put("tuning.check.worse", str(int((dd > 0).sum())))
        pv = wilcoxon(dd[dd != 0]).pvalue if (dd != 0).any() else 1.0
        put("tuning.check.p", fmt_p(pv))
        put("tuning.selected.mean", piv["order=random"].mean())


# ---------------------------------------------------------------- output
def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w") as fh:
        fh.write(body + "\n")


def holm():
    f = pd.DataFrame(FAMILY)
    order = np.argsort(f.p.to_numpy())
    m, run, adj = len(f), 0.0, np.empty(len(f))
    for k, i in enumerate(order):
        run = max(run, (m - k) * f.p.iloc[i])
        adj[i] = min(1.0, run)
    f["p_holm"] = adj
    lines = []
    for i, r in f.iterrows():
        key = f"cmp.{r.label}.{r.vs}".replace("$", "").replace(" ", "").replace("=", "")
        put(f"{key}.diff", r["diff"])
        put(f"{key}.ci", f"[{r.lo:.2f}, {r.hi:.2f}]")
        put(f"{key}.bw", f"{r.better}/{r.worse}")
        put(f"{key}.better", str(r.better))
        put(f"{key}.worse", str(r.worse))
        put(f"{key}.p", fmt_p(r.p_holm))
        if i == 0 or f.family.iloc[i - 1] != r.family:
            lines.append(("\\midrule\n" if i else "") + f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{r.family}}}}}\\\\")
        lines.append(f"{r.label} & {NAMES.get(r.vs, r.vs)} & {r.n} & "
                     f"{r['diff']:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {r.better}/{r.worse} & ${fmt_p(r.p_holm)}$\\\\")
    head = ("\\linespread{1}\\selectfont\\scriptsize\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
            "Set & Other method & inst. & difference [95\\% CI] & b/w & $p$\\\\\n\\midrule\n")
    tail = "\n\\botrule\n\\end{tabular}"
    starts = [i for i, l in enumerate(lines) if "multicolumn" in l]       # family headers
    cut = min(starts, key=lambda i: abs(i - len(lines) / 2))              # boundary closest to the middle
    first, second = lines[:cut], lines[cut:]
    second[0] = second[0].replace("\\midrule\n", "", 1)
    write("tab_family1", head + "\n".join(first) + tail)
    write("tab_family2", head + "\n".join(second) + tail)
    put("family.size", str(len(f)))
    f.to_csv(os.path.join(OUT, "family.csv"), index=False)


def claims(data, report=False):
    """Qualitative statements of the text, checked against the data; a failure means the text must change."""
    v = lambda k: float(NUM[k].replace(",", "").replace("{", "").replace("}", ""))
    fails = []
    def need(ok, text):
        if not ok:
            fails.append(text)
    r1000 = v("large.SACILS") / v("large.ILS[adj]")
    r10k = v("scal10000.SACILS") / v("scal10000.ILS[adj]")
    need(0.4 <= r1000 <= 0.6, f"abstract/conclusion: chains halve singles at n=1000 (ratio {r1000:.2f})")
    need(r10k < 0.2, f"abstract: more than 80% fewer at n=10,000 (ratio {r10k:.2f})")
    need(0.15 <= r10k <= 0.185, f"discussion: about a sixth at n=10,000 (ratio {r10k:.3f})")
    need(v("opt.SACILS.gap") < 0.5, "abstract: less than half a pair short of the optimum")
    for s in ("large", "opt", "scal2000", "scal5000", "scal10000"):
        a, b = v(f"{s}.ILS[adj]"), v(f"{s}.TBLS+ILS")
        need(abs(a - b) / max(a, b) <= 0.25, f"RQ2: the two controls are close on {s} ({a}, {b})")
        need(v(f"{s}.SAC") < v(f"{s}.TBLS"), f"RQ2: SAC alone beats TBLS on {s}")
        need(v(f"{s}.SACILS[both]") > v(f"{s}.SACILS"), f"RQ4: both sides worse on {s}")
    need(v("large.SACILS[lns]") > v("large.SACILS") and v("scal10000.SACILS[lns]") > v("scal10000.SACILS"),
         "RQ3: SAC-ILS-LNS worse on large and at n=10,000")
    if "x8.1000.CP-SAT.at100" in NUM:
        need(abs(v("x8.1000.CP-SAT.at100") - v("x8.1000.ILSx8.at100")) < 0.2, "RQ3: CP-SAT x8 as good at n=1000")
        need(v("x8.1000.CP-SAT.optimal") > v("x8.1000.CP-SAT.count") / 2, "RQ3: CP-SAT proves most optima at n=1000")
        for n in (5000, 10000, 50000, 100000):
            if f"x8.{n}.CP-SAT.at100" in NUM:
                need(v(f"x8.{n}.ILSx8.at100") < v(f"x8.{n}.CP-SAT.at100"), f"RQ3: SAC-ILS x8 beats CP-SAT at n={n}")
                need(v(f"x8.{n}.ILSx8.at100") <= v(f"x8.{n}.HYB.at100"), f"RQ3: SAC-ILS x8 not worse than HYB at n={n}")
    if "x8.1000.CPSAT-M1pre.at100" in NUM:
        for n in (1000, 2000):
            need(v(f"x8.{n}.CPSAT-M1pre.at100") <= v(f"x8.{n}.ILSx8.at100") + 0.05,
                 f"with preprocessing CP-SAT matches or slightly improves on SAC-ILS at n={n}")
            need(v(f"x8.{n}.ILSx8.at100") - v(f"x8.{n}.CPSAT-M1pre.at100") <= 0.3, f"... only slightly at n={n}")
        for n in (5000, 10000, 50000, 100000):
            need(v(f"x8.{n}.ILSx8.at100") < v(f"x8.{n}.CPSAT-M1pre.at100"), f"SAC-ILS beats CP-SAT with preprocessing at n={n}")
            need(NUM[f"pre.{n}.worse"] == "0" and NUM[f"pre.{n}.better"] == NUM[f"x8.{n}.CP-SAT.count"],
                 f"SAC-ILS better than CP-SAT with preprocessing on every instance at n={n}")
        for n in (1000, 2000, 5000, 10000, 50000, 100000):
            need(v(f"x8.{n}.CPSAT-M3pre.at100") >= v(f"x8.{n}.CPSAT-M1pre.at100"), f"M3 not better than M1 at n={n}")
        need(v("x8.1000.CPSAT-M1pre.optimal") >= v("x8.1000.CP-SAT.optimal"), "preprocessing proves at least as many optima")
    pub = {a: v(f"pub.large.{a}") for a in ("CKY", "HR", "MCS", "AS")}
    need(min(pub, key=pub.get) == "CKY" and max(pub, key=pub.get) == "AS", "RQ3: published range CKY..AS on large")
    need(min(pub.values()) > v("large.TBLS"), "RQ3: published methods leave more singles than TBLS")
    need(v("large.SACILS[path].cutpct") > v("large.SACILS.cutpct") and v("large.SACILS[path].applied") < v("large.SACILS.applied"),
         "path mode: more searches cut by the budget and fewer improvements applied")
    need(v("tbls.p197.max") < 0.5, "setup: little room for improvement for p1 <= 0.97")
    need(NUM["chains.short.median"] == "7", "discussion: shortest chains seven edges in the median")
    need(NUM["why.chain"] == NUM["why.stable_path"], "RQ1: all stable paths are chains")
    sm = data["small"]
    cp = sm[sm.method == "CPSAT1"]
    need(len(cp) == 120 and (cp.status == "OPTIMAL").all(), "RQ3: CP-SAT proves all 120 small optima")
    with open(os.path.join(OUT, "claims.txt"), "w") as fh:
        fh.write("".join(f"FAILED {x}\n" for x in fails) or "all claims of the text are supported by the data\n")
    if fails and not report:
        raise SystemExit("claims not supported by the data:\n  " + "\n  ".join(fails))
    for x in fails:
        print("claim not supported:", x)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=None, help="result directory (default: results/reference)")
    ap.add_argument("--out", default=None, help="output directory (default: analysis/out/<name of the result directory>)")
    ap.add_argument("--report", action="store_true", help="list unsupported claims instead of stopping")
    args = ap.parse_args()
    configure(args.results, args.out)
    os.makedirs(OUT, exist_ok=True)
    data, rows = one_thread()
    published(data)
    eight_threads()
    analyses()
    tuning()
    holm()
    claims(data, args.report)
    lines = ["% generated by make_numbers.py -- do not edit",
             "\\providecommand{\\N}[1]{\\ifcsname num@#1\\endcsname\\csname num@#1\\endcsname"
             "\\else\\textcolor{red}{??#1}\\fi}"]
    for k, v in sorted(NUM.items()):
        lines.append(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{v}}}")
    write("numbers", "\n".join(lines))
    print(len(NUM), "numbers,", len(FAMILY), "comparisons ->", OUT)


if __name__ == "__main__":
    main()
