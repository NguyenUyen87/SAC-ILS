"""Run every experiment of the paper, one stage after the other, then compute all numbers,
tables and figures from the new results.

    python run_experiments.py --list                      # stages, groups and estimated hours
    python run_experiments.py                             # everything (about 1.5 days)
    python run_experiments.py --only quick                # a group, or stage names
    python run_experiments.py --results results/xeon --workers 8
    python run_experiments.py --smoke                     # 20-minute test of every stage (results/smoke)

Each stage is a command in experiments/ that appends its rows as runs finish, so an
interrupted stage resumes where it stopped; a finished stage is marked in
<results>/stages/<stage>.done and skipped next time (delete the marker to run it again).
Output of stage s goes to the console and to <results>/logs/<s>.log. The machine is kept
awake while this script runs (Windows, macOS).

Timing: one-thread stages run --workers single-thread runs in parallel (the paper used 4
on a 10-core laptop; on a 12-core server 8 leaves room for the system). Eight-thread
stages run one instance and one method at a time and need the machine to themselves:
do not run anything else while they run. Budgets are wall-clock times, so results on a
slower or busier machine are not those of the paper; the text claims that the new
results do not support are listed at the end (analysis/out/<results>/claims.txt).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ONE = ["SACILS", "SAC", "TBLS", "TBLS+ILS", "ILS[adj]", "ILS", "CPSAT1"]
REP = ["SACILS", "TBLS", "TBLS+ILS", "ILS[adj]"]


def stages(workers, smoke=False):
    """(name, group, estimated hours, command); commands are relative to the repository.
    smoke: every stage on its first instance(s) only (with budgets scaled down by the caller)."""
    w = ["--workers", str(workers)]
    ot = ["experiments/one_thread.py"]
    s = [("selftest", "quick", 0.02, ["-m", "pytest", "-q", "tests"]),
         # deterministic analyses (no time limit)
         ("chain_enum", "quick", 0.01, ["experiments/chain_enum.py"]),
         ("chain_stats", "quick", 0.2, ["experiments/chain_stats.py"]),
         ("chain_sides", "quick", 0.3, ["experiments/chain_sides.py"]),
         ("cfg_bounds", "quick", 1.0, ["experiments/cfg_bounds.py"]),
         # parameter selection on development instances
         ("tuning", "tuning", 9 / workers, ["experiments/tuning.py", "run", *w]),
         # one thread, first seed
         ("small", "one-thread", 0.6 / workers, [*ot, "small", "--methods", "SACILS", "SAC", "TBLS", "CPSAT1", *w]),
         ("large", "one-thread", 6 / workers, [*ot, "large", "--methods", *ONE, "SACILS[path]", "SACILS[both]",
                                                "SACILS[lns]", *w]),
         ("opt", "one-thread", 5 / workers, [*ot, "opt", "--methods", *ONE, "SACILS[both]", "SACILS[lns]", *w]),
         ("scal", "one-thread", 25 / workers, [*ot, "scal", "--methods", *ONE, "SACILS[both]", "SACILS[lns]", *w]),
         ("cfg", "one-thread", 7 / workers, [*ot, "cfg", "--methods", "SACILS", "SAC", "TBLS", "--limit", "5", *w]),
         ("cpsat1_vs_gs", "one-thread", 0.05, ["experiments/cpsat1_vs_gs.py"])]
    # one thread, four more seeds (the paper's large set also has a second seed of "ILS")
    for r in (1, 2, 3, 4):
        rr = ["--rep", str(r)]
        s += [(f"small_r{r}", "repetitions", 0.2 / workers, [*ot, "small", "--methods", "SACILS", "TBLS", *rr, *w]),
              (f"large_r{r}", "repetitions", 3 / workers, [*ot, "large", "--methods", *REP, *(["ILS"] if r == 1 else []),
                                                          *rr, *w]),
              (f"opt_r{r}", "repetitions", 2.3 / workers, [*ot, "opt", "--methods", *REP, *rr, *w]),
              (f"scal_r{r}", "repetitions", 11.3 / workers, [*ot, "scal", "--methods", *REP, *rr, *w]),
              (f"cfg_r{r}", "repetitions", 4.4 / workers, [*ot, "cfg", "--methods", "SACILS", "TBLS", "--limit", "5",
                                                          *rr, *w])]
    s += [("verify", "repetitions", 0.3, ["experiments/verify.py"]),
          # published methods, 3000 iterations of TBLS on the whole benchmark
          ("tbls_small", "benchmarks", 0.5 / workers, ["experiments/tbls_benchmark.py", "small", *w]),
          ("tbls_large", "benchmarks", 5 / workers, ["experiments/tbls_benchmark.py", "large", *w]),
          ("baselines_small", "benchmarks", 5 / workers, ["experiments/baselines.py", "small", *w]),
          ("baselines_large", "benchmarks", 10 / workers, ["experiments/baselines.py", "large", *w]),
          # eight threads, nothing else on the machine
          ("repair_paths", "eight-threads", 0.8, ["experiments/repair_paths.py"]),
          ("eight_large", "eight-threads", 0.8, ["experiments/eight_threads.py", "large"]),
          ("eight_scal", "eight-threads", 2.9, ["experiments/eight_threads.py", "scal"]),
          ("eight_huge", "eight-threads", 2.1, ["experiments/eight_threads.py", "huge"]),
          ("delorme_large", "eight-threads", 0.4, ["experiments/delorme.py", "large"]),
          ("delorme_scal", "eight-threads", 1.5, ["experiments/delorme.py", "scal"]),
          ("delorme_huge", "eight-threads", 1.4, ["experiments/delorme.py", "huge"])]
    if smoke:
        first = {"experiments/one_thread.py": "2", "experiments/eight_threads.py": "2", "experiments/delorme.py": "2",
                 "experiments/tbls_benchmark.py": "1", "experiments/baselines.py": "1", "experiments/tuning.py": "2",
                 "experiments/cfg_bounds.py": "1", "experiments/chain_sides.py": "1", "experiments/chain_stats.py": "1",
                 "experiments/repair_paths.py": "1"}
        s = [(n, g, h, c + (["--first", first[c[0]]] if c[0] in first else [])) for n, g, h, c in s
             if not n.endswith("_huge")]
    return s


def run_stage(name, cmd, env, results):
    logs = os.path.join(results, "logs")
    os.makedirs(logs, exist_ok=True)
    with open(os.path.join(logs, f"{name}.log"), "a", encoding="utf-8") as log:
        head = f"== {name} start {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}"
        print(head, flush=True)
        log.write(head + "\n")
        p = subprocess.Popen([sys.executable, "-u", *cmd], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = p.wait()
        tail = f"== {name} {'done' if rc == 0 else f'FAILED (exit code {rc})'} {time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(tail, flush=True)
        log.write(tail + "\n")
    return rc == 0


def data_ok(data):
    bench = os.path.join(data, "Instances", "SMTI")
    scal = os.path.join(data, "scal", "Instances", "SMTI")
    n_b = len([d for d in os.listdir(bench) if d.startswith(("n_100_", "n_1000_"))]) if os.path.isdir(bench) else 0
    n_s = len(os.listdir(scal)) if os.path.isdir(scal) else 0
    return n_b >= 330 and n_s >= 3, f"{n_b} benchmark folders (330 expected), {n_s} scalability folders (3 expected)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="data directory (see data/get_data.py)")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", socket.gethostname().lower()),
                    help="result directory (default: results/<computer name>)")
    ap.add_argument("--workers", type=int, default=8, help="parallel single-thread runs in one-thread stages")
    ap.add_argument("--only", nargs="+", default=None, help="groups or stage names to run")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--no-analysis", action="store_true", help="do not compute the numbers and figures at the end")
    ap.add_argument("--smoke", action="store_true",
                    help="quick test of every stage: first instances only, budgets x 0.05, very large instances "
                         "skipped, results in results/smoke")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")          # Windows consoles with a legacy code page
    except AttributeError:
        pass
    if args.smoke:
        args.results = os.path.join(ROOT, "results", "smoke")
    data, results = os.path.abspath(args.data), os.path.abspath(args.results)
    todo = stages(args.workers, args.smoke)
    if args.only:
        known = {s[0] for s in todo} | {s[1] for s in todo}
        bad = [x for x in args.only if x not in known]
        if bad:
            raise SystemExit(f"unknown stage or group: {bad}; see --list")
        todo = [s for s in todo if s[0] in args.only or s[1] in args.only]
    marks = os.path.join(results, "stages")
    if args.list:
        tot = 0.0
        for name, group, hours, cmd in todo:
            done = os.path.exists(os.path.join(marks, f"{name}.done"))
            tot += 0 if done else hours
            print(f"{'done ' if done else '     '}{name:16s} {group:14s} ~{hours:5.1f} h   python {' '.join(cmd)}")
        print(f"remaining: about {tot:.0f} hours (estimates for the paper's settings with --workers {args.workers})")
        return
    ok, msg = data_ok(data)
    if not ok and any(s[0] not in ("selftest", "chain_enum") for s in todo):
        raise SystemExit(f"data missing in {data}: {msg}\nrun: python data/get_data.py")
    from sacsmti.sysinfo import keep_awake, machine
    keep_awake()
    os.makedirs(marks, exist_ok=True)
    info = machine()
    info.update({"workers": args.workers, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    with open(os.path.join(results, "machine.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    print("machine:", info, flush=True)
    env = dict(os.environ, SACSMTI_DATA=data, SACSMTI_RESULTS=results, PYTHONUTF8="1",
               SACSMTI_BUDGET_SCALE="0.05" if args.smoke else "1")
    failed = []
    for name, group, hours, cmd in todo:
        mark = os.path.join(marks, f"{name}.done")
        if os.path.exists(mark):
            print(f"== {name}: done before, skipped", flush=True)
            continue
        if run_stage(name, cmd, env, results):
            open(mark, "w").close()
        else:
            failed.append(name)
    if args.smoke:
        print("== smoke test: the analysis needs complete results and is skipped", flush=True)
    elif not args.no_analysis:
        print("== analysis: numbers, tables and figures from", results, flush=True)
        r1 = subprocess.call([sys.executable, "analysis/make_numbers.py", "--results", results, "--report"], cwd=ROOT, env=env)
        r2 = subprocess.call([sys.executable, "analysis/make_figures.py", "--results", results], cwd=ROOT, env=env)
        if r1 or r2:
            print("== analysis incomplete (some result files are missing until every stage has run)", flush=True)
    if failed:
        raise SystemExit(f"failed stages: {failed} (see {os.path.join(results, 'logs')}); rerun to retry them")
    print("all stages done")


if __name__ == "__main__":
    main()
