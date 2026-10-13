# Stable augmenting chains for MAX-SMTI

Code, data references and result files for the paper *Stable augmenting chains: a local
search for maximum weakly stable matchings with ties and incomplete lists*.

The repository contains the algorithms (SAC, SAC-ILS and every method they are compared
with), one script per experiment, a runner that executes all experiments of the paper in
order, the result files from which the paper's numbers were computed, and the scripts that
compute every number, table and figure of the paper from result files.

## Layout

| Path | Content |
|---|---|
| `sacsmti/sac.py` | stable augmenting chains: chain search, SAC (`sac_search`), SAC-ILS (`sac_ils`) and its variants |
| `sacsmti/tbls.py` | TBLS (Viet et al.) and the TBLS-ILS control; Gale–Shapley start and TBLS disruption |
| `sacsmti/exact.py`, `sacsmti/ilp_delorme.py` | CP-SAT on the integer programming model; preprocessing and models M1–M4 of Delorme et al. (2019) |
| `sacsmti/lns.py` | exact neighbourhood re-optimisation (SAC-ILS-LNS, analysis of exact repairs) |
| `sacsmti/baselines/` | published methods: CKY, HR, MCS, AS, LTIU |
| `sacsmti/gen.py`, `sacsmti/io.py`, `sacsmti/stability.py` | instance generators, the benchmark's JSON format, independent stability checkers |
| `experiments/` | one script per experiment (see the table below) |
| `run_experiments.py` | runs every experiment, then the analysis |
| `analysis/make_numbers.py`, `analysis/make_figures.py` | every number, table and figure of the paper from a result directory |
| `data/get_data.py` | downloads the benchmark of Qiu et al. (2025); `data/opt_set.csv` lists the 68 instances with a proven optimum |
| `results/reference/` | result files of the paper (Apple M5 laptop; see `machine.json`) |
| `tests/` | unit tests, brute-force checks on small instances, and reference values of the paper's code |

## Installation

Python 3.9–3.12 (tested with 3.9.13 and 3.11). On Windows, install Python from python.org
and run the commands below in PowerShell (use `.venv\Scripts\python` instead of
`.venv/bin/python`).

```
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q                      # about one minute
```

`tests/test_reference.py` compares the generators and the algorithms with reference values
of the code that produced the paper's results. If it fails, the NumPy version draws other
random numbers and the generated instances differ: install the pinned versions.

## Data

```
.venv/bin/python data/get_data.py
```

downloads `Instances.rar` (2.2 GB) and `Instances-scalability.rar` (0.4 GB) from Science
Data Bank (https://www.scidb.cn/en/s/ZfauMn), checks their MD5 and unpacks the folders
used here (about 4.5 GB) into `data/`. The download resumes after an interruption. The
archives are RAR 5: unpacking uses 7-Zip, unrar or bsdtar; on Windows install 7-Zip if
needed (`winget install 7zip.7zip`). The very large instances (n = 50,000 and 100,000)
and all other generated instances are created from their seeds when needed.

## Running the experiments

```
.venv/bin/python run_experiments.py --list         # stages, groups, estimated hours
.venv/bin/python run_experiments.py --smoke        # every stage on 1-2 instances, short budgets, no analysis (about 15 min)
.venv/bin/python run_experiments.py                # everything, about 30-35 hours with 8 workers
```

Results go to `results/<computer name>/` (or `--results DIR`). Every stage appends rows
as runs finish; after an interruption, run the same command again and it continues where
it stopped. `--only GROUP_OR_STAGE ...` runs part of the pipeline. The machine is kept
awake while the runner works (Windows, macOS); switch off automatic restarts for updates.

Timing matters. All methods are compared with the same wall-clock budget, so:

* one-thread stages run `--workers` single-thread runs in parallel (default 8; the paper
  ran 4 at a time on a 10-core laptop). Keep the number at or below the number of
  physical cores minus a few for the system;
* eight-thread stages (group `eight-threads`) run one instance and one method at a time
  and need the machine to themselves: do not use the machine while they run;
* CP-SAT on the instances with n = 100,000 needs up to about 12 GB of memory.

After the last stage, the runner computes all numbers, tables and figures of the paper
from the new results into `analysis/out/<results name>/` and lists the statements of the
paper that the new results do not support in `claims.txt`. To compute them by hand:

```
.venv/bin/python analysis/make_numbers.py --results results/<name> --report
.venv/bin/python analysis/make_figures.py --results results/<name>
```

Without `--results`, both scripts use `results/reference` and reproduce the paper's
`numbers.tex`, tables and figures exactly.

## Experiments

| Paper | Stage(s) | Script | Result files |
|---|---|---|---|
| Sec. 5.1, reproduction of TBLS | `tbls_small`, `tbls_large` | `tbls_benchmark.py` | `tbls_benchmark_*.csv` |
| Sec. 5.1 and 5.5, parameter selection (Table "tuning") | `tuning` | `tuning.py` | `tuning.csv` |
| Sec. 5.2, improvements of a stalled search (Table "why") | `repair_paths` | `repair_paths.py` | `repair_paths.log` |
| Sec. 5.3–5.5, one thread (Tables "chains", "opt", "cfg", "variants", Fig. 1) | `small` … `cfg`, `*_r1` … `*_r4` | `one_thread.py` | `one_thread_<set>[_r<k>].csv` |
| Sec. 3.3 and 5.2, chain statistics and chains from single women | `chain_stats`, `chain_sides`, `chain_enum` | same names | `chain_*.csv` |
| Sec. 5.4, LP bounds of the generated configurations | `cfg_bounds` | `cfg_bounds.py` | `cfg_bounds.csv` |
| Sec. 5.4, CP-SAT with one thread against its start | `cpsat1_vs_gs` | `cpsat1_vs_gs.py` | `cpsat1_vs_gs.csv` |
| Sec. 5.4, eight threads (Table "x8", Fig. 2) | `eight_*`, `delorme_*` | `eight_threads.py`, `delorme.py` | `eight_threads_*.csv`, `delorme_*.csv` |
| Sec. 5.4, published methods (Table "published") | `baselines_small`, `baselines_large` | `baselines.py` | `baselines_*.csv` |
| independent check of all one-thread matchings | `verify` | `verify.py` | – |

Every matching is checked for weak stability when it is produced and saved to
`matchings_*.jsonl.gz`; `experiments/verify.py` checks the one-thread matchings again
against the original instance files with a checker that uses only the standard library.

## Notes on reproducibility

* Seeds: the seed of every run is fixed by its instance, method set and repetition; the
  generated instances depend only on their seeds (and on NumPy, see above).
* The methods stop on wall-clock time, and TBLS (also inside SAC-ILS) limits its repair
  step by a measured time. Runs are therefore not bit-for-bit reproducible, and on a
  different machine the numbers change; the comparisons between methods are made under
  equal conditions on each machine.
* `data/opt_set.csv` fixes the optimum set: the 68 benchmark instances (n = 1000) whose
  optimum was proven by CP-SAT on the integer programming model (600 s, warm start), with
  the optimal sizes.
