"""Instance input and output.

Lists: men[m] = {w: rank}, women[w] = {m: rank}; 0-based ids, ranks >= 1, equal
ranks are ties, smaller is better.
Files: the JSON format of the benchmark of Qiu (2025): {"men_preference":
{"m_i": {"w_j": rank from 0}}, "women_preference": {"w_j": {"m_i": rank from 0}}}.
"""
import json

import numpy as np


def from_matrices(men_rank_list, women_rank_list):
    """Rank matrices (0 = unacceptable) -> lists of {partner: rank} dicts."""
    R, W = np.asarray(men_rank_list), np.asarray(women_rank_list)
    men = [{int(w): int(R[m, w]) for w in np.flatnonzero(R[m] > 0)} for m in range(R.shape[0])]
    women = [{int(m): int(W[w, m]) for m in np.flatnonzero(W[w] > 0)} for w in range(W.shape[0])]
    return men, women


def from_json(d):
    """Instance file of Qiu (men_preference / women_preference, ranks from 0)."""
    n = len(d["men_preference"])
    men = [dict() for _ in range(n)]
    women = [dict() for _ in range(len(d["women_preference"]))]
    for m, pref in d["men_preference"].items():
        men[int(m[2:])] = {int(w[2:]): r + 1 for w, r in pref.items()}
    for w, pref in d["women_preference"].items():
        women[int(w[2:])] = {int(m[2:]): r + 1 for m, r in pref.items()}
    return men, women


def read_json(path):
    with open(path) as fh:
        return from_json(json.load(fh))


def write_json(path, men, women):
    d = {"men_preference": {f"m_{m}": {f"w_{w}": r - 1 for w, r in pr.items()} for m, pr in enumerate(men)},
         "women_preference": {f"w_{w}": {f"m_{m}": r - 1 for m, r in pr.items()} for w, pr in enumerate(women)}}
    with open(path, "w") as fh:
        json.dump(d, fh)
