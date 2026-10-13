"""HRT instances; SMTI is the special case where every capacity is 1.

Residents s = 0..n-1 and hospitals h = 0..m-1 (0-based). Ranks are 1-based,
smaller is better, equal means tied, 0 means unacceptable:
    r_rank[s, h]  rank of h in s's list
    h_rank[h, s]  rank of s in h's list
Acceptability must be mutual: r_rank[s, h] > 0  <=>  h_rank[h, s] > 0.
"""
import json

import numpy as np


class Instance:
    def __init__(self, r_rank, h_rank, cap):
        self.r_rank = np.asarray(r_rank, dtype=np.int64)
        self.h_rank = np.asarray(h_rank, dtype=np.int64)
        self.cap = np.asarray(cap, dtype=np.int64)
        self.n, self.m = self.r_rank.shape
        if self.h_rank.shape != (self.m, self.n) or self.cap.shape != (self.m,):
            raise ValueError("shapes of r_rank, h_rank and cap do not match")
        if not np.array_equal(self.r_rank > 0, self.h_rank.T > 0):
            raise ValueError("acceptability is not mutual")
        # preference lists as tie blocks: blocks[v] = {rank: array of agents}
        self.r_list = [self._sorted(self.r_rank[s]) for s in range(self.n)]
        self.h_list = [self._sorted(self.h_rank[h]) for h in range(self.m)]
        self.r_blocks = [self._blocks(self.r_rank[s], self.r_list[s]) for s in range(self.n)]
        self.h_blocks = [self._blocks(self.h_rank[h], self.h_list[h]) for h in range(self.m)]
        self.L = int((self.r_rank > 0).sum())

    @staticmethod
    def _sorted(row):
        idx = np.flatnonzero(row > 0)
        return idx[np.argsort(row[idx], kind="stable")]

    @staticmethod
    def _blocks(row, lst):
        out = {}
        for x in lst:
            out.setdefault(int(row[x]), []).append(int(x))
        return out

    @property
    def is_smti(self):
        return bool((self.cap == 1).all())

    @classmethod
    def from_smti(cls, men_rank_list, women_rank_list):
        """men = residents, women = hospitals of capacity 1 (rank matrices as in smti/)."""
        men = np.asarray(men_rank_list)
        return cls(men, np.asarray(women_rank_list), np.ones(men.shape[1], dtype=np.int64))

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump({"r_rank": self.r_rank.tolist(), "h_rank": self.h_rank.tolist(),
                       "cap": self.cap.tolist()}, fh)

    @classmethod
    def from_json(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["r_rank"], d["h_rank"], d["cap"])


def size(assign):
    return int((np.asarray(assign) >= 0).sum())
