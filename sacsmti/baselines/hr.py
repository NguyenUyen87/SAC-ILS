"""HR: heuristic repair algorithm for MAX-SMTI.

Viet, Uyen, Cao, Chung. A heuristic repair algorithm for the maximum stable
marriage problem with ties and incomplete lists. AI 2021, LNCS 13151,
pp. 494-506 (2022). doi:10.1007/978-3-030-97546-3_40

No code was published; this follows Algorithms 1 and 2 of the paper.
  Algorithm 1  men-proposing Gale-Shapley for SMTI (Kiraly's variant): an
               active man proposes to a single woman of his best remaining
               tie if there is one; otherwise to the first woman wj of that
               tie, who is given to him if her partner mk has a single woman
               tied with wj (mk becomes active) or if she strictly prefers him.
               When no man is active, Improve is called (one iteration).
  Algorithm 2  Improve: every single man recovers his original list and takes
               the woman wj with the smallest h(wj) = 1/k + (r_wj(mi) -
               r_wj(mk)) (1 - c(mk)) among his candidates (r_mi(wj) <= r_mk(wj)
               or r_wj(mi) = r_wj(mk), mk = partner of wj); mk becomes single
               and active and deletes wj from his list.

Choices where the pseudocode is not explicit (checked against the example of
Section 3.2, which this implementation reproduces step by step):
  * the active man taken is the one with the smallest index; within a tie the
    woman with the smallest index is taken;
  * line 27 compares original ranks, and a man keeps his partner in his list
    (the text: "If wj is assigned to mi ... otherwise mi deletes wj");
  * k (line 14 of Algorithm 2) counts the women still in mk's list with the
    same original rank as wj.
Improve can make the matching unstable: the first candidate condition,
r_mi(wj) <= r_mk(wj), compares ranks in two different lists and lets mi take
wj although wj may strictly prefer mk, who then deletes her. The paper
reports only the share of perfect matchings and returns the matching right
after the last Improve. We return the largest weakly stable matching found at
the end of a Gale-Shapley phase (the first phase is plain Gale-Shapley, so it
always exists).

candidate="tie" is the stability-preserving reading r_wj(mi) <= r_wj(mk):
since the matching after a Gale-Shapley phase is stable and mi is single,
only r_wj(mi) = r_wj(mk) can hold, i.e. wj is indifferent between mi and mk,
and every phase then ends with a weakly stable matching.
"""
import numpy as np


def _weakly_stable(R, W, pm, pw):
    n = R.shape[0]
    for m in range(n):
        cur = pm[m]
        for w in np.flatnonzero(R[m] > 0):
            if w == cur or (cur >= 0 and R[m, w] >= R[m, cur]):
                continue
            h = pw[w]
            if h < 0 or W[w, m] < W[w, h]:
                return False
    return True


def hr(R, W, max_iters=50, max_steps=None, trace=None, candidate="paper"):
    """R[m, w], W[w, m]: ranks (1 = best, 0 = unacceptable).

    candidate: "paper" (Algorithm 2 as printed) or "tie" (see the module docstring).
    Returns a dict: assign (man -> woman or -1) of the best weakly stable
    matching found, singles, iters (Improve calls), phases, and last_stable
    (whether the matching returned by the paper's loop is weakly stable).
    """
    R = np.asarray(R)
    W = np.asarray(W)
    n = R.shape[0]
    cur = R.copy()                       # current rank lists, 0 = deleted
    pm = np.full(n, -1, dtype=np.int64)
    pw = np.full(W.shape[0], -1, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    count = np.zeros(n, dtype=np.int64)
    best, best_size, phases, steps = pm.copy(), -1, 0, 0
    max_steps = max_steps or 50 * (int((R > 0).sum()) + n) * (max_iters + 1)

    def repair(mi, mk):
        wj = pm[mk]
        pm[mk] = -1
        pm[mi], pw[wj] = wj, mi
        active[mi], active[mk] = False, True

    def improve():
        for mi in np.flatnonzero(pm < 0):
            if pm[mi] >= 0 or not (R[mi] > 0).any():
                continue
            cur[mi] = R[mi]                                  # recover the original list
            ws = np.flatnonzero(R[mi] > 0)
            cand, h = [], []
            for wj in ws:
                mk = pw[wj]
                if mk < 0:                                   # not reached after a GS phase
                    cand.append(wj)
                    h.append(-np.inf)
                    continue
                ok = W[wj, mi] == W[wj, mk]
                if candidate == "paper":
                    ok = ok or R[mi, wj] <= R[mk, wj]
                if ok:
                    k = int(((cur[mk] > 0) & (R[mk] == R[mk, wj])).sum()) or 1
                    cand.append(wj)
                    h.append(1.0 / k + (W[wj, mi] - W[wj, mk]) * (1 - count[mk]))
            if not cand:
                continue
            wj = cand[int(np.argmin(h))]
            mk = pw[wj]
            if mk < 0:
                pm[mi], pw[wj] = wj, mi
                active[mi] = False
                continue
            repair(mi, mk)
            cur[mk, wj] = 0

    it = 1
    while it <= max_iters:
        steps += 1
        if steps > max_steps:
            break
        act = np.flatnonzero(active)
        if not len(act):
            phases += 1
            size = int((pm >= 0).sum())
            if trace is not None:
                trace.append(pm.copy())
            if size > best_size and _weakly_stable(R, W, pm, pw):
                best, best_size = pm.copy(), size
            if size == n:
                break
            it += 1
            improve()
            continue
        mi = act[0]
        ws = np.flatnonzero(cur[mi] > 0)
        if not len(ws):
            active[mi] = False
            count[mi] += 1
            continue
        top = ws[cur[mi, ws] == cur[mi, ws].min()]
        free = top[pw[top] < 0]
        if len(free):
            wj = free[0]
            pm[mi], pw[wj] = wj, mi
            active[mi] = False
            continue
        wj = top[0]
        mk = pw[wj]
        tied_free = (cur[mk] > 0) & (R[mk] == R[mk, wj]) & (pw < 0)
        if tied_free.any():
            repair(mi, mk)
        if pm[mi] < 0 and W[wj, mi] < W[wj, mk]:
            repair(mi, mk)
            cur[mk, wj] = 0
        elif pm[mi] != wj:
            cur[mi, wj] = 0
    last_stable = bool(not active.any() and _weakly_stable(R, W, pm, pw))
    return {"assign": best, "singles": int((best < 0).sum()), "iters": it - 1, "phases": phases,
            "last_size": int((pm >= 0).sum()),
            "last_stable": last_stable, "steps": steps}
