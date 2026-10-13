"""The 1.5-approximation of Csaji, Kiraly and Yokoi for MAX-SMTI (CKY baseline).

Csaji, Kiraly, Yokoi: "A Simple 1.5-Approximation Algorithm for a Wide Range of
Max-SMTI Problems" (arXiv:2304.02558), Section 3.1 with no critical agents and
0 < gamma < delta < 1, i.e. weak stability. Written for HRT (hospitals cloned into
cap(h) identical seats); SMTI is the case cap = 1.
"""
import numpy as np

# copy types of an edge in the extended instance
A, B0, B1, C = 0, 1, 2, 3


def cky(inst, rng):
    """Returns assign (resident -> hospital or -1)."""
    n = inst.n
    seats = [(h, j) for h in range(inst.m) for j in range(int(inst.cap[h]))]
    seat_of = {}
    for i, (h, j) in enumerate(seats):
        seat_of.setdefault(h, []).append(i)
    # arbitrary but fixed tie-breaking inside equal keys, one random value
    # per (resident, seat) pair that is acceptable
    tie = {}

    def tb(s, i):
        if (s, i) not in tie:
            tie[(s, i)] = rng.random()
        return tie[(s, i)]

    # Resident u ranks copies:  a ~gamma b0 ~(delta-gamma) b1  >  c
    # With integer ranks and gamma, delta < 1 this is, level by level,
    # a(level) > b0(level) > b1(level) > a(next level) > ... and then all c.
    proposals = []
    for s in range(n):
        lst = []
        for h in inst.r_list[s]:
            lv = int(inst.r_rank[s, h])
            for i in seat_of.get(int(h), ()):
                for t, order in ((A, 0), (B0, 1), (B1, 2)):
                    lst.append(((0, lv, order, tb(s, i)), t, i))
                lst.append(((1, lv, 0, tb(s, i)), C, i))
        lst.sort(key=lambda x: x[0])
        proposals.append([(t, i) for _, t, i in lst])

    # Seat w ranks copies:  c ~gamma b1 ~(delta-gamma) b0  >  a
    def seat_key(i, s, t):
        h = seats[i][0]
        lv = int(inst.h_rank[h, s])
        if t == A:
            return (1, lv, 0, -tb(s, i))
        return (0, lv, {C: 0, B1: 1, B0: 2}[t], -tb(s, i))

    holder = [None] * len(seats)  # (key, resident)
    nxt = [0] * n
    free = list(range(n - 1, -1, -1))
    while free:
        s = free.pop()
        if nxt[s] >= len(proposals[s]):
            continue
        t, i = proposals[s][nxt[s]]
        nxt[s] += 1
        key = seat_key(i, s, t)
        if holder[i] is None:
            holder[i] = (key, s)
        elif key < holder[i][0]:
            free.append(holder[i][1])
            holder[i] = (key, s)
        else:
            free.append(s)
    assign = np.full(n, -1, dtype=np.int64)
    for i, hd in enumerate(holder):
        if hd is not None:
            assign[hd[1]] = seats[i][0]
    return assign
