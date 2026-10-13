"""Independent weak-stability checkers, written from the definitions only, without any
helper used by the algorithms.

is_stable         sparse lists men[m] = {w: rank}, women[w] = {m: rank}; assign[m] = w or -1
is_weakly_stable  dense Instance (rank matrices)
"""


def is_stable(men, women, a):
    """Weak stability of the matching a (man -> woman or -1) for the sparse lists."""
    pw = {}
    for m, w in enumerate(a):
        if w >= 0:
            if w in pw or w not in men[m] or m not in women[w]:
                return False
            pw[int(w)] = m
    for m, prefs in enumerate(men):
        cur = int(a[m])
        for w, r in prefs.items():
            if w == cur or (cur >= 0 and r >= prefs[cur]):
                continue
            her = pw.get(w)
            if her is None or women[w][m] < women[w][her]:
                return False
    return True


def violations(inst, assign):
    """List of human-readable problems; an empty list means weakly stable."""
    problems = []
    n, m = inst.n, inst.m
    count = [0] * m
    worst = [0] * m  # worst (largest) rank among h's residents
    for s in range(n):
        h = int(assign[s])
        if h < 0:
            continue
        if not (0 <= h < m):
            problems.append(f"resident {s} assigned to unknown hospital {h}")
            continue
        if inst.r_rank[s, h] <= 0 or inst.h_rank[h, s] <= 0:
            problems.append(f"unacceptable pair ({s}, {h})")
        count[h] += 1
        worst[h] = max(worst[h], int(inst.h_rank[h, s]))
    for h in range(m):
        if count[h] > inst.cap[h]:
            problems.append(f"hospital {h} over capacity ({count[h]} > {inst.cap[h]})")
    for s in range(n):
        cur = int(assign[s])
        for h in range(m):
            rs = int(inst.r_rank[s, h])
            if rs <= 0 or h == cur:
                continue
            resident_wants = cur < 0 or rs < int(inst.r_rank[s, cur])
            if not resident_wants:
                continue
            hospital_wants = count[h] < inst.cap[h] or int(inst.h_rank[h, s]) < worst[h]
            if hospital_wants:
                problems.append(f"blocking pair ({s}, {h})")
    return problems


def is_weakly_stable(inst, assign):
    return not violations(inst, assign)
