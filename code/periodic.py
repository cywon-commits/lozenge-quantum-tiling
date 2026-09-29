"""Exhaustive enumeration of PERIODIC lozenge tilings with small supercells.
Supercell lattice vectors (in triangular-lattice integer coords): v1=(p,0), v2=(s,q).
Edges are labelled (site, d) with d=0: +a1, 1: +a2, 2: +a2-a1.
A tiling = one removed edge per up triangle such that each down triangle is used once.
Only tilings that are bipartite on the supercell torus are kept (their replicas are bipartite).
For each: degree histogram, Lieb imbalance per site m, and the LSWT-fitted coordination-model energy."""
import numpy as np, itertools, sys, json
from collections import deque

EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}
C_M = 0.0304
DIRS = [(1, 0), (0, 1), (-1, 1)]


class Cell:
    def __init__(self, p, q, s):
        self.p, self.q, self.s = p, q, s; self.N = p * q

    def canon(self, i, j):
        jj = j % self.q; k = (j - jj) // self.q
        return ((i - k * self.s) % self.p, jj)

    def idx(self, i, j):
        a, b = self.canon(i, j); return a * self.q + b

    def site(self, n):
        return divmod(n, self.q)


def enumerate_tilings(C, limit=800_000):
    N = C.N
    # up triangle x=(i,j): edges (x,0), (x,1), (x-a1, 2)... use explicit incidence
    up_edges = []; down_of = {}
    for n in range(N):
        i, j = C.site(n)
        e0 = (C.idx(i, j), 0)            # (i,j)-(i+1,j)       in up(i,j) ; down(i, j-1)
        e1 = (C.idx(i, j), 1)            # (i,j)-(i,j+1)       in up(i,j) ; down(i-1, j)
        e2 = (C.idx(i + 1, j), 2)        # (i+1,j)-(i,j+1)     in up(i,j) ; down(i,j)
        up_edges.append([e0, e1, e2])
        down_of[e0] = C.idx(i, j - 1); down_of[e1] = C.idx(i - 1, j); down_of[e2] = C.idx(i, j)
    used = [False] * N; choice = [None] * N; out = []

    def rec(u):
        if len(out) >= limit: return
        if u == N:
            out.append(tuple(choice)); return
        for e in up_edges[u]:
            d = down_of[e]
            if not used[d]:
                used[d] = True; choice[u] = e; rec(u + 1); used[d] = False
    sys.setrecursionlimit(10000)
    rec(0)
    return out


def analyse(C, tiling):
    """degrees, bipartiteness on supercell torus, imbalance"""
    N = C.N; rem = set(tiling)
    adj = [[] for _ in range(N)]
    for n in range(N):
        i, j = C.site(n)
        for d, (di, dj) in enumerate(DIRS):
            if (n, d) in rem: continue
            m = C.idx(i + di, j + dj)
            adj[n].append(m); adj[m].append(n)
    z = np.array([len(a) for a in adj])
    if z.min() < 3 or z.max() > 6: return None
    c = np.zeros(N, int); c[0] = 1; q = deque([0]); seen = {0}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); c[v] = -c[u]; q.append(v)
            elif c[v] == c[u]:
                return None
    m = abs(c.sum()) / 2 / N
    e = sum(EPS[int(k)] for k in z) / N + C_M * m
    return dict(z=z, m=m, e_model=e, hist={int(k): int(v) for k, v in zip(*np.unique(z, return_counts=True))})


def cells(maxN, L):
    """supercells compatible with an LxL torus: v1=(p,0) | L, v2=(s,q), q | L, (L/q)*s % p == 0"""
    out = []
    for p in range(3, L + 1):
        if L % p: continue
        for q in range(3, L + 1):
            if L % q or p * q > maxN: continue
            for s in range(p):
                if ((L // q) * s) % p == 0:
                    out.append((p, q, s))
    return out


def replicate(C, tiling, L):
    from dice_string import Tiling
    T = Tiling(L); rem = set(tiling); T.removed = set()
    for i in range(L):
        for j in range(L):
            n = C.idx(i, j)
            for d, (di, dj) in enumerate(DIRS):
                if (n, d) in rem:
                    T.removed.add(tuple(sorted((T.s(i, j), T.s(i + di, j + dj)))))
    assert not T.check_pairing()
    return T


if __name__ == "__main__":
    maxN = int(sys.argv[1]); L = int(sys.argv[2])
    res = []; KEEP = 4
    minN = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    for (p, q, s) in cells(maxN, L):
        if p * q <= minN: continue
        C = Cell(p, q, s); tl = enumerate_tilings(C)
        n_ok = 0; best = {}
        for t in tl:
            a = analyse(C, t)
            if a is None: continue
            n_ok += 1; k = round(a['m'], 6)
            lst = best.setdefault(k, [])
            if len(lst) < KEEP or a['e_model'] < lst[-1][0]:
                lst.append((a['e_model'], t, a)); lst.sort(key=lambda x: x[0]); del lst[KEEP:]
        for k, lst in best.items():
            for em, t, a in lst:
                res.append(dict(cell=[p, q, s], tiling=[list(x) for x in t], m=a['m'], e_model=em, hist=a['hist']))
        del tl
        print(f"cell p={p} q={q} s={s}: {n_ok} bipartite, distinct m {len(best)}", flush=True)
        json.dump(res, open(f'periodic_N{maxN}_L{L}_partial.json', 'w'))
    json.dump(res, open(f'periodic_N{maxN}_L{L}.json', 'w'))
    print("total", len(res))
