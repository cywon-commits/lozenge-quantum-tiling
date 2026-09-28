"""Streaming enumeration of periodic tilings for given cells (no storage), keep top-K per m by coordination model."""
import sys, json, time
from periodic import Cell, analyse
K = 6
def run(p, q, s, tmax=3000):
    C = Cell(p, q, s); N = C.N
    up = []; down_of = {}
    for n in range(N):
        i, j = C.site(n)
        e0 = (C.idx(i, j), 0); e1 = (C.idx(i, j), 1); e2 = (C.idx(i + 1, j), 2)
        up.append([e0, e1, e2])
        down_of[e0] = C.idx(i, j - 1); down_of[e1] = C.idx(i - 1, j); down_of[e2] = C.idx(i, j)
    used = [False] * N; ch = [None] * N; best = {}; cnt = [0, 0]; t0 = time.time(); stop = [False]
    sys.setrecursionlimit(10000)
    def rec(u):
        if stop[0]: return
        if u == N:
            cnt[0] += 1
            if cnt[0] % 20000 == 0 and time.time() - t0 > tmax: stop[0] = True
            a = analyse(C, tuple(ch))
            if a is None: return
            cnt[1] += 1; k = round(a['m'], 6); L = best.setdefault(k, [])
            if len(L) < K or a['e_model'] < L[-1][0]:
                L.append((a['e_model'], tuple(ch), a['hist'])); L.sort(key=lambda x: x[0]); del L[K:]
            return
        for e in up[u]:
            d = down_of[e]
            if not used[d]:
                used[d] = True; ch[u] = e; rec(u + 1); used[d] = False
    rec(0)
    out = [dict(cell=[p, q, s], tiling=[list(x) for x in t], m=k, e_model=em, hist=h)
           for k, L in best.items() for em, t, h in L]
    return out, cnt, stop[0], time.time() - t0
if __name__ == "__main__":
    p, q = int(sys.argv[1]), int(sys.argv[2])
    for s in range(p):
        if ((36 // q) * s) % p: continue
        out, cnt, trunc, dt = run(p, q, s)
        json.dump(out, open(f'big_{p}x{q}_s{s}.json', 'w'))
        print(f"{p}x{q} s={s}: tilings {cnt[0]} bipartite {cnt[1]} trunc {trunc} {dt:.0f}s", flush=True)
