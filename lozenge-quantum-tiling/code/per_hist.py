"""Stream all tilings of every supercell (pq<=36, compatible with L=36): record, per (m, z-histogram),
the number of bipartite tilings and the lowest coordination-model energy. Resumable (one json per cell)."""
import sys, json, os, time
from periodic import Cell, analyse, cells
os.makedirs('hist_cells', exist_ok=True)
def run(p, q, s):
    C = Cell(p, q, s); N = C.N; up = []; down_of = {}
    for n in range(N):
        i, j = C.site(n)
        e0 = (C.idx(i, j), 0); e1 = (C.idx(i, j), 1); e2 = (C.idx(i + 1, j), 2)
        up.append([e0, e1, e2]); down_of[e0] = C.idx(i, j - 1); down_of[e1] = C.idx(i - 1, j); down_of[e2] = C.idx(i, j)
    used = [False] * N; ch = [None] * N; stat = {}; cnt = [0]
    sys.setrecursionlimit(10000)
    def rec(u):
        if u == N:
            cnt[0] += 1; a = analyse(C, tuple(ch))
            if a is None: return
            key = f"{round(a['m'],6)}|" + ",".join(f"{k}:{a['hist'].get(k,0)}" for k in (3,4,5,6))
            st = stat.get(key)
            if st is None: stat[key] = [1, a['e_model'], [list(x) for x in ch]]
            else:
                st[0] += 1
                if a['e_model'] < st[1] - 1e-12: st[1] = a['e_model']; st[2] = [list(x) for x in ch]
            return
        for e in up[u]:
            d = down_of[e]
            if not used[d]: used[d] = True; ch[u] = e; rec(u + 1); used[d] = False
    rec(0); return stat, cnt[0]
for (p, q, s) in sorted(cells(36, 36), key=lambda c: c[0] * c[1]):
    f = f'hist_cells/{p}_{q}_{s}.json'
    if os.path.exists(f): continue
    t = time.time(); stat, n = run(p, q, s)
    json.dump(dict(cell=[p, q, s], ntilings=n, stat=stat), open(f, 'w'))
    print(p, q, s, n, len(stat), f"{time.time()-t:.0f}s", flush=True)
