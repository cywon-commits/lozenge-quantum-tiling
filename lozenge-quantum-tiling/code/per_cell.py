"""One pass over every supercell (pq<=36, compatible with L=36):
 (a) per (m, z-histogram): number of bipartite tilings and lowest coordination-model energy;
 (b) for m in TARGETS: all tilings within DE of the lowest model energy (at most CAP, lowest first)
     are re-ranked with thermodynamic-limit LSWT (Bloch, nk=6).  Resumable: one json per cell."""
import sys, json, os, time, heapq
import numpy as np
from periodic import Cell, analyse, cells
from bloch_lswt import energy
TARGETS = [1/36, 1/24, 1/18, 1/12, 1/9]; DE = 1.5e-3; CAP = 250
os.makedirs('cell_scan', exist_ok=True)
def run(p, q, s):
    C = Cell(p, q, s); N = C.N; up = []; down_of = {}
    for n in range(N):
        i, j = C.site(n)
        e0 = (C.idx(i, j), 0); e1 = (C.idx(i, j), 1); e2 = (C.idx(i + 1, j), 2)
        up.append([e0, e1, e2]); down_of[e0] = C.idx(i, j - 1); down_of[e1] = C.idx(i - 1, j); down_of[e2] = C.idx(i, j)
    used = [False] * N; ch = [None] * N; stat = {}; cnt = [0]; heaps = {t: [] for t in range(len(TARGETS))}; seq = [0]
    sys.setrecursionlimit(10000)
    def rec(u):
        if u == N:
            cnt[0] += 1; a = analyse(C, tuple(ch))
            if a is None: return
            key = f"{round(a['m'],6)}|" + ",".join(f"{k}:{a['hist'].get(k,0)}" for k in (3, 4, 5, 6))
            st = stat.get(key)
            if st is None: stat[key] = [1, a['e_model']]
            else: st[0] += 1; st[1] = min(st[1], a['e_model'])
            for t, mt in enumerate(TARGETS):
                if abs(a['m'] - mt) < 1e-6:
                    h = heaps[t]; seq[0] += 1; item = (-a['e_model'], seq[0], [list(x) for x in ch])
                    if len(h) < CAP: heapq.heappush(h, item)
                    elif -a['e_model'] > h[0][0]: heapq.heapreplace(h, item)
            return
        for e in up[u]:
            d = down_of[e]
            if not used[d]: used[d] = True; ch[u] = e; rec(u + 1); used[d] = False
    rec(0)
    cand = {}
    for t, h in heaps.items():
        if not h: continue
        L = sorted(((-x[0], x[2]) for x in h), key=lambda x: x[0]); emin = L[0][0]
        L = [x for x in L if x[0] <= emin + DE]
        res = []; seen = set()
        for em, til in L:
            E, m = energy(C, til, nk=6)
            k = round(E, 7)
            if k in seen: continue
            seen.add(k); res.append([em, E, til])
        res.sort(key=lambda x: x[1])
        cand[str(TARGETS[t])] = dict(n_within=len(L), n_distinct=len(res), best=res[:3], model_min=emin,
                                     lswt_of_model_best=[r[1] for r in res if abs(r[0] - emin) < 1e-12][:1],
                                     all=[[r[0], r[1]] for r in res])
    return stat, cnt[0], cand
for (p, q, s) in sorted(cells(36, 36), key=lambda c: c[0] * c[1]):
    f = f'cell_scan/{p}_{q}_{s}.json'
    if os.path.exists(f): continue
    t = time.time(); stat, n, cand = run(p, q, s)
    json.dump(dict(cell=[p, q, s], ntilings=n, stat=stat, cand=cand), open(f + '.tmp', 'w')); os.replace(f + '.tmp', f)
    print(p, q, s, n, len(stat), {k: v['n_distinct'] for k, v in cand.items()}, f"{time.time()-t:.0f}s", flush=True)
print('ALLDONE', flush=True)
