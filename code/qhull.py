"""QMC hull with the new crystals (cell-graph SSE runs) + published points."""
import json, glob, numpy as np
ESQ, ED = -0.669437, -0.63786; lin = lambda m: ESQ + (ED - ESQ) * 6 * m
B = json.load(open('/home/claude/w/lozenge-quantum-tiling/data/big_staircase.json'))
pts = {round(6 * m, 4): [(m, e, er, 'paper')] for m, e, er in B['points']}
new = {}
for f in glob.glob('/mnt/user-data/outputs/.qdec/c_*.log'):
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l)
            if 'sdl' in r['key']: continue
            m = r['m_lieb']; e0 = r['E'][0] - r['Ezee'][0]
            new.setdefault((r['key'], round(6 * m, 4)), []).append((m, e0, r['E'][1]))
for (k, M), v in new.items():
    m = v[0][0]; e = np.mean([x[1] for x in v]); er = max(np.std([x[1] for x in v]) / np.sqrt(len(v)), max(x[2] for x in v) / np.sqrt(len(v)))
    pts.setdefault(M, []).append((m, e, er, f'{k} n={len(v)}'))
best = sorted([min(v, key=lambda t: t[1]) for v in pts.values()])
print("M/Md     dev(1e-3J)   source")
for m, e, er, s in best: print(f"{6*m:.4f}  {1e3*(e-lin(m)):+.2f}({1e3*er:.2f})  {s}")
m = np.array([b[0] for b in best]); e = np.array([b[1] for b in best]); er = np.array([b[2] for b in best])
hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0)
def stair(es): return (es[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2).argmin(1)
sel = stair(e)
print('steps:', [(round(hs[i], 3), round(6 * m[sel[i - 1]], 4), round(6 * m[sel[i]], 4)) for i in range(1, len(hs)) if sel[i] != sel[i - 1]])
rng = np.random.default_rng(1); pres = np.zeros(len(m))
for _ in range(4000):
    for k in set(stair(e + rng.normal(0, er))): pres[k] += 1
print('presence:', {round(6 * mm, 4): round(p / 4000, 3) for mm, p in zip(m, pres)})
